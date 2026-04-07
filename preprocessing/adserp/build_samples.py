"""
AdSERP sample builder: full_page (whole screenshot + gaze map) and/or scroll_stop
(viewport crops per dwell step). Coordinates: SCALE = 1280/1422; fixations are screenshot px.

CLI: build_samples.py --mode full_page|scroll_stop|both [--trials ID ...] [--n N]
"""

import os
import re
import csv
import json
import math
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import cv2
import pandas as pd
from scipy.ndimage import gaussian_filter

_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("ADSERP_DATA_DIR", _REPO_ROOT / "datasets" / "Adserp" / "data"))
OUTPUT_DIR = Path(os.environ.get("ADSERP_SAMPLES_OUT", _REPO_ROOT / "datasets" / "Adserp" / "samples"))

SCALE = 1280 / 1422
VIEWPORT_H = round(1024 * SCALE)
VIEWPORT_W = 1280
GAZE_SIGMA = 25
MIN_DWELL_MS = 400
SCROLL_GAP_PX = 80


def get_all_trial_ids() -> list[str]:
    files = sorted((DATA_DIR / 'fixation-data').glob('*.csv'))
    return [f.stem for f in files]


def load_participants() -> dict:
    p = {}
    with open(DATA_DIR / 'participants.csv') as f:
        for row in csv.DictReader(f):
            p[row['participant_id']] = {
                'id': row['participant_id'],
                'age': int(row['age']),
                'gender': row['gender'],
                'english': row['english'],
                'education': row['education'],
                'pref_device': row['pref_device'],
                'pref_website': row['pref_website'],
                'online_shopping_freq': row['online_shopping_freq'],
            }
    return p


def load_trial_meta(trial_id: str) -> dict:
    root = ET.parse(DATA_DIR / 'trial-metadata' / f'{trial_id}.xml').getroot()
    slug = root.find('task').text.split(' | ')[1].strip()
    doc_w, doc_h = root.find('document').text.split('x')
    return {
        'slug':   slug,
        'query':  slug.replace('-', ' '),
        'doc_h_css': int(doc_h),
        'doc_w_css': int(doc_w),
        'doc_h_px': round(int(doc_h) * SCALE),
    }


def load_fixations(trial_id: str) -> pd.DataFrame:
    """Fixation table; coordinates are already screenshot pixels."""
    df = pd.read_csv(DATA_DIR / 'fixation-data' / f'{trial_id}.csv')
    return df[['timestamp', 'FPOGX', 'FPOGY', 'FPOGD']].copy()


def load_mouse(trial_id: str) -> pd.DataFrame:
    """Raw mouse event CSV."""
    df = pd.read_csv(DATA_DIR / 'mouse-movement-data' / f'{trial_id}.csv')
    return df


def get_scroll_timeline(mouse_df: pd.DataFrame) -> pd.DataFrame:
    """Scroll events: timestamp, scroll_y_px (screenshot px, CSS ypos * SCALE)."""
    sc = mouse_df[mouse_df['event'] == 'scroll'][['timestamp', 'ypos']].copy()
    sc = sc.rename(columns={'ypos': 'scroll_y_px'})
    sc['scroll_y_px'] = (sc['scroll_y_px'] * SCALE).round().astype(int)
    sc = sc.sort_values('timestamp').reset_index(drop=True)
    return sc


def get_scroll_at(timeline: pd.DataFrame, timestamp: int) -> int:
    """Scroll offset in screenshot px at or before timestamp."""
    before = timeline[timeline['timestamp'] <= timestamp]
    return int(before['scroll_y_px'].iloc[-1]) if len(before) > 0 else 0


def get_click_event(mouse_df: pd.DataFrame) -> Optional[dict]:
    """First click row; page coords use clientY * SCALE + scroll_y."""
    clicks = mouse_df[mouse_df['event'] == 'click']
    if len(clicks) == 0:
        return None
    row = clicks.iloc[0]
    return {
        'timestamp': int(row['timestamp']),
        'xpos_css':  float(row['xpos']),
        'ypos_css':  float(row['ypos']),
        'xpath':     str(row['xpath']),
    }


def resolve_click_page_coords(click: dict, timeline: pd.DataFrame) -> tuple[int, int]:
    """Map click client coords to full-page screenshot (page_x, page_y)."""
    scroll_y = get_scroll_at(timeline, click['timestamp'])
    screen_x = round(click['xpos_css'] * SCALE)
    screen_y = round(click['ypos_css'] * SCALE)
    page_x   = screen_x
    page_y   = screen_y + scroll_y
    return page_x, page_y


def classify_click(xpath: str) -> dict:
    """Infer click_type and organic_rank from xpath."""
    result = {'click_type': 'other', 'organic_rank': None}
    if '#rso' in xpath or "[@id='rso']" in xpath:
        result['click_type'] = 'organic'
        m = re.search(r"rso'\]/div\[(\d+)\]", xpath)
        if m:
            result['organic_rank'] = int(m.group(1))
        else:
            result['organic_rank'] = 1  # omitted div[1] in xpath
    elif '#tads' in xpath or "[@id='tads']" in xpath:
        result['click_type'] = 'ad_top'
    elif '#tadsb' in xpath or "[@id='tadsb']" in xpath:
        result['click_type'] = 'ad_bottom'
    elif 'vplaurlt' in xpath or 'platop' in xpath:
        result['click_type'] = 'shopping_ad'
    elif 'dimg' in xpath:
        result['click_type'] = 'image'
    return result


def build_gaze_map(fixations: pd.DataFrame,
                   img_h: int,
                   img_w: int = VIEWPORT_W,
                   sigma: int = GAZE_SIGMA,
                   y_offset: int = 0,
                   t_start: Optional[int] = None,
                   t_end:   Optional[int] = None,
                   y_min_content: int = 0) -> np.ndarray:
    """Gaussian-smoothed dwell heatmap, normalized to [0, 1]. y_min_content drops top-of-page fixations."""
    df = fixations.copy()
    if t_start is not None:
        df = df[df['timestamp'] >= t_start]
    if t_end is not None:
        df = df[df['timestamp'] <= t_end]

    if y_min_content > 0:
        df = df[df['FPOGY'] >= y_min_content]

    gaze_raw = np.zeros((img_h, img_w), dtype=np.float32)

    for _, row in df.iterrows():
        x = int(row['FPOGX'])
        y = int(row['FPOGY']) - y_offset
        d = float(row['FPOGD'])
        if 0 <= x < img_w and 0 <= y < img_h:
            gaze_raw[y, x] += d

    gaze_map = gaussian_filter(gaze_raw, sigma=sigma).astype(np.float32)

    vmax = gaze_map.max()
    if vmax > 0:
        gaze_map /= vmax

    return gaze_map


def gaze_map_to_overlay(img_bgr: np.ndarray,
                         gaze_map: np.ndarray,
                         alpha: float = 0.5) -> np.ndarray:
    """Overlay gaze heatmap on BGR image for visualization."""
    heatmap = (gaze_map * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    mask = (gaze_map > 0.01).astype(np.float32)[..., None]
    overlay = (img_bgr.astype(np.float32) * (1 - alpha * mask)
               + heatmap_color.astype(np.float32) * alpha * mask)
    return overlay.astype(np.uint8)


def detect_scroll_stops(mouse_df: pd.DataFrame,
                         timeline: pd.DataFrame,
                         click: Optional[dict],
                         min_dwell_ms: int = MIN_DWELL_MS) -> list[dict]:
    """Dwell segments between scrolls; gaps >= min_dwell_ms count as stops (incl. load→first scroll)."""
    stops = []

    t_load = int(mouse_df[mouse_df['event'].isin(['load', 'pageshow'])]['timestamp'].min())
    t_click = int(click['timestamp']) if click else int(mouse_df['timestamp'].max())

    scroll_events = timeline.to_dict('records')

    t_first_scroll = scroll_events[0]['timestamp'] if scroll_events else t_click
    initial_dwell = t_first_scroll - t_load
    if initial_dwell >= min_dwell_ms:
        stops.append({
            'scroll_y_px': 0,
            't_start': t_load,
            't_end':   t_first_scroll,
            'dwell_ms': initial_dwell,
        })

    for i in range(len(scroll_events) - 1):
        t_curr  = scroll_events[i]['timestamp']
        t_next  = scroll_events[i + 1]['timestamp']
        gap     = t_next - t_curr
        y_curr  = scroll_events[i]['scroll_y_px']
        if gap >= min_dwell_ms:
            stops.append({
                'scroll_y_px': y_curr,
                't_start': t_curr,
                't_end':   t_next,
                'dwell_ms': gap,
            })

    if scroll_events:
        last = scroll_events[-1]
        final_dwell = t_click - last['timestamp']
        if final_dwell >= min_dwell_ms:
            stops.append({
                'scroll_y_px': last['scroll_y_px'],
                't_start': last['timestamp'],
                't_end':   t_click,
                'dwell_ms': final_dwell,
            })

    if not stops:
        stops.append({
            'scroll_y_px': 0,
            't_start': t_load,
            't_end':   t_click,
            'dwell_ms': t_click - t_load,
        })

    return stops


def infer_action_from_stops(stops: list[dict],
                             step_idx: int,
                             click: Optional[dict],
                             click_page_x: int,
                             click_page_y: int) -> dict:
    """Label for this step from current vs next stop; last step is click or end."""
    is_last = (step_idx == len(stops) - 1)

    if is_last:
        if click:
            return {
                'action':   'click',
                'page_x':   click_page_x,
                'page_y':   click_page_y,
                'xpath':    click['xpath'],
                **classify_click(click['xpath']),
            }
        else:
            return {'action': 'end'}

    curr_y = stops[step_idx]['scroll_y_px']
    next_y = stops[step_idx + 1]['scroll_y_px']
    delta  = next_y - curr_y

    if delta >= 0:
        return {'action': 'scroll_down', 'delta_px': delta, 'target_scroll_y': next_y}
    else:
        return {'action': 'scroll_up',   'delta_px': abs(delta), 'target_scroll_y': next_y}


def build_fullpage_sample(trial_id: str,
                           participants: dict,
                           out_dir: Path,
                           save_viz: bool = True) -> Optional[dict]:
    """One full-page sample per trial (requires a click)."""
    pid = trial_id.split('-')[0]
    meta       = load_trial_meta(trial_id)
    fixations  = load_fixations(trial_id)
    mouse_df   = load_mouse(trial_id)
    timeline   = get_scroll_timeline(mouse_df)
    click      = get_click_event(mouse_df)

    if click is None:
        return None

    click_px, click_py = resolve_click_page_coords(click, timeline)
    click_info = {**classify_click(click['xpath']),
                  'page_x': click_px, 'page_y': click_py,
                  'xpath': click['xpath']}

    screenshot = cv2.imread(str(DATA_DIR / 'full-page-screenshots' / f'{trial_id}.png'))
    h, w = screenshot.shape[:2]

    gaze_map = build_gaze_map(fixations, img_h=h, img_w=w, y_min_content=150)

    sample_dir = out_dir / 'gaze_maps'
    sample_dir.mkdir(parents=True, exist_ok=True)
    gaze_path = sample_dir / f'{trial_id}.npz'
    np.savez_compressed(gaze_path, g=(gaze_map * 255).astype(np.uint8))

    if save_viz:
        viz_dir = out_dir / 'viz'
        viz_dir.mkdir(parents=True, exist_ok=True)
        overlay = gaze_map_to_overlay(screenshot, gaze_map, alpha=0.45)
        cv2.circle(overlay, (click_px, click_py), 18, (0, 255, 0), 3)
        cv2.putText(overlay, f"CLICK ({click_info['click_type']})",
                    (click_px + 20, click_py), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
        cv2.imwrite(str(viz_dir / f'{trial_id}.jpg'), overlay,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

    sample = {
        'sample_id': f'fullpage__{trial_id}',
        'trial_id':  trial_id,
        'format':    'full_page',
        'vlm_input': {
            'task_description': (
                f"A user with purchase intent is searching Google for: "
                f"'{meta['query']}'. "
                f"Given the full search results page and the user's gaze signal, "
                f"predict which result they would click."
            ),
            'image_path':     str(DATA_DIR / 'full-page-screenshots' / f'{trial_id}.png'),
            'gaze_map_path':  str(gaze_path),
            'participant_persona': (
                f"{participants[pid]['gender']}, "
                f"age {participants[pid]['age']}, "
                f"{participants[pid]['english']} English, "
                f"{participants[pid]['education']}, "
                f"prefers {participants[pid]['pref_device']}, "
                f"shops {participants[pid]['online_shopping_freq']} on "
                f"{participants[pid]['pref_website']}"
            ),
        },
        'label': click_info,
        'metadata': {
            'query':          meta['query'],
            'slug':           meta['slug'],
            'participant_id': pid,
            'n_fixations':    len(fixations),
            'total_gaze_ms':  int(fixations['FPOGD'].sum()),
            'doc_h_px':       h,
            'doc_w_px':       w,
            'n_scroll_events': len(timeline),
        }
    }
    return sample


def build_scrollstop_samples(trial_id: str,
                              participants: dict,
                              out_dir: Path,
                              save_viz: bool = True) -> list[dict]:
    """One sample per scroll stop (viewport crop + gaze); N samples per trial."""
    pid = trial_id.split('-')[0]
    meta       = load_trial_meta(trial_id)
    fixations  = load_fixations(trial_id)
    mouse_df   = load_mouse(trial_id)
    timeline   = get_scroll_timeline(mouse_df)
    click      = get_click_event(mouse_df)

    screenshot = cv2.imread(str(DATA_DIR / 'full-page-screenshots' / f'{trial_id}.png'))
    img_h, img_w = screenshot.shape[:2]

    stops = detect_scroll_stops(mouse_df, timeline, click)

    if not stops:
        return []

    if click:
        click_px, click_py = resolve_click_page_coords(click, timeline)
    else:
        click_px, click_py = -1, -1

    trial_out = out_dir / trial_id
    trial_out.mkdir(parents=True, exist_ok=True)

    samples = []
    history = []

    task_desc = (
        f"You are simulating a user with purchase intent searching Google for: "
        f"'{meta['query']}'. "
        f"At each step you see the current browser viewport. "
        f"Decide the next action: scroll_down / scroll_up / click."
    )

    for step_idx, stop in enumerate(stops):
        scroll_y = stop['scroll_y_px']

        y1 = max(0, scroll_y)
        y2 = min(img_h, y1 + VIEWPORT_H)
        if y2 - y1 < 100:
            continue
        viewport_crop = screenshot[y1:y2, :]

        vp_gaze = build_gaze_map(
            fixations,
            img_h=y2 - y1,
            img_w=img_w,
            y_offset=y1,
            t_start=stop['t_start'],
            t_end=stop['t_end'],
            y_min_content=150,
        )

        step_dir = trial_out / f'step_{step_idx:02d}'
        step_dir.mkdir(exist_ok=True)

        viewport_path  = step_dir / 'viewport.jpg'
        gaze_map_path  = step_dir / 'gaze_map.npz'
        cv2.imwrite(str(viewport_path), viewport_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        np.savez_compressed(gaze_map_path, g=(vp_gaze * 255).astype(np.uint8))

        if save_viz:
            overlay = gaze_map_to_overlay(viewport_crop, vp_gaze, alpha=0.45)
            label_action = infer_action_from_stops(
                stops, step_idx, click, click_px, click_py)
            if label_action['action'] == 'click':
                cy_vp = click_py - y1
                if 0 <= cy_vp < overlay.shape[0]:
                    cv2.circle(overlay, (click_px, cy_vp), 15, (0, 255, 0), 3)
            cv2.imwrite(str(step_dir / 'viz.jpg'), overlay,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])

        label_action = infer_action_from_stops(
            stops, step_idx, click, click_px, click_py)

        sample = {
            'sample_id':   f'scrollstop__{trial_id}__step{step_idx:02d}',
            'trial_id':    trial_id,
            'step_idx':    step_idx,
            'total_steps': len(stops),
            'format':      'scroll_stop',
            'vlm_input': {
                'task_description':    task_desc,
                'current_scroll_y_px': scroll_y,
                'viewport_crop_rect':  {'y1': y1, 'y2': y2, 'x1': 0, 'x2': img_w},
                'viewport_path':       str(viewport_path),
                'gaze_map_path':       str(gaze_map_path),
                'action_history':      list(history),
                'participant_persona': (
                    f"{participants[pid]['gender']}, "
                    f"age {participants[pid]['age']}, "
                    f"{participants[pid]['english']} English, "
                    f"{participants[pid]['education']}, "
                    f"shops {participants[pid]['online_shopping_freq']} on "
                    f"{participants[pid]['pref_website']}"
                ),
            },
            'label': label_action,
            'metadata': {
                'query':           meta['query'],
                'slug':            meta['slug'],
                'participant_id':  pid,
                'dwell_ms':        stop['dwell_ms'],
                'n_fixations_in_stop': int(
                    ((fixations['timestamp'] >= stop['t_start']) &
                     (fixations['timestamp'] <= stop['t_end']) &
                     (fixations['FPOGY'] >= y1) &
                     (fixations['FPOGY'] < y2)).sum()
                ),
                'doc_h_px':        img_h,
            }
        }
        samples.append(sample)

        history.append({
            'step':         step_idx,
            'scroll_y_px':  scroll_y,
            'dwell_ms':     stop['dwell_ms'],
            'action':       label_action['action'],
            **({k: v for k, v in label_action.items() if k != 'action'}),
        })

    with open(trial_out / 'steps.json', 'w') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full_page', 'scroll_stop', 'both'],
                        default='both')
    parser.add_argument('--trials', nargs='+', default=None,
                        help='Trial ids, e.g. p005-b6-t3 p004-b2-t4')
    parser.add_argument('--n', type=int, default=5,
                        help='Random sample size when --trials is omitted')
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip visualization images')
    args = parser.parse_args()

    save_viz = not args.no_viz

    if args.trials:
        trial_ids = args.trials
    else:
        all_ids = get_all_trial_ids()
        random.seed(42)
        trial_ids = random.sample(all_ids, min(args.n, len(all_ids)))

    participants = load_participants()

    fp_out  = OUTPUT_DIR / 'full_page'
    ss_out  = OUTPUT_DIR / 'scroll_stops'
    fp_out.mkdir(parents=True, exist_ok=True)
    ss_out.mkdir(parents=True, exist_ok=True)

    fp_samples = []
    ss_samples = []

    for tid in trial_ids:
        if args.mode in ('full_page', 'both'):
            s = build_fullpage_sample(tid, participants, fp_out, save_viz)
            if s:
                fp_samples.append(s)

        if args.mode in ('scroll_stop', 'both'):
            steps = build_scrollstop_samples(tid, participants, ss_out, save_viz)
            ss_samples.extend(steps)

    if fp_samples:
        fp_jsonl = fp_out / 'samples.jsonl'
        with open(fp_jsonl, 'w') as f:
            for s in fp_samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')

    if ss_samples:
        ss_jsonl = ss_out / 'samples.jsonl'
        with open(ss_jsonl, 'w') as f:
            for s in ss_samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
