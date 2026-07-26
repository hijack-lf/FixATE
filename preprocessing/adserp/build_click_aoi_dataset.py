import os
import json
import shutil
import argparse
import random
from pathlib import Path
from typing import Optional

import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent))
from build_samples import (
    get_all_trial_ids,
    load_fixations,
    load_mouse,
    get_scroll_timeline,
    get_scroll_at,
    get_click_event,
    detect_scroll_stops,
    SCALE,
    VIEWPORT_H,
    VIEWPORT_W,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("ADSERP_DATA_DIR", _REPO_ROOT / "datasets" / "Adserp" / "data"))
SAMPLES_DIR = Path(
    os.environ.get(
        "ADSERP_SAMPLES_SCROLL_STOPS",
        _REPO_ROOT / "datasets" / "Adserp" / "samples" / "scroll_stops",
    )
)
OUTPUT_DIR = Path(os.environ.get("ADSERP_CLICK_AOI_OUT", _REPO_ROOT / "datasets" / "Adserp"))


def load_organic_aois(trial_id: str) -> list[dict]:
    path = DATA_DIR / "organic-aoi-data" / f"{trial_id}.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    aois = []
    for item in data.get("organic", []):
        aois.append(
            {
                "type": "organic",
                "rank": item["rank"],
                "title": item.get("title"),
                "site": item.get("site"),
                "bbox": {
                    "x1": item["x_left"],
                    "y1": item["y_top"],
                    "x2": item["x_right"],
                    "y2": item["y_bot"],
                },
            }
        )
    return aois


def load_ad_aois(trial_id: str) -> list[dict]:
    path = DATA_DIR / "ad-boundary-data" / f"{trial_id}.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    aois = []
    for ad_type, ads in data.items():
        for ad in ads:
            x = ad["location"]["x"]
            y = ad["location"]["y"]
            w = ad["size"]["width"]
            h = ad["size"]["height"]
            aois.append(
                {
                    "type": ad_type,
                    "rank": None,
                    "title": None,
                    "site": None,
                    "bbox": {"x1": x, "y1": y, "x2": x + w, "y2": y + h},
                }
            )
    return aois


def compute_gaze_dwell(
    fixations: pd.DataFrame,
    bbox: dict,
    t_start: int,
    t_end: int,
) -> int:
    df = fixations[(fixations["timestamp"] >= t_start) & (fixations["timestamp"] <= t_end)]
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    in_aoi = (df["FPOGX"] >= x1) & (df["FPOGX"] <= x2) & (df["FPOGY"] >= y1) & (df["FPOGY"] <= y2)
    return int(df.loc[in_aoi, "FPOGD"].sum())


def compute_cursor_dwell(
    mouse_df: pd.DataFrame,
    bbox: dict,
    scroll_timeline: pd.DataFrame,
    t_start: int,
    t_end: int,
) -> int:
    events = mouse_df[
        mouse_df["event"].isin(["mousemove", "mouseover"])
        & (mouse_df["timestamp"] >= t_start)
        & (mouse_df["timestamp"] <= t_end)
    ].sort_values("timestamp")

    if len(events) < 2:
        return 0

    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    total_ms = 0
    rows = events[["timestamp", "xpos", "ypos"]].values

    for i in range(len(rows) - 1):
        t_i, cx, cy = rows[i]
        t_i1, _, _ = rows[i + 1]
        scroll_y = get_scroll_at(scroll_timeline, int(t_i))
        page_x = round(cx * SCALE)
        page_y = round(cy * SCALE) + scroll_y
        if x1 <= page_x <= x2 and y1 <= page_y <= y2:
            total_ms += int(t_i1) - int(t_i)

    return total_ms


def find_clicked_aoi(aois: list[dict], click_label: dict) -> Optional[int]:
    if click_label.get("action") != "click":
        return None

    page_x = click_label.get("page_x")
    page_y = click_label.get("page_y")

    if click_label.get("click_type") == "organic" and click_label.get("organic_rank"):
        for i, aoi in enumerate(aois):
            if aoi["type"] == "organic" and aoi["rank"] == click_label["organic_rank"]:
                return i

    if page_x is not None and page_y is not None:
        for i, aoi in enumerate(aois):
            b = aoi["bbox"]
            if b["x1"] <= page_x <= b["x2"] and b["y1"] <= page_y <= b["y2"]:
                return i

    return None


def build_sample(trial_id: str) -> Optional[tuple[dict, Path]]:
    steps_path = SAMPLES_DIR / trial_id / "steps.json"
    if not steps_path.exists():
        return None

    with open(steps_path) as f:
        steps = json.load(f)

    click_steps = [s for s in steps if s["label"]["action"] == "click"]
    if not click_steps:
        return None

    step = click_steps[-1]
    step_idx = step["step_idx"]

    try:
        mouse_df = load_mouse(trial_id)
        timeline = get_scroll_timeline(mouse_df)
        click = get_click_event(mouse_df)
    except Exception:
        return None

    if click is None:
        return None

    stops = detect_scroll_stops(mouse_df, timeline, click)
    if step_idx >= len(stops):
        return None

    stop = stops[step_idx]
    t_start = stop["t_start"]
    t_end = stop["t_end"]
    scroll_y = stop["scroll_y_px"]

    try:
        fixations = load_fixations(trial_id)
    except Exception:
        return None

    organic_aois = load_organic_aois(trial_id)
    ad_aois = load_ad_aois(trial_id)
    all_aois = organic_aois + ad_aois

    viewport_y1 = scroll_y
    viewport_y2 = scroll_y + VIEWPORT_H

    for i, aoi in enumerate(all_aois):
        aoi["aoi_id"] = i
        aoi["gaze_dwell_ms"] = compute_gaze_dwell(fixations, aoi["bbox"], t_start, t_end)
        aoi["cursor_dwell_ms"] = compute_cursor_dwell(mouse_df, aoi["bbox"], timeline, t_start, t_end)
        b = aoi["bbox"]
        aoi["visible_in_viewport"] = not (b["y2"] < viewport_y1 or b["y1"] > viewport_y2)

    click_label = step["label"]
    clicked_aoi_id = find_clicked_aoi(all_aois, click_label)

    step_dir = SAMPLES_DIR / trial_id / f"step_{step_idx:02d}"
    viewport_src = step_dir / "viewport.jpg"
    if not viewport_src.exists():
        return None

    pid = trial_id.split("-")[0]
    meta = step["metadata"]

    sample = {
        "sample_id": trial_id,
        "trial_id": trial_id,
        "user_id": pid,
        "query": meta["query"],
        "scroll_y_px": scroll_y,
        "doc_h_px": meta["doc_h_px"],
        "viewport_rect": {
            "x1": 0,
            "y1": scroll_y,
            "x2": VIEWPORT_W,
            "y2": scroll_y + VIEWPORT_H,
        },
        "image_path": f"images/{trial_id}.jpg",
        "dwell_ms": stop["dwell_ms"],
        "aois": all_aois,
        "clicked_aoi_id": clicked_aoi_id,
        "click": {
            "page_x": click_label.get("page_x"),
            "page_y": click_label.get("page_y"),
            "click_type": click_label.get("click_type"),
            "organic_rank": click_label.get("organic_rank"),
            "xpath": click_label.get("xpath"),
        },
    }

    return sample, viewport_src


def main():
    parser = argparse.ArgumentParser(description="Build click-AOI sub-dataset from AdSERP")
    parser.add_argument("--trials", nargs="+", help="Specific trial IDs")
    parser.add_argument("--n", type=int, help="Randomly sample N trials")
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="all",
        help="Predefined split file under DATA_DIR/splits/",
    )
    parser.add_argument("--out", default=str(OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    if args.trials:
        trial_ids = args.trials
    elif args.split != "all":
        split_file = DATA_DIR / "splits" / f"{args.split}_trials.txt"
        with open(split_file) as sf:
            trial_ids = sf.read().splitlines()
    else:
        trial_ids = get_all_trial_ids()

    if args.n:
        random.seed(42)
        trial_ids = random.sample(trial_ids, min(args.n, len(trial_ids)))

    trial_ids = sorted(trial_ids)

    samples = []
    n_skipped = 0
    n_no_organic = 0

    for trial_id in trial_ids:
        result = build_sample(trial_id)
        if result is None:
            n_skipped += 1
            continue

        sample, viewport_src = result

        dst = img_dir / f"{trial_id}.jpg"
        if not dst.exists():
            shutil.copy2(viewport_src, dst)

        has_organic = any(a["type"] == "organic" for a in sample["aois"])
        if not has_organic:
            n_no_organic += 1

        samples.append(sample)

    jsonl_path = out_dir / "samples.jsonl"
    with open(jsonl_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    index = []
    for s in samples:
        index.append(
            {
                "sample_id": s["sample_id"],
                "user_id": s["user_id"],
                "query": s["query"],
                "n_aois": len(s["aois"]),
                "n_organic_aois": sum(1 for a in s["aois"] if a["type"] == "organic"),
                "n_ad_aois": sum(1 for a in s["aois"] if a["type"] != "organic"),
                "clicked_aoi_id": s["clicked_aoi_id"],
                "click_type": s["click"]["click_type"],
                "dwell_ms": s["dwell_ms"],
                "image_path": s["image_path"],
            }
        )

    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    stats = {
        "total_samples": len(samples),
        "skipped_trials": n_skipped,
        "trials_without_organic": n_no_organic,
        "click_type_distribution": {},
        "user_distribution": {},
    }
    for s in samples:
        ct = s["click"]["click_type"] or "unknown"
        stats["click_type_distribution"][ct] = stats["click_type_distribution"].get(ct, 0) + 1
        uid = s["user_id"]
        stats["user_distribution"][uid] = stats["user_distribution"].get(uid, 0) + 1

    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
