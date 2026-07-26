from __future__ import annotations
import csv
import os
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import argparse
import sys
from dataclasses import dataclass, field


_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_RECGAZE_DIR = os.path.join(_ROOT, "datasets", "RecGaze")
SUMMARY_PATH = os.path.join(_RECGAZE_DIR, "raw/summary_feedback.csv")
sf_NON_PUBLIC_PATH = os.path.join(_RECGAZE_DIR, "raw/non_public_feedback_dataset.csv")
sf_AOI_PATH = os.path.join(_RECGAZE_DIR, "raw/aoi_data.csv")
OUTPUT_DIR = _RECGAZE_DIR
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "summary_feedback_page_divided.csv")
NEW_COL = "Fixation_AOI_Visible_Carousel_rows"
HORIZONTAL_COL = "Fixation_AOI_Visible_Carousel_horizontal_pages"
HORIZONTAL_ALL_COL = "Fixation_AOI_All_Carousel_horizontal_pages"


def sf_to_float(v: str) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def normalize_task_id(v: str) -> Optional[int]:
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def parse_page(v: str) -> Optional[int]:
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def parse_int_like(v: str) -> Optional[int]:
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def clamp(v: int, low: int, high: int) -> int:
    return max(low, min(high, v))


def sf_load_task_row_geometry(aoi_path: str) -> Dict[int, Tuple[List[float], float]]:
    task_row_tops: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    task_row_heights: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

    with open(aoi_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("AOI_type") != "Movie":
                continue
            if parse_page(row.get("Page", "")) != 1:
                continue

            task_id = normalize_task_id(row.get("task_id", ""))
            row_num = normalize_task_id(row.get("Row_num", ""))
            min_y = sf_to_float(row.get("AOI_min_y", ""))
            max_y = sf_to_float(row.get("AOI_max_y", ""))

            if task_id is None or row_num is None or min_y is None or max_y is None:
                continue

            task_row_tops[task_id][row_num].append(min_y)
            task_row_heights[task_id][row_num].append(max_y - min_y)

    task_geom: Dict[int, Tuple[List[float], float]] = {}
    global_row_tops: Optional[List[float]] = None
    global_height: Optional[float] = None

    for task_id in sorted(task_row_tops.keys()):
        row_top_map = task_row_tops[task_id]
        row_height_map = task_row_heights[task_id]

        tops_sorted: List[float] = []
        heights_for_poster: List[float] = []
        for row_num in range(1, 11):
            if row_num not in row_top_map or not row_top_map[row_num]:
                continue
            tops_sorted.append(statistics.median(row_top_map[row_num]))
            if row_num in row_height_map and row_height_map[row_num]:
                heights_for_poster.append(statistics.median(row_height_map[row_num]))

        if len(tops_sorted) != 10 or not heights_for_poster:
            continue

        poster_height = statistics.median(heights_for_poster)
        task_geom[task_id] = (tops_sorted, poster_height)
        global_row_tops = tops_sorted
        global_height = poster_height

    if global_row_tops is not None and global_height is not None:
        for task_id in range(1, 41):
            if task_id not in task_geom:
                task_geom[task_id] = (list(global_row_tops), global_height)

    return task_geom


def sf_compute_top_row(scroll_y: float, row_tops: List[float], poster_height: float) -> int:
    half = poster_height / 2.0
    top_row = 1
    for r in range(1, 10):
        if scroll_y > row_tops[r - 1] + half:
            top_row = r + 1
        else:
            break
    return clamp(top_row, 1, 8)


def visible_rows_str(scroll_y: float, task_id: int, task_geom: Dict[int, Tuple[List[float], float]]) -> str:
    if task_id not in task_geom:
        return ""
    row_tops, poster_height = task_geom[task_id]
    if len(row_tops) < 10 or poster_height <= 0:
        return ""
    top = sf_compute_top_row(scroll_y, row_tops, poster_height)
    return f"{top},{top + 1},{top + 2}"


def row_key3(row: Dict[str, str]) -> Tuple[str, str, str]:
    return (
        row.get("UserID", ""),
        row.get("TaskID", ""),
        row.get("Timestamp", ""),
    )


def row_key4(row: Dict[str, str], occ: int) -> Tuple[str, str, str, int]:
    k3 = row_key3(row)
    return (k3[0], k3[1], k3[2], occ)


def parse_visible_rows(visible_rows: str) -> List[int]:
    out: List[int] = []
    s = str(visible_rows).strip()
    if not s:
        return out
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except ValueError:
            continue
    return out


def has_fixation_non_public(row: Dict[str, str]) -> bool:
    return bool(str(row.get("Fixation_LocY", "")).strip()) or bool(str(row.get("Fixation_Duration", "")).strip())


def build_fixation_visible_rows_map(
    non_public_path: str, task_geom: Dict[int, Tuple[List[float], float]]
) -> Dict[Tuple[str, str, str, int], str]:
    out_map: Dict[Tuple[str, str, str, int], str] = {}
    occ_counter = defaultdict(int)
    indexed: List[Tuple[Dict[str, str], int]] = []

    with open(non_public_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        def attach_occ(r: Dict[str, str]) -> Tuple[Dict[str, str], int]:
            k3 = row_key3(r)
            occ = occ_counter[k3]
            occ_counter[k3] += 1
            return r, occ

        for raw in reader:
            indexed.append(attach_occ(raw))

    n = len(indexed)
    for i in range(n):
        cur_row, occ = indexed[i]
        if not has_fixation_non_public(cur_row):
            continue
        fix_y = sf_to_float(cur_row.get("Fixation_LocY", ""))
        if fix_y is None:
            continue
        task_id = normalize_task_id(cur_row.get("TaskID", ""))
        if task_id is None:
            continue
        uid = cur_row.get("UserID", "")
        tid = cur_row.get("TaskID", "")
        gaze_y: Optional[float] = None
        for offset in (1, 2, 3):
            j = i + offset
            if j >= n:
                break
            cand_row, _ = indexed[j]
            if cand_row.get("UserID", "") != uid or cand_row.get("TaskID", "") != tid:
                continue
            gy = sf_to_float(cand_row.get("Gaze_LocY", ""))
            if gy is not None:
                gaze_y = gy
                break
        if gaze_y is None:
            continue
        scroll_y = fix_y - gaze_y
        out_map[row_key4(cur_row, occ)] = visible_rows_str(scroll_y, task_id, task_geom)

    return out_map


def build_horizontal_pages_map(
    summary_path: str, visible_map: Dict[Tuple[str, str, str, int], str]
) -> Tuple[
    Dict[Tuple[str, str, str, int], str],
    Dict[Tuple[str, str, str, int], str],
]:
    out_map: Dict[Tuple[str, str, str, int], str] = {}
    out_all_map: Dict[Tuple[str, str, str, int], str] = {}
    occ_counter = defaultdict(int)
    horizontal_state: Dict[Tuple[str, str, int], int] = defaultdict(int)

    with open(summary_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k3 = row_key3(row)
            occ = occ_counter[k3]
            occ_counter[k3] += 1
            k4 = row_key4(row, occ)

            uid = row.get("UserID", "")
            tid = row.get("TaskID", "")

            visible_rows = parse_visible_rows(visible_map.get(k4, ""))
            if visible_rows:
                pages = [str(horizontal_state[(uid, tid, r)]) for r in visible_rows]
                out_map[k4] = ",".join(pages)

            all_states = [str(horizontal_state[(uid, tid, r)]) for r in range(1, 11)]
            out_all_map[k4] = ",".join(all_states)

            click_type = str(row.get("Click_AOI_type", "")).strip().lower()
            carousel_pos = parse_int_like(row.get("Click_AOI_Carousel_position", ""))
            if carousel_pos is None or not (1 <= carousel_pos <= 10):
                continue
            key = (uid, tid, carousel_pos)
            if click_type == "forward":
                horizontal_state[key] = (horizontal_state[key] + 1) % 3
            elif click_type == "backward":
                horizontal_state[key] = (horizontal_state[key] - 1) % 3

    return out_map, out_all_map


def process_summary_and_write(
    summary_path: str,
    output_path: str,
    visible_map: Dict[Tuple[str, str, str, int], str],
    horizontal_map: Dict[Tuple[str, str, str, int], str],
    horizontal_all_map: Dict[Tuple[str, str, str, int], str],
    new_col: str,
    horizontal_col: str,
    horizontal_all_col: str,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(summary_path, "r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        original_cols = reader.fieldnames or []
        if "Fixation_AOI_Carousel_position" not in original_cols:
            raise ValueError("summary_feedback.csv missing column: Fixation_AOI_Carousel_position")

        insert_idx = original_cols.index("Fixation_AOI_Carousel_position") + 1
        out_cols = (
            original_cols[:insert_idx]
            + [new_col, horizontal_col, horizontal_all_col]
            + original_cols[insert_idx:]
        )

        occ_counter = defaultdict(int)

        with open(output_path, "w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=out_cols)
            writer.writeheader()

            for row in reader:
                k3 = row_key3(row)
                occ = occ_counter[k3]
                occ_counter[k3] += 1
                k4 = (k3[0], k3[1], k3[2], occ)
                row[new_col] = visible_map.get(k4, "")
                row[horizontal_col] = horizontal_map.get(k4, "")
                row[horizontal_all_col] = horizontal_all_map.get(k4, "")
                writer.writerow(row)


def run_summary_pages() -> None:
    task_geom = sf_load_task_row_geometry(sf_AOI_PATH)
    if not task_geom:
        raise RuntimeError("no valid row geometry extracted from aoi_data.csv")

    visible_map = build_fixation_visible_rows_map(sf_NON_PUBLIC_PATH, task_geom)
    horizontal_map, horizontal_all_map = build_horizontal_pages_map(SUMMARY_PATH, visible_map)
    process_summary_and_write(
        SUMMARY_PATH,
        OUTPUT_PATH,
        visible_map,
        horizontal_map,
        horizontal_all_map,
        NEW_COL,
        HORIZONTAL_COL,
        HORIZONTAL_ALL_COL,
    )

GroupKey = Tuple[str, str, str, str, str, int]
SessionKey = Tuple[str, str, str]


INPUT_FILTER_PATH = os.path.join(_RECGAZE_DIR, "summary_feedback_page_divided_filter.csv"
)
INPUT_SUMMARY_PATH = os.path.join(_RECGAZE_DIR, "summary_feedback_page_divided.csv"
)
NON_PUBLIC_PATH = os.path.join(_RECGAZE_DIR, "raw/non_public_feedback_dataset.csv"
)
AOI_PATH = os.path.join(_RECGAZE_DIR, "raw/aoi_data.csv")
BASE_OUTPUT_DIR = _RECGAZE_DIR


RULE_CHOICES = ("50", "100", "2row", "real")

RULE_CONFIG = {
    "50": {
        "subdir": None,
        "requires_scroll": False,
        "output_filename": "human_attention_page_divided_50%split.csv",
    },
    "100": {
        "subdir": "page_divided_100%split",
        "requires_scroll": True,
        "output_filename": "human_attention_page_divided_100%split.csv",
    },
    "2row": {
        "subdir": "page_divided_2row",
        "requires_scroll": True,
        "output_filename": "human_attention_page_divided_2row.csv",
    },
    "real": {
        "subdir": "page_divide_real",
        "requires_scroll": True,
        "output_filename": "human_attention_page_divided_real.csv",
    },
}

MAX_SLOTS = 20

REAL_DURATION_THRESHOLD_MS = 150.0
REAL_SCROLL_TOLERANCE_PX = 17.0

REAL_PARTIAL_MIN_PX = 3.0
REAL_FULL_VISIBLE_PX = 192.5
REAL_VIEWPORT_H = 1080.0


def output_path_for(rule: str) -> str:
    cfg = RULE_CONFIG[rule]
    filename = cfg["output_filename"]
    if cfg["subdir"]:
        return os.path.join(BASE_OUTPUT_DIR, cfg["subdir"], filename)
    return os.path.join(BASE_OUTPUT_DIR, filename)


def to_float(v: str) -> Optional[float]:
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def to_int_like(v: str) -> Optional[int]:
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def parse_triplet(s: str) -> Optional[List[int]]:
    raw = str(s).strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if not parts:
        return None
    out: List[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            return None
    return out


def load_task_row_geometry(
    aoi_path: str,
) -> Dict[int, Tuple[List[float], float]]:
    task_row_tops: Dict[int, Dict[int, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    task_row_heights: Dict[int, Dict[int, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    with open(aoi_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("AOI_type") != "Movie":
                continue
            if parse_page(row.get("Page", "")) != 1:
                continue
            task_id = to_int_like(row.get("task_id", ""))
            row_num = to_int_like(row.get("Row_num", ""))
            min_y = to_float(row.get("AOI_min_y", ""))
            max_y = to_float(row.get("AOI_max_y", ""))
            if (
                task_id is None
                or row_num is None
                or min_y is None
                or max_y is None
            ):
                continue
            task_row_tops[task_id][row_num].append(min_y)
            task_row_heights[task_id][row_num].append(max_y - min_y)

    task_geom: Dict[int, Tuple[List[float], float]] = {}
    global_row_tops: Optional[List[float]] = None
    global_height: Optional[float] = None
    for task_id in sorted(task_row_tops.keys()):
        row_top_map = task_row_tops[task_id]
        row_height_map = task_row_heights[task_id]
        tops_sorted: List[float] = []
        heights_for_poster: List[float] = []
        for row_num in range(1, 11):
            if row_num not in row_top_map or not row_top_map[row_num]:
                continue
            tops_sorted.append(statistics.median(row_top_map[row_num]))
            if row_num in row_height_map and row_height_map[row_num]:
                heights_for_poster.append(
                    statistics.median(row_height_map[row_num])
                )
        if len(tops_sorted) != 10 or not heights_for_poster:
            continue
        poster_height = statistics.median(heights_for_poster)
        task_geom[task_id] = (tops_sorted, poster_height)
        global_row_tops = tops_sorted
        global_height = poster_height

    if global_row_tops is not None and global_height is not None:
        for task_id in range(1, 41):
            if task_id not in task_geom:
                task_geom[task_id] = (list(global_row_tops), global_height)
    return task_geom


def compute_top_row(
    scroll_y: float, row_tops: List[float], poster_height: float
) -> int:
    half = poster_height / 2.0
    top_row = 1
    for r in range(1, 10):
        if scroll_y > row_tops[r - 1] + half:
            top_row = r + 1
        else:
            break
    return max(1, min(8, top_row))


def build_fixation_scroll_map(
    non_public_path: str,
) -> Dict[Tuple[str, str, str, int], float]:
    out_map: Dict[Tuple[str, str, str, int], float] = {}
    occ_counter: Dict[Tuple[str, str, str], int] = defaultdict(int)
    indexed: List[Tuple[Dict[str, str], int]] = []

    with open(non_public_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k3 = (
                row.get("UserID", ""),
                row.get("TaskID", ""),
                row.get("Timestamp", ""),
            )
            occ = occ_counter[k3]
            occ_counter[k3] += 1
            indexed.append((row, occ))

    n = len(indexed)
    for i in range(n):
        cur_row, occ = indexed[i]
        fix_y_raw = str(cur_row.get("Fixation_LocY", "")).strip()
        dur_raw = str(cur_row.get("Fixation_Duration", "")).strip()
        if not (fix_y_raw or dur_raw):
            continue
        fix_y = to_float(fix_y_raw)
        if fix_y is None:
            continue
        uid = cur_row.get("UserID", "")
        tid = cur_row.get("TaskID", "")
        gaze_y: Optional[float] = None
        for offset in (1, 2, 3):
            j = i + offset
            if j >= n:
                break
            cand_row, _ = indexed[j]
            if (
                cand_row.get("UserID", "") != uid
                or cand_row.get("TaskID", "") != tid
            ):
                continue
            gy = to_float(cand_row.get("Gaze_LocY", ""))
            if gy is not None:
                gaze_y = gy
                break
        if gaze_y is None:
            continue
        scroll_y = fix_y - gaze_y
        key = (uid, tid, cur_row.get("Timestamp", ""), occ)
        out_map[key] = scroll_y
    return out_map


def resolve_effective_page(
    *,
    rule: str,
    task_id: Optional[int],
    scroll_y: Optional[float],
    task_geom: Optional[Dict[int, Tuple[List[float], float]]],
    raw_visible: Optional[List[int]],
    raw_horizontal: Optional[List[int]],
    all_horizontal: Optional[List[int]] = None,
) -> Optional[Tuple[List[int], List[int]]]:
    if raw_visible is None or raw_horizontal is None:
        return None
    if len(raw_visible) != 3 or len(raw_horizontal) != 3:
        return None

    if rule == "50":
        return list(raw_visible), list(raw_horizontal)

    if task_id is None or task_geom is None or scroll_y is None:
        return None
    if task_id not in task_geom:
        return None
    row_tops, poster_height = task_geom[task_id]
    if len(row_tops) < 10 or poster_height <= 0:
        return None

    if rule == "real":
        if all_horizontal is None or len(all_horizontal) < 10:
            return list(raw_visible), list(raw_horizontal)

        candidate: List[int] = []
        candidate_pages: List[int] = []
        for r_idx in range(1, len(row_tops) + 1):
            top_canvas = row_tops[r_idx - 1]
            y_top_raw = top_canvas - scroll_y
            y_bot_raw = y_top_raw + poster_height
            y_top_c = max(0.0, min(REAL_VIEWPORT_H, y_top_raw))
            y_bot_c = max(0.0, min(REAL_VIEWPORT_H, y_bot_raw))
            h_visible = y_bot_c - y_top_c
            if h_visible >= REAL_PARTIAL_MIN_PX:
                candidate.append(r_idx)
                if 0 <= (r_idx - 1) < len(all_horizontal):
                    candidate_pages.append(all_horizontal[r_idx - 1])
                else:
                    candidate_pages.append(0)
        if not candidate:
            return None
        max_rows = MAX_SLOTS // 5
        if len(candidate) > max_rows:
            candidate = candidate[:max_rows]
            candidate_pages = candidate_pages[:max_rows]
        return candidate, candidate_pages

    summary_top = raw_visible[0]
    if 1 <= summary_top <= len(row_tops):
        top = summary_top
    else:
        top = compute_top_row(scroll_y, row_tops, poster_height)

    crop_top = scroll_y - row_tops[top - 1]
    spacing = row_tops[1] - row_tops[0] if len(row_tops) >= 2 else poster_height
    gap = max(0.0, spacing - poster_height)
    eps = max(1.0, spacing * 0.05)

    aligned = (-gap - eps) <= crop_top <= eps
    top_cropped = crop_top > eps
    prev_intrude = crop_top < -gap - eps

    if top <= 1 and prev_intrude:
        aligned = True
        prev_intrude = False

    if rule == "100":
        if aligned:
            return list(raw_visible), list(raw_horizontal)
        return None

    if rule == "2row":
        if aligned:
            return list(raw_visible), list(raw_horizontal)
        if top_cropped:
            return raw_visible[1:], raw_horizontal[1:]
        if prev_intrude:
            return raw_visible[:2], raw_horizontal[:2]
        return raw_visible[:2], raw_horizontal[:2]

    raise ValueError(f"Unknown rule: {rule}")


def compute_local_index(
    carousel_pos: int,
    movie_pos: int,
    visible_rows: List[int],
    horizontal_pages: List[int],
) -> Optional[int]:
    if carousel_pos not in visible_rows:
        return None
    row_rank = visible_rows.index(carousel_pos)
    if row_rank >= len(horizontal_pages):
        return None
    page_state = horizontal_pages[row_rank]
    if page_state not in (0, 1, 2):
        return None
    movie_start = page_state * 5 + 1
    movie_end = movie_start + 4
    if not (movie_start <= movie_pos <= movie_end):
        return None
    col = movie_pos - movie_start + 1
    return row_rank * 5 + col


def run_page_divided(rule: str) -> None:
    if rule not in RULE_CONFIG:
        raise ValueError(f"Unsupported rule: {rule}. Choose from {RULE_CHOICES}.")
    cfg = RULE_CONFIG[rule]
    output_path = output_path_for(rule)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    input_path = INPUT_FILTER_PATH if rule == "50" else INPUT_SUMMARY_PATH

    task_geom: Optional[Dict[int, Tuple[List[float], float]]] = None
    scroll_map: Dict[Tuple[str, str, str, int], float] = {}
    if cfg["requires_scroll"]:
        task_geom = load_task_row_geometry(AOI_PATH)
        if not task_geom:
            raise RuntimeError("Failed to load task geometry from aoi_data.csv")
        scroll_map = build_fixation_scroll_map(NON_PUBLIC_PATH)

    attention_map: Dict[GroupKey, List[float]] = {}
    first_ts_map: Dict[GroupKey, Optional[float]] = {}
    group_order: List[GroupKey] = []
    group_first_seq: Dict[GroupKey, int] = {}
    visit_scroll_y_map: Dict[GroupKey, float] = {}

    click_cols = [
        "Click_AOI_type",
        "Click_AOI_MovieID",
        "Click_AOI_Movie_position_in_carousel",
        "Click_AOI_Carousel_genre",
        "Click_AOI_Carousel_genre_is_top_genre",
        "Click_AOI_Carousel_genre_is_preferred_genre",
        "Click_AOI_Carousel_genre_rating",
    ]
    click_action_rows: List[Tuple[SessionKey, float, int, Dict[str, str]]] = []
    last_fix_page_by_session: Dict[SessionKey, Tuple[str, str, int]] = {}
    last_page_key_by_session: Dict[SessionKey, Tuple[str, str]] = {}
    page_visit_by_session: Dict[SessionKey, int] = {}
    session_order: List[SessionKey] = []
    session_seen: set = set()
    event_seq = 0

    summary_occ_counter: Dict[Tuple[str, str, str], int] = defaultdict(int)

    dropped_by_rule = 0
    filtered_mismatch = 0

    real_first_visits = 0
    real_new_visits_diff_sig = 0
    real_new_visits_same_sig_scroll = 0
    real_accumulate_fixations = 0

    last_fixation_scroll_y_by_session: Dict[SessionKey, float] = {}

    def ensure_group(
        key: GroupKey, ts: Optional[float], seq: int,
        scroll_y_now: Optional[float] = None,
    ) -> None:
        if key not in attention_map:
            attention_map[key] = [0.0] * MAX_SLOTS
            first_ts_map[key] = ts
            group_order.append(key)
            group_first_seq[key] = seq
            if scroll_y_now is not None:
                visit_scroll_y_map[key] = scroll_y_now
        else:
            if ts is not None:
                cur = first_ts_map.get(key)
                if cur is None or ts < cur:
                    first_ts_map[key] = ts

    with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            event_seq += 1
            user_id = row.get("UserID", "")
            task_id_str = row.get("TaskID", "")
            subject_id = row.get("SubjectID", "")
            ts_raw = row.get("Timestamp", "")
            ts = to_float(ts_raw)

            k3 = (user_id, task_id_str, ts_raw)
            occ = summary_occ_counter[k3]
            summary_occ_counter[k3] += 1

            raw_visible_str = row.get(
                "Fixation_AOI_Visible_Carousel_rows", ""
            ).strip()
            raw_horizontal_str = row.get(
                "Fixation_AOI_Visible_Carousel_horizontal_pages", ""
            ).strip()
            all_horizontal_str = row.get(
                "Fixation_AOI_All_Carousel_horizontal_pages", ""
            ).strip()

            raw_visible = (
                parse_triplet(raw_visible_str) if raw_visible_str else None
            )
            raw_horizontal = (
                parse_triplet(raw_horizontal_str) if raw_horizontal_str else None
            )
            all_horizontal = (
                parse_triplet(all_horizontal_str) if all_horizontal_str else None
            )

            fix_type = row.get("Fixation_AOI_type", "").strip().lower()
            closest_type = row.get("Fixation_AOI_Closest_type", "").strip().lower()
            use_closest_movie = fix_type == "background" and closest_type == "movie"
            effective_type = "movie" if use_closest_movie else fix_type
            fix_duration = to_float(row.get("Fixation_Duration", ""))
            has_fixation_event = fix_duration is not None or bool(fix_type)

            click_type_raw = str(row.get("Click_AOI_type", "")).strip()

            session: SessionKey = (user_id, task_id_str, subject_id)
            if session not in session_seen:
                session_seen.add(session)
                session_order.append(session)

            if rule != "50" and effective_type == "movie":
                if use_closest_movie:
                    car_for_filter = to_int_like(
                        row.get("Fixation_AOI_Closest_Carousel_position", "")
                    )
                else:
                    car_for_filter = to_int_like(
                        row.get("Fixation_AOI_Carousel_position", "")
                    )
                if (
                    car_for_filter is not None
                    and raw_visible is not None
                    and car_for_filter not in raw_visible
                ):
                    filtered_mismatch += 1
                    continue

            effective: Optional[Tuple[List[int], List[int]]] = None
            if raw_visible is not None and raw_horizontal is not None:
                task_id_int = to_int_like(task_id_str)
                scroll_y: Optional[float] = None
                if cfg["requires_scroll"]:
                    scroll_y = scroll_map.get((user_id, task_id_str, ts_raw, occ))
                effective = resolve_effective_page(
                    rule=rule,
                    task_id=task_id_int,
                    scroll_y=scroll_y,
                    task_geom=task_geom,
                    raw_visible=raw_visible,
                    raw_horizontal=raw_horizontal,
                    all_horizontal=all_horizontal,
                )

            if effective is not None:
                visible_rows_eff, horizontal_pages_eff = effective
                visible_rows_str = ",".join(str(x) for x in visible_rows_eff)
                horizontal_pages_str = ",".join(str(x) for x in horizontal_pages_eff)
            else:
                visible_rows_eff = []
                horizontal_pages_eff = []
                visible_rows_str = ""
                horizontal_pages_str = ""
                if raw_visible is not None and has_fixation_event and cfg["requires_scroll"]:
                    dropped_by_rule += 1

            has_page = bool(visible_rows_str and horizontal_pages_str)

            if rule != "real":
                allow_visit_update = has_page
                if allow_visit_update:
                    cur_page = (visible_rows_str, horizontal_pages_str)
                    prev = last_page_key_by_session.get(session)
                    if prev is not None and cur_page != prev:
                        page_visit_by_session[session] = (
                            page_visit_by_session.get(session, 0) + 1
                        )
                    last_page_key_by_session[session] = cur_page
            else:
                allow_visit_update = False
                if has_page and has_fixation_event:
                    cur_sig = (visible_rows_str, horizontal_pages_str)
                    prev_sig = last_page_key_by_session.get(session)
                    is_stable = (
                        fix_duration is not None
                        and fix_duration >= REAL_DURATION_THRESHOLD_MS
                    )
                    cur_scroll_y = scroll_map.get(
                        (user_id, task_id_str, ts_raw, occ)
                    )
                    prev_scroll_y = last_fixation_scroll_y_by_session.get(session)
                    scroll_moved = (
                        prev_scroll_y is not None
                        and cur_scroll_y is not None
                        and abs(cur_scroll_y - prev_scroll_y)
                        > REAL_SCROLL_TOLERANCE_PX
                    )

                    if prev_sig is None:
                        page_visit_by_session[session] = 0
                        last_page_key_by_session[session] = cur_sig
                        allow_visit_update = True
                        real_first_visits += 1
                    elif cur_sig != prev_sig:
                        page_visit_by_session[session] = (
                            page_visit_by_session.get(session, 0) + 1
                        )
                        last_page_key_by_session[session] = cur_sig
                        allow_visit_update = True
                        real_new_visits_diff_sig += 1
                    elif is_stable and scroll_moved:
                        page_visit_by_session[session] = (
                            page_visit_by_session.get(session, 0) + 1
                        )
                        allow_visit_update = True
                        real_new_visits_same_sig_scroll += 1
                    else:
                        real_accumulate_fixations += 1

                    if cur_scroll_y is not None:
                        last_fixation_scroll_y_by_session[session] = cur_scroll_y

            visit = page_visit_by_session.get(session, 0)

            if allow_visit_update:
                last_fix_page_by_session[session] = (
                    visible_rows_str,
                    horizontal_pages_str,
                    visit,
                )
            elif rule == "real" and has_page and has_fixation_event:
                last_fix_page_by_session[session] = (
                    visible_rows_str,
                    horizontal_pages_str,
                    visit,
                )

            allow_attach = has_page

            if allow_attach and has_fixation_event:
                key_fe = (
                    user_id,
                    task_id_str,
                    subject_id,
                    visible_rows_str,
                    horizontal_pages_str,
                    visit,
                )
                ensure_group(key_fe, ts, event_seq, scroll_y)

            if effective_type == "movie" and fix_duration is not None and allow_attach:
                if use_closest_movie:
                    carousel_pos = to_int_like(
                        row.get("Fixation_AOI_Closest_Carousel_position", "")
                    )
                    movie_pos = to_int_like(
                        row.get(
                            "Fixation_AOI_Closest_Movie_position_in_carousel", ""
                        )
                    )
                else:
                    carousel_pos = to_int_like(
                        row.get("Fixation_AOI_Carousel_position", "")
                    )
                    movie_pos = to_int_like(
                        row.get("Fixation_AOI_Movie_position_in_carousel", "")
                    )
                if carousel_pos is not None and movie_pos is not None:
                    local_idx = compute_local_index(
                        carousel_pos=carousel_pos,
                        movie_pos=movie_pos,
                        visible_rows=visible_rows_eff,
                        horizontal_pages=horizontal_pages_eff,
                    )
                    if local_idx is not None and 1 <= local_idx <= MAX_SLOTS:
                        key = (
                            user_id,
                            task_id_str,
                            subject_id,
                            visible_rows_str,
                            horizontal_pages_str,
                            visit,
                        )
                        ensure_group(key, ts, event_seq)
                        attention_map[key][local_idx - 1] += fix_duration

            if click_type_raw:
                page = last_fix_page_by_session.get(session)
                if page is None and has_page:
                    page = (visible_rows_str, horizontal_pages_str, visit)
                if page is not None:
                    click_vr, click_hp, click_visit = page
                    key = (
                        user_id,
                        task_id_str,
                        subject_id,
                        click_vr,
                        click_hp,
                        click_visit,
                    )
                    ensure_group(key, ts, event_seq)
                    click_row: Dict[str, str] = {
                        "UserID": user_id,
                        "TaskID": task_id_str,
                        "SubjectID": subject_id,
                        "Timestamp": ts_raw,
                        "Fixation_AOI_Visible_Carousel_rows": click_vr,
                        "Fixation_AOI_Visible_Carousel_horizontal_pages": click_hp,
                        "Page_visit_index": str(click_visit),
                    }
                    for idx in range(1, MAX_SLOTS + 1):
                        click_row[f"slot{idx}"] = ""
                    for c in click_cols:
                        click_row[c] = str(row.get(c, "")).strip()
                    click_action_rows.append(
                        (
                            session,
                            ts if ts is not None else float("inf"),
                            event_seq,
                            click_row,
                        )
                    )

    slot_cols = [f"slot{i}" for i in range(1, MAX_SLOTS + 1)]
    out_cols = [
        "UserID",
        "TaskID",
        "SubjectID",
        "Timestamp",
        "Fixation_AOI_Visible_Carousel_rows",
        "Fixation_AOI_Visible_Carousel_horizontal_pages",
        "Page_visit_index",
    ] + slot_cols + click_cols

    def fix_aware_remap(
        vr_str_in: str,
        hp_str_in: str,
        slot_data_in: List[float],
        scroll_y_in: Optional[float],
        task_id_str_in: str,
    ) -> Tuple[str, str, List[float]]:
        cand = [int(x) for x in vr_str_in.split(",") if x.strip().isdigit()]
        cand_hp_raw = [x.strip() for x in hp_str_in.split(",") if x.strip()]
        if not cand:
            return vr_str_in, hp_str_in, slot_data_in
        row_has_fix: List[bool] = []
        for i in range(len(cand)):
            base = i * 5
            row_has_fix.append(any(slot_data_in[base + c] > 0 for c in range(5)))
        row_is_full: List[bool] = [False] * len(cand)
        if scroll_y_in is not None and task_geom is not None:
            try:
                tid_int = int(float(task_id_str_in))
            except (TypeError, ValueError):
                tid_int = None
            if tid_int is not None and tid_int in task_geom:
                row_tops, poster_h = task_geom[tid_int]
                for i, r in enumerate(cand):
                    if 1 <= r <= len(row_tops):
                        y_top_raw = row_tops[r - 1] - scroll_y_in
                        y_bot_raw = y_top_raw + poster_h
                        y_top_c = max(0.0, min(REAL_VIEWPORT_H, y_top_raw))
                        y_bot_c = max(0.0, min(REAL_VIEWPORT_H, y_bot_raw))
                        h_v = y_bot_c - y_top_c
                        if h_v >= REAL_FULL_VISIBLE_PX:
                            row_is_full[i] = True
        keep_idx = [i for i in range(len(cand)) if row_is_full[i] or row_has_fix[i]]
        if not keep_idx:
            keep_idx = list(range(len(cand)))
        out_rows = [cand[i] for i in keep_idx]
        out_hp = [cand_hp_raw[i] if i < len(cand_hp_raw) else "0" for i in keep_idx]
        out_slots = [0.0] * MAX_SLOTS
        for new_i, old_i in enumerate(keep_idx):
            for c in range(5):
                out_slots[new_i * 5 + c] = slot_data_in[old_i * 5 + c]
        return ",".join(str(r) for r in out_rows), ",".join(str(p) for p in out_hp), out_slots

    rows_by_session: Dict[SessionKey, List[Tuple[float, int, Dict[str, str]]]] = {}
    for key in group_order:
        user_id, task_id_str, subject_id, vr_str, hp_str, visit_idx = key
        ts0 = first_ts_map.get(key)
        ts_out = "" if ts0 is None else f"{ts0:.3f}".rstrip("0").rstrip(".")

        att = attention_map[key]

        if rule == "real":
            scroll_y_visit = visit_scroll_y_map.get(key)
            vr_str_out, hp_str_out, att_out = fix_aware_remap(
                vr_str, hp_str, att, scroll_y_visit, task_id_str
            )
        else:
            vr_str_out, hp_str_out, att_out = vr_str, hp_str, att

        vr_parts = [p for p in vr_str_out.split(",") if p.strip()] if vr_str_out else []
        active_slots = len(vr_parts) * 5
        out_row: Dict[str, str] = {
            "UserID": user_id,
            "TaskID": task_id_str,
            "SubjectID": subject_id,
            "Timestamp": ts_out,
            "Fixation_AOI_Visible_Carousel_rows": vr_str_out,
            "Fixation_AOI_Visible_Carousel_horizontal_pages": hp_str_out,
            "Page_visit_index": str(visit_idx),
        }
        for local_idx in range(1, MAX_SLOTS + 1):
            if local_idx <= active_slots:
                out_row[f"slot{local_idx}"] = f"{att_out[local_idx - 1]:.6f}"
            else:
                out_row[f"slot{local_idx}"] = ""
        for c in click_cols:
            out_row[c] = ""
        sess: SessionKey = (user_id, task_id_str, subject_id)
        rows_by_session.setdefault(sess, []).append(
            (
                ts0 if ts0 is not None else float("inf"),
                group_first_seq.get(key, 0),
                out_row,
            )
        )

    for session, ts_click, seq, row in click_action_rows:
        rows_by_session.setdefault(session, []).append((ts_click, seq, row))

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()
        for session in session_order:
            for _, _, row in sorted(
                rows_by_session.get(session, []), key=lambda x: (x[0], x[1])
            ):
                writer.writerow(row)

    if rule == "real":
        total_real = (
            real_first_visits
            + real_new_visits_diff_sig
            + real_new_visits_same_sig_scroll
            + real_accumulate_fixations
        )

INPUT_CSV = os.path.join(_RECGAZE_DIR, "page_divide_real", "human_attention_page_divided_real.csv")
OUTPUT_CSV = INPUT_CSV

VIEWPORT_H = 1080.0

MIN_VISIBLE_PX = 3.0

csv.field_size_limit(min(2**31 - 1, sys.maxsize))


def geom_visible_rows(
    scroll_y: float,
    row_tops: List[float],
    poster_height: float,
    threshold_px: float = MIN_VISIBLE_PX,
) -> List[Tuple[int, str, int]]:
    if scroll_y is None or row_tops is None or poster_height is None:
        return []
    out: List[Tuple[int, str, int]] = []
    for i, top in enumerate(row_tops, start=1):
        y_top_raw = top - scroll_y
        y_bot_raw = y_top_raw + poster_height
        y_top_c = max(0.0, min(VIEWPORT_H, y_top_raw))
        y_bot_c = max(0.0, min(VIEWPORT_H, y_bot_raw))
        h = y_bot_c - y_top_c
        if h < threshold_px:
            continue
        if h >= poster_height - 0.5:
            status = "full"
        elif y_top_raw < 0:
            status = "partial_top"
        else:
            status = "partial_bottom"
        out.append((i, status, int(round(h))))
    return out


def case_label(geom_rows: List[Tuple[int, str, int]]) -> str:
    if not geom_rows:
        return "NONE"
    statuses = [s for _, s, _ in geom_rows]
    n_full = sum(1 for s in statuses if s == "full")
    n_ptop = sum(1 for s in statuses if s == "partial_top")
    n_pbot = sum(1 for s in statuses if s == "partial_bottom")
    parts: List[str] = []
    if n_ptop:
        parts.append(f"{n_ptop}P_top")
    if n_full:
        parts.append(f"{n_full}F")
    if n_pbot:
        parts.append(f"{n_pbot}P_bot")
    return "+".join(parts)


def run_geometry_augment():
    if not os.path.isfile(INPUT_CSV):
        raise FileNotFoundError(INPUT_CSV)

    task_geom = load_task_row_geometry(AOI_PATH)

    scroll_map = build_fixation_scroll_map(NON_PUBLIC_PATH)

    ts_to_scroll: Dict[Tuple[str, str, str], List[Optional[float]]] = defaultdict(list)
    for (uid, tid, ts, occ), sy in scroll_map.items():
        key = (uid, tid, ts)
        if len(ts_to_scroll[key]) <= occ:
            ts_to_scroll[key].extend([None] * (occ + 1 - len(ts_to_scroll[key])))
        ts_to_scroll[key][occ] = sy

    with open(INPUT_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        old_cols = list(reader.fieldnames or [])
        rows = list(reader)

    new_cols = [
        "visible_rows_geom",
        "visible_row_status",
        "visible_row_heights",
        "n_visible_rows",
        "geom_case",
        "scroll_y_recovered",
    ]
    out_cols = old_cols + [c for c in new_cols if c not in old_cols]

    n_filled = 0
    n_no_scroll = 0
    n_no_geom = 0
    n_no_ts = 0

    for row in rows:
        for c in new_cols:
            row.setdefault(c, "")

        uid = row.get("UserID", "")
        tid = row.get("TaskID", "")
        ts  = row.get("Timestamp", "")
        if not ts:
            n_no_ts += 1
            continue

        try:
            tid_int = int(float(tid))
        except (TypeError, ValueError):
            tid_int = None

        scrolls = ts_to_scroll.get((uid, tid, ts), [])
        scroll_y: Optional[float] = None
        for sy in scrolls:
            if sy is not None:
                scroll_y = sy
                break
        if scroll_y is None:
            n_no_scroll += 1
            continue

        if tid_int is None or tid_int not in task_geom:
            n_no_geom += 1
            continue

        row_tops, poster_height = task_geom[tid_int]
        if not row_tops or poster_height <= 0:
            n_no_geom += 1
            continue

        geom = geom_visible_rows(scroll_y, row_tops, poster_height, MIN_VISIBLE_PX)
        row["visible_rows_geom"]   = ",".join(str(r) for r, _, _ in geom)
        row["visible_row_status"]  = ",".join(s for _, s, _ in geom)
        row["visible_row_heights"] = ",".join(str(h) for _, _, h in geom)
        row["n_visible_rows"]      = str(len(geom))
        row["geom_case"]           = case_label(geom)
        row["scroll_y_recovered"]  = f"{scroll_y:.2f}"
        n_filled += 1


    tmp = OUTPUT_CSV + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, OUTPUT_CSV)

    from collections import Counter
    case_counter = Counter(r.get("geom_case", "") for r in rows if r.get("geom_case"))
    n_counter = Counter(r.get("n_visible_rows", "") for r in rows if r.get("n_visible_rows"))

INPUT_PATH = os.path.join(_RECGAZE_DIR, "summary_feedback_page_divided_filter.csv"
)

RULE_OUTPUT = {
    "50": os.path.join(BASE_OUTPUT_DIR, "human_attention_page_row_analysis.csv"),
    "100": os.path.join(
        BASE_OUTPUT_DIR,
        "page_divided_100%split",
        "human_attention_page_row_analysis_100%split.csv",
    ),
    "2row": os.path.join(
        BASE_OUTPUT_DIR,
        "page_divided_2row",
        "human_attention_page_row_analysis_2row.csv",
    ),
    "real": os.path.join(
        BASE_OUTPUT_DIR,
        "page_divide_real",
        "human_attention_page_row_analysis_real.csv",
    ),
}


def parse_row_list(s: str, strict3: bool = False) -> Optional[List[int]]:
    raw = str(s).strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if strict3 and len(parts) != 3:
        return None
    if not (2 <= len(parts) <= 3):
        return None
    out: List[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            return None
    return out


def fixation_row_rank(
    carousel_pos: int,
    movie_pos: int,
    visible_rows: List[int],
    horizontal_pages: List[int],
) -> Optional[int]:
    if carousel_pos not in visible_rows:
        return None
    row_rank = visible_rows.index(carousel_pos)
    if row_rank >= len(horizontal_pages):
        return None
    page_state = horizontal_pages[row_rank]
    if page_state not in (0, 1, 2):
        return None
    movie_start = page_state * 5 + 1
    movie_end = movie_start + 4
    if not (movie_start <= movie_pos <= movie_end):
        return None
    return row_rank


def click_row_rank(
    click_carousel_pos: Optional[int],
    visible_rows: Optional[List[int]],
) -> Optional[int]:
    if click_carousel_pos is None or visible_rows is None:
        return None
    if click_carousel_pos not in visible_rows:
        return None
    return visible_rows.index(click_carousel_pos)


@dataclass
class FixationRecord:
    row_rank: int
    duration: float
    ts: float


@dataclass
class ClickRecord:
    click_type: str
    row_rank: Optional[int]
    col: Optional[int]
    absolute_row: Optional[int]
    movie_id: Optional[str]
    ts: float
    genre: str


@dataclass
class Visit:
    user_id: str
    task_id: str
    visible_rows_str: str
    horizontal_pages_str: str
    visible_rows: List[int]
    horizontal_pages: List[int]
    start_ts: float
    end_ts: float
    fixations: List[FixationRecord] = field(default_factory=list)
    clicks: List[ClickRecord] = field(default_factory=list)


def classify_state_transition(cur: Visit, nxt: Visit) -> Tuple[str, str]:
    cur_vr = cur.visible_rows
    nxt_vr = nxt.visible_rows
    cur_hp = cur.horizontal_pages
    nxt_hp = nxt.horizontal_pages

    vr_changed = cur_vr != nxt_vr

    n = min(len(cur_hp), len(nxt_hp))
    hp_diff_idx = [i for i in range(n) if cur_hp[i] != nxt_hp[i]]
    if len(cur_hp) != len(nxt_hp):
        vr_changed = True

    if vr_changed and not hp_diff_idx:
        if nxt_vr[0] > cur_vr[0]:
            return ("scroll_down", "")
        if nxt_vr[0] < cur_vr[0]:
            return ("scroll_up", "")
        return ("scroll_other", "")

    if not vr_changed and len(hp_diff_idx) == 1:
        i = hp_diff_idx[0]
        direction = "right" if nxt_hp[i] > cur_hp[i] else "left"
        return (f"swipe_row{i + 1}_{direction}", str(i + 1))

    if not vr_changed and len(hp_diff_idx) > 1:
        return ("swipe_multi_row", ",".join(str(i + 1) for i in hp_diff_idx))

    if vr_changed and hp_diff_idx:
        if nxt_vr[0] > cur_vr[0]:
            return ("scroll_down", "")
        if nxt_vr[0] < cur_vr[0]:
            return ("scroll_up", "")
        return ("state_change_other", "")

    return ("state_change_none", "")


def classify_next_action(
    visit: Visit, next_visit: Optional[Visit]
) -> Tuple[str, str, str]:
    for c in sorted(visit.clicks, key=lambda x: x.ts):
        ctype = c.click_type.lower()
        if ctype == "forward":
            row = c.row_rank + 1 if c.row_rank is not None else ""
            return ("swipe_forward", str(row), "")
        if ctype == "backward":
            row = c.row_rank + 1 if c.row_rank is not None else ""
            return ("swipe_backward", str(row), "")
        if ctype == "movie":
            row = c.row_rank + 1 if c.row_rank is not None else ""
            detail = c.movie_id or ""
            return ("click_movie", str(row), detail)
        if ctype == "background":
            return ("click_background", "", "")
        return (f"click_{ctype}", "", "")

    if next_visit is None:
        return ("end_session", "", "")

    action, row_str = classify_state_transition(visit, next_visit)
    return (action, row_str, "")


def summarise_visit(visit: Visit) -> Dict[str, str]:
    n_rows = len(visit.visible_rows)
    row_time = [0.0] * n_rows
    row_count = [0] * n_rows

    for fx in visit.fixations:
        if fx.row_rank < n_rows:
            row_time[fx.row_rank] += fx.duration
            row_count[fx.row_rank] += 1

    total_time = sum(row_time)
    total_count = sum(row_count)

    shares = [
        (row_time[i] / total_time) if total_time > 0 else 0.0
        for i in range(n_rows)
    ]

    last_row = ""
    if visit.fixations:
        last_row = str(visit.fixations[-1].row_rank + 1)

    last3 = visit.fixations[-3:]
    n3 = len(last3)
    last3_counts = [0] * n_rows
    for fx in last3:
        if fx.row_rank < n_rows:
            last3_counts[fx.row_rank] += 1
    last3_shares = [
        (last3_counts[i] / n3) if n3 > 0 else 0.0 for i in range(n_rows)
    ]
    last3_rows = ",".join(str(fx.row_rank + 1) for fx in last3)

    if total_time > 0:
        dominant_row = shares.index(max(shares)) + 1
        dominant_share = max(shares)
    else:
        dominant_row = ""
        dominant_share = 0.0

    def _t(lst: List[float], i: int) -> str:
        return f"{lst[i]:.3f}" if i < len(lst) else "0.000"

    def _c(lst: List[int], i: int) -> str:
        return str(lst[i]) if i < len(lst) else "0"

    def _s(lst: List[float], i: int) -> str:
        return f"{lst[i]:.4f}" if i < len(lst) else "0.0000"

    return {
        "row1_total_time": _t(row_time, 0),
        "row2_total_time": _t(row_time, 1),
        "row3_total_time": _t(row_time, 2),
        "row1_fix_count": _c(row_count, 0),
        "row2_fix_count": _c(row_count, 1),
        "row3_fix_count": _c(row_count, 2),
        "row1_attention_share": _s(shares, 0),
        "row2_attention_share": _s(shares, 1),
        "row3_attention_share": _s(shares, 2),
        "total_fix_time": f"{total_time:.3f}",
        "total_fix_count": str(total_count),
        "last_fix_row": last_row,
        "last3_fix_rows": last3_rows,
        "last3_row1_share": _s(last3_shares, 0),
        "last3_row2_share": _s(last3_shares, 1),
        "last3_row3_share": _s(last3_shares, 2),
        "dominant_row": str(dominant_row),
        "dominant_share": f"{dominant_share:.4f}",
    }


def flush_visit(
    session_visits: Dict[Tuple[str, str], List[Visit]],
    session_key: Tuple[str, str],
    visit: Optional[Visit],
) -> None:
    if visit is None:
        return
    session_visits.setdefault(session_key, []).append(visit)


def run_row_analysis(rule: str) -> None:
    if rule not in RULE_OUTPUT:
        raise ValueError(f"Unknown rule: {rule!r}. Choose from {list(RULE_OUTPUT)}")
    output_path = RULE_OUTPUT[rule]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    task_geom = None
    scroll_map: Dict[Tuple[str, str, str, int], float] = {}
    if rule != "50":

        task_geom = load_task_row_geometry(AOI_PATH)
        scroll_map = build_fixation_scroll_map(NON_PUBLIC_PATH)
        input_path = INPUT_SUMMARY_PATH
    else:
        resolve_effective_page = None
        input_path = INPUT_PATH

    session_visits: Dict[Tuple[str, str], List[Visit]] = {}
    session_order: List[Tuple[str, str]] = []
    session_seen: set = set()
    current_visit: Dict[Tuple[str, str], Optional[Visit]] = {}

    occ_counter: Dict[Tuple[str, str, str], int] = defaultdict(int)

    last_fix_scroll_y_by_session: Dict[Tuple[str, str], float] = {}

    dropped_rule = 0
    filtered_mismatch = 0

    with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_id = row.get("UserID", "")
            task_id = row.get("TaskID", "")
            ts_raw = row.get("Timestamp", "")
            session = (user_id, task_id)

            if session not in session_seen:
                session_seen.add(session)
                session_order.append(session)
                current_visit[session] = None

            ts = to_float(ts_raw)
            ts_val = ts if ts is not None else 0.0

            k3 = (user_id, task_id, ts_raw)
            occ = occ_counter[k3]
            occ_counter[k3] += 1

            vr_str = row.get("Fixation_AOI_Visible_Carousel_rows", "").strip()
            hp_str = row.get(
                "Fixation_AOI_Visible_Carousel_horizontal_pages", ""
            ).strip()

            fix_type = row.get("Fixation_AOI_type", "").strip().lower()
            closest_type = row.get("Fixation_AOI_Closest_type", "").strip().lower()
            use_closest = fix_type == "background" and closest_type == "movie"
            effective_type = "movie" if use_closest else fix_type
            fix_duration = to_float(row.get("Fixation_Duration", ""))

            click_type = row.get("Click_AOI_type", "").strip()

            if rule != "50":
                if effective_type == "movie" and vr_str:
                    raw_vr = parse_row_list(vr_str)
                    if use_closest:
                        car_filter = to_int_like(
                            row.get("Fixation_AOI_Closest_Carousel_position", "")
                        )
                    else:
                        car_filter = to_int_like(
                            row.get("Fixation_AOI_Carousel_position", "")
                        )
                    if (
                        raw_vr is not None
                        and car_filter is not None
                        and car_filter not in raw_vr
                    ):
                        filtered_mismatch += 1
                        continue

                if vr_str and hp_str:
                    raw_vr = parse_row_list(vr_str)
                    raw_hp = parse_row_list(hp_str)
                    scroll_y = scroll_map.get((user_id, task_id, ts_raw, occ))
                    task_id_int = to_int_like(task_id)
                    effective = resolve_effective_page(
                        rule=rule,
                        task_id=task_id_int,
                        scroll_y=scroll_y,
                        task_geom=task_geom,
                        raw_visible=raw_vr,
                        raw_horizontal=raw_hp,
                    )
                    if effective is None:
                        if fix_duration is not None or fix_type:
                            dropped_rule += 1
                        vr_str = ""
                        hp_str = ""
                    else:
                        vr_eff, hp_eff = effective
                        vr_str = ",".join(str(x) for x in vr_eff)
                        hp_str = ",".join(str(x) for x in hp_eff)

            has_page_state = bool(vr_str and hp_str)

            if has_page_state:
                visible_rows = parse_row_list(vr_str)
                horizontal_pages = parse_row_list(hp_str)
                if visible_rows is None or horizontal_pages is None:
                    continue

                visit = current_visit.get(session)
                same_sig = (
                    visit is not None
                    and visit.visible_rows_str == vr_str
                    and visit.horizontal_pages_str == hp_str
                )

                if rule == "real":
                    is_stable = (
                        fix_duration is not None
                        and fix_duration >= REAL_DURATION_THRESHOLD_MS
                    )
                    cur_scroll_y_real = scroll_map.get(
                        (user_id, task_id, ts_raw, occ)
                    )
                    prev_scroll_y_real = last_fix_scroll_y_by_session.get(session)
                    scroll_moved = (
                        prev_scroll_y_real is not None
                        and cur_scroll_y_real is not None
                        and abs(cur_scroll_y_real - prev_scroll_y_real)
                        > REAL_SCROLL_TOLERANCE_PX
                    )
                    if visit is None:
                        start_new_visit = True
                    elif not same_sig:
                        start_new_visit = True
                    elif is_stable and scroll_moved:
                        start_new_visit = True
                    else:
                        start_new_visit = False

                    if cur_scroll_y_real is not None and (
                        fix_duration is not None or fix_type
                    ):
                        last_fix_scroll_y_by_session[session] = cur_scroll_y_real
                else:
                    start_new_visit = (visit is None) or (not same_sig)

                if start_new_visit:
                    if visit is not None:
                        flush_visit(session_visits, session, visit)
                    visit = Visit(
                        user_id=user_id,
                        task_id=task_id,
                        visible_rows_str=vr_str,
                        horizontal_pages_str=hp_str,
                        visible_rows=visible_rows,
                        horizontal_pages=horizontal_pages,
                        start_ts=ts_val,
                        end_ts=ts_val,
                    )
                    current_visit[session] = visit
                else:
                    visit.end_ts = ts_val

                if effective_type == "movie" and fix_duration is not None:
                    if use_closest:
                        carousel_pos = to_int_like(
                            row.get("Fixation_AOI_Closest_Carousel_position", "")
                        )
                        movie_pos = to_int_like(
                            row.get(
                                "Fixation_AOI_Closest_Movie_position_in_carousel", ""
                            )
                        )
                    else:
                        carousel_pos = to_int_like(
                            row.get("Fixation_AOI_Carousel_position", "")
                        )
                        movie_pos = to_int_like(
                            row.get("Fixation_AOI_Movie_position_in_carousel", "")
                        )
                    if carousel_pos is not None and movie_pos is not None:
                        rank = fixation_row_rank(
                            carousel_pos, movie_pos, visible_rows, horizontal_pages
                        )
                        if rank is not None:
                            visit.fixations.append(
                                FixationRecord(
                                    row_rank=rank,
                                    duration=fix_duration,
                                    ts=ts_val,
                                )
                            )

            if click_type:
                visit = current_visit.get(session)
                if visit is None:
                    continue
                click_car = to_int_like(row.get("Click_AOI_Carousel_position", ""))
                rank = click_row_rank(click_car, visit.visible_rows)
                col = to_int_like(
                    row.get("Click_AOI_Movie_position_in_carousel", "")
                )
                mid_raw = row.get("Click_AOI_MovieID", "").strip()
                movie_id: Optional[str] = mid_raw if mid_raw else None
                visit.clicks.append(
                    ClickRecord(
                        click_type=click_type,
                        row_rank=rank,
                        col=col,
                        absolute_row=click_car,
                        movie_id=movie_id,
                        ts=ts_val,
                        genre=row.get("Click_AOI_Carousel_genre", "").strip(),
                    )
                )

    for session, visit in current_visit.items():
        if visit is not None:
            flush_visit(session_visits, session, visit)

    out_cols = [
        "UserID",
        "TaskID",
        "visit_index",
        "start_timestamp",
        "end_timestamp",
        "Fixation_AOI_Visible_Carousel_rows",
        "Fixation_AOI_Visible_Carousel_horizontal_pages",
        "row1_total_time",
        "row2_total_time",
        "row3_total_time",
        "row1_fix_count",
        "row2_fix_count",
        "row3_fix_count",
        "row1_attention_share",
        "row2_attention_share",
        "row3_attention_share",
        "total_fix_time",
        "total_fix_count",
        "last_fix_row",
        "last3_fix_rows",
        "last3_row1_share",
        "last3_row2_share",
        "last3_row3_share",
        "dominant_row",
        "dominant_share",
        "next_action",
        "next_action_row",
        "next_action_detail",
    ]

    total_visits = 0
    action_counter: Dict[str, int] = {}

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()

        for session in session_order:
            visits = session_visits.get(session, [])
            for idx, visit in enumerate(visits):
                nxt = visits[idx + 1] if idx + 1 < len(visits) else None
                metrics = summarise_visit(visit)
                action, action_row, action_detail = classify_next_action(visit, nxt)
                action_counter[action] = action_counter.get(action, 0) + 1

                out: Dict[str, str] = {
                    "UserID": visit.user_id,
                    "TaskID": visit.task_id,
                    "visit_index": str(idx + 1),
                    "start_timestamp": f"{visit.start_ts:.3f}",
                    "end_timestamp": f"{visit.end_ts:.3f}",
                    "Fixation_AOI_Visible_Carousel_rows": visit.visible_rows_str,
                    "Fixation_AOI_Visible_Carousel_horizontal_pages": visit.horizontal_pages_str,
                    "next_action": action,
                    "next_action_row": action_row,
                    "next_action_detail": action_detail,
                }
                out.update(metrics)
                writer.writerow(out)
                total_visits += 1

CURSOR_SUMMARY_PATH = os.path.join(_RECGAZE_DIR, "summary_feedback_page_divided.csv"
)
VISIT_PATH = os.path.join(_RECGAZE_DIR, "page_divide_real/human_attention_page_divided_real.csv",
)
OUT_PATH = os.path.join(_RECGAZE_DIR, "page_divide_real/human_attention_page_divided_real_with_cursor.csv",
)

NUM_SLOTS = 20
CURSOR_DURATION_SCALE_MS = 1000.0


VisitKey = Tuple[str, str, str]


def _load_visit_table() -> Tuple[List[dict], Dict[VisitKey, List[Tuple[float, int]]]]:
    rows: List[dict] = []
    session_visits: Dict[VisitKey, List[Tuple[float, int]]] = defaultdict(list)
    with open(VISIT_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            rows.append(row)
            uid = row.get("UserID", "").strip()
            tid = row.get("TaskID", "").strip()
            sid = row.get("SubjectID", "").strip()
            ts = to_float(row.get("Timestamp", ""))
            if uid and tid and sid and ts is not None:
                session_visits[(uid, tid, sid)].append((ts, row_idx))
    for key in session_visits:
        session_visits[key].sort(key=lambda x: x[0])
    return rows, session_visits


def _find_matching_visit(
    session_list: List[Tuple[float, int]],
    event_ts: float,
) -> Optional[int]:
    lo, hi = 0, len(session_list) - 1
    if hi < 0 or event_ts < session_list[0][0]:
        return None
    result = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if session_list[mid][0] <= event_ts:
            result = session_list[mid][1]
            lo = mid + 1
        else:
            hi = mid - 1
    return result if result >= 0 else None


def run_cursor_aggregation() -> None:
    visit_rows, session_visits = _load_visit_table()

    cursor_slots: Dict[int, List[float]] = defaultdict(lambda: [0.0] * NUM_SLOTS)

    n_events = 0
    n_with_cursor = 0
    n_no_session = 0
    n_no_visit = 0
    n_no_aoi = 0
    n_outside_visible = 0
    n_attached = 0
    used_closest = 0

    with open(CURSOR_SUMMARY_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_events += 1
            dur = to_float(row.get("Cursor_Duration", ""))
            if dur is None:
                continue
            n_with_cursor += 1

            uid = row.get("UserID", "").strip()
            tid = row.get("TaskID", "").strip()
            sid = row.get("SubjectID", "").strip()
            ts = to_float(row.get("Timestamp", ""))
            if not (uid and tid and sid) or ts is None:
                n_no_session += 1
                continue

            session_list = session_visits.get((uid, tid, sid))
            if not session_list:
                n_no_session += 1
                continue

            visit_row_idx = _find_matching_visit(session_list, ts)
            if visit_row_idx is None:
                n_no_visit += 1
                continue
            visit = visit_rows[visit_row_idx]
            vr = parse_triplet(visit.get("Fixation_AOI_Visible_Carousel_rows", ""))
            hp = parse_triplet(visit.get("Fixation_AOI_Visible_Carousel_horizontal_pages", ""))
            if vr is None or hp is None:
                n_no_visit += 1
                continue

            cur_type = row.get("Cursor_AOI_type", "").strip().lower()
            closest_type = row.get("Cursor_AOI_Closest_type", "").strip().lower()
            use_closest = cur_type != "movie" and closest_type == "movie"
            if cur_type == "movie":
                carousel_pos = to_int_like(row.get("Cursor_AOI_Carousel_position", ""))
                movie_pos = to_int_like(row.get("Cursor_AOI_Movie_position_in_carousel", ""))
            elif use_closest:
                carousel_pos = to_int_like(row.get("Cursor_AOI_Closest_Carousel_position", ""))
                movie_pos = to_int_like(row.get("Cursor_AOI_Closest_Movie_position_in_carousel", ""))
                used_closest += 1
            else:
                n_no_aoi += 1
                continue

            if carousel_pos is None or movie_pos is None:
                n_no_aoi += 1
                continue

            local_idx = compute_local_index(
                carousel_pos=carousel_pos,
                movie_pos=movie_pos,
                visible_rows=vr,
                horizontal_pages=hp,
            )
            if local_idx is None:
                n_outside_visible += 1
                continue

            cursor_slots[visit_row_idx][local_idx - 1] += dur * CURSOR_DURATION_SCALE_MS
            n_attached += 1


    out_dir = os.path.dirname(OUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cursor_cols = [f"cursor_slot{i}" for i in range(1, NUM_SLOTS + 1)]

    with open(VISIT_PATH, "r", encoding="utf-8", newline="") as fin, \
         open(OUT_PATH, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        in_fields = list(reader.fieldnames or [])
        out_fields = in_fields + cursor_cols + ["cursor_sum"]
        writer = csv.DictWriter(fout, fieldnames=out_fields)
        writer.writeheader()
        rows_with_cursor = 0
        for row_idx, row in enumerate(reader):
            slots = cursor_slots.get(row_idx)
            if slots is None:
                for c in cursor_cols:
                    row[c] = 0.0
                row["cursor_sum"] = 0.0
            else:
                for i, c in enumerate(cursor_cols):
                    row[c] = slots[i]
                row["cursor_sum"] = float(sum(slots))
                if row["cursor_sum"] > 0:
                    rows_with_cursor += 1
            writer.writerow(row)


def main():
    run_summary_pages()
    run_page_divided("real")
    run_geometry_augment()
    run_row_analysis("real")
    run_cursor_aggregation()


if __name__ == "__main__":
    main()
