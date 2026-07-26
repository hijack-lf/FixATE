from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_RECGAZE_DIR = os.path.join(_ROOT, "datasets", "RecGaze")

ACTION_CSV = os.path.join(_RECGAZE_DIR, "page_divide_real/human_attention_page_row_analysis_real.csv")
_PAGE_CSV_DEFAULT = os.path.join(_RECGAZE_DIR, "page_divide_real/human_attention_page_divided_real.csv")
_PAGE_CSV_WITH_CURSOR = os.path.join(_RECGAZE_DIR, "page_divide_real/human_attention_page_divided_real_with_cursor.csv")
PAGE_CSV = _PAGE_CSV_WITH_CURSOR if os.path.isfile(_PAGE_CSV_WITH_CURSOR) else _PAGE_CSV_DEFAULT

TASK_MAX = 40

_IMG_ROOT = os.path.join(_RECGAZE_DIR, "page_divide_real", "image")
OUT_DIR = os.path.join(_RECGAZE_DIR, "action_manifest")

ACTION_LABELS = ["scroll_down", "scroll_up", "swipe_forward", "swipe_backward", "click_movie"]
ACTION2ID = {a: i for i, a in enumerate(ACTION_LABELS)}


def _resolve_page_visit_index(a: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    a = a.copy()
    p_sorted = p[["TaskID", "Timestamp", "Page_visit_index"]].sort_values(
        ["TaskID", "Timestamp"])
    pvi_out: list = []
    ts_diff_out: list = []
    for tid, grp_a in a.groupby("TaskID", sort=False):
        grp_p = p_sorted[p_sorted.TaskID == tid]
        if len(grp_p) == 0:
            pvi_out.extend([None] * len(grp_a))
            ts_diff_out.extend([None] * len(grp_a))
            continue
        ts_arr = grp_p.Timestamp.to_numpy()
        pvi_arr = grp_p.Page_visit_index.to_numpy()
        for st in grp_a.start_timestamp.to_numpy():
            import numpy as _np
            idx = _np.searchsorted(ts_arr, st + 1e-6, side="right") - 1
            if idx < 0:
                idx = 0
            pvi_out.append(int(pvi_arr[idx]))
            ts_diff_out.append(float(st - ts_arr[idx]) * 1000.0)
    a["page_visit_index"] = pvi_out
    a["pvi_ts_diff_ms"] = ts_diff_out
    return a


def build_user(user_id, df_a, df_p):
    img_dir = os.path.join(_IMG_ROOT, user_id)
    out_csv = os.path.join(OUT_DIR, f"action_manifest_{user_id}_T1-{TASK_MAX}.csv")
    a = df_a[(df_a.UserID == user_id) & (df_a.TaskID <= TASK_MAX)].copy()
    p = df_p[(df_p.UserID == user_id) & (df_p.TaskID <= TASK_MAX)].copy()

    a = a[a.next_action.isin(ACTION_LABELS)].copy()
    a = _resolve_page_visit_index(a, p)
    n_no_pvi = int(a.page_visit_index.isna().sum())
    if n_no_pvi:
        a = a[a.page_visit_index.notna()].copy()
    big_diff = int((a.pvi_ts_diff_ms.abs() > 500).sum())

    a["image_filename"]   = a.apply(lambda r: f"{user_id}_{int(r.TaskID)}_{int(r.page_visit_index)}.jpg", axis=1)
    a["image_path"]       = a.image_filename.apply(lambda f: os.path.join(img_dir, f))
    a["image_exists"]     = a.image_path.apply(os.path.isfile)
    a["action_id"]        = a.next_action.map(ACTION2ID)

    n_before = len(a)
    a = a[a.image_exists].copy()

    slot_cols = [f"slot{i}" for i in range(1, 21)]
    cursor_slot_cols = [f"cursor_slot{i}" for i in range(1, 21)]
    extra_cols = [
        "Fixation_AOI_Visible_Carousel_rows",
        "Fixation_AOI_Visible_Carousel_horizontal_pages",
    ]
    geom_cols = [
        "visible_rows_geom",
        "visible_row_status",
        "visible_row_heights",
        "n_visible_rows",
        "geom_case",
        "scroll_y_recovered",
    ]
    geom_cols_present = [c for c in geom_cols if c in p.columns]
    if len(geom_cols_present) < len(geom_cols):
        missing = set(geom_cols) - set(geom_cols_present)
    a = a.drop(columns=[c for c in extra_cols if c in a.columns])
    cursor_cols_present = [c for c in cursor_slot_cols if c in p.columns]
    if "cursor_sum" in p.columns:
        cursor_cols_present = cursor_cols_present + ["cursor_sum"]
    p_slim_cols = (["TaskID", "Page_visit_index"] + slot_cols + extra_cols
                   + geom_cols_present + cursor_cols_present)
    p_slim = p[p_slim_cols].copy()
    p_first = p_slim.groupby(["TaskID", "Page_visit_index"], as_index=False).first()

    merged = a.merge(
        p_first,
        left_on=["TaskID", "page_visit_index"],
        right_on=["TaskID", "Page_visit_index"],
        how="left",
    )

    merged["fixation_sum"] = merged[slot_cols].sum(axis=1)
    merged["has_fixation"] = merged["fixation_sum"] > 0

    merged["row1_time"] = merged[[f"slot{i}" for i in range(1, 6)]].sum(axis=1)
    merged["row2_time"] = merged[[f"slot{i}" for i in range(6, 11)]].sum(axis=1)
    merged["row3_time"] = merged[[f"slot{i}" for i in range(11, 16)]].sum(axis=1)
    merged["row4_time"] = merged[[f"slot{i}" for i in range(16, 21)]].sum(axis=1)

    merged = merged.rename(columns={
        "Fixation_AOI_Visible_Carousel_rows": "visible_rows",
        "Fixation_AOI_Visible_Carousel_horizontal_pages": "visible_pages",
    })

    from render_images import collect_visits
    visits = collect_visits(user_id)
    visit_geom: dict = {}
    for v in visits:
        uid, tid, _sid, visit_idx, sy, hp_state = v
        try:
            tid_int = int(float(tid))
        except (TypeError, ValueError):
            continue
        visit_geom[(uid, tid_int, int(visit_idx))] = (
            float(sy) if sy is not None else None,
            {int(k): int(v) for k, v in hp_state.items()},
        )

    def _lookup_geom(row, kind):
        v = visit_geom.get((row.UserID, int(row.TaskID), int(row.page_visit_index)))
        if v is None:
            return None if kind == "sy" else "{}"
        sy, hp = v
        if kind == "sy":
            return sy
        return json.dumps(hp, separators=(",", ":"))

    merged["scroll_y"] = merged.apply(lambda r: _lookup_geom(r, "sy"), axis=1)
    merged["hp_state_json"] = merged.apply(lambda r: _lookup_geom(r, "hp"), axis=1)
    n_missing_sy = int(merged.scroll_y.isna().sum())

    extra_action_cols = [c for c in ("next_action_row", "next_action_detail")
                          if c in merged.columns]
    out_cols = (
        ["UserID", "TaskID", "visit_index", "image_filename", "image_path",
         "next_action", "action_id"]
        + extra_action_cols
        + ["dominant_row", "fixation_sum", "has_fixation",
           "visible_rows", "visible_pages",
           "row1_time", "row2_time", "row3_time", "row4_time",
           "scroll_y", "hp_state_json"]
        + slot_cols
        + geom_cols_present
        + cursor_cols_present
    )
    out = merged[out_cols].rename(columns={"UserID": "user_id", "TaskID": "task_id"})
    out.to_csv(out_csv, index=False)


def run_manifests():
    os.makedirs(OUT_DIR, exist_ok=True)
    df_a = pd.read_csv(ACTION_CSV)
    df_p = pd.read_csv(PAGE_CSV)
    users = sorted(df_a.UserID.astype(str).unique())
    for user_id in users:
        build_user(user_id, df_a, df_p)

MANIFEST_DIR = os.path.join(_RECGAZE_DIR, "action_manifest")
CURSOR_CSV = os.path.join(_RECGAZE_DIR, "page_divide_real", "human_attention_page_divided_real_with_cursor.csv")

CURSOR_SLOT_COLS = [f"cursor_slot{i}" for i in range(1, 21)]
CURSOR_EXTRA_COLS = ["cursor_sum"]
JOIN_COLS = ["UserID", "TaskID", "Page_visit_index"]


def run_cursor_patch() -> None:
    if not os.path.isfile(CURSOR_CSV):
        sys.exit(2)

    cursor = pd.read_csv(CURSOR_CSV, usecols=JOIN_COLS + CURSOR_SLOT_COLS + CURSOR_EXTRA_COLS)

    paths = sorted(glob.glob(os.path.join(MANIFEST_DIR, "action_manifest_*_T1-*.csv")))
    paths = [p for p in paths if not p.endswith(".bak") and not p.endswith(".precursor.bak")]

    n_patched = 0
    n_skipped = 0
    n_warned = 0
    for path in paths:
        manifest = pd.read_csv(path)
        if "cursor_slot1" in manifest.columns:
            n_skipped += 1
            continue

        if "user_id" not in manifest.columns or "task_id" not in manifest.columns \
                or "image_filename" not in manifest.columns:
            n_warned += 1
            continue

        def _pvi(fn: str) -> int:
            stem = fn.rsplit(".", 1)[0]
            return int(stem.rsplit("_", 1)[-1])
        manifest["_page_visit_index"] = manifest["image_filename"].map(_pvi)

        manifest = manifest.merge(
            cursor.rename(columns={
                "UserID": "user_id",
                "TaskID": "task_id",
                "Page_visit_index": "_page_visit_index",
            }),
            on=["user_id", "task_id", "_page_visit_index"],
            how="left",
        )
        manifest = manifest.drop(columns=["_page_visit_index"])
        miss = manifest[CURSOR_SLOT_COLS[0]].isna().sum()
        if miss:
            for c in CURSOR_SLOT_COLS + CURSOR_EXTRA_COLS:
                manifest[c] = manifest[c].fillna(0.0)

        bak = path + ".precursor.bak"
        if not os.path.isfile(bak):
            shutil.copy2(path, bak)

        manifest.to_csv(path, index=False)
        n_patched += 1

ROW_Y_TOPS = [176, 516, 855, 1195, 1534, 1874, 2213, 2553, 2892, 3232]
COL_X_STARTS = [76, 427, 778, 1129, 1480]
POSTER_W, POSTER_H = 342, 193
VIEWPORT_H = 1080
SLOTS_PER_ROW = 5
MAX_STAGE2_SLOTS = 15

ANALYSIS_CSV = os.path.join(_RECGAZE_DIR, "page_divide_real", "human_attention_page_row_analysis_real.csv")
ITEM_FEATURES_CSV = os.path.join(_RECGAZE_DIR, "raw", "item_features.csv")
MANIFEST_GLOB = os.path.join(_RECGAZE_DIR, "action_manifest", "action_manifest_*.csv")


def split_of(t):
    if 1 <= t <= 25:  return "train"
    if 26 <= t <= 30: return "val"
    if 31 <= t <= 35: return "ood1"
    if 36 <= t <= 40: return "ood2"
    return "other"


def build_click_target_index(analysis_csv, item_features_csv):
    items = pd.read_csv(item_features_csv)
    ml = {}
    for _, r in items.iterrows():
        try:
            ml[(int(float(r["TaskID"])), int(float(r["MovieID"])))] = (
                int(float(r["Carousel_position"])), int(float(r["Movie_position_in_carousel"])))
        except (ValueError, KeyError, TypeError):
            continue
    analysis = pd.read_csv(analysis_csv)
    out = {}
    for _, r in analysis.iterrows():
        if r.get("next_action") != "click_movie":
            continue
        try:
            uid = str(r["UserID"]); tid = int(float(r["TaskID"]))
            vix = int(float(r["visit_index"]))
            crow = int(float(r["next_action_row"])); mid = int(float(r["next_action_detail"]))
        except (ValueError, KeyError, TypeError):
            continue
        vr = [int(x) for x in str(r.get("Fixation_AOI_Visible_Carousel_rows", "")).split(",")
              if x.strip().lstrip("-").isdigit()]
        if not vr or crow < 1 or crow > len(vr):
            continue
        acr = vr[crow - 1]; pos = ml.get((tid, mid))
        if pos is None or pos[0] != acr:
            continue
        out[(uid, tid, vix)] = (acr, pos[1])
    return out


def resolve_slot(visible_rows_geom, visible_row_status, hp_state, clicked_carousel_row, clicked_movie_position):
    if not visible_rows_geom or len(visible_rows_geom) != len(visible_row_status):
        return None
    letter_idx = 0
    for r, status in zip(visible_rows_geom, visible_row_status):
        if status != "full":
            continue
        if r == clicked_carousel_row:
            page = hp_state.get(r, 0); page = page if page in (0, 1, 2) else 0
            col0 = (clicked_movie_position - 1) - page * 5
            return letter_idx + col0 if 0 <= col0 < 5 else None
        letter_idx += 5
        if letter_idx >= 20:
            return None
    return None


def candidate_bboxes(visible_rows_geom, visible_row_status, scroll_y):
    bboxes = []
    for r, status in zip(visible_rows_geom, visible_row_status):
        if status != "full":
            continue
        y_top = ROW_Y_TOPS[r - 1] - scroll_y
        y_top_c = max(0.0, min(float(VIEWPORT_H), y_top))
        y_bot_c = max(0.0, min(float(VIEWPORT_H), y_top + POSTER_H))
        for col in range(SLOTS_PER_ROW):
            x0 = COL_X_STARTS[col]
            bboxes.append([int(x0), int(y_top_c), int(x0 + POSTER_W), int(y_bot_c)])
        if len(bboxes) >= MAX_STAGE2_SLOTS:
            break
    return bboxes[:MAX_STAGE2_SLOTS]


def parse_ints(s):
    return [int(x) for x in str(s).split(",") if x.strip().lstrip("-").isdigit()]


def parse_status(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def gaze_target(row, visible_rows_geom, visible_row_status):
    vis_fix = parse_ints(row.get("visible_rows", ""))
    slot_fix_raw = []
    for i in range(1, 21):
        try:
            v = float(row.get(f"slot{i}", 0.0))
            v = 0.0 if pd.isna(v) else v
        except (ValueError, TypeError):
            v = 0.0
        slot_fix_raw.append(v)
    fix = [0.0] * MAX_STAGE2_SLOTS
    valid = [False] * MAX_STAGE2_SLOTS
    out_row = 0
    for r, status in zip(visible_rows_geom, visible_row_status):
        if status != "full":
            continue
        if r in vis_fix:
            rank = vis_fix.index(r)
            for c in range(SLOTS_PER_ROW):
                s = out_row * SLOTS_PER_ROW + c
                fi = rank * SLOTS_PER_ROW + c
                if s < MAX_STAGE2_SLOTS and fi < 20:
                    fix[s] = slot_fix_raw[fi]
                    valid[s] = True
        out_row += 1
        if out_row * SLOTS_PER_ROW >= MAX_STAGE2_SLOTS:
            break
    return fix, valid


def run_stage2():
    args = argparse.Namespace(
        manifest_glob=MANIFEST_GLOB,
        out=os.path.join(_RECGAZE_DIR, "stage2_dataset.jsonl"))

    files = sorted(glob.glob(args.manifest_glob))
    man = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    man = man.drop_duplicates(subset=["user_id", "task_id", "visit_index", "action_id"])
    clk = man[man["action_id"] == 4].copy()

    cti = build_click_target_index(ANALYSIS_CSV, ITEM_FEATURES_CSV)

    analysis = pd.read_csv(ANALYSIS_CSV)
    mid_map = {}
    for _, r in analysis[analysis["next_action"] == "click_movie"].iterrows():
        try:
            mid_map[(str(r["UserID"]), int(float(r["TaskID"])), int(float(r["visit_index"])))] = \
                int(float(r["next_action_detail"]))
        except (ValueError, TypeError):
            continue
    items = pd.read_csv(ITEM_FEATURES_CSV)
    movie_info = {}
    for _, r in items.iterrows():
        try:
            movie_info[(int(float(r["TaskID"])), int(float(r["MovieID"])))] = (
                str(r.get("TMDB_title", "")).strip(),
                str(r.get("TMDB_genres", "")).strip(),
                str(r.get("Carousel_genre", "")).strip())
        except (ValueError, TypeError):
            continue

    records = []
    drop = defaultdict(int)
    for _, row in clk.iterrows():
        uid = str(row["user_id"]); tid = int(row["task_id"]); vix = int(row["visit_index"])
        pos = cti.get((uid, tid, vix))
        if pos is None:
            drop["unresolved"] += 1; continue
        vg = parse_ints(row["visible_rows_geom"]); vs = parse_status(row["visible_row_status"])
        try:
            hp = {int(k): int(v) for k, v in json.loads(str(row.get("hp_state_json", "{}"))).items()}
        except (ValueError, TypeError):
            hp = {}
        slot = resolve_slot(vg, vs, hp, pos[0], pos[1])
        if slot is None:
            drop["no_slot"] += 1; continue
        if slot >= MAX_STAGE2_SLOTS:
            drop["slot>=15"] += 1; continue
        try:
            scroll_y = float(row["scroll_y"]) if not pd.isna(row["scroll_y"]) else 0.0
        except (ValueError, TypeError):
            scroll_y = 0.0
        bb = candidate_bboxes(vg, vs, scroll_y)
        if slot >= len(bb):
            drop["slot_no_bbox"] += 1; continue
        img = str(row["image_path"])
        mid = mid_map.get((uid, tid, vix))
        title, tmdb_g, car_g = movie_info.get((tid, mid), ("", "", "")) if mid is not None else ("", "", "")
        slot_fix, slot_fix_valid = gaze_target(row, vg, vs)
        records.append({
            "user_id": uid, "task_id": tid, "visit_index": vix, "split": split_of(tid),
            "image_path": img,
            "image_index_path": img.replace("/page_divide_real/image/", "/page_divide_real/image_index/"),
            "label": slot, "row": slot // 5, "col": slot % 5,
            "n_candidates": len(bb), "bboxes": bb,
            "clicked_movie_id": mid, "clicked_title": title,
            "clicked_genres": tmdb_g, "clicked_carousel_genre": car_g,
            "slot_fix": slot_fix, "slot_fix_valid": slot_fix_valid,
        })

    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    by_split = defaultdict(int); slot_hist = defaultdict(lambda: defaultdict(int)); ncand = defaultdict(int)
    for r in records:
        by_split[r["split"]] += 1
        slot_hist[r["split"]][r["label"]] += 1
        ncand[r["n_candidates"]] += 1

STAGE2_JSONL = os.path.join(_RECGAZE_DIR, "stage2_dataset.jsonl")
FIXATE_OUT_DIR = _RECGAZE_DIR
INDEX_IMAGES_DIR = os.path.join(_RECGAZE_DIR, "page_divide_real", "image_index")

SPLIT_SEED = 42
N_TEST_FREE, N_TEST_SEMI = 5, 1
N_VAL_FREE, N_VAL_SEMI = 3, 1
MAX_SLOTS = 15
HISTORY_MAX_ITEMS = 25


def slots_from_manifest(row, col_prefix):
    vis_fix = parse_ints(row.get("visible_rows", ""))
    raw = []
    for i in range(1, 21):
        try:
            v = float(row.get(f"{col_prefix}{i}", 0.0))
            v = 0.0 if pd.isna(v) else v
        except (ValueError, TypeError):
            v = 0.0
        raw.append(v)
    vals = [0.0] * MAX_SLOTS
    valid = [False] * MAX_SLOTS
    out_row = 0
    for r, status in zip(parse_ints(row.get("visible_rows_geom", "")),
                         parse_status(row.get("visible_row_status", ""))):
        if status != "full":
            continue
        if r in vis_fix:
            rank = vis_fix.index(r)
            for c in range(SLOTS_PER_ROW):
                s = out_row * SLOTS_PER_ROW + c
                fi = rank * SLOTS_PER_ROW + c
                if s < MAX_SLOTS and fi < 20:
                    vals[s] = raw[fi]
                    valid[s] = True
        out_row += 1
        if out_row * SLOTS_PER_ROW >= MAX_SLOTS:
            break
    return vals, valid


def build_taste_text(uid, current_task, user_clicks, max_items=HISTORY_MAX_ITEMS):
    cl = [(t, title, cg) for (t, title, cg) in user_clicks.get(uid, [])
          if t != current_task and title]
    if not cl:
        return ""
    cl.sort(key=lambda x: x[0])
    gc = Counter()
    for _, _, cg in cl:
        for tok in str(cg).split(","):
            tok = tok.strip()
            if tok:
                gc[tok] += 1
    lines = ["Movies this user clicked in other browsing sessions, with the genre "
             "category each was shown under:"]
    for _, title, cg in cl[:max_items]:
        g = ", ".join(x.strip() for x in str(cg).split(",")[:3] if x.strip())
        lines.append(f"  - {title}" + (f" ({g})" if g else ""))
    if len(cl) > max_items:
        lines.append(f"  ...and {len(cl) - max_items} more")
    top = ", ".join(f"{name} x{n}" for name, n in gc.most_common(5))
    if top:
        lines.append(f"Most-clicked genre shelves: {top}.")
    return "\n".join(lines)


def run_fixate():
    os.makedirs(FIXATE_OUT_DIR, exist_ok=True)

    recs = [json.loads(l) for l in open(STAGE2_JSONL)]
    recs = [r for r in recs if int(r["task_id"]) <= 35]

    rng = random.Random(SPLIT_SEED)
    free, semi = list(range(1, 31)), list(range(31, 36))
    test_tasks = sorted(rng.sample(free, N_TEST_FREE)) + sorted(rng.sample(semi, N_TEST_SEMI))
    free_rest = [t for t in free if t not in test_tasks]
    semi_rest = [t for t in semi if t not in test_tasks]
    val_tasks = sorted(rng.sample(free_rest, N_VAL_FREE)) + sorted(rng.sample(semi_rest, N_VAL_SEMI))
    train_tasks = sorted(t for t in free + semi if t not in test_tasks and t not in val_tasks)

    def split_of(t):
        return "test" if t in test_tasks else ("val" if t in val_tasks else "train")

    for r in recs:
        r["split"] = split_of(int(r["task_id"]))

    user_clicks = defaultdict(list)
    for r in recs:
        if r["split"] == "train" and r.get("clicked_title"):
            user_clicks[r["user_id"]].append(
                (int(r["task_id"]), r["clicked_title"], r.get("clicked_carousel_genre", "")))
    for r in recs:
        r["history_text"] = build_taste_text(r["user_id"], int(r["task_id"]), user_clicks)
    n_empty_hist = sum(1 for r in recs if not r["history_text"])

    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(MANIFEST_GLOB))],
                   ignore_index=True)
    df = (df.sort_values("cursor_sum", ascending=False, na_position="last")
            .drop_duplicates(subset=["user_id", "task_id", "visit_index"], keep="first"))
    df = df.set_index([df.user_id.astype(str), df.task_id.astype(int), df.visit_index.astype(int)])

    n_fix_ok = n_fix_bad = n_no_manifest = 0
    for r in recs:
        key = (str(r["user_id"]), int(r["task_id"]), int(r["visit_index"]))
        if key not in df.index:
            n_no_manifest += 1
            r["slot_cursor"] = [0.0] * MAX_SLOTS
            r["slot_cursor_valid"] = [False] * MAX_SLOTS
            r["has_cursor"] = False
            continue
        row = df.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        fix_re, _ = slots_from_manifest(row, "slot")
        stored = [float(x) for x in r["slot_fix"]]
        if all(abs(a - b) < 1e-6 for a, b in zip(fix_re, stored)):
            n_fix_ok += 1
        else:
            n_fix_bad += 1
        cur, cur_valid = slots_from_manifest(row, "cursor_slot")
        r["slot_cursor"] = cur
        r["slot_cursor_valid"] = cur_valid
        r["has_cursor"] = sum(cur) > 0

    n_img_missing = 0
    for r in recs:
        uid = str(r["user_id"])
        r["image_path"] = os.path.join(uid, os.path.basename(r.get("image_path", "")))
        r["image_index_path"] = os.path.join(uid, os.path.basename(r.get("image_index_path", "")))
        if not os.path.isfile(os.path.join(INDEX_IMAGES_DIR, r["image_index_path"])):
            n_img_missing += 1

    by_split = Counter(r["split"] for r in recs)
    stats = {"seed": SPLIT_SEED, "test_tasks": test_tasks, "val_tasks": val_tasks,
             "train_tasks": train_tasks, "counts": dict(by_split),
             "users": len(set(r["user_id"] for r in recs)),
             "empty_history": n_empty_hist, "images_missing": n_img_missing,
             "fix_remap_ok": n_fix_ok, "fix_remap_mismatch": n_fix_bad,
             "no_manifest_row": n_no_manifest}
    for s in ("train", "val", "test"):
        rs = [r for r in recs if r["split"] == s]
        pu = Counter(r["user_id"] for r in rs)
        stats[f"{s}_per_user_min_med_max"] = [
            min(pu.values()), sorted(pu.values())[len(pu) // 2], max(pu.values())]
        stats[f"{s}_users"] = len(pu)
        stats[f"{s}_has_fixation"] = sum(1 for r in rs if sum(r["slot_fix"]) > 0)
        stats[f"{s}_has_cursor"] = sum(1 for r in rs if r["has_cursor"])
        stats[f"{s}_fix_and_cursor"] = sum(1 for r in rs
                                           if sum(r["slot_fix"]) > 0 and r["has_cursor"])
        stats[f"{s}_free_semi"] = [sum(1 for r in rs if r["task_id"] <= 30),
                                   sum(1 for r in rs if r["task_id"] > 30)]

    out_jsonl = os.path.join(FIXATE_OUT_DIR, "fixate_dataset.jsonl")
    with open(out_jsonl, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(FIXATE_OUT_DIR, "fixate_split.json"), "w") as f:
        json.dump(stats, f, indent=2)


def main():
    run_manifests()
    run_cursor_patch()
    run_stage2()
    run_fixate()


if __name__ == "__main__":
    main()
