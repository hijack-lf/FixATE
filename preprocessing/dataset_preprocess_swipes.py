#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build init_interface_user_gaze(swipes).csv from summary_feedback.csv (matches GAZE_DATA_CSV)."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_RECGAZE = os.path.join(_REPO_ROOT, "datasets", "RecGaze")
DEFAULT_SUMMARY = os.path.join(_RECGAZE, "summary_feedback.csv")
DEFAULT_OUT = os.path.join(_RECGAZE, "init_interface_user_gaze(swipes).csv")

TASK_LO, TASK_HI = 1, 35


def _parse_task_id(s: str) -> Optional[int]:
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _parse_carousel_pos(s: Optional[str]) -> Optional[int]:
    if s is None or not str(s).strip():
        return None
    try:
        return int(round(float(s)))
    except ValueError:
        return None


def _ts(row: dict) -> float:
    try:
        return float(row.get("Timestamp") or 0.0)
    except ValueError:
        return 0.0


def _last_click_type(rows: List[dict]) -> Optional[str]:
    """Last non-empty Click_AOI_type by Timestamp; None if none."""
    events: List[Tuple[float, str]] = []
    for row in rows:
        ct = (row.get("Click_AOI_type") or "").strip()
        if not ct:
            continue
        events.append((_ts(row), ct))
    if not events:
        return None
    events.sort(key=lambda x: x[0])
    return events[-1][1]


def trial_kept(rows: List[dict]) -> bool:
    has_fix = False
    for row in rows:
        if (row.get("Fixation_Duration") or "").strip() or (
            row.get("Fixation_AOI_type") or ""
        ).strip():
            has_fix = True
        p = _parse_carousel_pos(row.get("Fixation_AOI_Carousel_position"))
        if p is not None and not (1 <= p <= 3):
            return False
    if not has_fix:
        return False
    return _last_click_type(rows) == "Movie"


@dataclass
class ExtractStats:
    n_trials: int
    n_trials_1_30: int
    n_trials_31_35: int
    n_rows: int


def extract(
    summary_path: str,
    out_path: str,
    task_lo: int = TASK_LO,
    task_hi: int = TASK_HI,
) -> ExtractStats:
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)

    with open(summary_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("empty CSV header row")
        for row in reader:
            tid = _parse_task_id(row.get("TaskID", ""))
            if tid is None or not (task_lo <= tid <= task_hi):
                continue
            groups[(row["UserID"], row["TaskID"])].append(row)

    kept: List[Tuple[str, str]] = []
    n_1_30 = n_31_35 = 0
    for key, rows in groups.items():
        if not trial_kept(rows):
            continue
        kept.append(key)
        tid = _parse_task_id(rows[0]["TaskID"])
        assert tid is not None
        if tid <= 30:
            n_1_30 += 1
        else:
            n_31_35 += 1

    out_rows: List[dict] = []
    for key in kept:
        chunk = groups[key]
        chunk.sort(key=_ts)
        out_rows.extend(chunk)
    out_rows.sort(key=lambda r: (r["UserID"], r["TaskID"], _ts(r)))

    d = os.path.dirname(os.path.abspath(out_path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    return ExtractStats(
        n_trials=len(kept),
        n_trials_1_30=n_1_30,
        n_trials_31_35=n_31_35,
        n_rows=len(out_rows),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default=DEFAULT_SUMMARY, help="summary_feedback.csv path")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output CSV path")
    ap.add_argument("--task-min", type=int, default=TASK_LO)
    ap.add_argument("--task-max", type=int, default=TASK_HI)
    args = ap.parse_args()

    extract(args.summary, args.out, args.task_min, args.task_max)


if __name__ == "__main__":
    main()
