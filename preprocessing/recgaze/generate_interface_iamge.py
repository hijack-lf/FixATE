#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expand RecGaze interface images from raw swipe sequences.
- Sequences: summary_feedback.csv (same columns as processed swipes; swipes may be in Click or Cursor fields)
- Movie metadata and poster URLs: item_features.csv
Default CLI run processes TaskID 1–35 in one pass; use positional mode or --only-* for subsets.
"""

import argparse
import csv
import os
import urllib.request
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RECGAZE_RAW = os.path.join(_REPO_ROOT, "datasets", "raw", "RecGaze")
SUMMARY_FEEDBACK_CSV = os.path.join(RECGAZE_RAW, "summary_feedback.csv")
ITEM_FEATURES_CSV = os.path.join(RECGAZE_RAW, "item_features.csv")
POSTER_CACHE_DIR = os.path.join(RECGAZE_RAW, "poster_cache")
OUTPUT_DIR = os.path.join(_REPO_ROOT, "datasets", "RecGaze", "interface_iamge")

POSTER_MAX_WIDTH = 200
POSTER_MAX_HEIGHT = 300
ROW_HEIGHT = 200
COL_WIDTH = 220
TITLE_HEIGHT = 30
CANVAS_WIDTH = COL_WIDTH * 5
CANVAS_HEIGHT = ROW_HEIGHT * 3
BG_COLOR = (20, 20, 20)
TEXT_COLOR = (255, 255, 255)


class CarouselState:
    """Carousel row state (three rows, five slots each)."""

    def __init__(self, task_id: int):
        self.task_id = task_id
        self.carousel_positions = {
            1: list(range(1, 6)),
            2: list(range(1, 6)),
            3: list(range(1, 6)),
        }

    def forward(self, carousel_pos: int):
        if carousel_pos not in self.carousel_positions:
            return
        current = self.carousel_positions[carousel_pos]
        new_positions = []
        for pos in current:
            new_pos = pos + 5
            if new_pos > 15:
                new_pos = new_pos - 15
            new_positions.append(new_pos)
        self.carousel_positions[carousel_pos] = new_positions

    def backward(self, carousel_pos: int):
        if carousel_pos not in self.carousel_positions:
            return
        current = self.carousel_positions[carousel_pos]
        new_positions = []
        for pos in current:
            new_pos = pos - 5
            if new_pos < 1:
                new_pos = new_pos + 15
            new_positions.append(new_pos)
        self.carousel_positions[carousel_pos] = new_positions

    def get_state(self) -> Dict[int, List[int]]:
        return self.carousel_positions.copy()


def normalize_carousel_row(carousel_pos_str: str) -> Optional[int]:
    """Map CSV carousel position to UI row 1–3 (handles row index or global slot 1–15)."""
    if not carousel_pos_str or not str(carousel_pos_str).strip():
        return None
    try:
        v = int(round(float(carousel_pos_str)))
    except ValueError:
        return None
    if 1 <= v <= 3:
        return v
    if 1 <= v <= 15:
        return (v - 1) // 5 + 1
    return None


def iter_row_interface_events(record: dict) -> Iterator[Tuple[str, Optional[int]]]:
    """Parse per-row UI events; prefer Click_AOI_* then Cursor_AOI_*; Movie click ends sequence."""
    click_t = (record.get("Click_AOI_type") or "").strip()
    if click_t == "Movie":
        yield ("movie", None)
        return
    if click_t == "Forward":
        row = normalize_carousel_row(record.get("Click_AOI_Carousel_position", ""))
        if row is not None:
            yield ("forward", row)
        return
    if click_t == "Backward":
        row = normalize_carousel_row(record.get("Click_AOI_Carousel_position", ""))
        if row is not None:
            yield ("backward", row)
        return

    cursor_t = (record.get("Cursor_AOI_type") or "").strip()
    if cursor_t == "Forward":
        row = normalize_carousel_row(record.get("Cursor_AOI_Carousel_position", ""))
        if row is not None:
            yield ("forward", row)
    elif cursor_t == "Backward":
        row = normalize_carousel_row(record.get("Cursor_AOI_Carousel_position", ""))
        if row is not None:
            yield ("backward", row)


def record_timestamp(record: dict) -> float:
    try:
        return float(record.get("Timestamp") or 0.0)
    except ValueError:
        return 0.0


def load_movie_data(task_id: int) -> Dict[Tuple[int, int], Dict]:
    """Load all movies for task_id from item_features.csv (key: carousel row, in-row position)."""
    movies: Dict[Tuple[int, int], Dict] = {}
    with open(ITEM_FEATURES_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(row["TaskID"]) != task_id:
                    continue
                carousel_pos = int(row["Carousel_position"])
                movie_pos = int(row["Movie_position_in_carousel"])
                movies[(carousel_pos, movie_pos)] = {
                    "movie_id": row["MovieID"],
                    "title": row["TMDB_title"],
                    "genre": row["Carousel_genre"],
                    "poster_url": row["TMDB_poster_path"],
                }
            except (ValueError, KeyError):
                continue
    return movies


def download_poster(url: str, save_path: str) -> bool:
    if os.path.exists(save_path):
        return True
    if not url or not url.strip():
        return False
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Mozilla/5.0")
        with urllib.request.urlopen(req, timeout=15) as response:
            data = response.read()
            with open(save_path, "wb") as f:
                f.write(data)
        return True
    except Exception:
        return False


def get_poster_filename(task_id: int, carousel_pos: int, movie_pos: int, genre: str, poster_url: str) -> str:
    genre_clean = genre.replace("/", "_").replace("\\", "_")
    ext = os.path.splitext(os.path.basename(poster_url))[1] or ".jpg"
    return (
        f"taskid_{task_id:02d}+Carousel_position_{carousel_pos}+Movie_position_in_carousel_{movie_pos}+"
        f"{genre_clean}{ext}"
    )


def load_poster_image(
    task_id: int,
    carousel_pos: int,
    movie_pos: int,
    genre: str,
    poster_url: str,
    poster_dir: str,
) -> Image.Image:
    filename = get_poster_filename(task_id, carousel_pos, movie_pos, genre, poster_url)
    local_path = os.path.join(poster_dir, filename)
    os.makedirs(poster_dir, exist_ok=True)
    if not os.path.exists(local_path):
        if poster_url and poster_url.strip():
            download_poster(poster_url, local_path)
    if os.path.exists(local_path):
        try:
            img = Image.open(local_path)
            img.thumbnail((POSTER_MAX_WIDTH, POSTER_MAX_HEIGHT), Image.Resampling.LANCZOS)
            return img
        except Exception:
            pass
    placeholder = Image.new("RGB", (POSTER_MAX_WIDTH, POSTER_MAX_HEIGHT), color=(100, 100, 100))
    draw = ImageDraw.Draw(placeholder)
    draw.text((POSTER_MAX_WIDTH // 2, POSTER_MAX_HEIGHT // 2), "No Image", fill=(255, 255, 255), anchor="mm")
    return placeholder


def generate_interface_image(
    task_id: int,
    carousel_state: CarouselState,
    movie_data: Dict[Tuple[int, int], Dict],
    output_path: str,
    poster_dir: str,
):
    canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), color=BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        genre_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        title_font = ImageFont.load_default()
        genre_font = ImageFont.load_default()
    state = carousel_state.get_state()
    for row_idx, carousel_pos in enumerate([1, 2, 3]):
        y_offset = row_idx * ROW_HEIGHT
        movie_positions = state[carousel_pos]
        genre_name = ""
        for movie_pos in movie_positions:
            key = (carousel_pos, movie_pos)
            if key in movie_data:
                genre_name = movie_data[key]["genre"]
                break
        if genre_name:
            draw.text((10, y_offset + 20), genre_name, fill=TEXT_COLOR, font=genre_font)
        for col_idx, movie_pos_in_carousel in enumerate(movie_positions):
            x_offset = col_idx * COL_WIDTH
            key = (carousel_pos, movie_pos_in_carousel)
            if key not in movie_data:
                continue
            movie_info = movie_data[key]
            poster = load_poster_image(
                task_id,
                carousel_pos,
                movie_pos_in_carousel,
                movie_info["genre"],
                movie_info["poster_url"],
                poster_dir,
            )
            poster_x = x_offset + (COL_WIDTH - poster.width) // 2
            poster_y = y_offset + TITLE_HEIGHT + (ROW_HEIGHT - TITLE_HEIGHT - poster.height) // 2
            canvas.paste(poster, (poster_x, poster_y))
    canvas.save(output_path)


def process_swipes(
    task_id_range: tuple = (1, 35),
    poster_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    only_user_task: Optional[tuple] = None,
):
    if poster_dir is None:
        poster_dir = POSTER_CACHE_DIR
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(poster_dir, exist_ok=True)
    with open(SUMMARY_FEEDBACK_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        swipes_data = list(reader)
    grouped = defaultdict(list)
    for row in swipes_data:
        user_id = row["UserID"]
        task_id = row["TaskID"]
        if task_id:
            grouped[(user_id, task_id)].append(row)
    for _key, recs in grouped.items():
        recs.sort(key=record_timestamp)
    generated_count = 0
    skipped_count = 0
    for (user_id, task_id_str), records in sorted(grouped.items()):
        try:
            task_id = int(task_id_str)
            if not (task_id_range[0] <= task_id <= task_id_range[1]):
                continue
        except Exception:
            continue
        if only_user_task is not None and (user_id, task_id) != (only_user_task[0], only_user_task[1]):
            continue
        carousel = CarouselState(task_id)
        movie_data = load_movie_data(task_id)
        if not movie_data:
            skipped_count += 1
            continue
        found_movie_click = False
        for record in records:
            for ev, _row in iter_row_interface_events(record):
                if ev == "movie":
                    found_movie_click = True
                    break
                if ev == "forward" and _row is not None:
                    carousel.forward(_row)
                elif ev == "backward" and _row is not None:
                    carousel.backward(_row)
            if found_movie_click:
                break
        if not found_movie_click:
            skipped_count += 1
            continue
        output_filename = f"User_{user_id}_TaskID_{task_id:02d}_final_interface.png"
        output_path = os.path.join(output_dir, output_filename)
        try:
            generate_interface_image(task_id, carousel, movie_data, output_path, poster_dir)
            generated_count += 1
        except Exception:
            skipped_count += 1
    return generated_count, skipped_count


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="Generate final interface PNGs from RecGaze summary_feedback + item_features.",
    )
    p.add_argument(
        "mode",
        nargs="?",
        default="1-35",
        choices=("1-35", "36-40", "all"),
        help="TaskID range preset (default: 1–35 if no --only)",
    )
    p.add_argument("--only-user", type=str, default=None, help="With --only-task: single user-task pair")
    p.add_argument("--only-task", type=int, default=None, help="With --only-user: task id 1–40")
    args = p.parse_args(argv)
    if args.only_user is not None or args.only_task is not None:
        if args.only_user is None or args.only_task is None:
            p.error("--only-user and --only-task must be used together")
        ot = args.only_task
        if 1 <= ot <= 35:
            task_range = (1, 35)
        elif 36 <= ot <= 40:
            task_range = (36, 40)
        else:
            p.error("TaskID must be in 1–40")
        process_swipes(task_id_range=task_range, only_user_task=(args.only_user, ot))
        return
    if args.mode == "all":
        process_swipes(task_id_range=(1, 35))
        process_swipes(task_id_range=(36, 40))
    elif args.mode == "1-35":
        process_swipes(task_id_range=(1, 35))
    elif args.mode == "36-40":
        process_swipes(task_id_range=(36, 40))


if __name__ == "__main__":
    main()
