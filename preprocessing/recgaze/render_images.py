import argparse
import csv
import os
import string
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

import build_gaze_tables as M


_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_RECGAZE_DIR = os.path.join(_ROOT, "datasets", "RecGaze")

ROW_Y_TOPS = [176, 516, 855, 1195, 1534, 1874, 2213, 2553, 2892, 3232]
POSTER_W, POSTER_H = 342, 193
COL_X_STARTS = [76, 427, 778, 1129, 1480]
VIEWPORT_W, VIEWPORT_H = 1920, 1080
BG_COLOR = (16, 16, 16)
TEXT_COLOR = (220, 220, 220)

ITEM_FEATURES_PATH = os.path.join(_RECGAZE_DIR, "raw/item_features.csv")
POSTER_CACHE_DIR = os.path.join(_RECGAZE_DIR, "posters_cache")
SINGLE_FOLDERS = [
    os.path.join(_RECGAZE_DIR, "TaskID1-30_init_singe_iamge"),
    os.path.join(_RECGAZE_DIR, "TaskID31-35_init_singe_iamge"),
    os.path.join(_RECGAZE_DIR, "TaskID36-40_init_singe_iamge"),
]
OUTPUT_DIR_DEFAULT = os.path.join(_RECGAZE_DIR, "page_divide_real/image"
)

os.makedirs(POSTER_CACHE_DIR, exist_ok=True)


def load_movie_info() -> Dict[Tuple[int, int, int], Tuple[int, str, str]]:
    info: Dict[Tuple[int, int, int], Tuple[int, str, str]] = {}
    with open(ITEM_FEATURES_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                tid = int(float(row["TaskID"]))
                car = int(float(row["Carousel_position"]))
                mov = int(float(row["Movie_position_in_carousel"]))
                mid = int(float(row["MovieID"]))
            except (ValueError, KeyError):
                continue
            genre = (row.get("Carousel_genre") or "").strip()
            url = (row.get("TMDB_poster_path") or "").strip()
            info[(tid, car, mov)] = (mid, genre, url)
    return info


def find_existing_poster(tid: int, car: int, mov: int, genre: str) -> Optional[str]:
    fname = f"taskid_{tid:02d}+Carousel_position_{car}+Movie_position_in_carousel_{mov}+{genre}"
    for d in SINGLE_FOLDERS + [POSTER_CACHE_DIR]:
        for ext in (".jpg", ".jpeg", ".png"):
            p = os.path.join(d, fname + ext)
            if os.path.exists(p):
                return p
    return None


def cache_path_for(tid: int, car: int, mov: int, genre: str, url: str) -> str:
    fname = f"taskid_{tid:02d}+Carousel_position_{car}+Movie_position_in_carousel_{mov}+{genre}"
    ext = os.path.splitext(os.path.basename(url))[1].lower() or ".jpg"
    if ext not in (".jpg", ".jpeg", ".png"):
        ext = ".jpg"
    return os.path.join(POSTER_CACHE_DIR, fname + ext)


def download_one(url: str, dst: str, timeout: int = 20) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        with open(dst, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False


def ensure_posters(
    info: Dict[Tuple[int, int, int], Tuple[int, str, str]],
    needed_keys: List[Tuple[int, int, int]],
    workers: int = 16,
) -> Dict[Tuple[int, int, int], Optional[str]]:
    out: Dict[Tuple[int, int, int], Optional[str]] = {}
    to_download: List[Tuple[Tuple[int, int, int], str, str]] = []
    for k in needed_keys:
        if k not in info:
            out[k] = None
            continue
        _, genre, url = info[k]
        local = find_existing_poster(k[0], k[1], k[2], genre)
        if local:
            out[k] = local
            continue
        if not url:
            out[k] = None
            continue
        dst = cache_path_for(k[0], k[1], k[2], genre, url)
        if os.path.exists(dst):
            out[k] = dst
            continue
        to_download.append((k, url, dst))

    if to_download:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(download_one, url, dst): (k, dst) for (k, url, dst) in to_download}
            ok = fail = 0
            for i, fut in enumerate(as_completed(futs), 1):
                k, dst = futs[fut]
                if fut.result() and os.path.exists(dst) and os.path.getsize(dst) > 1024:
                    out[k] = dst
                    ok += 1
                else:
                    out[k] = None
                    if os.path.exists(dst):
                        try:
                            os.remove(dst)
                        except OSError:
                            pass
                    fail += 1
    return out


VisitTuple = Tuple[str, str, str, int, Optional[float], Dict[int, int]]


_GLOBAL_CACHE: Dict[str, object] = {}


def _get_scroll_map_and_geom():
    if "scroll_map" not in _GLOBAL_CACHE:
        _GLOBAL_CACHE["scroll_map"] = M.build_fixation_scroll_map(M.NON_PUBLIC_PATH)
        _GLOBAL_CACHE["task_geom"] = M.load_task_row_geometry(M.AOI_PATH)
    return _GLOBAL_CACHE["scroll_map"], _GLOBAL_CACHE["task_geom"]


def collect_visits(target_user: str) -> List[VisitTuple]:
    scroll_map, task_geom = _get_scroll_map_and_geom()

    visits: List[VisitTuple] = []
    last_sig: Dict[Tuple[str, str, str], Tuple[str, str]] = {}
    visit_idx_map: Dict[Tuple[str, str, str], int] = {}
    last_scroll_y: Dict[Tuple[str, str, str], float] = {}
    occ_counter: Dict[Tuple[str, str, str], int] = {}
    hp_state_map: Dict[Tuple[str, str, str], Dict[int, int]] = {}

    def get_hp_state(sess):
        if sess not in hp_state_map:
            hp_state_map[sess] = {r: 0 for r in range(1, 11)}
        return hp_state_map[sess]

    with open(M.INPUT_SUMMARY_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = row["UserID"]
            if uid != target_user:
                continue
            tid = row["TaskID"]
            sid = row["SubjectID"]
            ts_raw = row["Timestamp"]
            session = (uid, tid, sid)
            hp_state = get_hp_state(session)

            occ_key = (uid, tid, ts_raw)
            occ = occ_counter.get(occ_key, 0)
            occ_counter[occ_key] = occ + 1

            scroll_y = scroll_map.get((uid, tid, ts_raw, occ))

            fix_dur = M.to_float(row.get("Fixation_Duration", ""))
            fix_type = (row.get("Fixation_AOI_type") or "").strip().lower()
            has_fix = fix_dur is not None or bool(fix_type)

            raw_vr = M.parse_triplet(row.get("Fixation_AOI_Visible_Carousel_rows", ""))
            raw_hp = M.parse_triplet(row.get("Fixation_AOI_Visible_Carousel_horizontal_pages", ""))
            all_hp = M.parse_triplet(row.get("Fixation_AOI_All_Carousel_horizontal_pages", ""))

            if all_hp is not None and len(all_hp) >= 10:
                for r in range(1, 11):
                    v = all_hp[r - 1]
                    if v in (0, 1, 2):
                        hp_state[r] = v
            elif raw_vr is not None and raw_hp is not None and len(raw_vr) == 3 == len(raw_hp):
                for i, r in enumerate(raw_vr):
                    if 1 <= r <= 10 and raw_hp[i] in (0, 1, 2):
                        hp_state[r] = raw_hp[i]

            click_type_raw = (row.get("Click_AOI_type") or "").strip()
            click_car = M.to_int_like(row.get("Click_AOI_Carousel_position", ""))
            if click_car is not None and 1 <= click_car <= 10:
                ct_low = click_type_raw.lower()
                if ct_low == "forward":
                    hp_state[click_car] = min(2, hp_state.get(click_car, 0) + 1)
                elif ct_low == "backward":
                    hp_state[click_car] = max(0, hp_state.get(click_car, 0) - 1)

            if not has_fix:
                continue

            closest_type = (row.get("Fixation_AOI_Closest_type") or "").strip().lower()
            use_closest = fix_type == "background" and closest_type == "movie"
            effective_type = "movie" if use_closest else fix_type
            if effective_type == "movie":
                if use_closest:
                    car_for_filter = M.to_int_like(row.get("Fixation_AOI_Closest_Carousel_position", ""))
                else:
                    car_for_filter = M.to_int_like(row.get("Fixation_AOI_Carousel_position", ""))
                raw_check = M.parse_triplet(row.get("Fixation_AOI_Visible_Carousel_rows", ""))
                if car_for_filter is not None and raw_check is not None and car_for_filter not in raw_check:
                    continue

            if raw_vr is None or raw_hp is None or len(raw_vr) != 3 or len(raw_hp) != 3:
                continue

            tid_int = M.to_int_like(tid)
            effective = M.resolve_effective_page(
                rule="real", task_id=tid_int, scroll_y=scroll_y,
                task_geom=task_geom, raw_visible=raw_vr, raw_horizontal=raw_hp,
                all_horizontal=all_hp,
            )
            if effective is None:
                continue
            vr_eff, hp_eff = effective
            cur_sig = (",".join(str(x) for x in vr_eff), ",".join(str(x) for x in hp_eff))
            prev_sig = last_sig.get(session)

            new_visit = False
            if prev_sig is None:
                visit_idx_map[session] = 0
                new_visit = True
            elif cur_sig != prev_sig:
                visit_idx_map[session] = visit_idx_map.get(session, 0) + 1
                new_visit = True
            else:
                is_stable = fix_dur is not None and fix_dur >= M.REAL_DURATION_THRESHOLD_MS
                last_sy = last_scroll_y.get(session)
                scrolled = (
                    last_sy is not None and scroll_y is not None
                    and abs(scroll_y - last_sy) > M.REAL_SCROLL_TOLERANCE_PX
                )
                if is_stable and scrolled:
                    visit_idx_map[session] = visit_idx_map.get(session, 0) + 1
                    new_visit = True

            if new_visit:
                visits.append((uid, tid, sid, visit_idx_map[session], scroll_y, dict(hp_state)))

            last_sig[session] = cur_sig
            if scroll_y is not None:
                last_scroll_y[session] = scroll_y

    return visits


def load_font(size: int = 16):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


GENRE_OFFSET_Y = 70
GENRE_TEXT_BBOX_H = 43

ARROW_CX_LEFT = 50
ARROW_CX_RIGHT = 1848
ARROW_RADIUS = 16


def _make_arrow_btn(direction: str) -> "Image.Image":
    pad = 4
    size = ARROW_RADIUS * 2 + pad * 2
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = cy = size // 2
    r = ARROW_RADIUS
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(70, 70, 70, 200))
    s = 7
    if direction == "left":
        pts = [(cx + s // 2, cy - s), (cx - s // 2, cy), (cx + s // 2, cy + s)]
    else:
        pts = [(cx - s // 2, cy - s), (cx + s // 2, cy), (cx - s // 2, cy + s)]
    d.line(pts, fill=(255, 255, 255, 255), width=3, joint="curve")
    return img


_ARROW_LEFT = None
_ARROW_RIGHT = None


def paste_arrow_button(canvas: "Image.Image", cx: int, cy: int, direction: str):
    global _ARROW_LEFT, _ARROW_RIGHT
    if _ARROW_LEFT is None:
        _ARROW_LEFT = _make_arrow_btn("left")
        _ARROW_RIGHT = _make_arrow_btn("right")
    btn = _ARROW_LEFT if direction == "left" else _ARROW_RIGHT
    half = btn.size[0] // 2
    canvas.paste(btn, (cx - half, cy - half), btn)


def render_visit(
    visit: VisitTuple,
    poster_paths: Dict[Tuple[int, int, int], Optional[str]],
    movie_info: Dict[Tuple[int, int, int], Tuple[int, str, str]],
    out_path: str,
):
    uid, tid, sid, vidx, sy, hp_state = visit
    sy_eff = float(sy) if sy is not None else 0.0
    tid_int = int(float(tid))

    canvas = Image.new("RGB", (VIEWPORT_W, VIEWPORT_H), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    poster_label_font = load_font(14)
    genre_font = load_font(36)

    for row_num in range(1, 11):
        screen_y_top = ROW_Y_TOPS[row_num - 1] - sy_eff
        if screen_y_top + POSTER_H < 0 or screen_y_top > VIEWPORT_H:
            continue

        hp = hp_state.get(row_num, 0)
        if hp not in (0, 1, 2):
            hp = 0

        _, genre, _ = movie_info.get((tid_int, row_num, 1), (0, "", ""))
        genre = genre or "?"

        genre_screen_y_top = screen_y_top - GENRE_OFFSET_Y
        if -GENRE_TEXT_BBOX_H < genre_screen_y_top < VIEWPORT_H:
            draw.text(
                (COL_X_STARTS[0], int(round(genre_screen_y_top))),
                genre,
                fill=(255, 255, 255),
                font=genre_font,
            )

        for col in range(5):
            mov_pos = hp * 5 + col + 1
            key = (tid_int, row_num, mov_pos)
            x = COL_X_STARTS[col]
            y = int(round(screen_y_top))
            poster_path = poster_paths.get(key)
            placed = False
            if poster_path and os.path.exists(poster_path):
                try:
                    p = Image.open(poster_path).convert("RGB").resize((POSTER_W, POSTER_H))
                    canvas.paste(p, (x, y))
                    placed = True
                except Exception:
                    placed = False
            if not placed:
                draw.rectangle([x, y, x + POSTER_W, y + POSTER_H], fill=(60, 60, 60), outline=(120, 120, 120))
                label = f"T{tid_int:02d} R{row_num} M{mov_pos}\n{genre}"
                draw.text((x + 8, y + 8), label, fill=TEXT_COLOR, font=poster_label_font)

        cy = int(round(screen_y_top + POSTER_H / 2))
        if 0 <= cy <= VIEWPORT_H:
            paste_arrow_button(canvas, ARROW_CX_LEFT, cy, "left")
            paste_arrow_button(canvas, ARROW_CX_RIGHT, cy, "right")

    canvas.save(out_path, "JPEG", quality=82, optimize=True)


def _discover_all_users() -> List[str]:
    seen: Dict[str, None] = {}
    with open(M.INPUT_SUMMARY_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = row.get("UserID", "").strip()
            if uid and uid not in seen:
                seen[uid] = None
    return list(seen.keys())


def _user_already_done(out_base: str, user: str) -> bool:
    udir = os.path.join(out_base, user)
    if not os.path.isdir(udir):
        return False
    for name in os.listdir(udir):
        if name.lower().endswith(".jpg"):
            return True
    return False


def render_one_user(
    user: str,
    out_base: str,
    info: Dict[Tuple[int, int, int], Tuple[int, str, str]],
    task_filter: str = "",
    limit: int = 0,
    workers: int = 24,
):
    out_dir = os.path.join(out_base, user)
    os.makedirs(out_dir, exist_ok=True)

    visits = collect_visits(user)
    if task_filter:
        visits = [v for v in visits if v[1] == task_filter]
    if limit:
        visits = visits[:limit]
    if not visits:
        return

    needed = set()
    for v in visits:
        _, tid, _, _, sy, hp_state = v
        tid_int = int(float(tid))
        sy_eff = float(sy) if sy is not None else 0.0
        for r in range(1, 11):
            screen_y_top = ROW_Y_TOPS[r - 1] - sy_eff
            if screen_y_top + POSTER_H < 0 or screen_y_top > VIEWPORT_H:
                continue
            hp = hp_state.get(r, 0)
            if hp not in (0, 1, 2):
                hp = 0
            for col in range(5):
                mov_pos = hp * 5 + col + 1
                needed.add((tid_int, r, mov_pos))
    poster_paths = ensure_posters(info, sorted(needed), workers=workers)

    n_done = 0
    for v in visits:
        uid, tid, sid, vidx, sy, hp_state = v
        out_path = os.path.join(out_dir, f"{uid}_{tid}_{vidx}.jpg")
        render_visit(v, poster_paths, info, out_path)
        n_done += 1


def run_plain(args):
    os.makedirs(OUTPUT_DIR_DEFAULT, exist_ok=True)

    info = load_movie_info()

    if args.user:
        users = [args.user]
    else:
        users = _discover_all_users()

    pending: List[str] = []
    for u in users:
        if not args.force and _user_already_done(OUTPUT_DIR_DEFAULT, u):
            continue
        pending.append(u)

    for i, u in enumerate(pending, 1):
        render_one_user(
            u, OUTPUT_DIR_DEFAULT, info,
            task_filter=args.task, limit=args.limit, workers=args.workers,
        )

INDEX_OUTPUT_DIR = os.path.join(_RECGAZE_DIR, "page_divide_real", "image_index")


LETTERS = string.ascii_uppercase

LABEL_GAP_TOP = 4
LABEL_FONT_SIZE = 26
LABEL_COLOR = (255, 255, 255)


def render_visit_indexed(
    visit: VisitTuple,
    poster_paths: Dict[Tuple[int, int, int], Optional[str]],
    movie_info: Dict[Tuple[int, int, int], Tuple[int, str, str]],
    out_path: str,
):
    tmp_path = out_path + ".base.jpg"
    render_visit(visit, poster_paths, movie_info, tmp_path)

    canvas = Image.open(tmp_path).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    label_font = load_font(LABEL_FONT_SIZE)

    uid, tid, sid, vidx, sy, hp_state = visit
    sy_eff = float(sy) if sy is not None else 0.0

    letter_iter = iter(LETTERS)
    for row_num in range(1, 11):
        screen_y_top = ROW_Y_TOPS[row_num - 1] - sy_eff
        if screen_y_top + POSTER_H < 0 or screen_y_top > VIEWPORT_H:
            continue
        if screen_y_top < 0 or screen_y_top + POSTER_H > VIEWPORT_H:
            continue
        label_y = int(round(screen_y_top + POSTER_H + LABEL_GAP_TOP))
        if label_y + LABEL_FONT_SIZE > VIEWPORT_H:
            label_y = VIEWPORT_H - LABEL_FONT_SIZE - 2
        for col in range(5):
            try:
                letter = next(letter_iter)
            except StopIteration:
                letter = "?"
            poster_cx = COL_X_STARTS[col] + POSTER_W // 2
            try:
                bbox = draw.textbbox((0, 0), letter, font=label_font)
                tw = bbox[2] - bbox[0]
                tx = poster_cx - tw // 2 - bbox[0]
            except Exception:
                tx = poster_cx - 8
            draw.text((tx, label_y), letter, font=label_font, fill=LABEL_COLOR)

    canvas.save(out_path, "JPEG", quality=85, optimize=True)
    try:
        os.remove(tmp_path)
    except OSError:
        pass


def render_user_indexed(user: str, out_base: str, limit: int, workers: int = 16):
    out_dir = os.path.join(out_base, user)
    os.makedirs(out_dir, exist_ok=True)

    info = load_movie_info()

    visits = collect_visits(user)
    if limit:
        visits = visits[:limit]
    if not visits:
        return

    needed = set()
    for v in visits:
        _, tid, _, _, sy, hp_state = v
        tid_int = int(float(tid))
        sy_eff = float(sy) if sy is not None else 0.0
        for r in range(1, 11):
            screen_y_top = ROW_Y_TOPS[r - 1] - sy_eff
            if screen_y_top + POSTER_H < 0 or screen_y_top > VIEWPORT_H:
                continue
            hp = hp_state.get(r, 0)
            if hp not in (0, 1, 2):
                hp = 0
            for col in range(5):
                needed.add((tid_int, r, hp * 5 + col + 1))
    poster_paths = ensure_posters(info, sorted(needed), workers=workers)

    for i, v in enumerate(visits, 1):
        uid, tid, sid, vidx, sy, hp_state = v
        out_path = os.path.join(out_dir, f"{uid}_{tid}_{vidx}.jpg")
        render_visit_indexed(v, poster_paths, info, out_path)


def run_indexed(args):
    os.makedirs(INDEX_OUTPUT_DIR, exist_ok=True)
    users = [args.user] if args.user else _discover_all_users()
    for u in users:
        if not args.force and not args.user and _user_already_done(INDEX_OUTPUT_DIR, u):
            continue
        render_user_indexed(u, INDEX_OUTPUT_DIR, args.limit, args.workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="")
    ap.add_argument("--task", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run_plain(args)
    run_indexed(args)


if __name__ == "__main__":
    main()
