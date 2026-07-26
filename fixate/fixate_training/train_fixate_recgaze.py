import os
import sys
import json
import time
import math
import copy
import logging
import importlib.util
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from itertools import product
from datetime import datetime, timezone, timedelta

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.spatial.distance import jensenshannon


import csv
import re
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config.recgaze_config import *


def slot_to_letter(slot_id: int) -> str:
    if slot_id < 1:
        slot_id = 1
    elif slot_id > 15:
        slot_id = 15

    return chr(ord('A') + slot_id - 1)


def _normalize_option_letter(s: str) -> str:
    if not s or not isinstance(s, str):
        return "OTHER"
    u = s.strip().upper()
    if len(u) == 1 and "A" <= u <= "O":
        return u
    return "OTHER"


def compute_option_ratios(letters: List[str], num_slots: int = 15) -> Dict[str, float]:
    from collections import Counter
    n = len(letters)
    if n == 0:
        return {}
    valid = [chr(ord("A") + i) for i in range(num_slots)]
    cnt = Counter(letters)
    return {k: cnt.get(k, 0) / n for k in valid + ["OTHER"]}


def define_slot_bboxes(
    image_width: int = UI_IMAGE_WIDTH,
    image_height: int = UI_IMAGE_HEIGHT,
    num_rows: int = NUM_ROWS,
    num_cols: int = NUM_COLS,
) -> Dict[int, Tuple[int, int, int, int]]:
    title_height_ratio = 0.1
    margin_width_ratio = 0.05
    margin_height_ratio = 0.05

    usable_y_start = int(image_height * title_height_ratio)
    usable_y_end = int(image_height * (1 - margin_height_ratio))
    usable_x_start = int(image_width * margin_width_ratio)
    usable_x_end = int(image_width * (1 - margin_width_ratio))

    usable_width = usable_x_end - usable_x_start
    usable_height = usable_y_end - usable_y_start

    slot_width = usable_width / num_cols
    slot_height = usable_height / num_rows

    slot_bboxes = {}
    slot_id = 1

    for row in range(num_rows):
        for col in range(num_cols):
            x1 = int(usable_x_start + col * slot_width)
            y1 = int(usable_y_start + row * slot_height)
            x2 = int(usable_x_start + (col + 1) * slot_width)
            y2 = int(usable_y_start + (row + 1) * slot_height)

            slot_bboxes[slot_id] = (x1, y1, x2, y2)
            slot_id += 1

    return slot_bboxes


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _build_internvl_transform(input_size: int = 448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def _dynamic_preprocess_internvl(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images, target_aspect_ratio

def _preprocess_image_for_internvl(image: Image.Image, input_size=448, max_num=12):
    transform = _build_internvl_transform(input_size)
    tiles, aspect_ratio = _dynamic_preprocess_internvl(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(t) for t in tiles])
    num_patches = len(tiles)
    return pixel_values, num_patches, aspect_ratio


def load_user_features(csv_path: str) -> Dict[str, Dict[str, str]]:
    user_features = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_id = row.get('UserID', '').strip()
            if user_id:
                user_features[user_id] = {
                    'Top_genre': row.get('Top_genre', '').strip(),
                    'Preferred_genres': row.get('Preferred_genres', '').strip()
                }
    return user_features


def load_movie_layout_from_item_features(
    csv_path: str,
    task_id_min: int = 1,
    task_id_max: int = 35,
    max_movie_pos: int = 15,
) -> Dict[Tuple[int, int, int], int]:
    layout = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                task_id = int(row.get('TaskID', ''))
                if not (task_id_min <= task_id <= task_id_max):
                    continue
                carousel_pos = int(row.get('Carousel_position', ''))
                movie_pos = int(row.get('Movie_position_in_carousel', ''))
                movie_id = int(row.get('MovieID', ''))

                if 1 <= movie_pos <= max_movie_pos and 1 <= carousel_pos <= 3:
                    key = (task_id, carousel_pos, movie_pos)
                    layout[key] = movie_id
            except (ValueError, TypeError):
                continue
    return layout


def load_gaze_data(csv_path: str, movie_layout: Dict, logger: Optional[logging.Logger] = None,
                   signal: str = "fixation") -> List[Dict]:
    assert signal in ("fixation", "cursor"), f"bad signal={signal!r}"
    if signal == "cursor":
        duration_col = "Cursor_Duration"
        carousel_col = "Cursor_AOI_Carousel_position"
        movie_col    = "Cursor_AOI_Movie_position_in_carousel"
        signal_scale = 1000.0
    else:
        duration_col = "Fixation_Duration"
        carousel_col = "Fixation_AOI_Carousel_position"
        movie_col    = "Fixation_AOI_Movie_position_in_carousel"
        signal_scale = 1.0

    samples = []
    grouped_data = defaultdict(lambda: {
        'fixations': defaultdict(float),
        'clicks': set()
    })

    total_rows = 0
    valid_taskids = set()
    taskid_with_fixations = set()
    taskid_with_clicks = set()

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            user_id = row.get('UserID', '').strip()
            task_id_str = row.get('TaskID', '').strip()

            if not user_id or not task_id_str:
                continue

            try:
                task_id = int(task_id_str)
            except ValueError:
                continue

            if not (1 <= task_id <= 35):
                continue

            valid_taskids.add(task_id)
            key = (user_id, task_id)

            fixation_duration_str = row.get(duration_col, '').strip()
            if fixation_duration_str:
                try:
                    fixation_duration = float(fixation_duration_str) * signal_scale
                    carousel_pos_str = row.get(carousel_col, '').strip()
                    movie_pos_str = row.get(movie_col, '').strip()

                    if carousel_pos_str and movie_pos_str:
                        carousel_pos = int(float(carousel_pos_str))
                        movie_pos = int(float(movie_pos_str))

                        if 1 <= carousel_pos <= 3 and 1 <= movie_pos <= 5:
                            slot_id = (carousel_pos - 1) * 5 + movie_pos
                            grouped_data[key]['fixations'][slot_id] += fixation_duration
                            taskid_with_fixations.add(task_id)
                except (ValueError, TypeError):
                    pass

            click_movie_id_str = row.get('Click_AOI_MovieID', '').strip()
            if click_movie_id_str:
                try:
                    click_movie_id = int(float(click_movie_id_str))
                    grouped_data[key]['clicks'].add(click_movie_id)
                    taskid_with_clicks.add(task_id)
                except (ValueError, TypeError):
                    pass


    samples_with_click = 0
    samples_with_fixation = 0
    samples_with_both = 0
    samples_click_not_found = 0

    for (user_id, task_id), data in grouped_data.items():
        dwell_time = np.zeros(NUM_SLOTS, dtype=np.float32)
        for slot_id, duration in data['fixations'].items():
            if 1 <= slot_id <= NUM_SLOTS:
                dwell_time[slot_id - 1] = duration

        has_fixation = dwell_time.sum() > 0
        if has_fixation:
            samples_with_fixation += 1

        choice_slot = None
        if len(data['clicks']) > 0:
            samples_with_click += 1
            click_found = False
            for click_movie_id in data['clicks']:
                for row_num in range(1, NUM_ROWS + 1):
                    for col_num in range(1, 16):
                        layout_key = (task_id, row_num, col_num)
                        if layout_key in movie_layout and movie_layout[layout_key] == click_movie_id:
                            visible_col = ((col_num - 1) % NUM_COLS) + 1
                            choice_slot = (row_num - 1) * NUM_COLS + visible_col
                            click_found = True
                            break
                    if choice_slot:
                        break
                if choice_slot:
                    break

            if not click_found:
                samples_click_not_found += 1

        if choice_slot is not None and dwell_time.sum() > 0:
            samples_with_both += 1
            samples.append({
                'user_id': user_id,
                'task_id': task_id,
                'choice_slot': choice_slot,
                'dwell_time': dwell_time,
            })


    return samples


def gaze_distribution_from_dwell_time(dwell_time: np.ndarray, tau: float = GAZE_TAU, eps: float = 1e-8) -> np.ndarray:
    t = dwell_time.astype(np.float32)
    g = t / (t.sum() + eps)

    g = np.exp(np.log(g + eps) / tau)
    g = g / (g.sum() + eps)

    return g


VAL_SAMPLES: List[Dict] = []


def _profile_text_for(uid, sample, user_features):
    prof = user_features.get(uid, {})
    txt = (f"Top_genre: {prof.get('Top_genre', '')}, "
           f"Preferred_genres: {prof.get('Preferred_genres', '')}")
    if USE_HISTORY and sample.get("history_text"):
        txt = txt + "\n" + sample["history_text"]
    return txt


def _open_page_image_and_bboxes(sample):
    p = sample.get("image_path", "")
    if not (p and os.path.exists(p)):
        return None, None
    img = Image.open(p).convert("RGB")
    scale = 1.0
    if IMAGE_MAX_W and img.width > IMAGE_MAX_W:
        scale = IMAGE_MAX_W / float(img.width)
        img = img.resize(
            (IMAGE_MAX_W, max(1, int(round(img.height * scale)))),
            Image.LANCZOS)
    bb = np.asarray(sample["bboxes"], dtype=np.float32) * scale
    return img, bb


def load_fixate_samples(logger=None):
    global VAL_SAMPLES
    samples_all = []
    n_drop = 0
    with open(FIXATE_JSONL, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            fix_ok = sum(r["slot_fix"]) > 0
            cur_ok = sum(r["slot_cursor"]) > 0
            if SIGNAL_INTERSECT and not (fix_ok and cur_ok):
                n_drop += 1
                continue
            sig_ok = fix_ok if SLOT_KL_SIGNAL == "fixation" else cur_ok
            if not sig_ok:
                n_drop += 1
                continue
            dwell = np.array(r["slot_fix"] if SLOT_KL_SIGNAL == "fixation"
                             else r["slot_cursor"], dtype=np.float32)
            bb = np.zeros((NUM_SLOTS, 4), dtype=np.float32)
            for j, box in enumerate(r["bboxes"][:NUM_SLOTS]):
                bb[j] = box
            samples_all.append({
                "user_id": str(r["user_id"]),
                "task_id": int(r["task_id"]),
                "visit_index": int(r["visit_index"]),
                "choice_slot": int(r["label"]) + 1,
                "dwell_time": dwell,
                "image_path": os.path.join(INTERFACE_IMAGES_DIR,
                    r["image_index_path"] if USE_INDEXED_IMAGES else r["image_path"]),
                "bboxes": bb,
                "n_candidates": int(r["n_candidates"]),
                "history_text": r.get("history_text", ""),
                "split": r["split"],
            })
    VAL_SAMPLES = [s for s in samples_all if s["split"] == "val"]
    samples = [s for s in samples_all if s["split"] in ("train", "test")]
    test_idx = [i for i, s in enumerate(samples) if s["split"] == "test"]
    if logger:
        n_tr = sum(1 for s in samples if s["split"] == "train")
    return samples, test_idx


def _compute_wilcoxon_per_user(bl_tf_path, tr_tf_path, bl_fg_path, tr_fg_path, logger=None):
    from scipy.stats import wilcoxon

    def _per_user_means(path, keys):
        with open(path, "r", encoding="utf-8") as f:
            recs = json.load(f)
        by_u = defaultdict(lambda: defaultdict(list))
        for r in recs:
            for k in keys:
                if k in r and r[k] is not None:
                    by_u[r["user_id"]][k].append(float(r[k]))
        return {u: {k: float(np.mean(v)) for k, v in d.items()}
                for u, d in by_u.items()}

    out = {}
    plan = {
        "tf": (bl_tf_path, tr_tf_path,
               ["click@1", "click@3", "click@5", "js_div", "kl_div", "cosine_sim",
                "attn_logloss", "attn_auc", "gaze@1", "gaze@3", "gaze@5"]),
        "fg": (bl_fg_path, tr_fg_path,
               ["answer_correct", "click@1", "attn_logloss", "attn_auc"]),
    }
    for tag, (bp, tp, keys) in plan.items():
        if not (os.path.exists(bp) and os.path.exists(tp)):
            continue
        bl_u = _per_user_means(bp, keys)
        tr_u = _per_user_means(tp, keys)
        users = sorted(set(bl_u) & set(tr_u))
        for k in keys:
            pair = [(bl_u[u][k], tr_u[u][k]) for u in users
                    if k in bl_u[u] and k in tr_u[u]]
            if len(pair) < 10:
                continue
            b = np.array([x[0] for x in pair])
            t = np.array([x[1] for x in pair])
            diffs = t - b
            if np.allclose(diffs, 0):
                p = 1.0
            else:
                try:
                    p = float(wilcoxon(t, b).pvalue)
                except ValueError:
                    p = float("nan")
            out[f"{tag}_{k}"] = {
                "n_users": int(len(pair)),
                "mean_backbone": float(b.mean()),
                "mean_fixate": float(t.mean()),
                "mean_delta": float(diffs.mean()),
                "p_wilcoxon": p,
            }
    return out


class GazeDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
        user_features: Dict[str, Dict[str, str]],
        poster_images_dir: str,
        expanded_dirs: Optional[List[Tuple[int, int, str]]] = None,
    ):
        self.samples = samples
        self.user_features = user_features
        self.poster_images_dir = poster_images_dir
        self.expanded_dirs = expanded_dirs or []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        user_id = sample['user_id']
        task_id = sample['task_id']
        choice_slot = sample['choice_slot']
        dwell_time = sample['dwell_time']

        image, _bb = _open_page_image_and_bboxes(sample)

        profile_text = _profile_text_for(user_id, sample, self.user_features)
        g = gaze_distribution_from_dwell_time(dwell_time)

        return {
            'user_id': user_id,
            'task_id': task_id,
            'image': image,
            'profile_text': profile_text,
            'choice_slot': choice_slot - 1,
            'gaze_dist': torch.from_numpy(g),
            'bboxes': torch.from_numpy(_bb),
            'n_candidates': int(sample['n_candidates']),
        }


class PromptBasisModule(nn.Module):
    def __init__(self, num_basis: int = NUM_BASIS, num_soft_tokens: int = NUM_SOFT_TOKENS, hidden_dim: int = 3584,
                 use_user_alpha: bool = USE_USER_ALPHA):
        super().__init__()
        self.num_basis = num_basis
        self.num_soft_tokens = num_soft_tokens
        self.hidden_dim = hidden_dim
        self.use_user_alpha = use_user_alpha

        self.basis = nn.Parameter(torch.randn(num_basis, num_soft_tokens, hidden_dim) * 0.01)
        self.user_alphas = nn.ParameterDict()

    def get_or_create_user_alpha(self, user_id: str) -> nn.Parameter:
        if user_id not in self.user_alphas:
            device = self.basis.device
            alpha = nn.Parameter(torch.zeros(self.num_basis, device=device))
            self.user_alphas[user_id] = alpha
        return self.user_alphas[user_id]

    def forward(self, user_ids: List[str]) -> torch.Tensor:
        batch_size = len(user_ids)
        soft_prompts = []

        for user_id in user_ids:
            if self.use_user_alpha:
                alpha = self.get_or_create_user_alpha(user_id)
                alpha = alpha.to(device=self.basis.device)
                pi = F.softmax(alpha, dim=0).to(dtype=self.basis.dtype)
            else:
                pi = torch.ones(self.num_basis, device=self.basis.device, dtype=self.basis.dtype) / self.num_basis
            user_prompt = torch.einsum('b,bmd->md', pi, self.basis)
            soft_prompts.append(user_prompt)

        soft_prompts = torch.stack(soft_prompts, dim=0)
        return soft_prompts

    def get_user_alpha_l2_loss(self, user_ids: List[str]) -> torch.Tensor:
        if not self.use_user_alpha:
            return torch.zeros((), device=self.basis.device)
        loss = 0.0
        for user_id in user_ids:
            alpha = self.get_or_create_user_alpha(user_id)
            loss += torch.sum(alpha ** 2)
        return loss / len(user_ids)


class AttnLRPTrainer:

    def __init__(self, model_name: str = MODEL_NAME, model_type: str = MODEL_TYPE,
                 prompt_basis: Optional[PromptBasisModule] = None,
                 logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.model_type = model_type

        if self.model_type == "internvl":
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, use_fast=False)
            self.processor = None
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": 0} if torch.cuda.is_available() else None,
                trust_remote_code=True,
                use_flash_attn=False,
            )
            self.model.to(DEVICE)
            self._internvl_num_image_token = self.model.num_image_token
            img_ctx_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            self.model.img_context_token_id = img_ctx_id
        else:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer = getattr(self.processor, "tokenizer", None)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": 0} if torch.cuda.is_available() else None,
                trust_remote_code=True,
                attn_implementation="eager",
            )
            self.model.to(DEVICE)

        self.config = TrainerConfig()

        if self.model_type == "internvl":
            orig_layers = self.config.attnlrp_max_layers
            self.config.attnlrp_max_layers = INTERNVL_ATTNLRP_MAX_LAYERS


        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        if prompt_basis is None:
            hidden_dim = None
            if hasattr(self.model.config, 'hidden_size'):
                hidden_dim = self.model.config.hidden_size
            elif hasattr(self.model.config, 'text_config') and self.model.config.text_config is not None:
                if hasattr(self.model.config.text_config, 'hidden_size'):
                    hidden_dim = self.model.config.text_config.hidden_size
            elif hasattr(self.model, 'get_input_embeddings'):
                try:
                    embedding = self.model.get_input_embeddings()
                    if hasattr(embedding, 'embedding_dim'):
                        hidden_dim = embedding.embedding_dim
                    elif hasattr(embedding, 'weight'):
                        hidden_dim = embedding.weight.shape[-1]
                except:
                    pass

            if hidden_dim is None:
                hidden_dim = 3584

            prompt_basis = PromptBasisModule(num_basis=NUM_BASIS, num_soft_tokens=NUM_SOFT_TOKENS, hidden_dim=hidden_dim)

        model_dtype = next(self.model.parameters()).dtype

        self.prompt_basis = prompt_basis.to(dtype=model_dtype)


        emb_dim = self.model.get_input_embeddings().weight.shape[-1]
        basis_dim = self.prompt_basis.basis.shape[-1]

        self.slot_bboxes = define_slot_bboxes(image_width=UI_IMAGE_WIDTH, image_height=UI_IMAGE_HEIGHT)

        self.eps = 1e-4

        self.global_step = 0
        self._answer_token_ids = self._build_answer_token_ids()

    def _build_answer_token_ids(self) -> List[int]:
        ids = []
        for i in range(NUM_SLOTS):
            letter = chr(ord('A') + i)
            token_ids = self.tokenizer.encode(letter, add_special_tokens=False)
            ids.append(token_ids[0])
        return ids

    def _find_special_id(self, token_str: str) -> Optional[int]:
        try:
            tid = self.tokenizer.convert_tokens_to_ids(token_str)
            if tid is None or tid == self.tokenizer.unk_token_id:
                return None
            return tid
        except Exception:
            return None

    def find_visual_token_indices(self, input_ids: torch.Tensor, soft_prompt_length: int) -> List[int]:
        ids = input_ids[0].tolist()
        vision_start = self._find_special_id("<|vision_start|>")
        vision_end = self._find_special_id("<|vision_end|>")
        image_pad = self._find_special_id("<|image_pad|>")

        visual_indices = []

        if vision_start is not None and vision_end is not None and vision_start in ids and vision_end in ids:
            s = ids.index(vision_start)
            e = ids.index(vision_end)
            if e > s + 1:
                visual_indices = list(range(s + 1, e))

        if not visual_indices and image_pad is not None:
            visual_indices = [i for i, t in enumerate(ids) if t == image_pad]

        if not visual_indices:
            img_ctx_tok = self._find_special_id("<IMG_CONTEXT>")
            if img_ctx_tok is not None:
                visual_indices = [i for i, t in enumerate(ids) if t == img_ctx_tok]

        if not visual_indices:
            image_tok = self._find_special_id("<image>")
            if image_tok is not None:
                visual_indices = [i for i, t in enumerate(ids) if t == image_tok]

        visual_indices = [idx + soft_prompt_length for idx in visual_indices]
        return visual_indices

    def attnlrp_layer_relevance(self, attn_l: torch.Tensor, grad_l: torch.Tensor) -> torch.Tensor:
        if ATTNLRP_DETACH_GRAD:
            grad_l = grad_l.detach()
        AW = attn_l * grad_l
        if ATTNLRP_RELU_RULE:
            AW = torch.relu(AW)
        denom = AW.sum(dim=-1, keepdim=True)
        denom_stable = denom + self.eps * denom.sign()
        denom_stable = torch.where(
            denom_stable.abs() < self.eps,
            torch.full_like(denom_stable, self.eps),
            denom_stable,
        )
        E_bilinear = AW / denom_stable

        E = E_bilinear.mean(dim=0)


        return E

    def attnlrp_propagation(self, E_list: List[torch.Tensor]) -> torch.Tensor:
        L = len(E_list)
        N = E_list[0].shape[0]

        if self.config.last_n_layers is not None:
            n_last = int(self.config.last_n_layers)
            if n_last < L:
                E_list = E_list[-n_last:]
                L = len(E_list)

        if self.config.attnlrp_max_layers is not None:
            m = int(self.config.attnlrp_max_layers)
            if m < L:
                E_list = E_list[-m:]
                L = len(E_list)

        dtype = E_list[0].dtype
        device = E_list[0].device

        weights = torch.arange(1, L + 1, device=device, dtype=dtype)
        weights = weights / weights.sum()

        R = torch.zeros(N, N, device=device, dtype=dtype)
        for w, E in zip(weights, E_list):
            R = R + w * E

        R = R + torch.eye(N, device=device, dtype=dtype)

        return R

    def _op_glimpse_layer_relevance(self, attn_l: torch.Tensor, grad_l: torch.Tensor) -> torch.Tensor:
        G = torch.relu(grad_l * attn_l)
        g_pos = torch.relu(grad_l)
        num = G.sum(dim=(1, 2))
        den = g_pos.sum(dim=(1, 2))
        score = num / (den + self.eps)
        w = torch.softmax(score / GLIMPSE_HEAD_TEMP, dim=0)
        E = torch.zeros(G.shape[1], G.shape[2], device=G.device, dtype=G.dtype)
        for h in range(G.shape[0]):
            E = E + w[h] * G[h]
        E = E / (E.sum(dim=-1, keepdim=True) + self.eps)
        return E

    def _op_glimpse_propagation(self, E_list: List[torch.Tensor], grad_list: List[torch.Tensor]) -> torch.Tensor:
        L = len(E_list)
        N = E_list[0].shape[0]
        g_layers = []
        for g in grad_list:
            g_sum = g.sum(dim=0)
            g_l = torch.sum(torch.abs(g_sum))
            g_layers.append(g_l)
        g_layers = torch.stack(g_layers)
        depth_logits = GLIMPSE_DEPTH_TEMP * torch.arange(1, L + 1, device=g_layers.device, dtype=g_layers.dtype)
        s = torch.softmax(depth_logits, dim=0)
        alpha_raw = g_layers * s
        alpha = alpha_raw / (alpha_raw.sum() + self.eps)
        dtype = E_list[0].dtype
        I = torch.eye(N, device=E_list[0].device, dtype=dtype)
        R = I.clone()
        for l in range(L):
            R = R + (alpha[l] * E_list[l]) @ R
        return R

    def _op_rollout_R(self, attentions, sample_idx: int = 0) -> torch.Tensor:
        L_all = len(attentions)
        m = self.config.attnlrp_max_layers
        if m is not None and int(m) < L_all:
            attn_used = attentions[L_all - int(m):]
        else:
            attn_used = attentions
        R = None
        for layer_attn in attn_used:
            attn_avg = layer_attn[sample_idx].mean(dim=0).to(torch.float32)
            if ROLLOUT_DISCARD_RATIO > 0.0:
                flat = attn_avg.reshape(-1)
                threshold = flat.quantile(ROLLOUT_DISCARD_RATIO)
                attn_avg = torch.where(attn_avg < threshold,
                                       torch.zeros_like(attn_avg), attn_avg)
            I = torch.eye(attn_avg.shape[0], device=attn_avg.device, dtype=attn_avg.dtype)
            attn_hat = 0.5 * attn_avg + 0.5 * I
            attn_hat = attn_hat / (attn_hat.sum(dim=-1, keepdim=True) + 1e-12)
            R = attn_hat if R is None else attn_hat @ R
        assert R is not None, "rollout: empty attentions"
        return R

    def _op_layer_relevance(self, A_l: torch.Tensor, g_l: torch.Tensor) -> torch.Tensor:
        if ATTN_METHOD == "glimpse":
            return self._op_glimpse_layer_relevance(A_l, g_l)
        return self.attnlrp_layer_relevance(A_l, g_l)

    def _op_compute_R(self, E_list, grad_list, attentions, sample_idx: int = 0) -> torch.Tensor:
        if ATTN_METHOD == "rollout":
            return self._op_rollout_R(attentions, sample_idx)
        if ATTN_METHOD == "glimpse":
            return self._op_glimpse_propagation(E_list, grad_list)
        return self.attnlrp_propagation(E_list)

    def _build_internvl_text(self, profile_text: str, image: Image.Image,
                              answer_text: Optional[str] = None):
        instruction = INSTRUCTION
        user_prompt = f"<image>\n{profile_text}\n\n{instruction}"

        pixel_values_cpu, num_patches, aspect_ratio = _preprocess_image_for_internvl(image)
        num_image_token = self._internvl_num_image_token

        sys.path.insert(0, INTERNVL_MODEL_PATH)
        from conversation import get_conv_template
        sys.path.pop(0)
        template = get_conv_template(self.model.template)
        template.system_message = "You are a sophisticated user behavior emulator."
        template.append_message(template.roles[0], user_prompt)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        image_tokens = '<img>' + '<IMG_CONTEXT>' * num_image_token * num_patches + '</img>'
        query = query.replace('<image>', image_tokens, 1)

        if answer_text is not None:
            query = query + str(answer_text)
        return query, pixel_values_cpu, num_patches, aspect_ratio

    def _build_inputs_internvl(self, image, profile_text, soft_prompt_embeds, answer_text=None):
        full_text, pv_cpu, num_patches, aspect_ratio = \
            self._build_internvl_text(profile_text, image, answer_text)

        tok_out = self.tokenizer(full_text, return_tensors="pt")
        orig_input_ids     = tok_out["input_ids"]
        orig_attention_mask = tok_out["attention_mask"]

        labels, answer_token_ids, answer_pos_orig, answer_pos_combined = None, None, None, None
        if answer_text is not None:
            answer_token_ids = self.tokenizer.encode(str(answer_text), add_special_tokens=False)
            m = len(answer_token_ids)
            L_full = orig_input_ids.shape[1]
            answer_start = L_full - m
            labels = torch.full_like(orig_input_ids, -100)
            labels[0, answer_start:] = orig_input_ids[0, answer_start:]
            answer_pos_orig = list(range(answer_start, L_full))

        emb_layer  = self.model.get_input_embeddings()
        emb_device = emb_layer.weight.device
        emb_dtype  = emb_layer.weight.dtype

        orig_input_ids      = orig_input_ids.to(emb_device)
        orig_attention_mask  = orig_attention_mask.to(emb_device)
        if labels is not None:
            labels = labels.to(emb_device)

        text_embeds = emb_layer(orig_input_ids).clone()
        pv = pv_cpu.to(device=emb_device, dtype=emb_dtype)
        with torch.no_grad():
            vit_embeds = self.model.extract_feature(pv)

        B, N_seq, C = text_embeds.shape
        flat_embeds = text_embeds.reshape(B * N_seq, C)
        flat_ids    = orig_input_ids.reshape(B * N_seq)
        selected    = (flat_ids == self.model.img_context_token_id)
        vit_flat    = vit_embeds.reshape(-1, C).to(flat_embeds.device)
        n_select, n_vit = int(selected.sum()), vit_flat.shape[0]
        n_tok = min(n_select, n_vit)
        if n_tok > 0:
            indices = selected.nonzero(as_tuple=True)[0][:n_tok]
            flat_embeds[indices] = vit_flat[:n_tok].detach()
        text_embeds = flat_embeds.reshape(B, N_seq, C)

        if not hasattr(self, "_printed_orig_inputs"):
            self._printed_orig_inputs = True

        del pv, vit_embeds, vit_flat
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        soft_len = soft_prompt_embeds.shape[0]
        soft_prompt_embeds = soft_prompt_embeds.to(device=emb_device, dtype=text_embeds.dtype)
        combined_embeds = torch.cat([soft_prompt_embeds.unsqueeze(0), text_embeds], dim=1)
        soft_mask = torch.ones(1, soft_len, dtype=torch.long, device=emb_device)

        inputs_for_model = {
            "inputs_embeds":  combined_embeds,
            "attention_mask": torch.cat([soft_mask, orig_attention_mask], dim=1),
        }
        if labels is not None:
            inputs_for_model["labels"] = torch.cat(
                [torch.full((1, soft_len), -100, dtype=labels.dtype, device=emb_device), labels], dim=1)
            answer_pos_combined = [p + soft_len for p in answer_pos_orig]

        orig_inputs = {
            "input_ids":     orig_input_ids,
            "attention_mask": orig_attention_mask,
            "_internvl_num_patches":  num_patches,
            "_internvl_aspect_ratio": aspect_ratio,
        }
        meta = {
            "orig_inputs": orig_inputs, "orig_input_ids": orig_input_ids,
            "soft_len": soft_len, "answer_token_ids": answer_token_ids,
            "answer_pos_orig": answer_pos_orig, "answer_pos_combined": answer_pos_combined,
            "emb_device": emb_device,
        }
        return inputs_for_model, meta

    def _model_forward(self, inputs_for_model: Dict, **kwargs):
        if self.model_type == "internvl":
            fwd_kwargs = {k: v for k, v in inputs_for_model.items()}
            fwd_kwargs.update(kwargs)
            return self.model.language_model(**fwd_kwargs)
        return self.model(**inputs_for_model, **kwargs)

    def _get_internvl_patch_grid_size(self, aspect_ratio, n_visual: int):
        tok_per_side = int(math.sqrt(self._internvl_num_image_token))
        rows_tiles, cols_tiles = aspect_ratio
        content_grid = (tok_per_side * rows_tiles, tok_per_side * cols_tiles)
        return content_grid

    def _get_internvl_content_token_count(self, aspect_ratio):
        tok_per_side = int(math.sqrt(self._internvl_num_image_token))
        rows_tiles, cols_tiles = aspect_ratio
        return tok_per_side * rows_tiles * tok_per_side * cols_tiles

    def build_inputs_with_soft_prompt(
        self,
        image: Image.Image,
        profile_text: str,
        soft_prompt_embeds: torch.Tensor,
        answer_text: Optional[str] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:

        if self.model_type == "internvl":
            return self._build_inputs_internvl(image, profile_text, soft_prompt_embeds, answer_text)

        instruction = INSTRUCTION
        user_prompt = f"{profile_text}\n\n{instruction}"

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt}
            ]
        }]

        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text if answer_text is None else (prompt_text + str(answer_text))

        orig_inputs = self.processor(text=full_text, images=image, return_tensors="pt")

        if not hasattr(self, "_printed_orig_inputs"):
            self._printed_orig_inputs = True

        orig_input_ids = orig_inputs["input_ids"]
        orig_attention_mask = orig_inputs["attention_mask"]

        labels = None
        answer_token_ids = None
        answer_pos_orig = None
        answer_pos_combined = None

        if answer_text is not None:
            answer_token_ids = self.tokenizer.encode(str(answer_text), add_special_tokens=False)
            m = len(answer_token_ids)
            L_full = orig_input_ids.shape[1]
            answer_start = L_full - m

            labels = torch.full_like(orig_input_ids, -100)
            labels[0, answer_start:] = orig_input_ids[0, answer_start:]
            answer_pos_orig = list(range(answer_start, L_full))

        emb_layer = self.model.get_input_embeddings()
        emb_device = emb_layer.weight.device
        emb_dtype = emb_layer.weight.dtype

        orig_input_ids = orig_input_ids.to(emb_device)
        orig_attention_mask = orig_attention_mask.to(emb_device)
        if labels is not None:
            labels = labels.to(emb_device)

        for k, v in list(orig_inputs.items()):
            if torch.is_tensor(v):
                orig_inputs[k] = v.to(emb_device)

        text_embeds = emb_layer(orig_input_ids)

        soft_len = soft_prompt_embeds.shape[0]
        soft_prompt_embeds = soft_prompt_embeds.to(device=emb_device, dtype=text_embeds.dtype)

        combined_embeds = torch.cat(
            [soft_prompt_embeds.unsqueeze(0), text_embeds],
            dim=1
        )

        soft_mask = torch.ones(1, soft_len, dtype=torch.long, device=emb_device)

        inputs_for_model = dict(orig_inputs)
        inputs_for_model["attention_mask"] = torch.cat([soft_mask, orig_attention_mask], dim=1)
        inputs_for_model.pop("input_ids", None)
        inputs_for_model["inputs_embeds"] = combined_embeds

        if labels is not None:
            labels_with_soft = torch.cat(
                [torch.full((1, soft_len), -100, dtype=labels.dtype, device=emb_device), labels],
                dim=1
            )
            inputs_for_model["labels"] = labels_with_soft
            answer_pos_combined = [p + soft_len for p in answer_pos_orig]


        if not hasattr(self, "_printed_input_keys"):
            self._printed_input_keys = True
            pv = orig_inputs.get("pixel_values", None)

        meta = {
            "orig_inputs": orig_inputs,
            "orig_input_ids": orig_input_ids,
            "soft_len": soft_len,
            "answer_token_ids": answer_token_ids,
            "answer_pos_orig": answer_pos_orig,
            "answer_pos_combined": answer_pos_combined,
            "emb_device": emb_device,
        }
        return inputs_for_model, meta

    def aggregate_patch_saliency_to_slots_torch(
        self,
        patch_saliency: torch.Tensor,
        image_size: Tuple[int, int],
        patch_grid_size: Tuple[int, int],
    ) -> torch.Tensor:
        bboxes = getattr(self, "_current_bboxes", None)
        n_cand = int(getattr(self, "_current_ncand", 0) or 0)
        assert bboxes is not None and n_cand > 0, (
            "FixATE-v2: _current_bboxes/_current_ncand not set before aggregation")

        device = patch_saliency.device
        dtype = patch_saliency.dtype
        H_p, W_p = patch_grid_size

        if patch_saliency.dim() == 1:
            n = patch_saliency.numel()
            if n != H_p * W_p:
                if H_p * W_p == 4 * n:
                    Hm, Wm = H_p // 2, W_p // 2
                    ps2d = patch_saliency.view(Hm, Wm)
                    ps2d = ps2d.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
                    ps2d = ps2d[:H_p, :W_p]
                else:
                    ps2d = patch_saliency.view(H_p, W_p)
            else:
                ps2d = patch_saliency.view(H_p, W_p)
        else:
            ps2d = patch_saliency
            H_p, W_p = ps2d.shape
        ps2d = torch.relu(ps2d)

        img_w, img_h = float(image_size[0]), float(image_size[1])
        ii = torch.arange(H_p, device=device, dtype=torch.float32)
        jj = torch.arange(W_p, device=device, dtype=torch.float32)
        gi, gj = torch.meshgrid(ii, jj, indexing="ij")
        cx = ((gj + 0.5) * (img_w / float(W_p))).reshape(-1)
        cy = ((gi + 0.5) * (img_h / float(H_p))).reshape(-1)
        vals = ps2d.reshape(-1)

        if torch.is_tensor(bboxes):
            bb = bboxes.detach().to(device=device, dtype=torch.float32)
        else:
            bb = torch.as_tensor(np.asarray(bboxes, dtype=np.float32), device=device)

        slot_sal = torch.zeros(NUM_SLOTS, device=device, dtype=dtype)
        assigned = torch.zeros_like(cx, dtype=torch.bool)
        for s in range(min(n_cand, NUM_SLOTS)):
            x1, y1, x2, y2 = bb[s]
            if float(x2) <= float(x1) or float(y2) <= float(y1):
                continue
            m = (~assigned) & (cx >= x1) & (cx < x2) & (cy >= y1) & (cy < y2)
            if bool(m.any()):
                slot_sal[s] = vals[m].sum()
                assigned = assigned | m

        slot_attn = slot_sal / (slot_sal.sum() + self.eps)
        return slot_attn


    @staticmethod
    def _tail_aggregate(dist: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        topk_vals = torch.gather(dist, 1, topk_idx)
        tail = 1.0 - topk_vals.sum(dim=1, keepdim=True)
        tail = tail.clamp(min=0.0)
        return torch.cat([topk_vals, tail], dim=1)

    @staticmethod
    def _power_weights(tk_p: torch.Tensor, gamma: float, eps: float) -> torch.Tensor:
        raw = (tk_p + eps).pow(gamma)
        return raw / raw.sum(dim=1, keepdim=True)

    def _weighted_kl_divergence(
        self,
        tk_p: torch.Tensor,
        tk_q: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        eps = self.eps
        kl_per_item = tk_p * (torch.log(tk_p + eps) - torch.log(tk_q + eps))
        weighted = (weights * kl_per_item).sum(dim=1)
        return weighted.mean()

    def _weighted_full_kl_divergence(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        gamma: float,
        eps_pw: float,
    ) -> torch.Tensor:
        eps = self.eps
        weights = self._power_weights(p, gamma, eps_pw)
        kl_per_item = p * (torch.log(p + eps) - torch.log(q + eps))
        weighted = (weights * kl_per_item).sum(dim=1)
        return weighted.mean()

    def compute_losses(
        self,
        model_attentions: torch.Tensor,
        gaze_dists: torch.Tensor,
        user_ids: List[str],
        lambda_attn: float,
        beta_reg: float,
        loss_choice: Optional[torch.Tensor] = None,
        choice_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        g = gaze_dists
        a = model_attentions

        k = self.config.topk_k
        gamma = self.config.power_gamma
        eps_pw = self.config.power_eps
        mode = self.config.attn_loss_mode

        if mode == "plain_kl":
            eps = self.eps
            loss_attn = (g * (torch.log(g + eps) - torch.log(a + eps))).sum(dim=1).mean()
        elif mode == "coarsened_kl":
            _, topk_idx = g.topk(k, dim=1)
            tk_p = self._tail_aggregate(g, topk_idx)
            tk_q = self._tail_aggregate(a, topk_idx)
            weights = self._power_weights(tk_p, gamma, eps_pw)
            loss_attn = self._weighted_kl_divergence(tk_p, tk_q, weights)
        elif mode == "weighted_full_kl":
            loss_attn = self._weighted_full_kl_divergence(g, a, gamma, eps_pw)
        else:
            raise ValueError(f"Unknown ATTN_LOSS_MODE: {mode!r}")

        loss_reg = self.prompt_basis.get_user_alpha_l2_loss(user_ids)

        if loss_choice is None:
            loss_choice_val = torch.zeros((), device=model_attentions.device)
        else:
            loss_choice_val = loss_choice

        total_loss = choice_weight * loss_choice_val + lambda_attn * loss_attn + beta_reg * loss_reg

        loss_dict = {
            "loss_choice": float(loss_choice_val.detach().cpu().item()),
            "loss_attn": float(loss_attn.detach().cpu().item()),
            "loss_reg": float(loss_reg.detach().cpu().item()),
            "total_loss": float(total_loss.detach().cpu().item()),
            "choice_weight": float(choice_weight),
        }
        return total_loss, loss_dict

    def get_lambda_attn(self, step: int, warmup_steps: int = None, target: float = None) -> float:
        if warmup_steps is None:
            warmup_steps = self.config.lambda_warmup_steps
        if target is None:
            target = self.config.lambda_attn_target
        if step < warmup_steps:
            return 0.0
        else:
            return min(target, (step - warmup_steps) / warmup_steps * target)

    def _get_pixel_values_hw(self, orig_inputs: dict):
        if not isinstance(orig_inputs, dict):
            return -1, -1

        pv = orig_inputs.get("pixel_values", None)

        if torch.is_tensor(pv):
            shp = pv.shape
            if pv.dim() == 4:
                return int(shp[-2]), int(shp[-1])
            elif pv.dim() == 3:
                return int(shp[-2]), int(shp[-1])
            elif pv.dim() == 2:
                return int(shp[-2]), int(shp[-1])
            else:
                return -1, -1

        if isinstance(pv, (list, tuple)) and len(pv) > 0 and torch.is_tensor(pv[0]):
            t = pv[0]
            shp = t.shape
            if t.dim() >= 2:
                return int(shp[-2]), int(shp[-1])
            return -1, -1

        return -1, -1

    def _fix_patch_grid_size_from_visual_tokens(
        self,
        patch_grid_size: Tuple[int, int],
        n_visual: int,
    ) -> Tuple[int, int]:
        H_p, W_p = patch_grid_size
        hw = H_p * W_p

        if hw == n_visual:
            return (H_p, W_p)

        if hw == 4 * n_visual and H_p % 2 == 0 and W_p % 2 == 0:
            return (H_p // 2, W_p // 2)

        if n_visual == 4 * hw:
            return (H_p * 2, W_p * 2)

        side = int(math.sqrt(n_visual))
        for h in range(side, 0, -1):
            if n_visual % h == 0:
                return (h, n_visual // h)
        return (1, n_visual)


    def compute_model_attention_single(
        self,
        user_id: str,
        image: Image.Image,
        profile_text: str,
        answer_text: Optional[str] = None,
    ) -> torch.Tensor:
        soft_prompt = self.prompt_basis([user_id])[0]

        inputs, meta = self.build_inputs_with_soft_prompt(
            image=image,
            profile_text=profile_text,
            soft_prompt_embeds=soft_prompt,
            answer_text=answer_text,
        )

        with torch.enable_grad():
            outputs = self._model_forward(
                inputs,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
            logits = outputs.logits
            attentions = outputs.attentions

            ans_pos = meta["answer_pos_combined"]
            ans_ids = meta["answer_token_ids"]
            if ans_pos is None or ans_ids is None or len(ans_pos) == 0:
                target_scalar = logits[0, -1].max()
            else:
                target_scalar = None
                for p, tid in zip(ans_pos, ans_ids):
                    prev = p - 1
                    if prev >= 0:
                        if target_scalar is None:
                            target_scalar = logits[0, prev, tid]
                        else:
                            target_scalar = target_scalar + logits[0, prev, tid]

                if target_scalar is None:
                    target_scalar = logits[0, -1].max()

            E_list, grad_list = [], []
            L = len(attentions)
            if ATTNLRP_MAX_LAYERS is not None and int(ATTNLRP_MAX_LAYERS) < L:
                start = L - int(ATTNLRP_MAX_LAYERS)
                layer_ids = list(range(start, L))
            else:
                layer_ids = list(range(L))
            if ATTN_METHOD == "rollout":
                layer_ids = []

            _diag_skipped = 0
            for idx, l in enumerate(layer_ids):
                attn = attentions[l]

                if (not torch.is_tensor(attn)) or (not attn.requires_grad):
                    _diag_skipped += 1
                    continue

                g_attn = torch.autograd.grad(
                    outputs=target_scalar,
                    inputs=attn,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )[0]
                if g_attn is None:
                    continue

                A_l = attn[0]
                g_l = g_attn[0]

                E_l = self._op_layer_relevance(A_l, g_l)

                E_list.append(E_l)
                grad_list.append(g_l)


            if _op_no_relevance(E_list, attentions):
                if self.logger:
                    rqs = [(l, bool(attentions[l].requires_grad) if torch.is_tensor(attentions[l]) else "N/A") for l in layer_ids[:5]]
                uniform = torch.ones(NUM_SLOTS, device=soft_prompt.device, dtype=soft_prompt.dtype) / NUM_SLOTS
                return (soft_prompt.sum() * 0.0 + uniform).detach()

            R = self._op_compute_R(E_list, grad_list, attentions, 0)

            last_pos = R.shape[0] - 1
            relevance = R[last_pos, :]

            visual_indices = self.find_visual_token_indices(meta["orig_input_ids"], meta["soft_len"])
            if len(visual_indices) == 0:
                uniform = torch.ones(NUM_SLOTS, device=soft_prompt.device, dtype=soft_prompt.dtype) / NUM_SLOTS
                return (soft_prompt.sum() * 0.0 + uniform).detach()

            visual_relevance = relevance[visual_indices]
            visual_relevance = visual_relevance.abs()

            if self.logger:
                vr = visual_relevance.detach().float()

            orig_inputs = meta["orig_inputs"]
            patch_grid_size = None
            if self.model_type == "internvl":
                ar = orig_inputs.get("_internvl_aspect_ratio")
                if ar is not None:
                    patch_grid_size = self._get_internvl_patch_grid_size(ar, len(visual_indices))
                    content_n = self._get_internvl_content_token_count(ar)
                    if len(visual_indices) > content_n:
                        visual_indices = visual_indices[:content_n]
                        visual_relevance = relevance[visual_indices]
                        visual_relevance = visual_relevance.abs()
            if patch_grid_size is None:
                for key in ["image_grid_thw", "grid_thw", "vision_grid_thw"]:
                    if key in orig_inputs and torch.is_tensor(orig_inputs[key]):
                        thw = orig_inputs[key].reshape(-1)
                        if thw.numel() >= 3:
                            H_p = int(thw[-2].item())
                            W_p = int(thw[-1].item())
                            patch_grid_size = (H_p, W_p)
                            break
            if patch_grid_size is None:
                n_patches = len(visual_indices)
                side = int(math.sqrt(n_patches))
                patch_grid_size = (max(1, side), max(1, n_patches // max(1, side)))

            patch_grid_size = self._fix_patch_grid_size_from_visual_tokens(
                patch_grid_size=patch_grid_size,
                n_visual=len(visual_indices),
            )

            image_size = (image.width, image.height)

            model_attention = self.aggregate_patch_saliency_to_slots_torch(
                patch_saliency=visual_relevance,
                image_size=image_size,
                patch_grid_size=patch_grid_size,
            )

            return model_attention.detach()


    def train_step(self, batch: Dict, optimizer: torch.optim.Optimizer,
                   accumulation_steps: int = 1,
                   zero_grad: bool = True,
                   do_optimizer_step: bool = True) -> Dict[str, float]:
        user_ids = batch["user_id"]
        images = batch["image"]
        profile_texts = batch["profile_text"]
        choice_slots = batch["choice_slot"].to(DEVICE)
        gaze_dists = batch["gaze_dist"].to(DEVICE)
        batch_size = len(user_ids)

        soft_prompts = self.prompt_basis(user_ids)

        all_choice_losses = []
        all_model_attentions = []

        for i in range(batch_size):
            image = images[i]
            profile_text = profile_texts[i]
            soft_prompt = soft_prompts[i]
            self._current_bboxes = batch["bboxes"][i]
            self._current_ncand = int(batch["n_candidates"][i])

            target_slot = int(choice_slots[i].item()) + 1

            if target_slot < 1 or target_slot > 15:
                target_slot = max(1, min(15, target_slot))

            answer_text = slot_to_letter(target_slot)

            inputs, meta = self.build_inputs_with_soft_prompt(
                image=image,
                profile_text=profile_text,
                soft_prompt_embeds=soft_prompt,
                answer_text=answer_text,
            )

            emb_device = meta["emb_device"]
            soft_prompt = soft_prompt.to(emb_device)


            with torch.enable_grad():
                outputs = self._model_forward(
                    inputs,
                    output_attentions=True,
                    use_cache=False,
                    return_dict=True,
                )
                logits = outputs.logits
                attentions = outputs.attentions

                if self.global_step < 5 and i == 0:

                    reqs = []
                    if attentions is not None:
                        for l in range(len(attentions)):
                            a = attentions[l]
                            if torch.is_tensor(a):
                                reqs.append((l, bool(a.requires_grad)))


                if self.config.train_choice:
                    all_choice_losses.append(outputs.loss)

                ans_pos = meta["answer_pos_combined"]
                ans_ids = meta["answer_token_ids"]

                src_positions = []

                if ans_pos is None or ans_ids is None or len(ans_pos) == 0:
                    target_scalar = logits[0, -1].max()
                    src_pos = logits.shape[1] - 1
                else:
                    target_scalar = None
                    for p, tid in zip(ans_pos, ans_ids):
                        prev = p - 1
                        if prev >= 0:
                            src_positions.append(prev)
                            v = logits[0, prev, tid]
                            target_scalar = v if target_scalar is None else (target_scalar + v)

                    if target_scalar is None:
                        target_scalar = logits[0, -1].max()
                        src_pos = logits.shape[1] - 1
                    else:
                        src_pos = src_positions[-1]

                E_list, grad_list = [], []

                L = len(attentions)
                if self.config.attnlrp_max_layers is not None and int(self.config.attnlrp_max_layers) < L:
                    start = L - int(self.config.attnlrp_max_layers)
                    layer_ids = list(range(start, L))
                else:
                    layer_ids = list(range(L))
                if ATTN_METHOD == "rollout":
                    layer_ids = []

                valid_layers = []
                valid_attns = []
                for l in layer_ids:
                    attn = attentions[l]
                    if (not torch.is_tensor(attn)) or (not attn.requires_grad):
                        continue
                    valid_layers.append(l)
                    valid_attns.append(attn)

                if len(valid_attns) > 0:
                    all_grads = torch.autograd.grad(
                        outputs=target_scalar,
                        inputs=valid_attns,
                        retain_graph=True,
                        create_graph=self.config.attnlrp_create_graph,
                        allow_unused=True,
                    )

                    for l, attn, g_attn in zip(valid_layers, valid_attns, all_grads):
                        if g_attn is None:
                            continue

                        A_l = attn[0]
                        g_l = g_attn[0]

                        if self.config.attnlrp_grad_scale != 1.0:
                            g_l = g_l * self.config.attnlrp_grad_scale

                        if self.global_step < 2 and i == 0 and (l == valid_layers[0] or l == valid_layers[-1]):
                            gl = g_l.detach().float()
                            Al = A_l.detach().float()
                            prod = gl * Al


                        E_l = self._op_layer_relevance(A_l, g_l)

                        if self.global_step < 2 and i == 0 and (l == valid_layers[0] or l == valid_layers[-1]):
                            El = E_l.detach().float()

                        E_list.append(E_l)
                        grad_list.append(g_l)

                if _op_no_relevance(E_list, attentions):
                    uniform = torch.ones(NUM_SLOTS, device=soft_prompt.device, dtype=soft_prompt.dtype) / NUM_SLOTS
                    model_attention = soft_prompt.sum() * 0.0 + uniform
                else:
                    R = self._op_compute_R(E_list, grad_list, attentions, 0)

                    src_pos = max(0, min(int(src_pos), R.shape[0] - 1))

                    if self.global_step < 3 and i == 0:
                        rrow = R[src_pos, :].detach().float()
                        vi = self.find_visual_token_indices(meta['orig_input_ids'], meta['soft_len'])
                        if len(vi) > 0:
                            rv = rrow[vi]

                    relevance = R[src_pos, :]

                    visual_indices = self.find_visual_token_indices(meta["orig_input_ids"], meta["soft_len"])

                    if len(visual_indices) == 0:
                        model_attention = torch.ones(NUM_SLOTS, device=DEVICE) / NUM_SLOTS
                    else:
                        visual_relevance = relevance[visual_indices]

                        if self.global_step < 5 and i == 0:
                            vr = visual_relevance.detach().float()

                        visual_relevance = visual_relevance.abs()

                        if self.global_step < 5 and i == 0:
                            vr2 = visual_relevance.detach().float()

                        image_size = (image.width, image.height)

                        orig_inputs = meta["orig_inputs"]
                        patch_grid_size = None
                        if self.model_type == "internvl":
                            ar = orig_inputs.get("_internvl_aspect_ratio")
                            if ar is not None:
                                patch_grid_size = self._get_internvl_patch_grid_size(ar, len(visual_indices))
                                content_n = self._get_internvl_content_token_count(ar)
                                if len(visual_indices) > content_n:
                                    visual_indices = visual_indices[:content_n]
                                    visual_relevance = relevance[visual_indices]
                                    visual_relevance = visual_relevance.abs()
                        if patch_grid_size is None:
                            for key in ["image_grid_thw", "grid_thw", "vision_grid_thw"]:
                                if key in orig_inputs and torch.is_tensor(orig_inputs[key]):
                                    thw = orig_inputs[key].reshape(-1)
                                    if thw.numel() >= 3:
                                        H_p = int(thw[-2].item())
                                        W_p = int(thw[-1].item())
                                        patch_grid_size = (H_p, W_p)
                                        break
                        if patch_grid_size is None:
                            n_patches = len(visual_indices)
                            side = int(math.sqrt(n_patches))
                            patch_grid_size = (max(1, side), max(1, n_patches // max(1, side)))

                        patch_grid_size = self._fix_patch_grid_size_from_visual_tokens(
                                patch_grid_size=patch_grid_size,
                                n_visual=len(visual_indices),
                        )

                        image_size = (image.width, image.height)

                        model_attention = self.aggregate_patch_saliency_to_slots_torch(
                            patch_saliency=visual_relevance,
                            image_size=image_size,
                            patch_grid_size=patch_grid_size,
                        )

                        if self.global_step < 5 and i == 0:
                            pv = meta["orig_inputs"].get("pixel_values", None)
                            seq_len = attentions[layer_ids[-1]].shape[-1]
                            soft_len = meta["soft_len"]
                            n_visual = len(visual_indices)
                            H_p, W_p = patch_grid_size
                            hw = H_p * W_p

                            H_in, W_in = self._get_pixel_values_hw(meta["orig_inputs"])

                            attn_sum = float(model_attention.detach().sum().cpu().item())
                            attn_max = float(model_attention.detach().max().cpu().item())
                            attn_argmax = int(model_attention.detach().argmax().cpu().item())

                            pv_shape = tuple(pv.shape) if torch.is_tensor(pv) else None


                all_model_attentions.append(model_attention)

        model_attentions_batch = torch.stack(all_model_attentions, dim=0)
        gaze_dists = gaze_dists.to(model_attentions_batch.device)

        loss_choice_batch = None
        if self.config.train_choice and len(all_choice_losses) > 0:
            loss_choice_batch = torch.stack(all_choice_losses).mean()

        if self.config.train_choice:
            lambda_attn = self.get_lambda_attn(self.global_step)
        else:
            lambda_attn = self.config.lambda_attn_target

        total_loss, loss_dict = self.compute_losses(
            model_attentions=model_attentions_batch,
            gaze_dists=gaze_dists,
            user_ids=user_ids,
            lambda_attn=lambda_attn,
            beta_reg=self.config.beta_reg,
            loss_choice=loss_choice_batch,
            choice_weight=self.config.choice_loss_weight,
        )

        if zero_grad:
            optimizer.zero_grad(set_to_none=True)

        scaled_loss = total_loss / accumulation_steps
        scaled_loss.backward()

        if do_optimizer_step:
            basis_params = [self.prompt_basis.basis] if self.prompt_basis.basis.grad is not None else []
            alpha_params = ([p for uid, p in self.prompt_basis.user_alphas.items() if p.grad is not None]
                           if self.config.use_user_alpha else [])

            try:
                basis_raw = self.prompt_basis.basis.grad.norm().item() if basis_params else None
                alpha_raw = None
                if self.config.use_user_alpha:
                    some_uid = user_ids[0]
                    a = self.prompt_basis.user_alphas.get(some_uid, None)
                    alpha_raw = a.grad.norm().item() if (a is not None and a.grad is not None) else None

                if basis_params:
                    torch.nn.utils.clip_grad_norm_(basis_params, max_norm=self.config.basis_clip_grad_norm)
                if alpha_params:
                    torch.nn.utils.clip_grad_norm_(alpha_params, max_norm=self.config.alpha_clip_grad_norm)

                basis_clipped = self.prompt_basis.basis.grad.norm().item() if basis_params else None
                alpha_clipped = None
                if self.config.use_user_alpha:
                    alpha_clipped = a.grad.norm().item() if (a is not None and a.grad is not None) else None

            except Exception as e:
                pass
                pass

            optimizer.step()
            self.global_step += 1

        loss_dict["lambda_attn"] = float(lambda_attn)
        return loss_dict

    def train_step_batched(self, batch: Dict, optimizer: torch.optim.Optimizer,
                           accumulation_steps: int = 1,
                           zero_grad: bool = True,
                           do_optimizer_step: bool = True) -> Dict[str, float]:

        user_ids = batch["user_id"]
        images = batch["image"]
        profile_texts = batch["profile_text"]
        choice_slots = batch["choice_slot"].to(DEVICE)
        gaze_dists = batch["gaze_dist"].to(DEVICE)
        batch_size = len(user_ids)
        soft_prompts = self.prompt_basis(user_ids)

        per_inputs = []
        per_metas = []
        answer_texts = []
        for i in range(batch_size):
            target_slot = max(1, min(15, int(choice_slots[i].item()) + 1))
            answer_text = slot_to_letter(target_slot)
            answer_texts.append(answer_text)
            inputs_i, meta_i = self.build_inputs_with_soft_prompt(
                image=images[i], profile_text=profile_texts[i],
                soft_prompt_embeds=soft_prompts[i], answer_text=answer_text,
            )
            per_inputs.append(inputs_i)
            per_metas.append(meta_i)

        if self.model_type != "internvl":
            image_pad_id = self._find_special_id("<|image_pad|>")
            if image_pad_id is None:
                return self.train_step(batch, optimizer, accumulation_steps,
                                       zero_grad, do_optimizer_step)
            for i in range(batch_size):
                pv = per_inputs[i].pop("pixel_values", None)
                gthw = per_inputs[i].pop("image_grid_thw", None)
                for _k in ("pixel_values_videos", "video_grid_thw"):
                    per_inputs[i].pop(_k, None)
                if pv is not None:
                    vis_dtype = (self.model.visual.get_dtype()
                                 if hasattr(self.model, "visual")
                                    and hasattr(self.model.visual, "get_dtype")
                                 else pv.dtype)
                    with torch.no_grad():
                        img_embeds = self.model.visual(
                            pv.to(vis_dtype), grid_thw=gthw)
                        if isinstance(img_embeds, (tuple, list)):
                            img_embeds = img_embeds[0]
                        if hasattr(img_embeds, "last_hidden_state"):
                            img_embeds = img_embeds.last_hidden_state
                        if isinstance(img_embeds, dict) and "last_hidden_state" in img_embeds:
                            img_embeds = img_embeds["last_hidden_state"]
                        if torch.is_tensor(img_embeds) and img_embeds.dim() == 3 and img_embeds.shape[0] == 1:
                            img_embeds = img_embeds[0]
                    orig_ids = per_metas[i]["orig_input_ids"][0]
                    soft_len = per_metas[i]["soft_len"]
                    embeds = per_inputs[i]["inputs_embeds"]
                    mask_orig = (orig_ids == image_pad_id)
                    full_len = embeds.shape[1]
                    mask_comb = torch.zeros(full_len, dtype=torch.bool,
                                            device=mask_orig.device)
                    mask_comb[soft_len:soft_len + mask_orig.shape[0]] = mask_orig
                    n_img = int(mask_comb.sum().item())
                    n_emb = img_embeds.shape[0]
                    n_tok = min(n_img, n_emb)
                    if n_tok > 0:
                        indices = mask_comb.nonzero(as_tuple=True)[0][:n_tok]
                        embeds[0, indices] = img_embeds[:n_tok].detach().to(
                            embeds.dtype)
                    del pv, img_embeds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        emb_device = per_metas[0]["emb_device"]
        emb_dtype = per_inputs[0]["inputs_embeds"].dtype
        emb_dim = per_inputs[0]["inputs_embeds"].shape[-1]
        seq_lens = [inp["inputs_embeds"].shape[1] for inp in per_inputs]
        max_len = max(seq_lens)

        batched_embeds = torch.zeros(batch_size, max_len, emb_dim,
                                     device=emb_device, dtype=emb_dtype)
        batched_mask = torch.zeros(batch_size, max_len,
                                   dtype=torch.long, device=emb_device)
        batched_labels = torch.full((batch_size, max_len), -100,
                                    dtype=torch.long, device=emb_device)

        for i in range(batch_size):
            L_i = seq_lens[i]
            batched_embeds[i, :L_i] = per_inputs[i]["inputs_embeds"][0]
            batched_mask[i, :L_i] = per_inputs[i]["attention_mask"][0]
            if "labels" in per_inputs[i]:
                batched_labels[i, :L_i] = per_inputs[i]["labels"][0]

        batched_input = {
            "inputs_embeds": batched_embeds,
            "attention_mask": batched_mask,
        }
        if self.config.train_choice:
            batched_input["labels"] = batched_labels

        with torch.enable_grad():
            outputs = self._model_forward(
                batched_input,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
            logits = outputs.logits
            attentions = outputs.attentions


            target_scalars = []
            src_positions_list = []
            for i in range(batch_size):
                meta = per_metas[i]
                ans_pos = meta["answer_pos_combined"]
                ans_ids = meta["answer_token_ids"]

                if ans_pos is None or ans_ids is None or len(ans_pos) == 0:
                    scalar = logits[i, seq_lens[i] - 1].max()
                    src_pos = seq_lens[i] - 1
                else:
                    scalar = None
                    _src_positions = []
                    for p, tid in zip(ans_pos, ans_ids):
                        prev = p - 1
                        if prev >= 0:
                            _src_positions.append(prev)
                            v = logits[i, prev, tid]
                            scalar = v if scalar is None else (scalar + v)
                    if scalar is None:
                        scalar = logits[i, seq_lens[i] - 1].max()
                        src_pos = seq_lens[i] - 1
                    else:
                        src_pos = _src_positions[-1]

                target_scalars.append(scalar)
                src_positions_list.append(src_pos)

            sum_scalar = sum(target_scalars)

            L = len(attentions)
            if self.config.attnlrp_max_layers is not None and int(self.config.attnlrp_max_layers) < L:
                start = L - int(self.config.attnlrp_max_layers)
                layer_ids = list(range(start, L))
            else:
                layer_ids = list(range(L))
            if ATTN_METHOD == "rollout":
                layer_ids = []

            valid_layers = []
            valid_attns = []
            for l in layer_ids:
                attn = attentions[l]
                if (not torch.is_tensor(attn)) or (not attn.requires_grad):
                    continue
                valid_layers.append(l)
                valid_attns.append(attn)

            all_grads = None
            if len(valid_attns) > 0:
                all_grads = torch.autograd.grad(
                    outputs=sum_scalar,
                    inputs=valid_attns,
                    retain_graph=True,
                    create_graph=self.config.attnlrp_create_graph,
                    allow_unused=True,
                )

            all_model_attentions = []
            all_choice_losses = []

            if self.config.train_choice:
                all_choice_losses.append(outputs.loss)

            for i in range(batch_size):
                meta = per_metas[i]
                src_pos = src_positions_list[i]
                soft_prompt = soft_prompts[i].to(emb_device)
                image = images[i]
                self._current_bboxes = batch["bboxes"][i]
                self._current_ncand = int(batch["n_candidates"][i])

                E_list, grad_list = [], []
                if all_grads is not None:
                    for l, attn, g_attn in zip(valid_layers, valid_attns, all_grads):
                        A_l = attn[i]
                        g_l = g_attn[i]

                        if self.config.attnlrp_grad_scale != 1.0:
                            g_l = g_l * self.config.attnlrp_grad_scale

                        E_l = self._op_layer_relevance(A_l, g_l)
                        E_list.append(E_l)
                        grad_list.append(g_l)

                if _op_no_relevance(E_list, attentions):
                    uniform = torch.ones(NUM_SLOTS, device=soft_prompt.device,
                                         dtype=soft_prompt.dtype) / NUM_SLOTS
                    model_attention = soft_prompt.sum() * 0.0 + uniform
                else:
                    R = self._op_compute_R(E_list, grad_list, attentions, i)
                    src_pos = max(0, min(int(src_pos), R.shape[0] - 1))
                    relevance = R[src_pos, :]

                    visual_indices = self.find_visual_token_indices(
                        meta["orig_input_ids"], meta["soft_len"])

                    if len(visual_indices) == 0:
                        model_attention = torch.ones(NUM_SLOTS, device=DEVICE) / NUM_SLOTS
                    else:
                        visual_relevance = relevance[visual_indices].abs()

                        image_size = (image.width, image.height)
                        orig_inputs = meta["orig_inputs"]
                        patch_grid_size = None
                        if self.model_type == "internvl":
                            ar = orig_inputs.get("_internvl_aspect_ratio")
                            if ar is not None:
                                patch_grid_size = self._get_internvl_patch_grid_size(
                                    ar, len(visual_indices))
                                content_n = self._get_internvl_content_token_count(ar)
                                if len(visual_indices) > content_n:
                                    visual_indices = visual_indices[:content_n]
                                    visual_relevance = relevance[visual_indices].abs()
                        if patch_grid_size is None:
                            for key in ["image_grid_thw", "grid_thw", "vision_grid_thw"]:
                                if key in orig_inputs and torch.is_tensor(orig_inputs[key]):
                                    thw = orig_inputs[key].reshape(-1)
                                    if thw.numel() >= 3:
                                        patch_grid_size = (int(thw[-2].item()),
                                                           int(thw[-1].item()))
                                        break
                        if patch_grid_size is None:
                            n_patches = len(visual_indices)
                            side = int(math.sqrt(n_patches))
                            patch_grid_size = (max(1, side),
                                               max(1, n_patches // max(1, side)))

                        patch_grid_size = self._fix_patch_grid_size_from_visual_tokens(
                            patch_grid_size=patch_grid_size,
                            n_visual=len(visual_indices),
                        )

                        model_attention = self.aggregate_patch_saliency_to_slots_torch(
                            patch_saliency=visual_relevance,
                            image_size=image_size,
                            patch_grid_size=patch_grid_size,
                        )

                all_model_attentions.append(model_attention)

        model_attentions_batch = torch.stack(all_model_attentions, dim=0)
        gaze_dists = gaze_dists.to(model_attentions_batch.device)

        loss_choice_batch = None
        if self.config.train_choice and len(all_choice_losses) > 0:
            loss_choice_batch = all_choice_losses[0]

        if self.config.train_choice:
            lambda_attn = self.get_lambda_attn(self.global_step)
        else:
            lambda_attn = self.config.lambda_attn_target

        total_loss, loss_dict = self.compute_losses(
            model_attentions=model_attentions_batch,
            gaze_dists=gaze_dists,
            user_ids=user_ids,
            lambda_attn=lambda_attn,
            beta_reg=self.config.beta_reg,
            loss_choice=loss_choice_batch,
            choice_weight=self.config.choice_loss_weight,
        )

        if zero_grad:
            optimizer.zero_grad(set_to_none=True)

        scaled_loss = total_loss / accumulation_steps
        scaled_loss.backward()

        if do_optimizer_step:
            basis_params = [self.prompt_basis.basis] if self.prompt_basis.basis.grad is not None else []
            alpha_params = ([p for uid, p in self.prompt_basis.user_alphas.items() if p.grad is not None]
                           if self.config.use_user_alpha else [])
            try:
                basis_raw = self.prompt_basis.basis.grad.norm().item() if basis_params else None
                alpha_raw = None
                if self.config.use_user_alpha:
                    some_uid = user_ids[0]
                    a = self.prompt_basis.user_alphas.get(some_uid, None)
                    alpha_raw = a.grad.norm().item() if (a is not None and a.grad is not None) else None
                if basis_params:
                    torch.nn.utils.clip_grad_norm_(basis_params, max_norm=self.config.basis_clip_grad_norm)
                if alpha_params:
                    torch.nn.utils.clip_grad_norm_(alpha_params, max_norm=self.config.alpha_clip_grad_norm)
                basis_clipped = self.prompt_basis.basis.grad.norm().item() if basis_params else None
                alpha_clipped = None
                if self.config.use_user_alpha:
                    alpha_clipped = a.grad.norm().item() if (a is not None and a.grad is not None) else None
            except Exception as e:
                pass
                pass
            optimizer.step()
            self.global_step += 1

        loss_dict["lambda_attn"] = float(lambda_attn)
        return loss_dict

    def save_checkpoint(self, path: str):
        checkpoint = {'prompt_basis': self.prompt_basis.state_dict(), 'global_step': self.global_step}
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        self.prompt_basis.load_state_dict(checkpoint['prompt_basis'], strict=False)
        self.global_step = checkpoint.get('global_step', 0)

        info_msg = f"checkpoint loaded: {path}"
        if 'epoch' in checkpoint:
            info_msg += f" (Epoch {checkpoint['epoch']})"
        if 'best_loss' in checkpoint:
            info_msg += f" (Best Loss: {checkpoint['best_loss']:.4f})"


def compute_primary_score(
    metrics: Dict[str, float],
    fg_mode: Optional[bool] = None,
) -> Tuple[float, str]:
    if PRIMARY_METRIC == "composite_align_v1":
        score = 0.0
        detail_parts = []

        for key, weight in PRIMARY_METRIC_POSITIVE.items():
            if key == "answer_accuracy":
                if fg_mode is False:
                    continue
                if fg_mode is None and key not in metrics:
                    continue
            value = float(metrics.get(key, float("nan")))
            if not np.isfinite(value):
                return float("nan"), f"missing_or_nan:{key}"
            score += weight * value
            detail_parts.append(f"+{weight:.2f}*{key}={value:.4f}")

        for key, weight in PRIMARY_METRIC_PENALTY.items():
            value = float(metrics.get(key, float("nan")))
            if not np.isfinite(value):
                return float("nan"), f"missing_or_nan:{key}"
            score -= weight * value
            detail_parts.append(f"-{weight:.2f}*{key}={value:.4f}")

        return float(score), " ".join(detail_parts)

    score = float(metrics.get(PRIMARY_METRIC, float("nan")))
    if not np.isfinite(score):
        return float("nan"), f"missing_or_nan:{PRIMARY_METRIC}"
    return score, f"{PRIMARY_METRIC}={score:.4f}"


def setup_logger(log_name: str = 'cv_pipeline') -> logging.Logger:
    logger = logging.getLogger(log_name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def patch_trainer_for_eval(trainer, logger=None):
    import types

    _orig_attnlrp = type(trainer).attnlrp_layer_relevance

    def _attnlrp_fp32(self, A_l, g_l):
        return _orig_attnlrp(self, A_l.to(torch.float32), g_l.to(torch.float32))

    trainer.attnlrp_layer_relevance = types.MethodType(_attnlrp_fp32, trainer)

    def _compute_with_answer_logit(self, user_id, image, profile_text, answer_text=None):
        self.model.eval()
        soft_prompt = self.prompt_basis([user_id])[0]
        inputs, meta = self.build_inputs_with_soft_prompt(
            image=image,
            profile_text=profile_text,
            soft_prompt_embeds=soft_prompt,
            answer_text=answer_text,
        )

        with torch.enable_grad():
            outputs = self._model_forward(
                inputs,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
            logits = outputs.logits
            attentions = outputs.attentions

            ans_pos = meta["answer_pos_combined"]
            ans_ids = meta["answer_token_ids"]

            src_positions = []
            if ans_pos is None or ans_ids is None or len(ans_pos) == 0:
                target_scalar = logits[0, -1].max()
                src_pos = logits.shape[1] - 1
            else:
                target_scalar = None
                for p, tid in zip(ans_pos, ans_ids):
                    prev = p - 1
                    if prev >= 0:
                        src_positions.append(prev)
                        v = logits[0, prev, tid]
                        target_scalar = v if target_scalar is None else (target_scalar + v)
                if target_scalar is None:
                    target_scalar = logits[0, -1].max()
                    src_pos = logits.shape[1] - 1
                else:
                    src_pos = src_positions[-1]

            E_list, grad_list = [], []
            L = len(attentions)
            _max_layers = trainer.config.attnlrp_max_layers
            if _max_layers is not None and int(_max_layers) < L:
                layer_ids = list(range(L - int(_max_layers), L))
            else:
                layer_ids = list(range(L))
            if ATTN_METHOD == "rollout":
                layer_ids = []

            for l in layer_ids:
                attn = attentions[l]
                if not torch.is_tensor(attn) or not attn.requires_grad:
                    continue
                g_attn = torch.autograd.grad(
                    outputs=target_scalar, inputs=attn,
                    retain_graph=True, create_graph=False, allow_unused=True,
                )[0]
                if g_attn is None:
                    continue
                A_l = attn[0].to(torch.float32)
                g_l = g_attn[0].to(torch.float32)
                E_l = self._op_layer_relevance(A_l, g_l)
                E_list.append(E_l)
                grad_list.append(g_l)

        if _op_no_relevance(E_list, attentions):
            return torch.ones(NUM_SLOTS, dtype=torch.float32) / NUM_SLOTS

        R = self._op_compute_R(E_list, grad_list, attentions, 0)
        src_pos = max(0, min(src_pos, R.shape[0] - 1))
        relevance = R[src_pos, :]

        visual_indices = self.find_visual_token_indices(
            meta["orig_input_ids"], meta["soft_len"],
        )
        if len(visual_indices) == 0:
            return torch.ones(NUM_SLOTS, dtype=torch.float32) / NUM_SLOTS

        visual_relevance = relevance[visual_indices].abs()

        orig_inputs = meta["orig_inputs"]
        patch_grid_size = None
        if self.model_type == "internvl":
            ar = orig_inputs.get("_internvl_aspect_ratio")
            if ar is not None:
                patch_grid_size = self._get_internvl_patch_grid_size(ar, len(visual_indices))
                content_n = self._get_internvl_content_token_count(ar)
                if len(visual_indices) > content_n:
                    visual_indices = visual_indices[:content_n]
                    visual_relevance = relevance[visual_indices].abs()
        if patch_grid_size is None:
            for key in ["image_grid_thw", "grid_thw", "vision_grid_thw"]:
                if key in orig_inputs and torch.is_tensor(orig_inputs[key]):
                    thw = orig_inputs[key].reshape(-1)
                    if thw.numel() >= 3:
                        patch_grid_size = (int(thw[-2].item()), int(thw[-1].item()))
                        break
        if patch_grid_size is None:
            n_p = len(visual_indices)
            side = int(math.sqrt(n_p))
            patch_grid_size = (max(1, side), max(1, n_p // max(1, side)))
        patch_grid_size = self._fix_patch_grid_size_from_visual_tokens(
            patch_grid_size=patch_grid_size, n_visual=len(visual_indices),
        )

        model_attention = self.aggregate_patch_saliency_to_slots_torch(
            patch_saliency=visual_relevance,
            image_size=(image.width, image.height),
            patch_grid_size=patch_grid_size,
        ).detach().float()

        s = model_attention.sum()
        if s > 1e-9:
            model_attention = model_attention / s
        else:
            model_attention = torch.ones(NUM_SLOTS, dtype=torch.float32) / NUM_SLOTS
        return model_attention

    trainer.compute_model_attention_single = types.MethodType(
        _compute_with_answer_logit, trainer
    )


def split_leave1_test_then_kfold(
    samples: List[Dict],
    K: int = 5,
    seed: int = 42,
) -> Tuple[List[int], List[Tuple[List[int], List[int]]]]:
    rng = np.random.RandomState(seed)
    by_user: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        by_user[s["user_id"]].append(i)

    test_idx_set: set = set()
    history_idx_by_user: Dict[str, List[int]] = {}
    always_train: set = set()

    for uid, idxs in by_user.items():
        arr = np.array(idxs)
        rng.shuffle(arr)
        n = len(arr)
        if n == 1:
            always_train.add(int(arr[0]))
            continue
        test_idx_set.add(int(arr[0]))
        history_idx_by_user[uid] = [int(x) for x in arr[1:]]

    fold_val: List[set] = [set() for _ in range(K)]

    for uid, hidxs in history_idx_by_user.items():
        arr = np.array(hidxs)
        rng.shuffle(arr)
        m = len(arr)
        if m == 1:
            always_train.add(int(arr[0]))
            continue
        offset = rng.randint(0, K)
        for j, ix in enumerate(arr):
            fold_val[(offset + j) % K].add(int(ix))

    history_all = set()
    for hidxs in history_idx_by_user.values():
        history_all.update(hidxs)
    train_pool_all = history_all | always_train

    folds = []
    for f in range(K):
        val = fold_val[f]
        train = list((train_pool_all - val) | always_train)
        folds.append((train, list(val)))

    return list(test_idx_set), folds


def verify_and_log_split(
    samples: List[Dict],
    test_idx: List[int],
    folds: List[Tuple[List[int], List[int]]],
    logger: logging.Logger,
):
    K = len(folds)
    test_set = set(test_idx)
    all_idx = set(range(len(samples)))

    by_user = defaultdict(list)
    for i, s in enumerate(samples):
        by_user[s["user_id"]].append(i)
    n_users = len(by_user)

    test_users  = set(samples[i]["user_id"] for i in test_idx)
    n1_users    = sum(1 for uid, idxs in by_user.items() if len(idxs) == 1)
    n2_users    = sum(1 for uid, idxs in by_user.items() if len(idxs) == 2)

    history_pool = set()
    for tr, va in folds:
        history_pool.update(tr)
        history_pool.update(va)
    always_train_count = sum(
        1 for i in history_pool
        if all(i not in set(va) for _, va in folds)
    )

    user_sample_counts = sorted([len(idxs) for idxs in by_user.values()])
    p25 = int(np.percentile(user_sample_counts, 25))
    p50 = int(np.percentile(user_sample_counts, 50))
    p75 = int(np.percentile(user_sample_counts, 75))


    for f in range(K):
        train_idx, val_idx = folds[f]
        train_users_f = set(samples[i]["user_id"] for i in train_idx)
        val_users_f   = set(samples[i]["user_id"] for i in val_idx)

    errors = []

    for f, (tr, va) in enumerate(folds):
        overlap = test_set & (set(tr) | set(va))
        if overlap:
            errors.append(f"Fold {f}: test samples leak into train/val: {overlap}")

    for f, (tr, va) in enumerate(folds):
        train_users_f = set(samples[i]["user_id"] for i in tr)
        for vi in va:
            uid = samples[vi]["user_id"]
            if uid not in train_users_f:
                errors.append(f"Fold {f}: val user {uid} has no train samples!")

    history_pool = set()
    for f, (tr, va) in enumerate(folds):
        history_pool.update(tr)
        history_pool.update(va)
    history_users = set(samples[i]["user_id"] for i in history_pool)
    for ti in test_idx:
        uid = samples[ti]["user_id"]
        if uid not in history_users:
            errors.append(f"Test user {uid} has no samples in history pool!")

    if errors:
        raise RuntimeError(f"split validation failed with {len(errors)} errors")


def compute_sample_metrics(
    model_attn: np.ndarray,
    gaze_dist: np.ndarray,
    choice_slot_0based: int,
    config: Optional[TrainerConfig] = None,
    logit_probs: Optional[np.ndarray] = None,
    n_candidates: Optional[int] = None,
) -> Dict[str, float]:
    eps = 1e-8
    if n_candidates is not None and 0 < int(n_candidates) < len(model_attn):
        nc = int(n_candidates)
        model_attn = np.asarray(model_attn)[:nc]
        gaze_dist = np.asarray(gaze_dist)[:nc]
        if logit_probs is not None and len(logit_probs) > nc:
            logit_probs = np.asarray(logit_probs)[:nc]
    s_p = float(model_attn.sum())
    s_q = float(gaze_dist.sum())
    if s_p < eps:
        p = np.ones_like(model_attn) / len(model_attn)
    else:
        p = model_attn / s_p
    if s_q < eps:
        q = np.ones_like(gaze_dist) / len(gaze_dist)
    else:
        q = gaze_dist / s_q

    js  = float(jensenshannon(q, p) ** 2)
    if np.isnan(js):
        js = 1.0
    kl  = float(np.sum(q * np.log((q + eps) / (p + eps))))
    cos = float(np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q) + eps))

    model_ranked = np.argsort(p)[::-1]
    gaze_ranked  = np.argsort(q)[::-1]

    click1 = 1.0 if model_ranked[0] == choice_slot_0based else 0.0
    click3 = 1.0 if choice_slot_0based in model_ranked[:3] else 0.0
    click5 = 1.0 if choice_slot_0based in model_ranked[:5] else 0.0

    gaze1 = float(len(set(model_ranked[:1]) & set(gaze_ranked[:1]))) / 1
    gaze3 = float(len(set(model_ranked[:3]) & set(gaze_ranked[:3]))) / 3
    gaze5 = float(len(set(model_ranked[:5]) & set(gaze_ranked[:5]))) / 5

    k = config.topk_k if config is not None else TOPK_K
    topk_idx = np.argsort(q)[::-1][:k]
    tk_q = np.append(q[topk_idx], max(1.0 - q[topk_idx].sum(), 0.0))
    tk_p = np.append(p[topk_idx], max(1.0 - p[topk_idx].sum(), 0.0))
    topk_js = float(jensenshannon(tk_q, tk_p) ** 2)
    if np.isnan(topk_js):
        topk_js = 1.0

    lp = logit_probs if logit_probs is not None else p
    p_choice = float(np.clip(lp[choice_slot_0based], eps, None))
    attn_logloss = -float(np.log(p_choice))

    p_pos = lp[choice_slot_0based]
    neg_mask = np.ones(len(lp), dtype=bool)
    neg_mask[choice_slot_0based] = False
    p_neg = lp[neg_mask]
    attn_auc = float((np.sum(p_neg < p_pos) + 0.5 * np.sum(p_neg == p_pos)) / len(p_neg))

    return {
        "js_div":     js,
        "kl_div":     kl,
        "cosine_sim": cos,
        "topk_js_div": topk_js,
        "attn_logloss": attn_logloss,
        "attn_auc":   attn_auc,
        "click@1":    click1,
        "click@3":    click3,
        "click@5":    click5,
        "gaze@1":     gaze1,
        "gaze@3":     gaze3,
        "gaze@5":     gaze5,
    }




def _op_no_relevance(E_list, attentions) -> bool:
    if ATTN_METHOD == "rollout":
        return attentions is None or len(attentions) == 0
    return len(E_list) == 0


def _attnlrp_from_inputs(
    trainer, inputs, input_ids, answer_text, image,
) -> np.ndarray:
    uniform = np.ones(NUM_SLOTS, dtype=np.float32) / NUM_SLOTS

    with torch.enable_grad():
        outputs = trainer._model_forward(
            inputs,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits
        attentions = outputs.attentions

        L_full = input_ids.shape[1]
        soft_len = inputs["inputs_embeds"].shape[1] - L_full if "inputs_embeds" in inputs else 0

        ans_token_ids = trainer.tokenizer.encode(answer_text, add_special_tokens=False)
        ids_list = input_ids[0].tolist()
        src_positions = []
        target_scalar = None
        for tid in ans_token_ids:
            for pos in range(len(ids_list) - 1, -1, -1):
                if ids_list[pos] == tid:
                    prev = pos + soft_len - 1
                    if prev >= 0:
                        src_positions.append(prev)
                        v = logits[0, prev, tid]
                        target_scalar = v if target_scalar is None else (target_scalar + v)
                    break
        if target_scalar is None:
            target_scalar = logits[0, -1].max()
            src_pos = logits.shape[1] - 1
        else:
            src_pos = src_positions[-1]

        E_list, grad_list = [], []
        L = len(attentions)
        max_layers = trainer.config.attnlrp_max_layers
        if max_layers is not None and int(max_layers) < L:
            layer_ids = list(range(L - int(max_layers), L))
        else:
            layer_ids = list(range(L))

        _diag_logger = getattr(trainer, "logger", None)
        if not hasattr(_attnlrp_from_inputs, "_diag_done"):
            _attnlrp_from_inputs._diag_done = True
            ie = inputs.get("inputs_embeds", None)

        if ATTN_METHOD == "rollout":
            layer_ids = []
        _skipped = 0
        for l in layer_ids:
            attn = attentions[l]
            if not torch.is_tensor(attn) or not attn.requires_grad:
                _skipped += 1
                continue
            g_attn = torch.autograd.grad(
                outputs=target_scalar, inputs=attn,
                retain_graph=True, create_graph=False, allow_unused=True,
            )[0]
            if g_attn is None:
                continue
            A_l = attn[0].to(torch.float32)
            g_l = g_attn[0].to(torch.float32)
            E_l = trainer._op_layer_relevance(A_l, g_l)
            E_list.append(E_l)
            grad_list.append(g_l)

    _logger = getattr(trainer, "logger", None)
    if _op_no_relevance(E_list, attentions):
        return uniform

    R = trainer._op_compute_R(E_list, grad_list, attentions, 0)
    src_pos = max(0, min(src_pos, R.shape[0] - 1))
    relevance = R[src_pos, :]

    ids = input_ids[0].tolist()
    visual_indices = []
    if trainer.model_type == "internvl":
        img_ctx_id = trainer.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        if img_ctx_id is not None:
            visual_indices = [i for i, t in enumerate(ids) if t == img_ctx_id]
    if not visual_indices:
        vs_id = trainer.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        ve_id = trainer.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        try:
            s_idx = ids.index(vs_id)
            e_idx = ids.index(ve_id, s_idx)
            visual_indices = list(range(s_idx + 1, e_idx))
        except ValueError:
            return uniform

    vis_indices_shifted = [idx + soft_len for idx in visual_indices]
    visual_relevance = relevance[vis_indices_shifted].abs()
    n_vis = len(visual_indices)

    patch_grid_size = None
    if trainer.model_type == "internvl":
        ar = inputs.get("_internvl_aspect_ratio", None)
        if ar is not None:
            patch_grid_size = trainer._get_internvl_patch_grid_size(ar, n_vis)
            content_n = trainer._get_internvl_content_token_count(ar)
            if n_vis > content_n:
                vis_indices_shifted = vis_indices_shifted[:content_n]
                visual_relevance = relevance[vis_indices_shifted].abs()
                n_vis = content_n
    if patch_grid_size is None:
        for key in ["image_grid_thw", "grid_thw", "vision_grid_thw"]:
            src = inputs.get(key, None)
            if src is not None and torch.is_tensor(src):
                thw = src.reshape(-1)
                if thw.numel() >= 3:
                    patch_grid_size = (int(thw[-2].item()), int(thw[-1].item()))
                    break
    if patch_grid_size is None:
        side = int(math.sqrt(n_vis))
        patch_grid_size = (max(1, side), max(1, n_vis // max(1, side)))
    patch_grid_size = trainer._fix_patch_grid_size_from_visual_tokens(
        patch_grid_size=patch_grid_size, n_visual=n_vis,
    )

    slot_attn = trainer.aggregate_patch_saliency_to_slots_torch(
        patch_saliency=visual_relevance,
        image_size=(image.width, image.height),
        patch_grid_size=patch_grid_size,
    ).detach().float().cpu().numpy()

    s = slot_attn.sum()
    return slot_attn / s if s > 1e-9 else uniform


def _build_baseline_inputs(trainer, image, profile_text, answer_text=None, for_generate=False):
    dev = next(trainer.model.parameters()).device
    emb_layer = trainer.model.get_input_embeddings()

    if trainer.model_type == "internvl":
        full_text, pv_cpu, num_patches, aspect_ratio = \
            trainer._build_internvl_text(profile_text, image, answer_text if not for_generate else None)
        tok_out = trainer.tokenizer(full_text, return_tensors="pt")
        input_ids = tok_out["input_ids"].to(dev)
        attn_mask = tok_out["attention_mask"].to(dev)
        text_embeds = emb_layer(input_ids).clone()
        pv = pv_cpu.to(device=dev, dtype=text_embeds.dtype)
        with torch.no_grad():
            vit_embeds = trainer.model.extract_feature(pv)
        B, N_seq, C = text_embeds.shape
        flat_embeds = text_embeds.reshape(B * N_seq, C)
        flat_ids = input_ids.reshape(B * N_seq)
        sel = (flat_ids == trainer.model.img_context_token_id)
        vit_flat = vit_embeds.reshape(-1, C).to(flat_embeds.device)
        n_tok = min(int(sel.sum()), vit_flat.shape[0])
        if n_tok > 0:
            indices = sel.nonzero(as_tuple=True)[0][:n_tok]
            flat_embeds[indices] = vit_flat[:n_tok].detach()
        text_embeds = flat_embeds.reshape(B, N_seq, C)
        del pv, vit_embeds, vit_flat
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        text_embeds = text_embeds.detach().requires_grad_(True)
        model_inputs = {
            "inputs_embeds": text_embeds,
            "attention_mask": attn_mask,
            "_internvl_aspect_ratio": aspect_ratio,
        }
        return model_inputs, input_ids, {"num_patches": num_patches, "aspect_ratio": aspect_ratio}
    else:
        if for_generate:
            text = (
                f"<|im_start|>system\n{INSTRUCTION}<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n"
                f"{profile_text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            inputs = trainer.processor(
                text=[text], images=[image], return_tensors="pt", padding=True,
            )
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            input_ids = inputs["input_ids"]
            return inputs, input_ids, {}
        else:
            text = (
                f"<|im_start|>system\n{INSTRUCTION}<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n"
                f"{profile_text}<|im_end|>\n"
                f"<|im_start|>assistant\n{answer_text}"
            )
            inputs = trainer.processor(
                text=[text], images=[image], return_tensors="pt", padding=True,
            )
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            input_ids = inputs["input_ids"]
            with torch.enable_grad():
                text_embeds = emb_layer(input_ids)
            text_embeds = text_embeds.detach().requires_grad_(True)
            model_inputs = dict(inputs)
            model_inputs.pop("input_ids", None)
            model_inputs["inputs_embeds"] = text_embeds
            return model_inputs, input_ids, {}


def compute_baseline_attnlrp_attention(
    trainer, image: Image.Image, profile_text: str, answer_text: str,
) -> np.ndarray:
    trainer.model.eval()
    model_inputs, input_ids, _ = _build_baseline_inputs(
        trainer, image, profile_text, answer_text, for_generate=False)
    return _attnlrp_from_inputs(trainer, model_inputs, input_ids, answer_text, image)


class _AnswerTokenConstraint:
    def __init__(self, valid_token_ids: List[int]):
        self.valid_token_ids = valid_token_ids

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float('-inf'))
        mask[:, self.valid_token_ids] = 0.0
        return scores + mask


def _generate_answer(trainer, gen_inputs, is_internvl=False):
    safe_inputs = {k: v for k, v in gen_inputs.items() if not k.startswith("_")}
    if "inputs_embeds" in safe_inputs:
        input_len = safe_inputs["inputs_embeds"].shape[1]
    elif "input_ids" in safe_inputs:
        input_len = safe_inputs["input_ids"].shape[1]
    else:
        input_len = 0

    constraint = _AnswerTokenConstraint(
        trainer._answer_token_ids[: int(getattr(trainer, "_current_ncand", NUM_SLOTS) or NUM_SLOTS)])
    gen_kwargs = dict(
        **safe_inputs,
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=trainer.tokenizer.pad_token_id,
        logits_processor=[constraint],
    )

    if is_internvl:
        gen_out = trainer.model.language_model.generate(**gen_kwargs)
    else:
        gen_out = trainer.model.generate(**gen_kwargs)

    if input_len > 0 and gen_out.shape[1] > input_len:
        new_tokens = gen_out[0][input_len:]
    else:
        new_tokens = gen_out[0]
    return trainer.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _compute_option_logprobs(trainer, image, profile_text, user_id=None):
    trainer.model.eval()
    dev = next(trainer.model.parameters()).device
    is_internvl = (trainer.model_type == "internvl")

    answer_tids = trainer._answer_token_ids[: int(getattr(trainer, "_current_ncand", NUM_SLOTS) or NUM_SLOTS)]

    if user_id is not None:
        if is_internvl:
            soft_prompt = trainer.prompt_basis([user_id])[0]
            inputs_gen, _ = trainer.build_inputs_with_soft_prompt(
                image=image, profile_text=profile_text,
                soft_prompt_embeds=soft_prompt, answer_text=None,
            )
            fwd = {k: v for k, v in inputs_gen.items()
                   if k != "labels" and not k.startswith("_")}
            with torch.no_grad():
                outputs = trainer.model.language_model(**fwd)
        else:
            text_gen = (
                f"<|im_start|>system\n{INSTRUCTION}<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n"
                f"{profile_text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            inputs_gen = trainer.processor(
                text=[text_gen], images=[image], return_tensors="pt", padding=True,
            )
            inputs_gen = {k: v.to(dev) for k, v in inputs_gen.items()}
            soft_prompt = trainer.prompt_basis([user_id])[0]
            emb_layer = trainer.model.get_input_embeddings()
            emb_device = emb_layer.weight.device
            text_embeds = emb_layer(inputs_gen["input_ids"].to(emb_device))
            sp = soft_prompt.to(device=emb_device, dtype=text_embeds.dtype)
            gen_embeds = torch.cat([sp.unsqueeze(0), text_embeds], dim=1)
            soft_len = sp.shape[0]
            fwd = {k: v for k, v in inputs_gen.items() if k != "input_ids"}
            fwd["inputs_embeds"] = gen_embeds
            fwd["attention_mask"] = torch.cat([
                torch.ones(1, soft_len, dtype=torch.long, device=emb_device),
                inputs_gen["attention_mask"].to(emb_device),
            ], dim=1)
            with torch.no_grad():
                outputs = trainer.model(**fwd)
    else:
        model_inputs, _, _ = _build_baseline_inputs(
            trainer, image, profile_text, answer_text=None, for_generate=True)
        fwd = {k: v for k, v in model_inputs.items() if not k.startswith("_")}
        with torch.no_grad():
            if is_internvl:
                outputs = trainer.model.language_model(**fwd)
            else:
                outputs = trainer.model(**fwd)

    logits = outputs.logits[0, -1, :]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return np.array([log_probs[tid].item() for tid in answer_tids])


def compute_freegeneration_attention(
    trainer, user_id: str, image: Image.Image, profile_text: str,
) -> Tuple[np.ndarray, str]:
    trainer.model.eval()
    dev = next(trainer.model.parameters()).device
    emb_device = trainer.model.get_input_embeddings().weight.device

    if trainer.model_type == "internvl":
        soft_prompt = trainer.prompt_basis([user_id])[0]
        inputs_gen, _ = trainer.build_inputs_with_soft_prompt(
            image=image, profile_text=profile_text,
            soft_prompt_embeds=soft_prompt, answer_text=None,
        )
        gen_inputs = {k: v for k, v in inputs_gen.items() if k != "labels"}
        with torch.no_grad():
            generated_text = _generate_answer(trainer, gen_inputs, is_internvl=True)
    else:
        text_gen = (
            f"<|im_start|>system\n{INSTRUCTION}<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n"
            f"{profile_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs_gen = trainer.processor(
            text=[text_gen], images=[image], return_tensors="pt", padding=True,
        )
        inputs_gen = {k: v.to(dev) for k, v in inputs_gen.items()}

        soft_prompt = trainer.prompt_basis([user_id])[0]
        emb_layer = trainer.model.get_input_embeddings()
        text_embeds = emb_layer(inputs_gen["input_ids"].to(emb_device))
        sp = soft_prompt.to(device=emb_device, dtype=text_embeds.dtype)
        gen_embeds = torch.cat([sp.unsqueeze(0), text_embeds], dim=1)
        soft_len = sp.shape[0]

        gen_inputs = dict(inputs_gen)
        gen_inputs.pop("input_ids", None)
        gen_inputs["inputs_embeds"] = gen_embeds
        gen_inputs["attention_mask"] = torch.cat([
            torch.ones(1, soft_len, dtype=torch.long, device=emb_device),
            inputs_gen["attention_mask"].to(emb_device),
        ], dim=1)

        with torch.no_grad():
            generated_text = _generate_answer(trainer, gen_inputs, is_internvl=False)

    slot_attn = trainer.compute_model_attention_single(
        user_id=user_id,
        image=image,
        profile_text=profile_text,
        answer_text=generated_text,
    )
    if torch.is_tensor(slot_attn):
        slot_attn = slot_attn.detach().float().cpu().numpy()

    s = float(slot_attn.sum())
    if s > 1e-9:
        slot_attn = slot_attn / s
    else:
        slot_attn = np.ones(NUM_SLOTS, dtype=np.float32) / NUM_SLOTS

    return slot_attn, generated_text


def compute_baseline_freegeneration_attention(
    trainer, image: Image.Image, profile_text: str,
) -> Tuple[np.ndarray, str]:
    trainer.model.eval()
    model_inputs, _, _ = _build_baseline_inputs(
        trainer, image, profile_text, answer_text=None, for_generate=True)
    with torch.no_grad():
        generated_text = _generate_answer(
            trainer, model_inputs, is_internvl=(trainer.model_type == "internvl"))
    slot_attn = compute_baseline_attnlrp_attention(
        trainer, image, profile_text, generated_text,
    )
    return slot_attn, generated_text


def evaluate_baseline_freegeneration_on_samples(
    trainer,
    samples_list: List[Dict],
    user_features: Dict,
    poster_dir: str,
    expanded_dirs: Optional[List] = None,
    logger: Optional[logging.Logger] = None,
    per_sample_save_path: Optional[str] = None,
) -> Dict[str, float]:
    all_metrics: List[Dict[str, float]] = []
    per_user_metrics: Dict[str, List[Dict]] = defaultdict(list)
    gt_letters: List[str] = []
    gen_letters: List[str] = []
    per_sample_keys: List[Dict] = []
    n_total = len(samples_list)
    n_skip = 0
    correct_count = 0
    recall3_count = 0
    recall5_count = 0


    for i, sample in enumerate(samples_list):
        uid     = sample["user_id"]
        task_id = sample["task_id"]
        choice  = sample["choice_slot"]
        dwell   = sample["dwell_time"]

        image, _scaled_bb = _open_page_image_and_bboxes(sample)
        trainer._current_bboxes = _scaled_bb
        trainer._current_ncand = int(sample["n_candidates"])
        if image is None:
            n_skip += 1
            continue

        profile_text = _profile_text_for(uid, sample, user_features)
        gt_answer = slot_to_letter(choice)

        try:
            freegen_attn, gen_answer = compute_baseline_freegeneration_attention(
                trainer, image, profile_text,
            )
        except Exception as e:
            n_skip += 1
            continue

        match = gen_answer.strip().upper() == gt_answer.strip().upper()
        if match:
            correct_count += 1

        logit_probs = None
        try:
            opt_lp = _compute_option_logprobs(trainer, image, profile_text, user_id=None)
            ranked = np.argsort(opt_lp)[::-1]
            gt_idx = choice - 1
            if gt_idx in ranked[:3]:
                recall3_count += 1
            if gt_idx in ranked[:5]:
                recall5_count += 1
            logit_probs = np.exp(opt_lp)
            logit_probs = logit_probs / logit_probs.sum()
        except Exception:
            pass

        gt_letters.append(_normalize_option_letter(gt_answer))
        gen_letters.append(_normalize_option_letter(gen_answer))


        gaze_dist = gaze_distribution_from_dwell_time(dwell)
        metrics = compute_sample_metrics(freegen_attn, gaze_dist, choice - 1, logit_probs=logit_probs,
                                         n_candidates=int(sample["n_candidates"]))
        all_metrics.append(metrics)
        per_user_metrics[uid].append(metrics)
        per_sample_keys.append({"user_id": uid, "task_id": int(task_id),
                                "visit_index": int(sample.get("visit_index", -1)),
                                "n_candidates": int(sample.get("n_candidates", NUM_SLOTS)),
                                "choice_slot": int(choice),
                                "answer_correct": int(match),
                                "gen_letter": _normalize_option_letter(gen_answer)})

        done = len(all_metrics)


    if not all_metrics:
        return {}

    result: Dict[str, float] = {}
    metric_names = list(all_metrics[0].keys())
    for name in metric_names:
        result[f"micro_{name}"] = float(np.mean([m[name] for m in all_metrics]))
    for name in metric_names:
        user_means = [
            float(np.mean([m[name] for m in ms]))
            for ms in per_user_metrics.values()
        ]
        result[f"macro_{name}"] = float(np.mean(user_means))
    result["n_samples"] = len(all_metrics)
    result["n_users"] = len(per_user_metrics)
    result["answer_accuracy"] = correct_count / len(all_metrics) if all_metrics else 0.0
    result["recall@3"] = recall3_count / len(all_metrics) if all_metrics else 0.0
    result["recall@5"] = recall5_count / len(all_metrics) if all_metrics else 0.0
    if per_sample_save_path:
        os.makedirs(os.path.dirname(per_sample_save_path), exist_ok=True)
        _records = [{**k, **m} for k, m in zip(per_sample_keys, all_metrics)]
        with open(per_sample_save_path, "w", encoding="utf-8") as _f:
            json.dump(_records, _f, ensure_ascii=False)
    gt_ratios = compute_option_ratios(gt_letters)
    bl_fg_ratios = compute_option_ratios(gen_letters)
    for k, v in gt_ratios.items():
        result[f"gt_ratio_{k}"] = float(v)
    for k, v in bl_fg_ratios.items():
        result[f"bl_fg_ratio_{k}"] = float(v)
    return result


def evaluate_baseline_on_samples(
    trainer,
    samples_list: List[Dict],
    user_features: Dict,
    poster_dir: str,
    expanded_dirs: Optional[List] = None,
    logger: Optional[logging.Logger] = None,
    per_sample_save_path: Optional[str] = None,
) -> Dict[str, float]:
    all_metrics: List[Dict[str, float]] = []
    per_user_metrics: Dict[str, List[Dict]] = defaultdict(list)
    per_sample_keys: List[Dict] = []
    n_total = len(samples_list)
    n_skip = 0
    recall3_count = 0
    recall5_count = 0


    for i, sample in enumerate(samples_list):
        uid     = sample["user_id"]
        task_id = sample["task_id"]
        choice  = sample["choice_slot"]
        dwell   = sample["dwell_time"]

        image, _scaled_bb = _open_page_image_and_bboxes(sample)
        trainer._current_bboxes = _scaled_bb
        trainer._current_ncand = int(sample["n_candidates"])
        if image is None:
            n_skip += 1
            continue

        profile_text = _profile_text_for(uid, sample, user_features)
        answer_text = slot_to_letter(choice)

        try:
            baseline_attn = compute_baseline_attnlrp_attention(
                trainer, image, profile_text, answer_text,
            )
        except Exception as e:
            n_skip += 1
            continue

        logit_probs = None
        try:
            opt_lp = _compute_option_logprobs(trainer, image, profile_text, user_id=None)
            ranked = np.argsort(opt_lp)[::-1]
            gt_idx = choice - 1
            if gt_idx in ranked[:3]:
                recall3_count += 1
            if gt_idx in ranked[:5]:
                recall5_count += 1
            logit_probs = np.exp(opt_lp)
            logit_probs = logit_probs / logit_probs.sum()
        except Exception:
            pass

        gaze_dist = gaze_distribution_from_dwell_time(dwell)
        metrics = compute_sample_metrics(baseline_attn, gaze_dist, choice - 1, logit_probs=logit_probs,
                                         n_candidates=int(sample["n_candidates"]))
        all_metrics.append(metrics)
        per_user_metrics[uid].append(metrics)
        per_sample_keys.append({"user_id": uid, "task_id": int(task_id),
                                "visit_index": int(sample.get("visit_index", -1)),
                                "n_candidates": int(sample.get("n_candidates", NUM_SLOTS)),
                                "choice_slot": int(choice)})

        done = len(all_metrics)

    if not all_metrics:
        return {}

    result: Dict[str, float] = {}
    metric_names = list(all_metrics[0].keys())
    for name in metric_names:
        result[f"micro_{name}"] = float(np.mean([m[name] for m in all_metrics]))
    for name in metric_names:
        user_means = [
            float(np.mean([m[name] for m in ms]))
            for ms in per_user_metrics.values()
        ]
        result[f"macro_{name}"] = float(np.mean(user_means))
    result["n_samples"] = len(all_metrics)
    result["n_users"] = len(per_user_metrics)
    result["recall@3"] = recall3_count / len(all_metrics) if all_metrics else 0.0
    result["recall@5"] = recall5_count / len(all_metrics) if all_metrics else 0.0
    if per_sample_save_path:
        os.makedirs(os.path.dirname(per_sample_save_path), exist_ok=True)
        records = [{**k, **m} for k, m in zip(per_sample_keys, all_metrics)]
        with open(per_sample_save_path, "w", encoding="utf-8") as _f:
            json.dump(records, _f, ensure_ascii=False)
    return result


def evaluate_freegeneration_on_samples(
    trainer,
    samples_list: List[Dict],
    user_features: Dict,
    poster_dir: str,
    expanded_dirs: Optional[List] = None,
    logger: Optional[logging.Logger] = None,
    per_sample_save_path: Optional[str] = None,
) -> Dict[str, float]:
    all_metrics: List[Dict[str, float]] = []
    per_user_metrics: Dict[str, List[Dict]] = defaultdict(list)
    gen_letters: List[str] = []
    per_sample_keys: List[Dict] = []
    n_total = len(samples_list)
    n_skip = 0
    correct_count = 0
    recall3_count = 0
    recall5_count = 0


    for i, sample in enumerate(samples_list):
        uid     = sample["user_id"]
        task_id = sample["task_id"]
        choice  = sample["choice_slot"]
        dwell   = sample["dwell_time"]

        image, _scaled_bb = _open_page_image_and_bboxes(sample)
        trainer._current_bboxes = _scaled_bb
        trainer._current_ncand = int(sample["n_candidates"])
        if image is None:
            n_skip += 1
            continue

        profile_text = _profile_text_for(uid, sample, user_features)
        gt_answer = slot_to_letter(choice)

        try:
            freegen_attn, gen_answer = compute_freegeneration_attention(
                trainer, uid, image, profile_text,
            )
        except Exception as e:
            n_skip += 1
            continue

        match = gen_answer.strip().upper() == gt_answer.strip().upper()
        if match:
            correct_count += 1

        logit_probs = None
        try:
            opt_lp = _compute_option_logprobs(trainer, image, profile_text, user_id=uid)
            ranked = np.argsort(opt_lp)[::-1]
            gt_idx = choice - 1
            if gt_idx in ranked[:3]:
                recall3_count += 1
            if gt_idx in ranked[:5]:
                recall5_count += 1
            logit_probs = np.exp(opt_lp)
            logit_probs = logit_probs / logit_probs.sum()
        except Exception:
            pass

        gen_letters.append(_normalize_option_letter(gen_answer))


        gaze_dist = gaze_distribution_from_dwell_time(dwell)
        metrics = compute_sample_metrics(freegen_attn, gaze_dist, choice - 1, logit_probs=logit_probs,
                                         n_candidates=int(sample["n_candidates"]))
        all_metrics.append(metrics)
        per_user_metrics[uid].append(metrics)
        per_sample_keys.append({"user_id": uid, "task_id": int(task_id),
                                "visit_index": int(sample.get("visit_index", -1)),
                                "n_candidates": int(sample.get("n_candidates", NUM_SLOTS)),
                                "choice_slot": int(choice),
                                "answer_correct": int(match),
                                "gen_letter": _normalize_option_letter(gen_answer)})

        done = len(all_metrics)


    if not all_metrics:
        return {}

    result: Dict[str, float] = {}
    metric_names = list(all_metrics[0].keys())
    for name in metric_names:
        result[f"micro_{name}"] = float(np.mean([m[name] for m in all_metrics]))
    for name in metric_names:
        user_means = [
            float(np.mean([m[name] for m in ms]))
            for ms in per_user_metrics.values()
        ]
        result[f"macro_{name}"] = float(np.mean(user_means))
    result["n_samples"] = len(all_metrics)
    result["n_users"] = len(per_user_metrics)
    result["answer_accuracy"] = correct_count / len(all_metrics) if all_metrics else 0.0
    result["recall@3"] = recall3_count / len(all_metrics) if all_metrics else 0.0
    result["recall@5"] = recall5_count / len(all_metrics) if all_metrics else 0.0
    if per_sample_save_path:
        os.makedirs(os.path.dirname(per_sample_save_path), exist_ok=True)
        _records = [{**k, **m} for k, m in zip(per_sample_keys, all_metrics)]
        with open(per_sample_save_path, "w", encoding="utf-8") as _f:
            json.dump(_records, _f, ensure_ascii=False)
    tr_fg_ratios = compute_option_ratios(gen_letters)
    for k, v in tr_fg_ratios.items():
        result[f"tr_fg_ratio_{k}"] = float(v)
    return result


def _load_image(sample) -> Optional[Image.Image]:
    if not isinstance(sample, dict):
        return None
    p = sample.get("image_path", "")
    if p and os.path.exists(p):
        return Image.open(p).convert("RGB")
    return None


def evaluate_on_samples(
    trainer,
    samples_list: List[Dict],
    user_features: Dict,
    poster_dir: str,
    expanded_dirs: Optional[List] = None,
    logger: Optional[logging.Logger] = None,
    per_sample_save_path: Optional[str] = None,
) -> Dict[str, float]:
    all_metrics: List[Dict[str, float]] = []
    per_user_metrics: Dict[str, List[Dict]] = defaultdict(list)
    per_sample_keys: List[Dict] = []
    n_total = len(samples_list)
    n_skip  = 0
    recall3_count = 0
    recall5_count = 0

    if logger:
        eval_users = set(s["user_id"] for s in samples_list)

    for i, sample in enumerate(samples_list):
        uid      = sample["user_id"]
        task_id  = sample["task_id"]
        choice   = sample["choice_slot"]
        dwell    = sample["dwell_time"]

        image, _scaled_bb = _open_page_image_and_bboxes(sample)
        trainer._current_bboxes = _scaled_bb
        trainer._current_ncand = int(sample["n_candidates"])
        if image is None:
            n_skip += 1
            continue

        profile_text = _profile_text_for(uid, sample, user_features)
        answer_text = slot_to_letter(choice)

        try:
            model_attn = trainer.compute_model_attention_single(
                user_id=uid,
                image=image,
                profile_text=profile_text,
                answer_text=answer_text,
            )
            if torch.is_tensor(model_attn):
                model_attn = model_attn.detach().float().cpu().numpy()
        except Exception as e:
            n_skip += 1
            continue

        logit_probs = None
        try:
            opt_lp = _compute_option_logprobs(trainer, image, profile_text, user_id=uid)
            ranked = np.argsort(opt_lp)[::-1]
            gt_idx = choice - 1
            if gt_idx in ranked[:3]:
                recall3_count += 1
            if gt_idx in ranked[:5]:
                recall5_count += 1
            logit_probs = np.exp(opt_lp)
            logit_probs = logit_probs / logit_probs.sum()
        except Exception:
            pass


        gaze_dist = gaze_distribution_from_dwell_time(dwell)


        metrics   = compute_sample_metrics(model_attn, gaze_dist, choice - 1, logit_probs=logit_probs,
                                             n_candidates=int(sample["n_candidates"]))
        all_metrics.append(metrics)
        per_user_metrics[uid].append(metrics)
        per_sample_keys.append({"user_id": uid, "task_id": int(task_id),
                                "visit_index": int(sample.get("visit_index", -1)),
                                "n_candidates": int(sample.get("n_candidates", NUM_SLOTS)),
                                "choice_slot": int(choice)})

        done = len(all_metrics)


    if not all_metrics:
        return {}

    result: Dict[str, float] = {}
    metric_names = list(all_metrics[0].keys())

    for name in metric_names:
        result[f"micro_{name}"] = float(np.mean([m[name] for m in all_metrics]))

    for name in metric_names:
        user_means = [
            float(np.mean([m[name] for m in ms]))
            for ms in per_user_metrics.values()
        ]
        result[f"macro_{name}"] = float(np.mean(user_means))

    result["n_samples"] = len(all_metrics)
    result["n_users"]   = len(per_user_metrics)
    result["recall@3"] = recall3_count / len(all_metrics) if all_metrics else 0.0
    result["recall@5"] = recall5_count / len(all_metrics) if all_metrics else 0.0

    if per_sample_save_path:
        os.makedirs(os.path.dirname(per_sample_save_path), exist_ok=True)
        records = [{**k, **m} for k, m in zip(per_sample_keys, all_metrics)]
        with open(per_sample_save_path, "w", encoding="utf-8") as _f:
            json.dump(records, _f, ensure_ascii=False)
    return result


_HP_TO_CONFIG = {
    "BETA_REG":           "beta_reg",
    "LAMBDA_ATTN_TARGET": "lambda_attn_target",
    "ATTNLRP_MAX_LAYERS": "attnlrp_max_layers",
    "CHOICE_LOSS_WEIGHT": "choice_loss_weight",
    "ATTN_LOSS_MODE":     "attn_loss_mode",
    "TOPK_K":             "topk_k",
    "POWER_GAMMA":        "power_gamma",
}


def apply_hyperparams_to_trainer(trainer, hp: Dict):
    cfg = trainer.config
    for hp_key, cfg_attr in _HP_TO_CONFIG.items():
        if hp_key in hp:
            setattr(cfg, cfg_attr, hp[hp_key])

    if MODEL_TYPE == "internvl":
        cap = INTERNVL_ATTNLRP_MAX_LAYERS
        if cfg.attnlrp_max_layers > cap:
            cfg.attnlrp_max_layers = cap


def apply_hyperparams_to_module(hp: Dict):
    for key in ("BETA_REG", "LAMBDA_ATTN_TARGET", "ATTNLRP_MAX_LAYERS",
                "CHOICE_LOSS_WEIGHT", "ATTN_LOSS_MODE", "TOPK_K", "POWER_GAMMA"):
        if key in hp:
            globals()[key] = hp[key]
    if MODEL_TYPE == "internvl":
        current = globals().get("ATTNLRP_MAX_LAYERS", 6)
        cap = INTERNVL_ATTNLRP_MAX_LAYERS
        if current > cap:
            globals()["ATTNLRP_MAX_LAYERS"] = cap


def reset_prompt_basis(trainer, hp: Dict):
    hidden_dim     = trainer.prompt_basis.hidden_dim
    model_dtype    = next(trainer.model.parameters()).dtype
    num_basis      = hp.get("NUM_BASIS",      NUM_BASIS)
    num_soft_tokens = hp.get("NUM_SOFT_TOKENS", NUM_SOFT_TOKENS)

    new_pb = PromptBasisModule(
        num_basis=num_basis,
        num_soft_tokens=num_soft_tokens,
        hidden_dim=hidden_dim,
        use_user_alpha=trainer.config.use_user_alpha,
    )
    trainer.prompt_basis = new_pb.to(dtype=model_dtype)
    trainer.global_step = 0


def build_optimizer(trainer, hp: Dict) -> torch.optim.Optimizer:
    if trainer.config.use_random_soft_prompt:
        _dummy = nn.Parameter(torch.zeros(1, device=trainer.prompt_basis.basis.device))
        return torch.optim.AdamW([{"params": [_dummy], "lr": 0.0}], weight_decay=0.0)
    param_groups = [{"params": [trainer.prompt_basis.basis], "lr": hp["BASIS_LR"]}]
    if trainer.config.use_user_alpha:
        alpha_params = list(trainer.prompt_basis.user_alphas.values())
        param_groups.append({"params": alpha_params, "lr": hp["ALPHA_LR"]})
    return torch.optim.AdamW(param_groups, weight_decay=hp.get("WEIGHT_DECAY", 1e-3))


def _collate_fn(batch):
    return {
        "user_id":     [s["user_id"]     for s in batch],
        "task_id":     [s["task_id"]     for s in batch],
        "image":       [s["image"]       for s in batch],
        "profile_text": [s["profile_text"] for s in batch],
        "choice_slot": torch.stack([torch.tensor(s["choice_slot"]) for s in batch]),
        "gaze_dist":   torch.stack([s["gaze_dist"]                 for s in batch]),
        "bboxes":      [s["bboxes"]       for s in batch],
        "n_candidates": [int(s["n_candidates"]) for s in batch],
    }


def train_one_fold(
    trainer,
    train_samples: List[Dict],
    val_samples: List[Dict],
    user_features: Dict,
    poster_dir: str,
    expanded_dirs,
    hp: Dict,
    num_epochs: int,
    eval_every: int,
    logger: logging.Logger,
) -> Dict:
    fold_start_time = time.time()
    apply_hyperparams_to_trainer(trainer, hp)
    reset_prompt_basis(trainer, hp)

    train_uids = set(s["user_id"] for s in train_samples)
    val_uids   = set(s["user_id"] for s in val_samples)
    all_uids   = train_uids | val_uids
    for uid in all_uids:
        trainer.prompt_basis.get_or_create_user_alpha(uid)

    optimizer = build_optimizer(trainer, hp)

    train_dataset = GazeDataset(
        train_samples, user_features, poster_dir,
        expanded_dirs=expanded_dirs,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=_collate_fn,
    )
    n_batches = len(train_loader)

    cfg = trainer.config

    best_val_score  = float("inf") if PRIMARY_LOWER_IS_BETTER else float("-inf")
    best_val_metrics = None
    best_epoch       = 0
    best_state       = None

    for epoch in range(1, num_epochs + 1):
        epoch_losses_total = []
        epoch_losses_attn  = []
        epoch_losses_reg   = []
        accum = GRADIENT_ACCUMULATION_STEPS
        _step_fn = trainer.train_step_batched if USE_BATCHED_FORWARD else trainer.train_step
        for bi, batch in enumerate(train_loader):
            is_first = (bi % accum == 0)
            is_last  = ((bi + 1) % accum == 0) or (bi + 1 == n_batches)
            loss_dict = _step_fn(
                batch, optimizer,
                accumulation_steps=accum,
                zero_grad=is_first,
                do_optimizer_step=is_last,
            )
            epoch_losses_total.append(loss_dict["total_loss"])
            epoch_losses_attn.append(loss_dict.get("loss_attn", 0.0))
            epoch_losses_reg.append(loss_dict.get("loss_reg", 0.0))


        avg_loss     = float(np.mean(epoch_losses_total))
        avg_attn     = float(np.mean(epoch_losses_attn))
        avg_reg      = float(np.mean(epoch_losses_reg))

        do_eval = (epoch % eval_every == 0) or (epoch == num_epochs)
        if do_eval:
            if USE_FG_FOR_BEST_EPOCH:
                val_metrics = evaluate_freegeneration_on_samples(
                    trainer, val_samples, user_features,
                    poster_dir, expanded_dirs, logger,
                )
            else:
                val_metrics = evaluate_on_samples(
                    trainer, val_samples, user_features,
                    poster_dir, expanded_dirs, logger,
                )
            score, score_detail = compute_primary_score(val_metrics, fg_mode=USE_FG_FOR_BEST_EPOCH)
            improved = False
            if np.isfinite(score):
                improved = (
                    (score < best_val_score) if PRIMARY_LOWER_IS_BETTER
                    else (score > best_val_score)
                )
            if improved:
                best_val_score  = score
                best_val_metrics = val_metrics
                best_epoch       = epoch
                best_state = {
                    k: v.cpu().clone()
                    for k, v in trainer.prompt_basis.state_dict().items()
                }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fold_elapsed = time.time() - fold_start_time

    return {
        "best_epoch":       best_epoch,
        "best_val_score":   best_val_score,
        "best_val_metrics": best_val_metrics or {},
        "best_state":       best_state,
    }


def cross_validate(
    trainer,
    samples: List[Dict],
    folds: List[Tuple[List[int], List[int]]],
    user_features: Dict,
    poster_dir: str,
    expanded_dirs,
    hp: Dict,
    num_epochs: int,
    eval_every: int,
    logger: logging.Logger,
) -> Dict:
    K = len(folds)
    fold_results = []


    for f in range(K):
        train_idx, val_idx = folds[f]
        train_samps = [samples[i] for i in train_idx]
        val_samps   = [samples[i] for i in val_idx]

        train_uids_f = set(s["user_id"] for s in train_samps)
        val_uids_f   = set(s["user_id"] for s in val_samps)


        result = train_one_fold(
            trainer, train_samps, val_samps,
            user_features, poster_dir, expanded_dirs,
            hp, num_epochs, eval_every, logger,
        )
        fold_results.append(result)


    valid_folds = [r for r in fold_results if r["best_val_metrics"]]
    if not valid_folds:
        return {"avg_metrics": {}, "avg_best_epoch": 0, "fold_results": fold_results}

    metric_keys = list(valid_folds[0]["best_val_metrics"].keys())
    avg_metrics: Dict[str, float] = {}
    for key in metric_keys:
        vals = [r["best_val_metrics"][key] for r in valid_folds]
        avg_metrics[key] = float(np.mean(vals))

    avg_best_epoch = float(np.mean([r["best_epoch"] for r in valid_folds]))
    avg_primary_score, avg_primary_detail = compute_primary_score(avg_metrics, fg_mode=USE_FG_FOR_BEST_EPOCH)


    return {
        "avg_metrics":    avg_metrics,
        "avg_primary_score": avg_primary_score,
        "avg_primary_detail": avg_primary_detail,
        "avg_best_epoch": avg_best_epoch,
        "fold_results":   fold_results,
    }


def grid_search_cv(
    trainer,
    samples: List[Dict],
    folds: List[Tuple[List[int], List[int]]],
    user_features: Dict,
    poster_dir: str,
    expanded_dirs,
    param_grid: Dict[str, List],
    num_epochs: int,
    eval_every: int,
    logger: logging.Logger,
) -> Tuple[Dict, Dict]:
    keys   = sorted(param_grid.keys())
    combos = list(product(*(param_grid[k] for k in keys)))


    best_score      = float("inf") if PRIMARY_LOWER_IS_BETTER else float("-inf")
    best_hp         = None
    best_cv         = None
    search_log      = []

    for ci, combo in enumerate(combos):
        hp = dict(DEFAULT_HYPERPARAMS)
        for k, v in zip(keys, combo):
            hp[k] = v


        cv_result = cross_validate(
            trainer, samples, folds, user_features,
            poster_dir, expanded_dirs, hp, num_epochs, eval_every, logger,
        )
        score = cv_result.get("avg_primary_score", float("nan"))

        search_log.append({
            "hyperparams": {k: hp[k] for k in keys},
            "primary_score": score,
            "primary_detail": cv_result.get("avg_primary_detail", ""),
            "avg_metrics": cv_result["avg_metrics"],
        })

        improved = False
        if np.isfinite(score):
            improved = (
                (score < best_score) if PRIMARY_LOWER_IS_BETTER
                else (score > best_score)
            )
        if improved:
            best_score = score
            best_hp    = hp.copy()
            best_cv    = cv_result

    if best_hp is None:
        raise RuntimeError("grid search failed: no valid primary metric found.")

    log_path = os.path.join(OUTPUT_DIR, f"attnlrp_grid_search_results_{RUN_TIMESTAMP}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(search_log, f, indent=2, ensure_ascii=False)

    return best_hp, best_cv


def _log_four_way_comparison(
    logger: logging.Logger,
    bl_tf_metrics: Dict,
    bl_fg_metrics: Dict,
    tr_tf_metrics: Dict,
    tr_fg_metrics: Dict,
    seed_label=None,
):
    cols = [
        ("BL-TF",  bl_tf_metrics),
        ("BL-FG",  bl_fg_metrics),
        ("TR-TF",  tr_tf_metrics),
        ("TR-FG",  tr_fg_metrics),
    ]

    seed_suffix = f" [seed={seed_label}]" if seed_label is not None else ""

    for label, m in cols:
        n = m.get("n_samples", 0)
        acc = m.get("answer_accuracy", None)
        acc_str = f", ans_acc={acc:.4f}" if acc is not None else ""

    option_letters = [chr(ord("A") + i) for i in range(NUM_SLOTS)] + ["OTHER"]
    opt_header = (f"  {'option':<8s} {'GroundTruth':>12s} {'Baseline-FG':>12s} "
                  f"{'Trained-FG':>12s}  (each column: fraction of samples per option)")
    for opt in option_letters:
        gt_r = bl_fg_metrics.get(f"gt_ratio_{opt}", 0.0) or 0.0
        bl_r = bl_fg_metrics.get(f"bl_fg_ratio_{opt}", 0.0) or 0.0
        tr_r = tr_fg_metrics.get(f"tr_fg_ratio_{opt}", 0.0) or 0.0

    n_users = tr_tf_metrics.get("n_users", 0)

    metric_keys = sorted(
        [k for k in tr_tf_metrics if k.startswith("micro_")]
    )
    def _format_delta_with_pct(trained: float, baseline: float) -> str:
        delta = trained - baseline
        if (not np.isfinite(baseline)) or abs(baseline) < 1e-12:
            pct_str = "NA"
        else:
            pct = (delta / baseline) * 100.0
            pct_str = f"{pct:+.2f}%"
        return f"{delta:+.4f} ({pct_str})"

    header = f"  {'metric':<18s}"
    for label, _ in cols:
        header += f" {label:>10s}"
    header += f" {'Δ(TR-BL)TF':>24s} {'Δ(TR-BL)FG':>24s}"

    for k in metric_keys:
        short = k.replace("micro_", "")
        vals = [m.get(k, float("nan")) for _, m in cols]
        line = f"  {short:<18s}"
        for v in vals:
            line += f" {v:>10.4f}"
        line += f" {_format_delta_with_pct(vals[2], vals[0]):>24s}"
        line += f" {_format_delta_with_pct(vals[3], vals[1]):>24s}"

    bl_fg_acc = bl_fg_metrics.get("answer_accuracy", None)
    tr_fg_acc = tr_fg_metrics.get("answer_accuracy", None)
    acc_line = f"  {'answer_accuracy':<18s}"
    acc_line += f" {'N/A':>10s}"
    acc_line += f" {bl_fg_acc:>10.4f}" if bl_fg_acc is not None else f" {'N/A':>10s}"
    acc_line += f" {'N/A':>10s}"
    acc_line += f" {tr_fg_acc:>10.4f}" if tr_fg_acc is not None else f" {'N/A':>10s}"
    acc_line += f" {'N/A':>24s}"
    if bl_fg_acc is not None and tr_fg_acc is not None:
        acc_line += f" {_format_delta_with_pct(tr_fg_acc, bl_fg_acc):>24s}"
    else:
        acc_line += f" {'N/A':>24s}"

    for rk in ("recall@3", "recall@5"):
        r_vals = [m.get(rk, float("nan")) for _, m in cols]
        r_line = f"  {rk:<18s}"
        for v in r_vals:
            r_line += f" {v:>10.4f}" if np.isfinite(v) else f" {'N/A':>10s}"
        r_line += f" {_format_delta_with_pct(r_vals[2], r_vals[0]):>24s}"
        r_line += f" {_format_delta_with_pct(r_vals[3], r_vals[1]):>24s}"

    macro_keys = sorted(
        [k for k in tr_tf_metrics if k.startswith("macro_")]
    )
    for k in macro_keys:
        short = k.replace("macro_", "")
        vals = [m.get(k, float("nan")) for _, m in cols]
        line = f"  {short:<18s}"
        for v in vals:
            line += f" {v:>10.4f}"
        line += f" {_format_delta_with_pct(vals[2], vals[0]):>24s}"
        line += f" {_format_delta_with_pct(vals[3], vals[1]):>24s}"


    for rk in ("recall@3", "recall@5"):
        r_vals = [m.get(rk, float("nan")) for _, m in cols]
        r_line = f"  {rk:<18s}"
        for v in r_vals:
            r_line += f" {v:>10.4f}" if np.isfinite(v) else f" {'N/A':>10s}"
        r_line += f" {_format_delta_with_pct(r_vals[2], r_vals[0]):>24s}"
        r_line += f" {_format_delta_with_pct(r_vals[3], r_vals[1]):>24s}"


def final_train_and_test(
    trainer,
    all_samples: List[Dict],
    test_idx: List[int],
    user_features: Dict,
    poster_dir: str,
    expanded_dirs,
    hp: Dict,
    num_epochs: int,
    logger: logging.Logger,
    val_ratio: float = 0.0,
    suppress_summary_log: bool = False,
    current_seed: Optional[int] = None,
) -> Dict[str, float]:

    test_set = set(test_idx)
    train_samples_all = [s for i, s in enumerate(all_samples) if i not in test_set]
    test_samples  = [all_samples[i] for i in test_idx]

    use_val = len(VAL_SAMPLES) > 0
    train_samples = train_samples_all
    val_samples   = list(VAL_SAMPLES)

    train_uids = set(s["user_id"] for s in train_samples)
    test_uids  = set(s["user_id"] for s in test_samples)


    apply_hyperparams_to_trainer(trainer, hp)
    reset_prompt_basis(trainer, hp)

    all_uids = set(s["user_id"] for s in all_samples)
    for uid in all_uids:
        trainer.prompt_basis.get_or_create_user_alpha(uid)

    optimizer = build_optimizer(trainer, hp)

    train_dataset = GazeDataset(
        train_samples, user_features, poster_dir,
        expanded_dirs=expanded_dirs,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=_collate_fn,
    )
    n_batches = len(train_loader)

    best_val_score  = float("inf") if PRIMARY_LOWER_IS_BETTER else float("-inf")
    best_epoch      = 0
    best_state      = None
    early_stop_patience = 5
    no_improve_count    = 0

    mode_desc = "val best-epoch selection" if use_val else "last-epoch parameters"
    if use_val:
        val_mode = "FG (Free Generation)" if USE_FG_FOR_BEST_EPOCH else "TF (Teacher Forcing)"

    for epoch in range(1, num_epochs + 1):
        epoch_losses_total = []
        epoch_losses_attn  = []
        epoch_losses_reg   = []
        accum = GRADIENT_ACCUMULATION_STEPS
        _step_fn = trainer.train_step_batched if USE_BATCHED_FORWARD else trainer.train_step
        for bi, batch in enumerate(train_loader):
            is_first = (bi % accum == 0)
            is_last  = ((bi + 1) % accum == 0) or (bi + 1 == n_batches)
            loss_dict = _step_fn(
                batch, optimizer,
                accumulation_steps=accum,
                zero_grad=is_first,
                do_optimizer_step=is_last,
            )
            epoch_losses_total.append(loss_dict["total_loss"])
            epoch_losses_attn.append(loss_dict.get("loss_attn", 0.0))
            epoch_losses_reg.append(loss_dict.get("loss_reg", 0.0))


        avg_loss = float(np.mean(epoch_losses_total))
        avg_attn = float(np.mean(epoch_losses_attn))
        avg_reg  = float(np.mean(epoch_losses_reg))

        do_eval = use_val and ((epoch % EVAL_EVERY_N_EPOCHS == 0) or (epoch == num_epochs))
        if do_eval:
            if USE_FG_FOR_BEST_EPOCH:
                val_metrics = evaluate_freegeneration_on_samples(
                    trainer, val_samples, user_features,
                    poster_dir, expanded_dirs, logger,
                )
            else:
                val_metrics = evaluate_on_samples(
                    trainer, val_samples, user_features,
                    poster_dir, expanded_dirs, logger,
                )
            score, score_detail = compute_primary_score(val_metrics, fg_mode=USE_FG_FOR_BEST_EPOCH)
            improved = False
            if np.isfinite(score):
                improved = (
                    (score < best_val_score) if PRIMARY_LOWER_IS_BETTER
                    else (score > best_val_score)
                )
            if improved:
                best_val_score = score
                best_epoch     = epoch
                best_state = {
                    k: v.cpu().clone()
                    for k, v in trainer.prompt_basis.state_dict().items()
                }
                no_improve_count = 0
            else:
                no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                break

    actual_epochs = epoch
    if use_val and best_state is not None:
        trainer.prompt_basis.load_state_dict(best_state)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"attnlrp_checkpoint_final_cv_{RUN_TIMESTAMP}.pt")
    trainer.save_checkpoint(ckpt_path)

    _per_sample_dir = os.path.join(OUTPUT_DIR, "per_sample")
    _seed = current_seed if current_seed is not None else SEED
    _bl_tf_path = os.path.join(_per_sample_dir,
                               f"per_sample_bl_tf_{RUN_TIMESTAMP}_seed{_seed}.json")
    _tr_tf_path = os.path.join(_per_sample_dir,
                               f"per_sample_tr_tf_{RUN_TIMESTAMP}_seed{_seed}.json")
    _bl_fg_path = os.path.join(_per_sample_dir,
                               f"per_sample_bl_fg_{RUN_TIMESTAMP}_seed{_seed}.json")
    _tr_fg_path = os.path.join(_per_sample_dir,
                               f"per_sample_tr_fg_{RUN_TIMESTAMP}_seed{_seed}.json")

    bl_tf_metrics = evaluate_baseline_on_samples(
        trainer, test_samples, user_features,
        poster_dir, expanded_dirs, logger,
        per_sample_save_path=_bl_tf_path,
    )

    bl_fg_metrics = evaluate_baseline_freegeneration_on_samples(
        trainer, test_samples, user_features,
        poster_dir, expanded_dirs, logger,
        per_sample_save_path=_bl_fg_path,
    )

    tr_tf_metrics = evaluate_on_samples(
        trainer, test_samples, user_features,
        poster_dir, expanded_dirs, logger,
        per_sample_save_path=_tr_tf_path,
    )

    tr_fg_metrics = evaluate_freegeneration_on_samples(
        trainer, test_samples, user_features,
        poster_dir, expanded_dirs, logger,
        per_sample_save_path=_tr_fg_path,
    )

    if not suppress_summary_log:
        _log_four_way_comparison(logger, bl_tf_metrics, bl_fg_metrics,
                                 tr_tf_metrics, tr_fg_metrics)

    _wilcoxon_block = {}
    try:
        _wilcoxon_block = _compute_wilcoxon_per_user(
            _bl_tf_path, _tr_tf_path, _bl_fg_path, _tr_fg_path, logger)
    except Exception as _we:
        pass
        pass

    _cfg = trainer.config

    _result_seed_suffix = f"_seed{current_seed}" if current_seed is not None else ""
    result_path = os.path.join(OUTPUT_DIR,
        f"att_attlrp_final_test_results_{RUN_TIMESTAMP}{_result_seed_suffix}.json")
    _saved_seeds = [current_seed] if current_seed is not None \
                   else (MULTI_SEEDS if USE_MULTI_SEED else [SEED])
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "attention_method": ATTN_METHOD,
                "attn_method_params": {
                    "GLIMPSE_HEAD_TEMP": GLIMPSE_HEAD_TEMP,
                    "GLIMPSE_DEPTH_TEMP": GLIMPSE_DEPTH_TEMP,
                    "ROLLOUT_DISCARD_RATIO": ROLLOUT_DISCARD_RATIO,
                    "MAX_LAYERS": _cfg.attnlrp_max_layers,
                },
                "model_type": MODEL_TYPE,
                "slot_kl_signal": SLOT_KL_SIGNAL,
                "signal_intersect": SIGNAL_INTERSECT,
                "lambda_attn_target": LAMBDA_ATTN_TARGET,
                "seeds": _saved_seeds,
                "training_params": {
                    "TRAIN_CHOICE": _cfg.train_choice,
                    "CHOICE_LOSS_WEIGHT": _cfg.choice_loss_weight,
                    "LAMBDA_ATTN_TARGET": _cfg.lambda_attn_target,
                    "ATTN_LOSS_MODE": _cfg.attn_loss_mode,
                    "TOPK_K": _cfg.topk_k,
                    "POWER_GAMMA": _cfg.power_gamma,
                },
                "hyperparams": {
                    k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in hp.items()
                },
                "baseline_tf_metrics": bl_tf_metrics,
                "baseline_fg_metrics": bl_fg_metrics,
                "trained_tf_metrics":  tr_tf_metrics,
                "trained_fg_metrics":  tr_fg_metrics,
                "wilcoxon_per_user":   _wilcoxon_block,
            },
            f, indent=2, ensure_ascii=False,
        )
    _model_label = "internvl" if MODEL_TYPE == "internvl" else "qwen"
    _printed = {
        "ATTN_LOSS_MODE", "TOPK_K", "POWER_GAMMA",
        "BETA_REG", "BASIS_LR", "ALPHA_LR", "WEIGHT_DECAY",
        "NUM_BASIS", "NUM_SOFT_TOKENS",
    }
    for k, v in hp.items():
        if k in _printed:
            continue
    combined_metrics = dict(tr_tf_metrics)
    if tr_fg_metrics:
        combined_metrics["answer_accuracy"] = tr_fg_metrics.get("answer_accuracy", float("nan"))
        for k, v in tr_fg_metrics.items():
            if k not in combined_metrics:
                combined_metrics[f"fg_{k}"] = v
    combined_metrics["_four_way_metrics"] = {
        "bl_tf": bl_tf_metrics,
        "bl_fg": bl_fg_metrics,
        "tr_tf": tr_tf_metrics,
        "tr_fg": tr_fg_metrics,
    }
    return combined_metrics


def _set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def multi_seed_final_train_and_test(
    trainer,
    all_samples: List[Dict],
    test_idx: List[int],
    user_features: Dict,
    poster_dir: str,
    expanded_dirs,
    hp: Dict,
    num_epochs: int,
    logger: logging.Logger,
    val_ratio: float = 0.0,
) -> Tuple[Dict[str, float], List[Tuple[int, Dict]]]:
    seeds = MULTI_SEEDS

    all_metrics = []
    all_seed_four_way = []
    best_score = float("inf") if PRIMARY_LOWER_IS_BETTER else float("-inf")
    best_metrics = None
    best_seed = None

    for si, seed in enumerate(seeds):
        _set_all_seeds(seed)

        metrics = final_train_and_test(
            trainer, all_samples, test_idx, user_features,
            poster_dir, expanded_dirs, hp, num_epochs, logger,
            val_ratio=val_ratio,
            suppress_summary_log=True,
            current_seed=seed,
        )
        all_metrics.append((seed, metrics))

        four_way = metrics.pop("_four_way_metrics", None)
        if four_way is not None:
            all_seed_four_way.append((seed, four_way))

        score, detail = compute_primary_score(metrics)
        if np.isfinite(score):
            improved = (score < best_score) if PRIMARY_LOWER_IS_BETTER else (score > best_score)
            if improved:
                best_score = score
                best_metrics = metrics
                best_seed = seed

    avg_metrics: Dict[str, float] = {}
    if all_metrics:
        all_keys: set = set()
        for _, m in all_metrics:
            all_keys.update(m.keys())
        for k in sorted(all_keys):
            vals = [
                m[k] for _, m in all_metrics
                if k in m and isinstance(m[k], (int, float)) and np.isfinite(m[k])
            ]
            if vals:
                avg_metrics[k] = float(np.mean(vals))

    avg_score, avg_detail = compute_primary_score(avg_metrics)

    for seed, m in all_metrics:
        s, d = compute_primary_score(m)
        marker = " *best*" if seed == best_seed else ""

    if best_metrics is not None:
        for k in sorted(best_metrics.keys()):
            v = best_metrics[k]

    return avg_metrics, all_seed_four_way


def main():
    total_start_time = time.time()
    logger = setup_logger()


    user_features = load_user_features(USER_FEATURES_CSV)
    samples, test_idx = load_fixate_samples(logger)

    by_user_raw = defaultdict(list)
    for s in samples:
        by_user_raw[s["user_id"]].append(s["task_id"])
    sample_counts = sorted([len(v) for v in by_user_raw.values()])

    expanded_dirs = None
    folds = []
    assert not RUN_GRID_SEARCH, "FixATE-v2: CV grid-search path not ported; calibrate on the fixed val split (E0)."

    trainer = AttnLRPTrainer(model_name=MODEL_NAME, model_type=MODEL_TYPE, logger=logger)
    patch_trainer_for_eval(trainer, logger)


    if RUN_GRID_SEARCH:
        keys = sorted(PARAM_GRID.keys())
        n_combos = 1
        for v in PARAM_GRID.values():
            n_combos *= len(v)

        cv_start_time = time.time()
        best_hp, best_cv = grid_search_cv(
            trainer, samples, folds, user_features,
            POSTER_IMAGES_DIR, expanded_dirs,
            PARAM_GRID, NUM_EPOCHS_CV, EVAL_EVERY_N_EPOCHS, logger,
        )
        cv_elapsed = time.time() - cv_start_time
        cv_avg_best_epoch = best_cv.get("avg_best_epoch", NUM_EPOCHS_FINAL)
        final_num_epochs = max(1, round(cv_avg_best_epoch))
    else:
        best_hp = dict(DEFAULT_HYPERPARAMS)
        final_num_epochs = NUM_EPOCHS_FINAL

    final_val_ratio = NO_CV_VAL_RATIO if not RUN_GRID_SEARCH else 0.0
    all_seed_four_way = []
    if USE_MULTI_SEED:
        final_start_time = time.time()
        test_metrics, all_seed_four_way = multi_seed_final_train_and_test(
            trainer, samples, test_idx, user_features,
            POSTER_IMAGES_DIR, expanded_dirs,
            best_hp, final_num_epochs, logger,
            val_ratio=final_val_ratio,
        )
    else:
        final_start_time = time.time()
        _set_all_seeds(SEED)
        test_metrics = final_train_and_test(
            trainer, samples, test_idx, user_features,
            POSTER_IMAGES_DIR, expanded_dirs,
            best_hp, final_num_epochs, logger,
            val_ratio=final_val_ratio,
        )
    final_elapsed = time.time() - final_start_time

    total_elapsed = time.time() - total_start_time
    log_file = os.path.join(OUTPUT_DIR, f"attnlrp_cv_pipeline_{RUN_TIMESTAMP}.log")
    _model_display = "internvl" if MODEL_TYPE == "internvl" else "qwen"

    if USE_MULTI_SEED and all_seed_four_way:
        for seed, fw in all_seed_four_way:
            _log_four_way_comparison(
                logger,
                fw["bl_tf"], fw["bl_fg"], fw["tr_tf"], fw["tr_fg"],
                seed_label=seed,
            )


if __name__ == "__main__":
    main()
