#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AdSERP SERP: AttnLRP training (variable-length AOIs, query-conditioned). Run from repo root."""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import json
import math
import logging

logging.disable(logging.CRITICAL)

from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from itertools import product

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.spatial.distance import jensenshannon

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from config.common_config import (
    DEVICE,
    EVAL_EVERY_N_EPOCHS,
    GAZE_TAU,
    K_FOLDS,
    MULTI_SEEDS,
    NUM_BASIS,
    NUM_SOFT_TOKENS,
    RUN_TIMESTAMP,
    SEED,
    compute_primary_score,
)
from config.attnlrp_config_adserp import (
    ADSERP_IMAGES_DIR,
    ADSERP_INSTRUCTION,
    ADSERP_SAMPLES_JSONL,
    ATTNLRP_CREATE_GRAPH,
    ATTNLRP_GRAD_SCALE,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    GRADIENT_ACCUMULATION_STEPS,
    INTERNVL_MODEL_PATH,
    MODEL_NAME,
    MODEL_TYPE,
    NUM_EPOCHS_CV,
    NUM_EPOCHS_FINAL,
    OUTPUT_DIR,
    PARAM_GRID,
    TrainerConfig,
    USE_USER_ALPHA,
    VIEWPORT_WIDTH,
)

INSTRUCTION = ADSERP_INSTRUCTION


def infer_model_type_from_name(model_name: str) -> Optional[str]:
    """Infer backbone type from checkpoint path or name."""
    name = (model_name or "").lower()
    if "internvl" in name:
        return "internvl"
    if "qwen" in name:
        return "qwen3vl"
    return None

def load_adserp_data(
    jsonl_path: str,
    images_dir: str,
    only_visible: bool = True,
) -> List[Dict]:
    """Load AdSERP samples from JSONL. Each dict has sample_id, user_id, query, image_path,
    aoi_bboxes (viewport coords), aoi_types, gaze_dwell_ms, clicked_aoi_idx, n_aois, viewport_size.
    """
    samples = []
    n_total = 0
    n_no_click = 0
    n_no_gaze = 0
    n_no_image = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            n_total += 1
            raw = json.loads(line.strip())

            if raw.get("clicked_aoi_id") is None:
                n_no_click += 1
                continue

            scroll_y = raw["scroll_y_px"]
            viewport = raw["viewport_rect"]
            vp_y1 = viewport["y1"]
            vp_y2 = viewport["y2"]
            vp_height = vp_y2 - vp_y1

            aoi_bboxes = []
            aoi_types = []
            aoi_gaze = []
            aoi_original_ids = []

            for aoi in raw["aois"]:
                if only_visible and not aoi["visible_in_viewport"]:
                    continue

                bx1 = aoi["bbox"]["x1"]
                by1 = aoi["bbox"]["y1"] - scroll_y
                bx2 = aoi["bbox"]["x2"]
                by2 = aoi["bbox"]["y2"] - scroll_y

                by1 = max(0, by1)
                by2 = min(vp_height, by2)
                if by2 <= by1:
                    continue

                aoi_bboxes.append((bx1, by1, bx2, by2))
                aoi_types.append(aoi["type"])
                aoi_gaze.append(aoi["gaze_dwell_ms"])
                aoi_original_ids.append(aoi["aoi_id"])

            if len(aoi_bboxes) == 0:
                continue

            gaze_arr = np.array(aoi_gaze, dtype=np.float32)
            if gaze_arr.sum() <= 0:
                n_no_gaze += 1
                continue

            clicked_orig_id = raw["clicked_aoi_id"]
            clicked_idx = None
            for i, oid in enumerate(aoi_original_ids):
                if oid == clicked_orig_id:
                    clicked_idx = i
                    break

            if clicked_idx is None:
                n_no_click += 1
                continue

            img_path = os.path.join(images_dir, f"{raw['trial_id']}.jpg")
            if not os.path.exists(img_path):
                n_no_image += 1
                continue

            samples.append({
                "sample_id": raw["sample_id"],
                "user_id": raw["user_id"],
                "query": raw["query"],
                "image_path": img_path,
                "aoi_bboxes": aoi_bboxes,
                "aoi_types": aoi_types,
                "gaze_dwell_ms": gaze_arr,
                "clicked_aoi_idx": clicked_idx,
                "n_aois": len(aoi_bboxes),
                "viewport_size": (VIEWPORT_WIDTH, vp_height),
            })

    return samples


def gaze_distribution_from_dwell(
    dwell_ms: np.ndarray,
    tau: float = GAZE_TAU,
    eps: float = 1e-8,
) -> np.ndarray:
    """Convert per-AOI dwell times to a tempered probability vector."""
    t = dwell_ms.astype(np.float32)
    g = t / (t.sum() + eps)
    g = np.exp(np.log(g + eps) / tau)
    g = g / (g.sum() + eps)
    return g



class AdSERPDataset(Dataset):
    """AdSERP samples with variable-length AOIs."""

    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s["image_path"]).convert("RGB")
        gaze_dist = gaze_distribution_from_dwell(s["gaze_dwell_ms"])

        return {
            "sample_id": s["sample_id"],
            "user_id": s["user_id"],
            "query": s["query"],
            "image": image,
            "aoi_bboxes": s["aoi_bboxes"],
            "aoi_types": s["aoi_types"],
            "gaze_dist": torch.from_numpy(gaze_dist),
            "clicked_aoi_idx": s["clicked_aoi_idx"],
            "n_aois": s["n_aois"],
            "viewport_size": s["viewport_size"],
        }


def adserp_collate_fn(batch: List[Dict]) -> Dict:
    """Collate without stacking variable-length fields."""
    return {
        "sample_id": [b["sample_id"] for b in batch],
        "user_id": [b["user_id"] for b in batch],
        "query": [b["query"] for b in batch],
        "image": [b["image"] for b in batch],
        "aoi_bboxes": [b["aoi_bboxes"] for b in batch],
        "aoi_types": [b["aoi_types"] for b in batch],
        "gaze_dist": [b["gaze_dist"] for b in batch],
        "clicked_aoi_idx": [b["clicked_aoi_idx"] for b in batch],
        "n_aois": [b["n_aois"] for b in batch],
        "viewport_size": [b["viewport_size"] for b in batch],
    }


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def _preprocess_image_for_internvl(image: Image.Image, input_size=448, max_num=4):
    transform = _build_internvl_transform(input_size)
    tiles, aspect_ratio = _dynamic_preprocess_internvl(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(t) for t in tiles])
    num_patches = len(tiles)
    return pixel_values, num_patches, aspect_ratio



class PromptBasisModule(nn.Module):
    def __init__(
        self,
        num_basis: int = NUM_BASIS,
        num_soft_tokens: int = NUM_SOFT_TOKENS,
        hidden_dim: int = 3584,
        use_user_alpha: bool = USE_USER_ALPHA,
    ):
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
        device = self.basis.device
        soft_prompts = []
        for user_id in user_ids:
            if self.use_user_alpha:
                alpha = self.get_or_create_user_alpha(user_id)
                alpha = alpha.to(device=device)
                pi = F.softmax(alpha, dim=0).to(dtype=self.basis.dtype)
            else:
                pi = torch.ones(self.num_basis, device=device, dtype=self.basis.dtype) / self.num_basis
            user_prompt = torch.einsum('b,bmd->md', pi, self.basis)
            soft_prompts.append(user_prompt)
        return torch.stack(soft_prompts, dim=0)

    def get_user_alpha_l2_loss(self, user_ids: List[str]) -> torch.Tensor:
        if not self.use_user_alpha:
            return torch.zeros((), device=self.basis.device)
        loss = 0.0
        for user_id in user_ids:
            alpha = self.get_or_create_user_alpha(user_id)
            loss += torch.sum(alpha ** 2)
        return loss / len(user_ids)



class AttnLRPTrainerAdSERP:
    """AttnLRP trainer for AdSERP (variable AOIs, patch→bbox aggregation)."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        model_type: str = MODEL_TYPE,
        prompt_basis: Optional[PromptBasisModule] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.model_type = model_type

        inferred_type = infer_model_type_from_name(model_name)
        if inferred_type is not None and inferred_type != self.model_type:
            self.model_type = inferred_type

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

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        if prompt_basis is None:
            hidden_dim = self._detect_hidden_dim()
            prompt_basis = PromptBasisModule(
                num_basis=NUM_BASIS, num_soft_tokens=NUM_SOFT_TOKENS, hidden_dim=hidden_dim)

        model_dtype = next(self.model.parameters()).dtype
        self.prompt_basis = prompt_basis.to(dtype=model_dtype)

        self.eps = 1e-4
        self.global_step = 0

    def _detect_hidden_dim(self) -> int:
        if hasattr(self.model.config, 'hidden_size'):
            return self.model.config.hidden_size
        if hasattr(self.model.config, 'text_config') and self.model.config.text_config is not None:
            if hasattr(self.model.config.text_config, 'hidden_size'):
                return self.model.config.text_config.hidden_size
        try:
            emb = self.model.get_input_embeddings()
            if hasattr(emb, 'embedding_dim'):
                return emb.embedding_dim
            if hasattr(emb, 'weight'):
                return emb.weight.shape[-1]
        except Exception:
            pass
        return 3584

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

        visual_indices = [idx + soft_prompt_length for idx in visual_indices]
        return visual_indices

    def attnlrp_layer_relevance(self, attn_l: torch.Tensor, grad_l: torch.Tensor) -> torch.Tensor:
        """AttnLRP per-layer relevance (Prop 3.3 gradient approximation)."""
        AW = attn_l * grad_l
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
        """Stack layer relevances with depth-weighted sum."""
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

    def aggregate_patch_saliency_to_aois_torch(
        self,
        patch_saliency: torch.Tensor,
        image_size: Tuple[int, int],
        patch_grid_size: Tuple[int, int],
        aoi_bboxes: List[Tuple[int, int, int, int]],
    ) -> torch.Tensor:
        """Map patch saliency to variable-length AOI boxes; normalize to a distribution."""
        device = patch_saliency.device
        dtype = patch_saliency.dtype
        H_p, W_p = patch_grid_size
        n_aois = len(aoi_bboxes)

        if patch_saliency.dim() == 1:
            n = patch_saliency.numel()
            if n != H_p * W_p:
                if H_p * W_p == 4 * n and H_p % 2 == 0 and W_p % 2 == 0:
                    Hm = H_p // 2
                    Wm = W_p // 2
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
        patch_w = img_w / float(W_p)
        patch_h = img_h / float(H_p)

        ii = torch.arange(H_p, device=device)
        jj = torch.arange(W_p, device=device)
        grid_i, grid_j = torch.meshgrid(ii, jj, indexing="ij")

        cx = (grid_j.to(torch.float32) + 0.5) * patch_w  # (H_p, W_p)
        cy = (grid_i.to(torch.float32) + 0.5) * patch_h

        cx_flat = cx.reshape(-1)
        cy_flat = cy.reshape(-1)
        vals_flat = ps2d.reshape(-1)

        aoi_sal = torch.zeros(n_aois, device=device, dtype=dtype)
        for a_idx, (bx1, by1, bx2, by2) in enumerate(aoi_bboxes):
            mask = (
                (cx_flat >= bx1) & (cx_flat < bx2) &
                (cy_flat >= by1) & (cy_flat < by2)
            )
            if mask.any():
                aoi_sal[a_idx] = vals_flat[mask].sum()

        total = aoi_sal.sum()
        if total > self.eps:
            aoi_attn = aoi_sal / total
        else:
            aoi_attn = torch.ones(n_aois, device=device, dtype=dtype) / n_aois

        return aoi_attn

    def _model_forward(self, inputs_for_model: Dict, **kwargs):
        if self.model_type == "internvl":
            fwd_kwargs = {k: v for k, v in inputs_for_model.items()}
            fwd_kwargs.update(kwargs)
            return self.model.language_model(**fwd_kwargs)
        return self.model(**inputs_for_model, **kwargs)

    def _get_internvl_patch_grid_size(self, aspect_ratio, n_visual: int):
        tok_per_side = int(math.sqrt(self._internvl_num_image_token))
        rows_tiles, cols_tiles = aspect_ratio
        return (tok_per_side * rows_tiles, tok_per_side * cols_tiles)

    def _get_internvl_content_token_count(self, aspect_ratio):
        tok_per_side = int(math.sqrt(self._internvl_num_image_token))
        rows_tiles, cols_tiles = aspect_ratio
        return tok_per_side * rows_tiles * tok_per_side * cols_tiles

    def _build_internvl_text(self, query: str, image: Image.Image,
                              answer_text: Optional[str] = None):
        user_prompt = f"<image>\nSearch query: {query}\n\n{INSTRUCTION}"

        pixel_values_cpu, num_patches, aspect_ratio = _preprocess_image_for_internvl(image)
        num_image_token = self._internvl_num_image_token

        sys.path.insert(0, INTERNVL_MODEL_PATH)
        from conversation import get_conv_template
        sys.path.pop(0)
        template = get_conv_template(self.model.template)
        template.system_message = "You are a search behavior simulator."
        template.append_message(template.roles[0], user_prompt)
        template.append_message(template.roles[1], None)
        full_query = template.get_prompt()

        image_tokens = '<img>' + '<IMG_CONTEXT>' * num_image_token * num_patches + '</img>'
        full_query = full_query.replace('<image>', image_tokens, 1)

        if answer_text is not None:
            full_query = full_query + str(answer_text)
        return full_query, pixel_values_cpu, num_patches, aspect_ratio

    def _build_inputs_internvl(self, image, query, soft_prompt_embeds, answer_text=None):
        full_text, pv_cpu, num_patches, aspect_ratio = \
            self._build_internvl_text(query, image, answer_text)

        tok_out = self.tokenizer(full_text, return_tensors="pt")
        orig_input_ids = tok_out["input_ids"]
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

        emb_layer = self.model.get_input_embeddings()
        emb_device = emb_layer.weight.device
        emb_dtype = emb_layer.weight.dtype

        orig_input_ids = orig_input_ids.to(emb_device)
        orig_attention_mask = orig_attention_mask.to(emb_device)
        if labels is not None:
            labels = labels.to(emb_device)

        text_embeds = emb_layer(orig_input_ids).clone()
        pv = pv_cpu.to(device=emb_device, dtype=emb_dtype)
        with torch.no_grad():
            vit_embeds = self.model.extract_feature(pv)

        B, N_seq, C = text_embeds.shape
        flat_embeds = text_embeds.reshape(B * N_seq, C)
        flat_ids = orig_input_ids.reshape(B * N_seq)
        selected = (flat_ids == self.model.img_context_token_id)
        vit_flat = vit_embeds.reshape(-1, C).to(flat_embeds.device)
        n_tok = min(int(selected.sum()), vit_flat.shape[0])
        if n_tok > 0:
            indices = selected.nonzero(as_tuple=True)[0][:n_tok]
            flat_embeds[indices] = vit_flat[:n_tok].detach()
        text_embeds = flat_embeds.reshape(B, N_seq, C)
        del pv, vit_embeds, vit_flat
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        soft_len = soft_prompt_embeds.shape[0]
        soft_prompt_embeds = soft_prompt_embeds.to(device=emb_device, dtype=text_embeds.dtype)
        combined_embeds = torch.cat([soft_prompt_embeds.unsqueeze(0), text_embeds], dim=1)
        soft_mask = torch.ones(1, soft_len, dtype=torch.long, device=emb_device)

        inputs_for_model = {
            "inputs_embeds": combined_embeds,
            "attention_mask": torch.cat([soft_mask, orig_attention_mask], dim=1),
        }
        if labels is not None:
            inputs_for_model["labels"] = torch.cat(
                [torch.full((1, soft_len), -100, dtype=labels.dtype, device=emb_device), labels], dim=1)
            answer_pos_combined = [p + soft_len for p in answer_pos_orig]

        orig_inputs = {
            "input_ids": orig_input_ids,
            "attention_mask": orig_attention_mask,
            "_internvl_num_patches": num_patches,
            "_internvl_aspect_ratio": aspect_ratio,
        }
        meta = {
            "orig_inputs": orig_inputs, "orig_input_ids": orig_input_ids,
            "soft_len": soft_len, "answer_token_ids": answer_token_ids,
            "answer_pos_orig": answer_pos_orig, "answer_pos_combined": answer_pos_combined,
            "emb_device": emb_device,
        }
        return inputs_for_model, meta

    def build_inputs_with_soft_prompt(
        self,
        image: Image.Image,
        query: str,
        soft_prompt_embeds: torch.Tensor,
        answer_text: Optional[str] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:

        if self.model_type == "internvl":
            return self._build_inputs_internvl(image, query, soft_prompt_embeds, answer_text)

        user_prompt = f"Search query: {query}\n\n{INSTRUCTION}"

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

        combined_embeds = torch.cat([soft_prompt_embeds.unsqueeze(0), text_embeds], dim=1)
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

    def _fix_patch_grid_size_from_visual_tokens(
        self, patch_grid_size: Tuple[int, int], n_visual: int,
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

    def _infer_patch_grid_size(self, meta: Dict, visual_indices: List[int]) -> Tuple[int, int]:
        orig_inputs = meta["orig_inputs"]
        patch_grid_size = None

        if self.model_type == "internvl":
            ar = orig_inputs.get("_internvl_aspect_ratio")
            if ar is not None:
                patch_grid_size = self._get_internvl_patch_grid_size(ar, len(visual_indices))

        if patch_grid_size is None:
            for key in ["image_grid_thw", "grid_thw", "vision_grid_thw"]:
                if key in orig_inputs and torch.is_tensor(orig_inputs[key]):
                    thw = orig_inputs[key].reshape(-1)
                    if thw.numel() >= 3:
                        patch_grid_size = (int(thw[-2].item()), int(thw[-1].item()))
                        break

        if patch_grid_size is None:
            n_patches = len(visual_indices)
            side = int(math.sqrt(n_patches))
            patch_grid_size = (max(1, side), max(1, n_patches // max(1, side)))

        patch_grid_size = self._fix_patch_grid_size_from_visual_tokens(
            patch_grid_size=patch_grid_size, n_visual=len(visual_indices))
        return patch_grid_size

    def compute_losses(
        self,
        model_attention: torch.Tensor,
        gaze_dist: torch.Tensor,
        user_ids: List[str],
        lambda_attn: float,
        beta_reg: float,
        loss_choice: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """CE (choice) + weighted KL to gaze + alpha L2. Choice enters with coefficient 1 (no extra weight)."""
        g = gaze_dist.unsqueeze(0)
        a = model_attention.unsqueeze(0)
        eps = self.eps
        gamma = self.config.power_gamma
        eps_pw = self.config.power_eps
        raw_w = (g + eps_pw).pow(gamma)
        weights = raw_w / raw_w.sum(dim=1, keepdim=True)
        kl_per = g * (torch.log(g + eps) - torch.log(a + eps))
        loss_attn = (weights * kl_per).sum(dim=1).mean()
        loss_reg = self.prompt_basis.get_user_alpha_l2_loss(user_ids)
        lc = loss_choice if loss_choice is not None else torch.zeros((), device=a.device, dtype=a.dtype)
        total_loss = lc + lambda_attn * loss_attn + beta_reg * loss_reg
        loss_dict = {
            "loss_choice": float(lc.detach().float().cpu().item()),
            "loss_attn": float(loss_attn.detach().cpu().item()),
            "loss_reg": float(loss_reg.detach().cpu().item()),
            "total_loss": float(total_loss.detach().cpu().item()),
        }
        return total_loss, loss_dict

    def train_step(
        self,
        batch: Dict,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int = 1,
        zero_grad: bool = True,
        do_optimizer_step: bool = True,
    ) -> Dict[str, float]:
        user_ids = batch["user_id"]
        images = batch["image"]
        queries = batch["query"]
        gaze_dists = batch["gaze_dist"]
        clicked_idxs = batch["clicked_aoi_idx"]
        aoi_bboxes_list = batch["aoi_bboxes"]
        n_aois_list = batch["n_aois"]
        viewport_sizes = batch["viewport_size"]
        batch_size = len(user_ids)

        soft_prompts = self.prompt_basis(user_ids)

        all_loss_dicts = []

        if zero_grad:
            optimizer.zero_grad(set_to_none=True)

        for i in range(batch_size):
            image = images[i]
            query = queries[i]
            soft_prompt = soft_prompts[i]
            gaze_dist_i = gaze_dists[i].to(DEVICE)
            clicked_idx = clicked_idxs[i]
            aoi_bboxes = aoi_bboxes_list[i]
            n_aois = n_aois_list[i]
            viewport_size = viewport_sizes[i]

            answer_text = str(clicked_idx)

            inputs, meta = self.build_inputs_with_soft_prompt(
                image=image, query=query,
                soft_prompt_embeds=soft_prompt, answer_text=answer_text,
            )

            emb_device = meta["emb_device"]
            soft_prompt = soft_prompt.to(emb_device)

            with torch.enable_grad():
                outputs = self._model_forward(
                    inputs, output_attentions=True, use_cache=False, return_dict=True,
                )
                logits = outputs.logits
                attentions = outputs.attentions

                choice_loss = None
                if self.config.train_choice and getattr(outputs, "loss", None) is not None:
                    choice_loss = outputs.loss

                ans_pos = meta["answer_pos_combined"]
                ans_ids = meta["answer_token_ids"]

                if ans_pos is None or ans_ids is None or len(ans_pos) == 0:
                    target_scalar = logits[0, -1].max()
                    src_pos = logits.shape[1] - 1
                else:
                    target_scalar = None
                    src_positions = []
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

                # AttnLRP backward
                E_list = []
                L = len(attentions)
                max_l = self.config.attnlrp_max_layers
                if max_l is not None and int(max_l) < L:
                    layer_ids = list(range(L - int(max_l), L))
                else:
                    layer_ids = list(range(L))

                valid_layers, valid_attns = [], []
                for l in layer_ids:
                    attn = attentions[l]
                    if torch.is_tensor(attn) and attn.requires_grad:
                        valid_layers.append(l)
                        valid_attns.append(attn)

                if len(valid_attns) > 0:
                    all_grads = torch.autograd.grad(
                        outputs=target_scalar, inputs=valid_attns,
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
                        E_l = self.attnlrp_layer_relevance(A_l, g_l)
                        E_list.append(E_l)

                if len(E_list) == 0:
                    uniform = torch.ones(n_aois, device=soft_prompt.device, dtype=soft_prompt.dtype) / n_aois
                    model_attention = soft_prompt.sum() * 0.0 + uniform
                else:
                    R = self.attnlrp_propagation(E_list)
                    src_pos = max(0, min(int(src_pos), R.shape[0] - 1))
                    relevance = R[src_pos, :]

                    visual_indices = self.find_visual_token_indices(meta["orig_input_ids"], meta["soft_len"])

                    if len(visual_indices) == 0:
                        model_attention = torch.ones(n_aois, device=DEVICE, dtype=soft_prompt.dtype) / n_aois
                    else:
                        if self.model_type == "internvl":
                            orig_inputs = meta["orig_inputs"]
                            ar = orig_inputs.get("_internvl_aspect_ratio")
                            if ar is not None:
                                content_n = self._get_internvl_content_token_count(ar)
                                if len(visual_indices) > content_n:
                                    visual_indices = visual_indices[:content_n]

                        visual_relevance = relevance[visual_indices].abs()
                        patch_grid_size = self._infer_patch_grid_size(meta, visual_indices)
                        image_size = (image.width, image.height)

                        model_attention = self.aggregate_patch_saliency_to_aois_torch(
                            patch_saliency=visual_relevance,
                            image_size=image_size,
                            patch_grid_size=patch_grid_size,
                            aoi_bboxes=aoi_bboxes,
                        )

            lambda_attn = self.config.lambda_attn_target

            sample_loss, loss_dict = self.compute_losses(
                model_attention=model_attention,
                gaze_dist=gaze_dist_i,
                user_ids=[user_ids[i]],
                lambda_attn=lambda_attn,
                beta_reg=self.config.beta_reg,
                loss_choice=choice_loss,
            )
            all_loss_dicts.append(loss_dict)
            scaled_sample_loss = sample_loss / batch_size / accumulation_steps
            scaled_sample_loss.backward()

            del outputs, logits, attentions, inputs, meta, sample_loss, scaled_sample_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if do_optimizer_step:
            optimizer.step()
            self.global_step += 1

        avg_dict = {}
        for k in all_loss_dicts[0]:
            avg_dict[k] = float(np.mean([d[k] for d in all_loss_dicts]))
        avg_dict["lambda_attn"] = float(lambda_attn)
        return avg_dict

    def save_checkpoint(self, path: str):
        checkpoint = {'prompt_basis': self.prompt_basis.state_dict(), 'global_step': self.global_step}
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        self.prompt_basis.load_state_dict(checkpoint['prompt_basis'], strict=False)
        self.global_step = checkpoint.get('global_step', 0)



def compute_sample_metrics(
    model_attn: np.ndarray,
    gaze_dist: np.ndarray,
    clicked_idx: int,
) -> Dict[str, float]:
    """Per-sample alignment metrics for variable-length AOIs."""
    eps = 1e-8
    n = len(model_attn)

    s_p = float(model_attn.sum())
    s_q = float(gaze_dist.sum())
    p = model_attn / s_p if s_p > eps else np.ones(n) / n
    q = gaze_dist / s_q if s_q > eps else np.ones(n) / n

    js = float(jensenshannon(q, p) ** 2)
    if np.isnan(js):
        js = 1.0
    kl = float(np.sum(q * np.log((q + eps) / (p + eps))))
    cos = float(np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q) + eps))

    model_ranked = np.argsort(p)[::-1]
    gaze_ranked = np.argsort(q)[::-1]

    click1 = 1.0 if (len(model_ranked) > 0 and model_ranked[0] == clicked_idx) else 0.0
    click3 = 1.0 if clicked_idx in model_ranked[:3] else 0.0
    click5 = 1.0 if clicked_idx in model_ranked[:5] else 0.0

    k1 = min(1, n)
    k3 = min(3, n)
    k5 = min(5, n)
    gaze1 = float(len(set(model_ranked[:k1]) & set(gaze_ranked[:k1]))) / k1 if k1 > 0 else 0
    gaze3 = float(len(set(model_ranked[:k3]) & set(gaze_ranked[:k3]))) / k3 if k3 > 0 else 0
    gaze5 = float(len(set(model_ranked[:k5]) & set(gaze_ranked[:k5]))) / k5 if k5 > 0 else 0

    return {
        "js_div": js,
        "kl_div": kl,
        "cosine_sim": cos,
        "attn_logloss": float(-np.log(np.clip(p[clicked_idx], eps, 1.0))),
        "attn_auc": float((np.sum(p[np.arange(n) != clicked_idx] < p[clicked_idx]) +
                           0.5 * np.sum(p[np.arange(n) != clicked_idx] == p[clicked_idx])) / max(1, n - 1)),
        "topk_js_div": float(
            jensenshannon(
                np.append(q[np.argsort(q)[::-1][:min(3, n)]], max(1.0 - q[np.argsort(q)[::-1][:min(3, n)]].sum(), 0.0)),
                np.append(p[np.argsort(q)[::-1][:min(3, n)]], max(1.0 - p[np.argsort(q)[::-1][:min(3, n)]].sum(), 0.0)),
            ) ** 2
        ) if n > 1 else 0.0,
        "click@1": click1,
        "click@3": click3,
        "click@5": click5,
        "gaze@1": gaze1,
        "gaze@3": gaze3,
        "gaze@5": gaze5,
        "recall@3": click3,
        "recall@5": click5,
    }


def evaluate_on_samples(
    trainer: AttnLRPTrainerAdSERP,
    samples: List[Dict],
    logger: Optional[logging.Logger] = None,
    free_generation: bool = False,
) -> Dict[str, float]:
    """Evaluate on a sample list (teacher forcing or free generation + AttnLRP)."""
    all_metrics = []

    n_correct = 0
    for idx, s in enumerate(samples):
        image = Image.open(s["image_path"]).convert("RGB")
        query = s["query"]
        user_id = s["user_id"]
        gaze_dist = gaze_distribution_from_dwell(s["gaze_dwell_ms"])
        clicked_idx = s["clicked_aoi_idx"]
        aoi_bboxes = s["aoi_bboxes"]
        n_aois = s["n_aois"]
        answer_text = None if free_generation else str(clicked_idx)

        soft_prompt = trainer.prompt_basis([user_id])[0]
        inputs, meta = trainer.build_inputs_with_soft_prompt(
            image=image, query=query,
            soft_prompt_embeds=soft_prompt, answer_text=answer_text,
        )

        with torch.enable_grad():
            outputs = trainer._model_forward(
                inputs, output_attentions=True, use_cache=False, return_dict=True,
            )
            logits = outputs.logits
            attentions = outputs.attentions

            ans_pos = meta["answer_pos_combined"]
            ans_ids = meta["answer_token_ids"]
            if ans_pos and ans_ids and len(ans_pos) > 0:
                target_scalar = None
                src_pos = logits.shape[1] - 1
                for p, tid in zip(ans_pos, ans_ids):
                    prev = p - 1
                    if prev >= 0:
                        v = logits[0, prev, tid]
                        target_scalar = v if target_scalar is None else (target_scalar + v)
                        src_pos = prev
                if target_scalar is None:
                    target_scalar = logits[0, -1].max()
            else:
                target_scalar = logits[0, -1].max()
                src_pos = logits.shape[1] - 1

            E_list = []
            L = len(attentions)
            max_l = trainer.config.attnlrp_max_layers
            layer_ids = list(range(max(0, L - int(max_l)), L)) if max_l and int(max_l) < L else list(range(L))

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
                E_l = trainer.attnlrp_layer_relevance(attn[0].float(), g_attn[0].float())
                E_list.append(E_l)

        if len(E_list) == 0:
            model_attn = np.ones(n_aois, dtype=np.float32) / n_aois
        else:
            R = trainer.attnlrp_propagation(E_list)
            src_pos = max(0, min(src_pos, R.shape[0] - 1))
            relevance = R[src_pos, :]

            visual_indices = trainer.find_visual_token_indices(meta["orig_input_ids"], meta["soft_len"])
            if len(visual_indices) == 0:
                model_attn = np.ones(n_aois, dtype=np.float32) / n_aois
            else:
                if trainer.model_type == "internvl":
                    ar = meta["orig_inputs"].get("_internvl_aspect_ratio")
                    if ar is not None:
                        content_n = trainer._get_internvl_content_token_count(ar)
                        if len(visual_indices) > content_n:
                            visual_indices = visual_indices[:content_n]

                visual_relevance = relevance[visual_indices].abs()
                patch_grid_size = trainer._infer_patch_grid_size(meta, visual_indices)

                model_attn_t = trainer.aggregate_patch_saliency_to_aois_torch(
                    patch_saliency=visual_relevance,
                    image_size=(image.width, image.height),
                    patch_grid_size=patch_grid_size,
                    aoi_bboxes=aoi_bboxes,
                ).detach().float().cpu().numpy()
                s = float(model_attn_t.sum())
                model_attn = model_attn_t / s if s > 1e-9 else np.ones(n_aois, dtype=np.float32) / n_aois

        del outputs, logits, attentions, inputs, meta, E_list, soft_prompt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        m = compute_sample_metrics(model_attn, gaze_dist, clicked_idx)
        all_metrics.append(m)
        pred_idx = int(np.argmax(model_attn))
        if pred_idx == int(clicked_idx):
            n_correct += 1

    if not all_metrics:
        return {}

    avg = {}
    for k in all_metrics[0]:
        vals = [m[k] for m in all_metrics if np.isfinite(m[k])]
        avg[f"micro_{k}"] = float(np.mean(vals)) if vals else float("nan")
    avg["answer_accuracy"] = float(n_correct / max(1, len(samples)))
    return avg


def _log_four_way_comparison(
    _logger: logging.Logger,
    _bl_tf: Dict[str, float],
    _bl_fg: Dict[str, float],
    _tr_tf: Dict[str, float],
    _tr_fg: Dict[str, float],
) -> None:
    return


def _aggregate_float_dicts(dicts: List[Dict[str, Any]]) -> Dict[str, float]:
    """Arithmetic mean over dicts for keys whose values are finite int/float."""
    if not dicts:
        return {}
    all_keys: set = set()
    for d in dicts:
        all_keys.update(d.keys())
    out: Dict[str, float] = {}
    for k in sorted(all_keys):
        vals = []
        for d in dicts:
            if k not in d:
                continue
            v = d[k]
            if isinstance(v, (int, float)) and np.isfinite(v):
                vals.append(float(v))
        if vals:
            out[k] = float(np.mean(vals))
    return out


def _log_multi_seed_summary(logger: logging.Logger, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Average validation-style metrics across MULTI_SEEDS runs; save JSON and print summary."""
    n = len(results)
    if n == 0:
        return {}

    scores = [float(r["final_score"]) for r in results if np.isfinite(r.get("final_score", float("nan")))]
    avg_final_score = float(np.mean(scores)) if scores else float("nan")

    avg_final_metrics = _aggregate_float_dicts([r.get("final_metrics") or {} for r in results])
    avg_bl_tf = _aggregate_float_dicts([r.get("bl_tf") or {} for r in results])
    avg_bl_fg = _aggregate_float_dicts([r.get("bl_fg") or {} for r in results])
    avg_tr_tf = _aggregate_float_dicts([r.get("tr_tf") or {} for r in results])
    avg_tr_fg = _aggregate_float_dicts([r.get("tr_fg") or {} for r in results])

    primary_tf, primary_tf_detail = compute_primary_score(avg_tr_tf)
    primary_fg, primary_fg_detail = compute_primary_score(avg_tr_fg)

    payload: Dict[str, Any] = {
        "n_seeds": n,
        "multi_seed_avg": {
            "final_score_val_best_js": avg_final_score,
            "final_metrics": avg_final_metrics,
            "bl_tf": avg_bl_tf,
            "bl_fg": avg_bl_fg,
            "tr_tf": avg_tr_tf,
            "tr_fg": avg_tr_fg,
            "primary_score_tr_tf": primary_tf,
            "primary_score_tr_tf_detail": primary_tf_detail,
            "primary_score_tr_fg": primary_fg,
            "primary_score_tr_fg_detail": primary_fg_detail,
        },
        "per_seed": results,
    }

    out_path = os.path.join(OUTPUT_DIR, f"adserp_attnlrp_multi_seed_{RUN_TIMESTAMP}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    msg = (
        f"[AdSERP multi-seed] n={n} | avg final_score (best val JS): {avg_final_score:.6f} | "
        f"primary tr_tf: {primary_tf:.6f} ({primary_tf_detail}) | "
        f"primary tr_fg: {primary_fg:.6f} ({primary_fg_detail}) | "
        f"wrote {out_path}"
    )
    logger.info(msg)
    print(msg)

    return payload



def split_train_test(
    samples: List[Dict],
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Per-user holdout: one test sample per user when n>=2; singleton users stay in train."""
    rng = np.random.RandomState(seed)
    by_user = defaultdict(list)
    for i, s in enumerate(samples):
        by_user[s["user_id"]].append(i)

    train_idx, test_idx = [], []
    for uid, idxs in by_user.items():
        arr = np.array(idxs)
        rng.shuffle(arr)
        if len(arr) == 1:
            train_idx.append(int(arr[0]))
        else:
            test_idx.append(int(arr[0]))
            train_idx.extend([int(x) for x in arr[1:]])

    return train_idx, test_idx


def build_kfold_indices(train_indices: List[int], k_folds: int, seed: int) -> List[Tuple[List[int], List[int]]]:
    """K-fold split over train indices; returns list of (train_idx, val_idx)."""
    rng = np.random.RandomState(seed)
    idx = np.array(train_indices, dtype=np.int64)
    rng.shuffle(idx)
    fold_sizes = [len(idx) // k_folds] * k_folds
    for i in range(len(idx) % k_folds):
        fold_sizes[i] += 1

    folds = []
    start = 0
    for fs in fold_sizes:
        val = idx[start:start + fs].tolist()
        train = np.concatenate([idx[:start], idx[start + fs:]]).tolist() if fs < len(idx) else []
        folds.append((train, val))
        start += fs
    return folds


def apply_hyperparams_to_trainer(trainer: "AttnLRPTrainerAdSERP", hp: Dict[str, Any]) -> None:
    """Apply grid-search hyperparameters to trainer.config."""
    cfg = trainer.config
    cfg.lambda_attn_target = hp.get("lambda_attn_target", cfg.lambda_attn_target)
    cfg.beta_reg = hp.get("beta_reg", cfg.beta_reg)
    cfg.power_gamma = hp.get("power_gamma", cfg.power_gamma)



def setup_logger(log_name: str = "adserp_attnlrp") -> logging.Logger:
    lg = logging.getLogger(log_name)
    lg.handlers.clear()
    lg.addHandler(logging.NullHandler())
    lg.propagate = False
    return lg



def train_epoch(
    trainer: AttnLRPTrainerAdSERP,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    logger: logging.Logger,
) -> Dict[str, float]:
    epoch_losses = defaultdict(list)
    n_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        ga_step = (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == n_batches
        loss_dict = trainer.train_step(
            batch, optimizer,
            accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            zero_grad=(batch_idx % GRADIENT_ACCUMULATION_STEPS == 0),
            do_optimizer_step=ga_step,
        )
        for k, v in loss_dict.items():
            epoch_losses[k].append(v)

    return {k: float(np.mean(v)) for k, v in epoch_losses.items()}


def run_training(
    trainer: AttnLRPTrainerAdSERP,
    train_samples: List[Dict],
    eval_samples: List[Dict],
    num_epochs: int,
    logger: logging.Logger,
    hyperparams: Optional[Dict[str, Any]] = None,
    save_best: bool = True,
    eval_name: str = "test",
):
    if hyperparams:
        apply_hyperparams_to_trainer(trainer, hyperparams)

    train_dataset = AdSERPDataset(train_samples)
    train_batch_size = BATCH_SIZE
    if trainer.model_type == "internvl":
        train_batch_size = 1
    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        collate_fn=adserp_collate_fn, num_workers=0,
    )

    basis_lr = hyperparams.get("basis_lr", 1e-3) if hyperparams else 1e-3
    alpha_lr = hyperparams.get("alpha_lr", 1e-3) if hyperparams else 1e-3
    weight_decay = hyperparams.get("weight_decay", 1e-3) if hyperparams else 1e-3

    param_groups = [
        {"params": [trainer.prompt_basis.basis], "lr": basis_lr, "weight_decay": weight_decay},
    ]
    if USE_USER_ALPHA:
        param_groups.append(
            {"params": list(trainer.prompt_basis.user_alphas.values()), "lr": alpha_lr, "weight_decay": 0},
        )
    optimizer = torch.optim.AdamW(param_groups)

    best_score = float("inf")
    best_epoch = -1
    best_metrics = {}

    for epoch in range(1, num_epochs + 1):
        if USE_USER_ALPHA:
            new_params = []
            for uid in set(s["user_id"] for s in train_samples):
                alpha = trainer.prompt_basis.get_or_create_user_alpha(uid)
                found = any(
                    alpha.data_ptr() == p.data_ptr()
                    for pg in optimizer.param_groups
                    for p in pg["params"]
                )
                if not found:
                    new_params.append(alpha)
            if new_params:
                optimizer.add_param_group({"params": new_params, "lr": alpha_lr, "weight_decay": 0})

        train_epoch(trainer, train_loader, optimizer, epoch, logger)

        if epoch % EVAL_EVERY_N_EPOCHS == 0 or epoch == num_epochs:
            eval_metrics = evaluate_on_samples(trainer, eval_samples, logger)

            js = eval_metrics.get("micro_js_div", float("inf"))
            if js < best_score:
                best_score = js
                best_epoch = epoch
                best_metrics = dict(eval_metrics)
                if save_best:
                    ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_epoch{epoch}.pt")
                    trainer.save_checkpoint(ckpt_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_score, best_metrics


def grid_search_cv(
    all_samples: List[Dict],
    train_idx: List[int],
    logger: logging.Logger,
    cv_fold_seed: int = SEED,
) -> Dict[str, Any]:
    """K-fold CV on train indices; select hyperparams by mean val micro_js_div (lower is better)."""
    keys = sorted(PARAM_GRID.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*[PARAM_GRID[k] for k in keys])]
    folds = build_kfold_indices(train_idx, K_FOLDS, cv_fold_seed)

    best_hp = None
    best_score = float("inf")

    for hp in combos:
        fold_scores = []
        for fi, (tr_idx, va_idx) in enumerate(folds, start=1):
            fold_train = [all_samples[i] for i in tr_idx]
            fold_val = [all_samples[i] for i in va_idx]

            trainer = AttnLRPTrainerAdSERP(model_name=MODEL_NAME, model_type=MODEL_TYPE, logger=logger)

            score, _ = run_training(
                trainer=trainer,
                train_samples=fold_train,
                eval_samples=fold_val,
                num_epochs=NUM_EPOCHS_CV,
                logger=logger,
                hyperparams=hp,
                save_best=False,
                eval_name=f"val-fold{fi}",
            )
            fold_scores.append(score)

        avg_score = float(np.mean(fold_scores))
        if avg_score < best_score:
            best_score = avg_score
            best_hp = hp

    return best_hp if best_hp is not None else {}



def main():
    logger = setup_logger()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    seed_list = list(MULTI_SEEDS)

    samples = load_adserp_data(ADSERP_SAMPLES_JSONL, ADSERP_IMAGES_DIR)

    selected_hp_shared: Dict[str, Any] = {}
    multi_seed_final = len(seed_list) > 1

    all_seed_results: List[Dict[str, Any]] = []

    for run_seed in seed_list:
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)

        train_idx, test_idx = split_train_test(samples, seed=run_seed)
        train_samples = [samples[i] for i in train_idx]
        test_samples = [samples[i] for i in test_idx]

        if not selected_hp_shared:
            selected_hp_shared = grid_search_cv(
                samples, train_idx, logger, cv_fold_seed=run_seed
            )
        selected_hp = dict(selected_hp_shared)

        trainer = AttnLRPTrainerAdSERP(model_name=MODEL_NAME, model_type=MODEL_TYPE, logger=logger)

        final_score, final_metrics = run_training(
            trainer=trainer,
            train_samples=train_samples,
            eval_samples=test_samples,
            num_epochs=NUM_EPOCHS_FINAL,
            logger=logger,
            hyperparams=selected_hp,
            save_best=True,
            eval_name="test",
        )
        apply_hyperparams_to_trainer(trainer, selected_hp)
        tr_tf = evaluate_on_samples(trainer, test_samples, logger, free_generation=False)
        tr_fg = evaluate_on_samples(trainer, test_samples, logger, free_generation=True)

        if torch.cuda.is_available():
            trainer.model.to("cpu")
            torch.cuda.empty_cache()

        baseline_trainer = AttnLRPTrainerAdSERP(model_name=MODEL_NAME, model_type=MODEL_TYPE, logger=logger)
        apply_hyperparams_to_trainer(baseline_trainer, selected_hp)
        bl_tf = evaluate_on_samples(baseline_trainer, test_samples, logger, free_generation=False)
        bl_fg = evaluate_on_samples(baseline_trainer, test_samples, logger, free_generation=True)

        if not multi_seed_final:
            _log_four_way_comparison(logger, bl_tf, bl_fg, tr_tf, tr_fg)

        all_seed_results.append(
            {
                "seed": run_seed,
                "final_score": final_score,
                "final_metrics": final_metrics,
                "bl_tf": bl_tf,
                "bl_fg": bl_fg,
                "tr_tf": tr_tf,
                "tr_fg": tr_fg,
            }
        )

        del trainer, baseline_trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if multi_seed_final:
        _log_multi_seed_summary(logger, all_seed_results)


if __name__ == "__main__":
    main()
