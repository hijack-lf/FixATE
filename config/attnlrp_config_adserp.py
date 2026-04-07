"""AdSERP AttnLRP: extends common_config with SERP paths and task-specific training overrides."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from config.common_config import (
    _ROOT,
    BETA_REG,
    INTERNVL_MODEL_PATH,
    LAMBDA_ATTN_WEIGHT,
    POWER_EPS,
    POWER_GAMMA,
    QWEN3VL_MODEL_PATH,
    TRAIN_CHOICE,
    USE_USER_ALPHA,
)

# Training bundle for fixate/fixate_training/train_fixate_attnlrp_adserp.py.
# Must match the directory passed to preprocessing/adserp/build_click_aoi_dataset.py --out
# (that script writes samples.jsonl + images/<trial_id>.jpg here).
# Prerequisite: preprocessing/adserp/build_samples.py (scroll_stops) must have been run so
# build_click_aoi_dataset can read viewport crops from SAMPLES_DIR / <trial_id> / step_XX / viewport.jpg.
_FIXATE_TRAINING = os.path.join(_ROOT, "fixate", "fixate_training")
ADSERP_SAMPLES_JSONL = os.path.join(_FIXATE_TRAINING, "samples.jsonl")
ADSERP_IMAGES_DIR = os.path.join(_FIXATE_TRAINING, "images")

OUTPUT_DIR = os.path.join(_ROOT, "outputs", "adserp_attnlrp")
CHECKPOINT_DIR = os.path.join(_ROOT, "checkpoints", "adserp_attnlrp")

MODEL_TYPE = "qwen3vl"
MODEL_NAME = QWEN3VL_MODEL_PATH if MODEL_TYPE == "qwen3vl" else INTERNVL_MODEL_PATH

VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 922
MAX_AOIS = 25

NUM_EPOCHS_FINAL = 30
NUM_EPOCHS_CV = 6
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
USE_BATCHED_FORWARD = False

USE_RANDOM_SOFT_PROMPT = False

ATTNLRP_CREATE_GRAPH = True
ATTNLRP_GRAD_SCALE = 0.1

PARAM_GRID = {
    "lambda_attn_target": [0.3, 0.5],
    "basis_lr": [5e-4, 1e-3],
    "alpha_lr": [1e-3],
    "weight_decay": [1e-3],
    "beta_reg": [BETA_REG],
    "power_gamma": [POWER_GAMMA],
}

ADSERP_INSTRUCTION = (
    "You are a search behavior simulator. A user is viewing a Google search results page (SERP). "
    "Based on the search query and the visual layout of the results, predict which result the user will click. "
    "Each clickable area is numbered starting from 0. Output only the number of the clicked area."
)


@dataclass
class TrainerConfig:
    beta_reg: float = BETA_REG
    lambda_attn_target: float = LAMBDA_ATTN_WEIGHT
    train_choice: bool = TRAIN_CHOICE
    attnlrp_max_layers: Optional[int] = None
    power_gamma: float = POWER_GAMMA
    power_eps: float = POWER_EPS
    use_user_alpha: bool = USE_USER_ALPHA
    use_random_soft_prompt: bool = USE_RANDOM_SOFT_PROMPT
    attnlrp_create_graph: bool = ATTNLRP_CREATE_GRAPH
    attnlrp_grad_scale: float = ATTNLRP_GRAD_SCALE
    last_n_layers: Optional[int] = None
