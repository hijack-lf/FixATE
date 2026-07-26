import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_TYPE = "qwen3vl"
QWEN3VL_MODEL_PATH = os.path.join(_ROOT, "llm_models", "Qwen3-VL-4B-Instruct")
INTERNVL_MODEL_PATH = os.path.join(_ROOT, "llm_models", "InternVL3_5-4B-Instruct")
MODEL_NAME = QWEN3VL_MODEL_PATH if MODEL_TYPE == "qwen3vl" else INTERNVL_MODEL_PATH

SLOT_KL_SIGNAL = "fixation"
SIGNAL_INTERSECT = False

ADSERP_DATASET_DIR = os.path.join(_ROOT, "datasets", "Adserp")
ADSERP_SAMPLES_JSONL = os.path.join(ADSERP_DATASET_DIR, "samples.jsonl")
ADSERP_IMAGES_DIR = os.path.join(ADSERP_DATASET_DIR, "images")

ADSERP_TEST_RATIO = 0.20
ADSERP_VAL_RATIO = 0.15

OUTPUT_DIR = os.path.join(_ROOT, "outputs", "adserp")
CHECKPOINT_DIR = os.path.join(_ROOT, "checkpoints", "adserp")
EVAL_ONLY = False
EVAL_CKPT_DIR = ""
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 922
MAX_AOIS = 25

K_FOLDS = 5
SEED = 42
USE_MULTI_SEED = True
MULTI_SEEDS = [42, 123, 456, 789, 101112]

NUM_EPOCHS_FINAL = 30
NUM_EPOCHS_CV = 6
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
USE_BATCHED_FORWARD = False
EVAL_EVERY_N_EPOCHS = 2

RUN_GRID_SEARCH = False
PARAM_GRID = {
    "attn_loss_mode": ["plain_kl", "weighted_full_kl"],
    "lambda_attn_target": [0.3, 0.5],
    "choice_loss_weight": [0.2, 0.3],
    "attnlrp_max_layers": [4, 6],
    "basis_lr": [5e-4, 1e-3],
    "alpha_lr": [1e-3],
    "weight_decay": [1e-3],
    "beta_reg": [1e-3],
    "topk_k": [3],
    "power_gamma": [1.5],
}

TRAIN_CHOICE = True
CHOICE_LOSS_WEIGHT = 0.5
LAMBDA_ATTN_TARGET = 0.5
LAMBDA_WARMUP_STEPS = 20

USE_USER_ALPHA = True
USE_RANDOM_SOFT_PROMPT = False

ATTNLRP_CREATE_GRAPH = True
ATTNLRP_MAX_LAYERS = 6
ATTNLRP_GRAD_SCALE = 0.1
INTERNVL_ATTNLRP_MAX_LAYERS = 3

ATTN_METHOD = "attnlrp"
GLIMPSE_HEAD_TEMP = 0.5
GLIMPSE_DEPTH_TEMP = 0.2
ROLLOUT_DISCARD_RATIO = 0.0
USE_FG_FOR_BEST_EPOCH = False

NUM_BASIS = 8
NUM_SOFT_TOKENS = 16

BETA_REG = 1e-3
GAZE_TAU = 1.0

BASIS_CLIP_GRAD_NORM = 100.0
ALPHA_CLIP_GRAD_NORM = 5.0

ATTN_LOSS_MODE = "weighted_full_kl"
TOPK_K = 3
POWER_GAMMA = 2
POWER_EPS = 1e-8

INSTRUCTION = (
    "You are a search behavior simulator. A user is viewing a Google search results page (SERP). "
    "Based on the search query and the visual layout of the results, predict which result the user "
    "will click. Each clickable area is numbered starting from 0. Output only the number of the "
    "clicked area."
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainerConfig:
    beta_reg: float = BETA_REG
    lambda_attn_target: float = LAMBDA_ATTN_TARGET
    attnlrp_max_layers: int = ATTNLRP_MAX_LAYERS
    choice_loss_weight: float = CHOICE_LOSS_WEIGHT
    attn_loss_mode: str = ATTN_LOSS_MODE
    topk_k: int = TOPK_K
    power_gamma: float = POWER_GAMMA
    power_eps: float = POWER_EPS
    lambda_warmup_steps: int = LAMBDA_WARMUP_STEPS
    train_choice: bool = TRAIN_CHOICE
    use_user_alpha: bool = USE_USER_ALPHA
    use_random_soft_prompt: bool = USE_RANDOM_SOFT_PROMPT
    attnlrp_create_graph: bool = ATTNLRP_CREATE_GRAPH
    attnlrp_grad_scale: float = ATTNLRP_GRAD_SCALE
    basis_clip_grad_norm: float = BASIS_CLIP_GRAD_NORM
    alpha_clip_grad_norm: float = ALPHA_CLIP_GRAD_NORM
    last_n_layers: Optional[int] = None
