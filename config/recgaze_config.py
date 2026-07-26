import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_TYPE = "internvl"
QWEN3VL_MODEL_PATH = os.path.join(_ROOT, "llm_models", "Qwen3-VL-4B-Instruct")
INTERNVL_MODEL_PATH = os.path.join(_ROOT, "llm_models", "InternVL3_5-4B-Instruct")
MODEL_NAME = QWEN3VL_MODEL_PATH if MODEL_TYPE == "qwen3vl" else INTERNVL_MODEL_PATH

SLOT_KL_SIGNAL = "fixation"
SIGNAL_INTERSECT = False
USE_HISTORY = False

_RECGAZE_DIR = os.path.join(_ROOT, "datasets", "RecGaze")
FIXATE_JSONL = os.path.join(_RECGAZE_DIR, "fixate_dataset.jsonl")
INTERFACE_IMAGES_DIR = os.path.join(_RECGAZE_DIR, "page_divide_real", "image_index")
POSTER_IMAGES_DIR = os.path.join(_RECGAZE_DIR, "page_divide_real", "image")
USER_FEATURES_CSV = os.path.join(_RECGAZE_DIR, "raw", "user_features.csv")
IMAGE_MAX_W = 1100

ATTN_METHOD = "attnlrp"
GLIMPSE_HEAD_TEMP = 0.5
GLIMPSE_DEPTH_TEMP = 0.2
ROLLOUT_DISCARD_RATIO = 0.0

USE_INDEXED_IMAGES = True

INSTRUCTION = (
    "You are a sophisticated user behavior emulator. Given a user profile, simulate the user's final "
    "selection on this movie selection interface. Your final answer should only output exactly one "
    "uppercase English letter shown below the selected poster; do not output any other content."
)

OUTPUT_DIR = os.path.join(_ROOT, "outputs", "recgaze")
CHECKPOINT_DIR = os.path.join(_ROOT, "checkpoints", "recgaze")
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

K_FOLDS = 5
SEED = 42
USE_MULTI_SEED = True
MULTI_SEEDS = [42, 123, 456, 789, 2024]

NUM_EPOCHS_CV = 8
NUM_EPOCHS_FINAL = 30
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
USE_BATCHED_FORWARD = True
EVAL_EVERY_N_EPOCHS = 2
NO_CV_VAL_RATIO = 0.15
USE_FG_FOR_BEST_EPOCH = False

RUN_GRID_SEARCH = False
PARAM_GRID = {
    "BASIS_LR": [5e-4, 1e-3],
    "ALPHA_LR": [3e-3, 5e-3],
    "BETA_REG": [5e-4, 1e-3],
    "LAMBDA_ATTN_TARGET": [0.2, 0.3, 0.5],
    "ATTNLRP_MAX_LAYERS": [6],
}

PRIMARY_METRIC = "composite_align_v1"
PRIMARY_LOWER_IS_BETTER = False
PRIMARY_METRIC_POSITIVE = {
    "micro_cosine_sim": 0.10,
    "micro_click@1": 0.20,
    "micro_click@3": 0.25,
    "micro_gaze@3": 0.20,
    "answer_accuracy": 0.25,
}
PRIMARY_METRIC_PENALTY = {
    "micro_topk_js_div": 0.10,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_CHOICE = True
CHOICE_LOSS_WEIGHT = 0.5
LAMBDA_ATTN_TARGET = 0.5
LAMBDA_WARMUP_STEPS = 30

USE_USER_ALPHA = True
USE_RANDOM_SOFT_PROMPT = False

LAST_N_LAYERS = None

ATTNLRP_CREATE_GRAPH = True
ATTNLRP_DETACH_GRAD = False
ATTNLRP_RELU_RULE = False
ATTNLRP_MAX_LAYERS = 6
ATTNLRP_GRAD_SCALE = 0.1
INTERNVL_ATTNLRP_MAX_LAYERS = 5

NUM_BASIS = 8
NUM_SOFT_TOKENS = 16

BETA_REG = 1e-3
GAZE_TAU = 1

BASIS_CLIP_GRAD_NORM = 100.0
ALPHA_CLIP_GRAD_NORM = 5.0

ATTN_LOSS_MODE = "weighted_full_kl"
TOPK_K = 5
POWER_GAMMA = 2
POWER_EPS = 1e-8

NUM_ROWS = 3
NUM_COLS = 5
NUM_SLOTS = NUM_ROWS * NUM_COLS

UI_IMAGE_WIDTH = 1100
UI_IMAGE_HEIGHT = 600

DEFAULT_HYPERPARAMS = {
    "ATTN_LOSS_MODE": ATTN_LOSS_MODE,
    "LAMBDA_ATTN_TARGET": LAMBDA_ATTN_TARGET,
    "POWER_GAMMA": POWER_GAMMA,
    "TOPK_K": TOPK_K,
    "BETA_REG": BETA_REG,
    "ATTNLRP_MAX_LAYERS": ATTNLRP_MAX_LAYERS,
    "CHOICE_LOSS_WEIGHT": CHOICE_LOSS_WEIGHT,
    "NUM_BASIS": NUM_BASIS,
    "NUM_SOFT_TOKENS": NUM_SOFT_TOKENS,
    "BASIS_LR": 0.001,
    "ALPHA_LR": 0.001,
    "WEIGHT_DECAY": 0.001,
}


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
    last_n_layers: Optional[int] = LAST_N_LAYERS
