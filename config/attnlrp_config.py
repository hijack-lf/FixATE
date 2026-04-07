"""AttnLRP-specific config."""
from common_config import *
from dataclasses import dataclass

MODEL_TYPE = "internvl"
MODEL_NAME = INTERNVL_MODEL_PATH if MODEL_TYPE == "internvl" else QWEN3VL_MODEL_PATH

# AttnLRP-specific
ATTNLRP_CREATE_GRAPH = True
ATTNLRP_GRAD_SCALE = 0.1
RUN_GRID_SEARCH = True

@dataclass
class TrainerConfig:
    beta_reg: float = BETA_REG
    lambda_attn_weight: float = LAMBDA_ATTN_WEIGHT
    power_gamma: float = POWER_GAMMA; power_eps: float = POWER_EPS
    train_choice: bool = TRAIN_CHOICE
    use_user_alpha: bool = USE_USER_ALPHA; attnlrp_create_graph: bool = ATTNLRP_CREATE_GRAPH
    attnlrp_grad_scale: float = ATTNLRP_GRAD_SCALE

DEFAULT_HYPERPARAMS = {
    "LAMBDA_ATTN_WEIGHT": LAMBDA_ATTN_WEIGHT,
    "POWER_GAMMA": POWER_GAMMA, "BETA_REG": BETA_REG,
    "NUM_BASIS": NUM_BASIS, "NUM_SOFT_TOKENS": NUM_SOFT_TOKENS,
    "BASIS_LR": 0.001, "ALPHA_LR": 0.001, "WEIGHT_DECAY": 0.001,
}
PARAM_GRID = {"BASIS_LR": [5e-4, 1e-3], "ALPHA_LR": [3e-3, 5e-3],
              "BETA_REG": [5e-4, 1e-3], "LAMBDA_ATTN_WEIGHT": [0.2, 0.3, 0.5]}
HP_TO_CFG = {"BETA_REG": "beta_reg", "LAMBDA_ATTN_WEIGHT": "lambda_attn_weight",
             "POWER_GAMMA": "power_gamma"}