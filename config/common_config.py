"""Shared config & utilities for GLIMPSE / AttnLRP / Rollout pipelines."""
import os, math, random
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import jensenshannon

# Repo root (parent of config/); data and model paths are relative to this.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(_ROOT, "checkpoints")

# ═══════════════════════════════════════════════════════════════
#  Shared paths (relative to repo root)
# ═══════════════════════════════════════════════════════════════
QWEN3VL_MODEL_PATH  = os.path.join(_ROOT, "llm_models", "Qwen3-VL-4B-Instruct")
INTERNVL_MODEL_PATH = os.path.join(_ROOT, "llm_models", "InternVL3_5-4B-Instruct")
_RECRAZE = os.path.join(_ROOT, "datasets", "RecGaze")
GAZE_DATA_CSV       = os.path.join(_RECRAZE, "init_interface_user_gaze(swipes).csv")
POSTER_IMAGES_DIR   = os.path.join(_RECRAZE, "interface_iamge")
USER_FEATURES_CSV   = os.path.join(_RECRAZE, "user_features.csv")
ITEM_FEATURES_CSV   = os.path.join(_RECRAZE, "item_features.csv")

# ═══════════════════════════════════════════════════════════════
#  Shared prompt (letter labels under posters)
# ═══════════════════════════════════════════════════════════════
INSTRUCTION = (
    "You are a sophisticated user behavior emulator. Given a user profile, "
    "simulate the user's final selection on this movie selection interface. "
    "Your final answer should only output exactly one uppercase English letter "
    "shown below the selected poster; do not output any other content."
)

# ═══════════════════════════════════════════════════════════════
#  Shared training / split parameters
# ═══════════════════════════════════════════════════════════════
K_FOLDS = 5; SEED = 42
USE_MULTI_SEED = True; 
MULTI_SEEDS = [42, 123, 456, 789, 2024]
NUM_EPOCHS_CV = 8; 
NUM_EPOCHS_FINAL = 30
BATCH_SIZE = 4; 
GRADIENT_ACCUMULATION_STEPS = 2
USE_BATCHED_FORWARD = True; 
EVAL_EVERY_N_EPOCHS = 2
NO_CV_VAL_RATIO = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════
#  Shared soft-prompt / loss parameters
# ═══════════════════════════════════════════════════════════════
TRAIN_CHOICE = True
LAMBDA_ATTN_WEIGHT = 1
USE_USER_ALPHA = True
NUM_BASIS = 8; NUM_SOFT_TOKENS = 16; BETA_REG = 1e-3; GAZE_TAU = 1
POWER_GAMMA = 2; POWER_EPS = 1e-8

# ═══════════════════════════════════════════════════════════════
#  Shared UI / slot parameters
# ═══════════════════════════════════════════════════════════════
NUM_ROWS, NUM_COLS, NUM_SLOTS = 3, 5, 15
UI_IMAGE_WIDTH, UI_IMAGE_HEIGHT = 1100, 600

# ═══════════════════════════════════════════════════════════════
#  Primary metric (composite; see compute_primary_score)
# ═══════════════════════════════════════════════════════════════
PRIMARY_LOWER_IS_BETTER = False
# Default weights (methods may override)
PRIMARY_METRIC_POSITIVE = {
    "micro_cosine_sim": 0.10, "micro_click@1": 0.20,
    "micro_click@3": 0.25, "micro_gaze@3": 0.20, "answer_accuracy": 0.25,
}

# ═══════════════════════════════════════════════════════════════
#  Timestamp for artifact filenames (local timezone)
# ═══════════════════════════════════════════════════════════════
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# ═══════════════════════════════════════════════════════════════
#  Shared utilities
# ═══════════════════════════════════════════════════════════════

def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def compute_primary_score(metrics):
    score = 0.0
    for key, w in PRIMARY_METRIC_POSITIVE.items():
        if key == "answer_accuracy" and key not in metrics:
            continue
        v = float(metrics.get(key, float("nan")))
        if not np.isfinite(v): return float("nan"), f"missing:{key}"
        score += w * v
    return float(score), f"composite={score:.4f}"


def compute_sample_metrics(model_attn, gaze_dist, choice_0, logit_probs=None):
    eps = 1e-8
    p = model_attn / max(float(model_attn.sum()), eps)
    q = gaze_dist / max(float(gaze_dist.sum()), eps)
    js = float(jensenshannon(q, p)**2); js = 1.0 if np.isnan(js) else js
    kl = float(np.sum(q * np.log((q+eps)/(p+eps))))
    cos = float(np.dot(p,q) / (np.linalg.norm(p)*np.linalg.norm(q)+eps))
    mr, gr = np.argsort(p)[::-1], np.argsort(q)[::-1]
    lp = logit_probs if logit_probs is not None else p
    pc = float(np.clip(lp[choice_0], eps, None))
    neg = lp[np.arange(len(lp)) != choice_0]; pp = lp[choice_0]
    return {
        "js_div": js, "kl_div": kl, "cosine_sim": cos,
        "attn_logloss": -float(np.log(pc)),
        "attn_auc": float((np.sum(neg<pp)+0.5*np.sum(neg==pp))/len(neg)),
        "click@1": float(mr[0]==choice_0), "click@3": float(choice_0 in mr[:3]), "click@5": float(choice_0 in mr[:5]),
        "gaze@1": float(len(set(mr[:1])&set(gr[:1]))),
        "gaze@3": float(len(set(mr[:3])&set(gr[:3])))/3,
        "gaze@5": float(len(set(mr[:5])&set(gr[:5])))/5,
    }


def build_optimizer(trainer, hp):
    groups = [{"params": [trainer.prompt_basis.basis], "lr": hp["BASIS_LR"]}]
    if trainer.config.use_user_alpha:
        groups.append({"params": list(trainer.prompt_basis.user_alphas.values()), "lr": hp["ALPHA_LR"]})
    return torch.optim.AdamW(groups, weight_decay=hp.get("WEIGHT_DECAY", 1e-3))


def collate_fn(batch):
    return {
        "user_id": [s["user_id"] for s in batch], "task_id": [s["task_id"] for s in batch],
        "image": [s["image"] for s in batch], "profile_text": [s["profile_text"] for s in batch],
        "choice_slot": torch.stack([torch.tensor(s["choice_slot"]) for s in batch]),
        "gaze_dist": torch.stack([s["gaze_dist"] for s in batch]),
    }


def format_delta(trained, baseline):
    d = trained - baseline
    pct = f"{(d/baseline)*100:+.2f}%" if np.isfinite(baseline) and abs(baseline)>1e-12 else "NA"
    return f"{d:+.4f} ({pct})"


def apply_hyperparams(trainer, hp, hp_to_cfg):
    """Apply hyperparameters; hp_to_cfg maps HP keys to trainer.config attribute names."""
    for hk, ca in hp_to_cfg.items():
        if hk in hp: setattr(trainer.config, ca, hp[hk])


def reset_prompt_basis(trainer, hp, PromptBasisModule):
    """Re-instantiate prompt_basis from hyperparameters."""
    trainer.prompt_basis = PromptBasisModule(
        num_basis=hp.get("NUM_BASIS", NUM_BASIS),
        num_soft_tokens=hp.get("NUM_SOFT_TOKENS", NUM_SOFT_TOKENS),
        hidden_dim=trainer.prompt_basis.hidden_dim,
        use_user_alpha=trainer.config.use_user_alpha,
    ).to(dtype=next(trainer.model.parameters()).dtype)
    trainer.global_step = 0


def split_leave1_test_then_kfold(samples, K=5, seed=42):
    """Leave-one-user-out test index plus K-fold CV on remaining pool."""
    from collections import defaultdict
    rng = np.random.RandomState(seed)
    by_user = defaultdict(list)
    for i, s in enumerate(samples): by_user[s["user_id"]].append(i)
    test_set, hist, at = set(), {}, set()
    for uid, idxs in by_user.items():
        arr = np.array(idxs); rng.shuffle(arr)
        if len(arr)==1: at.add(int(arr[0])); continue
        test_set.add(int(arr[0])); hist[uid] = [int(x) for x in arr[1:]]
    fv = [set() for _ in range(K)]
    for uid, hidxs in hist.items():
        arr = np.array(hidxs); rng.shuffle(arr)
        if len(arr)==1: at.add(int(arr[0])); continue
        off = rng.randint(0, K)
        for j, ix in enumerate(arr): fv[(off+j)%K].add(int(ix))
    ha = set(); [ha.update(h) for h in hist.values()]
    pool = ha | at
    return list(test_set), [(list((pool-v)|at), list(v)) for v in fv]
