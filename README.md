# 👁️ FixATE: Fixation-Aligned Tuning for Personalized User Emulation
[![Dataset: RecGaze](https://img.shields.io/badge/Dataset-RecGaze-blue)](https://github.com/santideleon/RecGaze_Dataset)
[![Dataset: AdSERP](https://img.shields.io/badge/Dataset-AdSERP-blue)](https://github.com/kayhan-latifzadeh/AdSERP)

---

This repository contains the official implementation of the paper:

> **Through Their Eyes: Fixation-aligned Tuning for Personalized User Emulation**
>
> *ACM Multimedia 2026 — Brave New Ideas Track*

We propose **FixATE**, a framework that aligns a frozen VLM's visual attention with each user's characteristic gaze pattern through interpretability-based probing and personalized soft prompt tuning, enabling more faithful user simulation in visual recommendation scenarios.

<p align="center">
  <img src="assets/framework.png" width="90%" alt="FixATE Framework Overview">
</p>

## 📚 Contents

- [Overview](#overview)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)

## 🔍 Overview

Existing LLM-based user simulators perceive recommendations through text or structured metadata, missing the visual attention signals that drive real user behavior. **FixATE** bridges this gap by:

1. **Probing** the VLM's internal visual attention via interpretability operators (Attention Rollout, GLIMPSE, AttnLRP) to obtain slot-level relevance distributions comparable with human fixation.
2. **Learning personalized soft prompts** through a factorized basis decomposition, steering the model's attention toward each user's characteristic fixation pattern.

<p align="center">
  <img src="assets/motivation.png" width="70%" alt="Motivation: Perceptual gap between text-based and visual interfaces">
</p>


### Dependencies

- PyTorch >= 2.1
- Transformers >= 4.40
- Two supported VLM backbones:
  - [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
  - [InternVL3.5-4B-Instruct](https://huggingface.co/OpenGVLab/InternVL3_5-4B-Instruct)

## 📦 Data Preparation

### 1. Download Datasets

We use two public eye-tracking datasets:

| Dataset | Domain | Layout | Source |
|---------|--------|--------|--------|
| **RecGaze** | Movie recommendation | 3×5 carousel grid | [GitHub](https://github.com/santideleon/RecGaze_Dataset) |
| **AdSERP** | Sponsored search | Vertical SERP | [GitHub](https://github.com/kayhan-latifzadeh/AdSERP) |

Download the datasets and place them under `data/`:

```
datasets/
├── recgaze/
└── adserp/
```

### 2. RecGaze: preprocess swipe / gaze trials

Filter `summary_feedback.csv` into a processed CSV (valid fixation rows, last click on Movie, etc.) for downstream use:

```bash
python preprocessing/recgaze/dataset_preprocess_swipes.py \
  --summary datasets/RecGaze/summary_feedback.csv \
  --out "datasets/RecGaze/init_interface_user_gaze(swipes).csv"
```

Defaults use the same paths under `datasets/RecGaze/`; use `--task-min` / `--task-max` to change the TaskID range (default 1–35).

### 3. RecGaze: generate interface images

Place raw CSVs under `datasets/raw/RecGaze/` (`summary_feedback.csv`, `item_features.csv`; posters go to `poster_cache/` as needed). The script replays swipe events, updates carousel state, and renders one final PNG per user–task:

```bash
python preprocessing/recgaze/generate_interface_iamge.py 1-35
# 或：python preprocessing/recgaze/generate_interface_iamge.py all
```

For a single user–task pair, pass both `--only-user` and `--only-task`.

### 4. AdSERP: build gaze samples (optional)

Raw AdSERP data defaults to `datasets/Adserp/data` (override with `ADSERP_DATA_DIR`). Build full-page and/or scroll-stop samples:

```bash
python preprocessing/adserp/build_samples.py --mode both --n 5
```

Optional click-AOI viewport subset (expects `scroll_stops` crops from `build_samples`; default output dir is set in `ADSERP_CLICK_AOI_OUT`):

```bash
python preprocessing/adserp/build_click_aoi_dataset.py --split all
```

Typical outputs:

```
datasets/RecGaze/
├── init_interface_user_gaze(swipes).csv   # Step 2: filtered rows
└── interface_iamge/                       # Step 3: PNGs (e.g. User_*_TaskID_*_final_interface.png)
datasets/raw/RecGaze/                      # Raw CSVs + poster_cache for step 3

datasets/Adserp/samples/
├── full_page/samples.jsonl
└── scroll_stops/samples.jsonl
```


## 🔧 Training

Training is implemented as scripts under `fixate/fixate_training/` with hyperparameters in `config/` (there is no top-level `train.py` CLI). Run commands from the **repository root** so imports like `config.*` resolve.

### Prerequisites

- Download VLM weights and point to them in `config/common_config.py` (`QWEN3VL_MODEL_PATH`, `INTERNVL_MODEL_PATH`, defaulting to `llm_models/Qwen3-VL-4B-Instruct` and `llm_models/InternVL3_5-4B-Instruct`).
- **RecGaze:** gaze CSV, posters, and metadata under `datasets/RecGaze/` (see `GAZE_DATA_CSV`, `POSTER_IMAGES_DIR`, `USER_FEATURES_CSV`, `ITEM_FEATURES_CSV` in `common_config.py`).
- **AdSERP:** after preprocessing, ensure `samples.jsonl` and `images/` match `ADSERP_SAMPLES_JSONL` / `ADSERP_IMAGES_DIR` in `config/attnlrp_config_adserp.py` (default: `fixate/fixate_training/`).

### RecGaze

| Probing operator | Script | Primary config |
|------------------|--------|----------------|
| AttnLRP | `python fixate/fixate_training/train_fixate_attnlrp.py` | `config/attnlrp_config.py` (default `MODEL_TYPE`: `internvl`) |
| GLIMPSE | `python fixate/fixate_training/train_fixate_glimpse.py` | `config/glimpse_config.py` (default `internvl`) |
| Attention Rollout | `python fixate/fixate_training/train_fixate_rollout.py` | `config/rollout_config.py` (default `qwen3vl`) |

Shared training defaults (batch size, epochs, soft-prompt size, loss weights, splits) live in `config/common_config.py`; each operator file sets `MODEL_TYPE` / `MODEL_NAME` and operator-specific options (e.g. `RUN_GRID_SEARCH`, `PARAM_GRID`).

### AdSERP (AttnLRP)

```bash
python fixate/fixate_training/train_fixate_attnlrp_adserp.py
```

Use `config/attnlrp_config_adserp.py` for backbone, paths (`OUTPUT_DIR`, `CHECKPOINT_DIR`), SERP batching, and grid-search ranges.

### Main configuration knobs (edit Python configs, not CLI flags)

| Setting | Typical location | Role |
|---------|------------------|------|
| Backbone | `MODEL_TYPE` + `MODEL_NAME` in each `config/*_config.py` | `internvl` vs `qwen3vl` and checkpoint directory |
| Basis & soft prompts | `NUM_BASIS`, `NUM_SOFT_TOKENS` in `common_config.py` | Factorized prompt size |
| Attention alignment | `LAMBDA_ATTN_WEIGHT`, `POWER_GAMMA`, `BETA_REG` | Loss weights and weighted KL / regularization |
| Optimization | `BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `NUM_EPOCHS_CV` / `NUM_EPOCHS_FINAL` | Throughput and training length |
| CV / search | `RUN_GRID_SEARCH`, `PARAM_GRID`, `K_FOLDS`, `SEED` / `MULTI_SEEDS` | Hyperparameter search and reproducibility |

Backbone choice is per config file: set `MODEL_TYPE` to `internvl` or `qwen3vl` and ensure `MODEL_NAME` points at the corresponding local folder.

## 📊 Evaluation

### Run Evaluation

```bash
python eval.py \
    --dataset recgaze \
    --data_dir data/processed/recgaze \
    --backbone qwen3-vl-4b \
    --backbone_path <PATH_TO_QWEN3_VL> \
    --probing_operator attnlrp \
    --checkpoint outputs/recgaze_qwen3vl_attnlrp/best.pt \
    --output_dir results/recgaze_qwen3vl_attnlrp \
    --num_runs 5
```

### Metrics

We evaluate from two perspectives:

**Attention Alignment Metrics:**
- KL Divergence (↓), Jensen–Shannon Divergence (↓), Cosine Similarity (↑)
- Clicked-Slot Hit@k (CSH@k), Top-Gaze Overlap@k (TGO@k)

**Prediction-Level Metrics:**
- Accuracy, LogLoss (↓), AUC

### Cross-Domain Evaluation (AdSERP)

```bash
python eval.py \
    --dataset adserp \
    --data_dir data/processed/adserp \
    --backbone qwen3-vl-4b \
    --backbone_path <PATH_TO_QWEN3_VL> \
    --probing_operator attnlrp \
    --checkpoint outputs/adserp_qwen3vl_attnlrp/best.pt \
    --output_dir results/adserp_qwen3vl_attnlrp
```

## 📁 Project Structure

```
FixATE/
├── assets/                     # Figures for README
├── configs/                    # Experiment configs
├── data/                       # Raw and processed data
├── preprocess/
│   ├── build_fixation.py       # Fixation data preprocessing
│   └── render_interface.py     # Interface screenshot rendering
├── src/
│   ├── models/
│   │   ├── fixate.py           # FixATE core module
│   │   ├── prompt_basis.py     # Personalized prompt basis decomposition
│   │   └── backbones/          # VLM backbone wrappers
│   ├── probing/
│   │   ├── attn_rollout.py     # Attention Rollout operator
│   │   ├── glimpse.py          # GLIMPSE operator
│   │   └── attnlrp.py          # AttnLRP operator
│   ├── data/
│   │   ├── dataset.py          # Dataset and dataloader
│   │   └── collator.py         # Multimodal collator
│   ├── losses.py               # Attention alignment & NTP losses
│   └── metrics.py              # Evaluation metrics
├── scripts/
│   └── run_fixate.sh           # End-to-end run script
├── train.py                    # Training entry point
├── eval.py                     # Evaluation entry point
├── requirements.txt
├── LICENSE
└── README.md
```

## 🙏 Acknowledgements

- [RecGaze](https://github.com/deleMartinez/RecGaze) for the eye-tracking dataset in carousel-based recommendation
- [AdSERP](https://github.com/nicolo-mn/ad-serp) for the eye-tracking dataset in sponsored search
- [Qwen3-VL](https://github.com/QwenLM/Qwen2.5-VL) and [InternVL](https://github.com/OpenGVLab/InternVL) for VLM backbones
- [GLIMPSE](https://github.com/gxshen/GLIMPSE), [AttnLRP](https://github.com/rachtibat/LRP-eXplains-Transformers), and [Attention Rollout](https://github.com/samiraabnar/attention_flow) for interpretability operators