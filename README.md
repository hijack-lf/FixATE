# 👁️ FixATE: Fixation-Aligned Tuning for Personalized User Emulation

[![Dataset: RecGaze](https://img.shields.io/badge/Dataset-RecGaze-blue)](https://github.com/santideleon/RecGaze_Dataset)
[![Dataset: AdSERP](https://img.shields.io/badge/Dataset-AdSERP-blue)](https://github.com/kayhan-latifzadeh/AdSERP)

Code for **"Through Their Eyes: Fixation-aligned Tuning for Personalized User Emulation"**.

**FixATE** aligns a frozen VLM's visual attention with each user's characteristic gaze pattern through interpretability-based probing and personalized soft prompt tuning, enabling more faithful user simulation in visual interfaces. We evaluate on two eye-tracking datasets covering two structurally distinct scenarios: carousel-based movie recommendation (**RecGaze**) and sponsored search (**AdSERP**). Beyond eye tracking, FixATE also supports **cursor dwell** as a low-cost supervision signal that approaches fixation-level performance without specialized hardware.

## 📚 Contents

- [Overview](#overview)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Reproducing Ablations & Sensitivity](#ablations)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)

<h2 id="overview">🔍 Overview</h2>

Existing LLM-based user simulators perceive recommendations through text or structured metadata, missing the visual attention signals that drive real user behavior. **FixATE** bridges this gap by:

1. **Probing** the VLM's internal visual attention via interpretability operators (Attention Rollout, GLIMPSE, AttnLRP) to obtain slot-level relevance distributions comparable with human fixation.
2. **Learning personalized soft prompts** through a factorized basis decomposition, steering the model's attention toward each user's characteristic fixation pattern.
3. **Supporting low-cost supervision**: cursor dwell time can replace eye-tracking fixation as the alignment target, trading a modest performance gap for far cheaper data collection.

### Dependencies

- PyTorch >= 2.1
- Transformers (a version supporting Qwen3-VL and InternVL3.5)
- Two supported VLM backbones, downloaded into `llm_models/`:
  - [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
  - [InternVL3.5-4B-Instruct](https://huggingface.co/OpenGVLab/InternVL3_5-4B-Instruct)

<h2 id="data-preparation">📦 Data Preparation</h2>

Download [RecGaze](https://github.com/santideleon/RecGaze_Dataset) and [AdSERP](https://github.com/kayhan-latifzadeh/AdSERP) and place them under `datasets/` (`RecGaze/`, `Adserp/`).

**RecGaze** (2,701 click sessions from 87 users; content-disjoint task-level holdout: 1,921 train / 311 val / 469 test)

Place the raw RecGaze files under `datasets/RecGaze/raw/` (`summary_feedback.csv`, `non_public_feedback_dataset.csv`, `aoi_data.csv`, `item_features.csv`, `user_features.csv`). Then run, from the repo root:

```bash
python preprocessing/recgaze/build_gaze_tables.py   # raw logs -> page-divided fixation & cursor tables
python preprocessing/recgaze/render_images.py       # 1920x1080 page renders + letter-labeled (A-O) variants
python preprocessing/recgaze/build_dataset.py       # action manifests -> stage2 -> fixate_dataset.jsonl
```

All intermediates stay under `datasets/RecGaze/`. The training entry consumes `datasets/RecGaze/fixate_dataset.jsonl` and the letter-labeled renders in `datasets/RecGaze/page_divide_real/image_index/`. The final build applies the fixed content-disjoint task split (seed 42), attaches slot-level cursor dwell, and self-checks the cursor mapping against the stored fixation vectors.

**AdSERP** (2,568 sessions; per-user query-disjoint split: 1,740 train / 309 val / 519 test)

```bash
python preprocessing/adserp/build_samples.py --mode both --n 5
python preprocessing/adserp/build_click_aoi_dataset.py --split all
```

Place the resulting `samples.jsonl` and `images/` under `datasets/Adserp/`. Each AOI record carries both `gaze_dwell_ms` and `cursor_dwell_ms`.

<h2 id="training">🔧 Training</h2>

Run from the **repo root**. All switches and hyperparameters live in `config/recgaze_config.py` and `config/adserp_config.py`; the defaults reproduce the main results in the paper.

```bash
python fixate/fixate_training/train_fixate_recgaze.py   # RecGaze
python fixate/fixate_training/train_fixate_adserp.py    # AdSERP
```

Key switches (edit the corresponding config file):

| Config entry | Values | Meaning |
|---|---|---|
| `ATTN_METHOD` | `attnlrp` / `glimpse` / `rollout` | probing operator |
| `MODEL_TYPE` | `qwen3vl` / `internvl` | VLM backbone |
| `SLOT_KL_SIGNAL` | `fixation` / `cursor` | supervision signal for the alignment loss |
| `SIGNAL_INTERSECT` | `True` / `False` | restrict to sessions with both signals (dual-signal subset: 1,830 / 298 / 448) |

Both trainers train only the soft-prompt parameters (basis + per-user coefficients) with the backbone frozen, average results over 5 seeds, and report the four-way comparison (Backbone / FixATE × teacher-forcing / free-generation) with per-user paired Wilcoxon tests.

<h2 id="ablations">🧪 Reproducing Ablations & Sensitivity</h2>

Each ablation variant in the paper corresponds to one config change:

| Paper variant | Config change |
|---|---|
| Rand. SP | `USE_RANDOM_SOFT_PROMPT = True` |
| z_u = 1 (no personalization) | `USE_USER_ALPHA = False` |
| w/o ω_n (no importance weights) | `POWER_GAMMA = 0` |
| w/o L_Attn | `LAMBDA_ATTN_TARGET = 0` |
| w/o L_NTP | `TRAIN_CHOICE = False` |
| Cursor supervision (§4.6) | `SLOT_KL_SIGNAL = "cursor"`, `SIGNAL_INTERSECT = True` |

Sensitivity sweeps (§4.4) vary `NUM_BASIS` (M), `NUM_SOFT_TOKENS` (N_soft), and `LAMBDA_ATTN_TARGET` (λ) in the same config files.

<h2 id="evaluation">📊 Evaluation</h2>

Training scripts write per-run metrics to JSON under `outputs/` and checkpoints under `checkpoints/`. Sample-level metrics are micro-averaged over the evaluation set (prefixed `micro_` in logs).

### Attention alignment metrics

How well the normalized model slot-attention vector **a** matches the normalized human dwell vector **g** on the same slots. **choice** is the ground-truth clicked slot index.

| Metric | Meaning | Better |
|--------|---------|--------|
| **KL divergence** | KL(*g* ∥ *a*) | Lower |
| **JS divergence** | Squared Jensen–Shannon distance between *g* and *a* | Lower |
| **Cosine similarity** | Cosine similarity between *g* and *a* | Higher |
| **CSH@k** | Whether **choice** is in the top-*k* slots ranked by *a* | Higher |
| **TGO@k** | Overlap between top-*k* by *a* and top-*k* by *g* (normalized by *k*) | Higher |

### Prediction-level metrics

| Metric | Meaning | Better |
|--------|---------|--------|
| **Log-Loss** | Negative log-probability assigned to the true slot over candidate answer tokens | Lower |
| **AUC** | One-vs-rest rank score of **choice** under the logit-based slot distribution | Higher |
| **Answer Accuracy** | Fraction of samples where the generated choice matches the label | Higher |

<h2 id="project-structure">📁 Project Structure</h2>

```
├── config/
│   ├── recgaze_config.py       # RecGaze paths, switches, hyperparameters
│   └── adserp_config.py        # AdSERP paths, switches, hyperparameters
├── datasets/                   # RecGaze / AdSERP data (downloaded & built locally)
├── fixate/fixate_training/
│   ├── train_fixate_recgaze.py # RecGaze trainer (all 3 operators x 2 backbones)
│   └── train_fixate_adserp.py  # AdSERP trainer (all 3 operators x 2 backbones)
├── preprocessing/
│   ├── recgaze/                # full pipeline: page-divide -> renders -> manifests -> dataset
│   └── adserp/                 # SERP sample & click-AOI dataset build
├── llm_models/                 # local VLM checkpoints (download here)
├── outputs/  checkpoints/      # run artifacts (created at runtime)
└── requirements.txt
```

## 😄 Acknowledgements

- [RecGaze](https://github.com/santideleon/RecGaze_Dataset) for the eye-tracking dataset in carousel-based recommendation
- [AdSERP](https://github.com/kayhan-latifzadeh/AdSERP) for the eye-tracking dataset in sponsored search
- [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) and [InternVL3_5-4B-Instruct](https://huggingface.co/OpenGVLab/InternVL3_5-4B-Instruct) for VLM backbones
- [GLIMPSE](https://arxiv.org/abs/2506.18985), [AttnLRP](https://dl.acm.org/doi/10.5555/3692070.3692076), and [Attention Rollout](https://aclanthology.org/2020.acl-main.385/) for interpretability operators
