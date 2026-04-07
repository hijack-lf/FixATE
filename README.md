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
  <img src="asserts/Fixate_framework.png" width="90%" alt="FixATE Framework Overview">
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


### Dependencies

- PyTorch >= 2.1
- Transformers >= 4.40
- Two supported VLM backbones:
  - [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
  - [InternVL3.5-4B-Instruct](https://huggingface.co/OpenGVLab/InternVL3_5-4B-Instruct)

## 📦 Data Preparation

Download [RecGaze](https://github.com/santideleon/RecGaze_Dataset) and [AdSERP](https://github.com/kayhan-latifzadeh/AdSERP) into `datasets/` (e.g. `RecGaze/`, `Adserp/`).

**RecGaze**

```bash
python preprocessing/recgaze/dataset_preprocess_swipes.py
python preprocessing/recgaze/generate_interface_iamge.py 1-35
```

Raw inputs for the second step: `datasets/raw/RecGaze/` (`summary_feedback.csv`, `item_features.csv`, `poster_cache/`). Outputs: `datasets/RecGaze/init_interface_user_gaze(swipes).csv` and `datasets/RecGaze/interface_iamge/*.png`.

**AdSERP (optional)**

```bash
python preprocessing/adserp/build_samples.py --mode both --n 5
python preprocessing/adserp/build_click_aoi_dataset.py --split all
```

## 🔧 Training

Run from the **repo root**. Hyperparameters are in `config/common_config.py` and operator-specific files (`config/attnlrp_config.py`, `glimpse_config.py`, `rollout_config.py`, `attnlrp_config_adserp.py`). Put VLM weights under `llm_models/` (paths in `common_config.py`).

**RecGaze**

```bash
python fixate/fixate_training/train_fixate_attnlrp.py      # AttnLRP
python fixate/fixate_training/train_fixate_glimpse.py      # GLIMPSE
python fixate/fixate_training/train_fixate_rollout.py      # Attention Rollout
```

**AdSERP**

```bash
python fixate/fixate_training/train_fixate_attnlrp_adserp.py
```

## 📊 Evaluation

Training scripts write per-run metrics to JSON under `outputs/` and `checkpoints/` (paths depend on `config/`). Below matches what `compute_sample_metrics` and the trainers aggregate (sample-level metrics are **micro-averaged** over the evaluation set, prefixed with `micro_` in logs).

### Attention alignment (model attention vs. human gaze)

Let **g** be the normalized human gaze (dwell) vector and **a** the normalized model slot-attention vector on the same slots. **choice** is the ground-truth clicked slot index.

| Metric | Meaning | Better |
|--------|---------|--------|
| **KL divergence** (`kl_div` / `micro_kl_div`) | KL(*g* ∥ *a*): how much human gaze *g* differs from model mass *a* | Lower |
| **JS divergence** (`js_div` / `micro_js_div`) | Squared Jensen–Shannon distance between *g* and *a* | Lower |
| **Cosine similarity** (`cosine_sim` / `micro_cosine_sim`) | Cosine similarity between vectors *g* and *a* | Higher |
| **Attention log-loss** (`attn_logloss`) | Negative log of *a* on the clicked slot (mass on the true choice) | Lower |
| **Attention AUC** (`attn_auc`) | One-vs-rest style rank score: other slots vs. clicked slot under *a* | Higher |
| **Click@k** (`click@1`, `click@3`, `click@5`) | Whether **choice** is in the top-*k* slots when ranked by model attention *a* | Higher |
| **Gaze@k** (`gaze@1`, `gaze@3`, `gaze@5`,) | Overlap between top-*k* by *a* and top-*k* by *g* (implementation normalizes by *k* for *k*>1) | Higher |

AdSERP runs may log extra variants (e.g. **top-k JS** over visible AOIs) when slot sets are variable.

### Prediction quality (answer / choice)

| Metric | Meaning | Better |
|--------|---------|--------|
| **Answer accuracy** (`answer_accuracy`) | Fraction of samples where the model’s generated choice (letter or index) matches the label | Higher |

### Composite score (RecGaze-style selection)

`config/common_config.py` defines a weighted **primary score** over several `micro_*` terms (e.g. cosine, click@k, gaze@k, answer accuracy) for model selection during CV; see `PRIMARY_METRIC_POSITIVE` and `compute_primary_score`.

## 📁 Project Structure

High-level layout:

```
├── config/                 # Training hyperparameters & paths
├── datasets/               # RecGaze / AdSERP data
├── fixate/                 # Core library + training scripts (fixate_training/)
├── preprocessing/          # Dataset-specific preprocessing
├── llm_models/             # Local VLM checkpoints (optional path)
├── outputs/                # Metrics / logs
└── requirements.txt
```

## 🙏 Acknowledgements

- [RecGaze](https://github.com/deleMartinez/RecGaze) for the eye-tracking dataset in carousel-based recommendation
- [AdSERP](https://github.com/nicolo-mn/ad-serp) for the eye-tracking dataset in sponsored search
- [Qwen3-VL](https://github.com/QwenLM/Qwen2.5-VL) and [InternVL](https://github.com/OpenGVLab/InternVL) for VLM backbones
- [GLIMPSE](https://github.com/gxshen/GLIMPSE), [AttnLRP](https://github.com/rachtibat/LRP-eXplains-Transformers), and [Attention Rollout](https://github.com/samiraabnar/attention_flow) for interpretability operators