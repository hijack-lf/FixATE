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
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Citation](#citation)

## 🔍 Overview

Existing LLM-based user simulators perceive recommendations through text or structured metadata, missing the visual attention signals that drive real user behavior. **FixATE** bridges this gap by:

1. **Probing** the VLM's internal visual attention via interpretability operators (Attention Rollout, GLIMPSE, AttnLRP) to obtain slot-level relevance distributions comparable with human fixation.
2. **Learning personalized soft prompts** through a factorized basis decomposition, steering the model's attention toward each user's characteristic fixation pattern.

<p align="center">
  <img src="assets/motivation.png" width="70%" alt="Motivation: Perceptual gap between text-based and visual interfaces">
</p>

## 🚧 Installation

```bash
git clone https://github.com/FixATE/FixATE.git
cd FixATE
conda create -n fixate python=3.10 -y
conda activate fixate
pip install -r requirements.txt
```

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
data/
├── recgaze/
│   ├── fixation_data.csv
│   ├── click_data.csv
│   └── interface_screenshots/
└── adserp/
    ├── fixation_data.csv
    ├── click_data.csv
    └── interface_screenshots/
```

### 2. Preprocess Fixation Data

Aggregate pixel-level fixation into slot-level dwell-time vectors:

```bash
python preprocess/build_fixation.py \
    --dataset recgaze \
    --data_dir data/recgaze \
    --output_dir data/processed/recgaze
```

### 3. Generate Interface Screenshots

Render recommendation interfaces as images for VLM input:

```bash
python preprocess/render_interface.py \
    --dataset recgaze \
    --data_dir data/recgaze \
    --output_dir data/processed/recgaze/screenshots
```

After preprocessing, you will get:

```
data/processed/recgaze/
├── train.jsonl          # Training interactions
├── test.jsonl           # Test interactions (leave-one-out)
├── fixation_dist.json   # Per-session slot-level fixation distributions
└── screenshots/         # Rendered interface images
```

## 🚀 Quick Start

Run FixATE training and evaluation with default settings on RecGaze:

```bash
bash scripts/run_fixate.sh
```

## 🔧 Training

### Train FixATE with Personalized Soft Prompts

```bash
python train.py \
    --dataset recgaze \
    --data_dir data/processed/recgaze \
    --backbone qwen3-vl-4b \
    --backbone_path <PATH_TO_QWEN3_VL> \
    --probing_operator attnlrp \
    --num_basis 8 \
    --soft_prompt_length 16 \
    --loss_weight_attn 1.0 \
    --power_exponent 2.0 \
    --batch_size 4 \
    --grad_accum_steps 2 \
    --epochs 30 \
    --lr 1e-4 \
    --output_dir outputs/recgaze_qwen3vl_attnlrp \
    --seed 42
```

#### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--backbone` | VLM backbone (`qwen3-vl-4b` / `internvl3.5-4b`) | `qwen3-vl-4b` |
| `--probing_operator` | Interpretability operator (`attnlrp` / `glimpse` / `attn_rollout`) | `attnlrp` |
| `--num_basis` | Number of prompt basis vectors *M* | `8` |
| `--soft_prompt_length` | Tokens per basis prompt *N_soft* | `16` |
| `--loss_weight_attn` | Weight λ for attention alignment loss | `1.0` |
| `--power_exponent` | Power exponent γ for weighted KL divergence | `2.0` |

#### Supported Configurations

FixATE is backbone- and operator-agnostic. All 6 combinations are supported:

| | Attention Rollout | GLIMPSE | AttnLRP |
|---|:---:|:---:|:---:|
| **Qwen3-VL-4B** | ✅ | ✅ | ✅ |
| **InternVL3.5-4B** | ✅ | ✅ | ✅ |

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