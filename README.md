<div align="center">

# 🌬️ A Physics-Informed Explainable Deep Learning Digital Twin for Wind Turbine System Identification, Condition Monitoring, and Predictive Maintenance



[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Mathematics%20MDPI-blue?style=for-the-badge&logo=read-the-docs&logoColor=white)](https://doi.org/10.3390/math1010000)
[![Dataset](https://img.shields.io/badge/Dataset-CARE%20Benchmark-orange?style=for-the-badge)](https://doi.org/10.3390/data9120138)

<br/>

**96.81% Accuracy · 96.69% F1 · AUC-ROC 0.9971 · FAR 3.04%**  
*Best result on every metric across 9 evaluated architectures on the CARE benchmark*

<br/>

<img src="figures/workflow_diagram V2.png" alt="PI-CTBA-Net Pipeline" width="90%"/>

*End-to-end pipeline: data ingestion → physics-informed preprocessing → sliding-window segmentation → model training → explainability → Digital Twin decision support*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Dataset Description](#-dataset-description)
- [Architecture](#-architecture)
- [Physics-Informed Loss](#-physics-informed-loss)
- [Folder Structure](#-folder-structure)
- [Experimental Setup](#-experimental-setup)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Explainability](#-explainability)
- [Citation](#-citation)

---

## 🔭 Overview

Wind turbine fault detection from SCADA data is challenging due to:
- High sensor dimensionality (86–957 channels per turbine)
- Cross-farm heterogeneity (onshore vs offshore environments)
- Absence of physics constraints in purely data-driven models

**PI-CTBA-Net** addresses these challenges by coupling:

| Component | Role |
|-----------|------|
| **Multi-Scale CNN** | Local temporal pattern extraction |
| **Transformer** | Global self-attention across the 24-h window |
| **Bidirectional LSTM** | Sequential dependency modelling |
| **Multi-Head Attention Pooling** | Adaptive feature aggregation |
| **5 Physics Auxiliary Losses** | Encoding Betz law, TSR, Newton cooling, power factor, pitch smoothness |

The model is evaluated as a **Digital Twin** fault-detection component on the publicly available [CARE benchmark](https://doi.org/10.3390/data9120138) — the first paper to combine multi-farm cross-evaluation, physics auxiliary losses, GradientSHAP explainability, and McNemar-validated comparisons in a single architecture.

---

## 📊 Key Results

### Table 8 — Test-Set Performance (10,971 windows)

| Model | Acc (%) | Prec (%) | Rec (%) | F1 (%) | AUC | MCC | FAR (%) |
|-------|---------|----------|---------|--------|-----|-----|---------|
| 1D-CNN | 79.82 | 78.64 | 79.05 | 78.84 | 0.8653 | 0.5962 | 18.72 |
| LSTM | 82.31 | 81.45 | 82.08 | 81.76 | 0.8926 | 0.6453 | 16.01 |
| Bi-LSTM | 84.76 | 84.02 | 84.31 | 84.16 | 0.9178 | 0.6941 | 13.54 |
| TCN | 87.18 | 86.71 | 86.95 | 86.83 | 0.9413 | 0.7428 | 11.02 |
| Transformer | 89.46 | 89.18 | 89.37 | 89.27 | 0.9586 | 0.7889 | 8.92 |
| CNN-LSTM | 91.87 | 91.56 | 91.81 | 91.68 | 0.9721 | 0.8375 | 6.73 |
| CNN-BiLSTM | 93.95 | 93.71 | 93.86 | 93.78 | 0.9842 | 0.8792 | 4.91 |
| CNN-TB-Att | 95.42 | 95.21 | 95.36 | 95.28 | 0.9918 | 0.9086 | 3.62 |
| **PI-CTBA-Net** | **96.81** | **96.65** | **96.74** | **96.69** | **0.9971** | **0.9368** | **3.04** |

> **Physics loss contribution** (PI-CTBA-Net vs CNN-TB-Att):  
> +1.41 pp F1 · +0.0053 AUC-ROC · **16.0% relative FAR reduction** (3.62% → 3.04%)  
> Confirmed by McNemar's test (*p* < 10⁻³¹)

### Cross-Farm Generalisation

| Farm | Acc (%) | F1 (%) | FAR (%) |
|------|---------|--------|---------|
| Farm A (Onshore, PT) | 96.70 | 96.88 | 2.22 |
| Farm B (Offshore, DE) | 96.95 | 96.12 | 2.21 |
| Farm C (Offshore, DE) | 96.80 | 96.59 | 2.23 |

*Inter-farm F1 spread < 0.77 pp — no farm-specific fine-tuning required*

---

## 🗃️ Dataset Description

The **CARE** (Condition monitoring And Reliability Evaluation) benchmark is a publicly available multi-farm SCADA dataset introduced by [Gück et al. (2024)](https://doi.org/10.3390/data9120138).

### Dataset Statistics

| Farm | Location | Turbines | Events | Anomaly | Normal | Raw Features | Rows |
|------|----------|----------|--------|---------|--------|--------------|------|
| Farm A | Onshore, Portugal | 5 | 22 | 11 | 11 | 86 | 1.20 M |
| Farm B | Offshore, Germany | 9 | 15 | 6 | 9 | 257 | 0.86 M |
| Farm C | Offshore, Germany | 22 | 58 | 27 | 31 | 957 | 3.21 M |
| **Total** | | **36** | **95** | **44** | **51** | **9 selected** | **5.27 M** |

### 9 Physics-Informed Features Selected

| Feature | Physics Equation | Farms |
|---------|-----------------|-------|
| `wind_speed` | Betz law: P = ½ρAC_pV³ | A, B, C |
| `active_power` | Betz law | A, B, C |
| `reactive_power` | Power factor: cos φ = P/S | A, B, C |
| `ambient_temp` | Newton's cooling | A, B, C |
| `grid_frequency` | Generator synchronisation | A, B, C |
| `rotor_speed` | Tip speed ratio: λ = ω_r R/V | A, B, C |
| `gearbox_oil_temp` | Newton's cooling | A, B, C |
| `generator_temp` | Newton's cooling | A, B, C |
| `pitch_angle` | Aerodynamic control | A, B, C |

**Download CARE dataset:** https://doi.org/10.3390/data9120138  
Place the extracted data under `data/CARE/` following the structure:
```
data/CARE/
├── Farm_A/
│   ├── datasets/          # Event CSV files (sep=";")
│   └── event_info.csv     # event_id → anomaly/normal labels
├── Farm_B/
└── Farm_C/
```



**Total parameters: 650,689 · Inference latency: 2.25 ms/window**

---

## ⚛️ Physics-Informed Loss

Five auxiliary loss terms encode domain knowledge:

| Loss | Physics Law | Formula | λ |
|------|------------|---------|---|
| **L₁** Betz Power Curve | P ∝ V³ | MSE(P̃, Ṽ³) | 0.10 |
| **L₂** TSR Stability | λ = ω_r R/V | Var_t(ω_r/\|V\|) | 0.05 |
| **L₃** Newton Cooling | dT/dt = -k(T - T_amb) | Var(ΔT_gb - T_amb) | 0.05 |
| **L₄** Power Factor | cos φ = P/S ≥ 0.9 | MSE(PF, 0.9) | 0.05 |
| **L₅** Pitch Smoothness | Rapid changes → faults | MSE(β_t, β_{t-1}) | 0.05 |

---

## 📁 Folder Structure

```
PI-CTBA-Net/
│
├── 📄 README.md                   ← You are here
├── 📄 requirements.txt            ← Python dependencies
├── 📄 .gitignore
│
├── 📁 config/
│   └── config.yaml                ← All hyperparameters & sensor maps
│
├── 📁 data/
│   ├── __init__.py
│   ├── preprocessing.py           ← CARE preprocessing pipeline
│   └── CARE/                      ← Dataset (download separately)
│
├── 📁 models/
│   ├── __init__.py
│   ├── pi_ctba_net.py             ← PI-CTBA-Net architecture (4 blocks)
│   ├── baselines.py               ← 8 baseline architectures
│   └── physics_loss.py            ← 5 physics auxiliary losses
│
├── 📁 training/
│   ├── __init__.py
│   ├── trainer.py                 ← Training loop + early stopping
│   └── evaluate.py                ← Metrics, McNemar, Bootstrap CI
│
├── 📁 explainability/
│   ├── __init__.py
│   └── shap_analysis.py           ← GradientSHAP + attention viz
│
├── 📁 utils/
│   ├── __init__.py
│   └── utils.py                   ← Seed, device, logging, I/O
│
├── 📁 eda/
│   ├── __init__.py
│   ├── dataset_builder.py         ← Build main_dataset.csv from raw CARE CSVs
│   └── eda_plots.py               ← 16 publication-ready EDA figures
│
├── 📁 scripts/
│   ├── train.py                   ← Train any model
│   ├── evaluate_all.py            ← Reproduce Table 8
│   ├── explain.py                 ← SHAP + attention analysis
│   └── run_eda.py                 ← Build dataset + generate all EDA figures
│
├── 📁 notebooks/                  ← Jupyter notebooks for EDA
├── 📁 checkpoints/                ← Saved model weights (git-ignored)
├── 📁 results/                    ← JSON results + CSV tables
└── 📁 figures/                    ← Generated plots
```

---

## ⚙️ Experimental Setup

| Hyperparameter | Value |
|----------------|-------|
| Window size T | 144 steps (24 hours) |
| Stride s | 72 steps (50% overlap) |
| Train / Val / Test | 70% / 15% / 15% |
| Anomaly threshold | > 5% of timesteps |
| Batch size | 64 |
| Max epochs | 100 |
| Early stopping patience | 30 (validation loss) |
| Learning rate | 3 × 10⁻⁴ |
| LR schedule | Cosine annealing |
| Optimiser | Adam (weight decay 10⁻⁵) |
| Gradient clipping | ‖∇‖₂ ≤ 1.0 |
| BCE pos_weight | 1.105 (N_normal / N_anomaly) |
| GPU | Tesla T4 (16 GB) |
| Training time | ~32.5 minutes |
| Random seed | 42 |

---

## 📊 EDA — Exploratory Data Analysis

The `eda/` module contains modular, publication-ready EDA code extracted from the original Jupyter notebooks.

### Figures Produced (16 total)

| Figure | Description |
|--------|-------------|
| Fig 1 | Dataset Overview (rows, events, labels, status) |
| Fig 2 | Feature Histograms with KDE overlay (3×3) |
| Fig 3 | Box Plots — outlier analysis (3×3) |
| Fig 4 | Correlation Heatmap (12×12) |
| Fig 5 | Power Curve Validation — P ∝ V³ (Betz law) |
| Fig 6 | Physics Constraint Validation (Cₚ, TSR, thermal, PF) |
| Fig 7 | Temporal Patterns — diurnal & seasonal |
| Fig 8 | Sample Time Series per farm |
| Fig 9 | Violin Plots — Normal vs Anomaly (3×3) |
| Fig 10 | Feature Discriminative Power (t-test + KS test) |
| Fig 11 | Cross-Farm KDE Distributions (3×3) |
| Fig 12 | Cross-Farm Anomaly Rate + Feature Mean Heatmap |
| Fig 13 | Data Quality — missing/zero rates |
| Fig 14 | 2D t-SNE projection |
| Fig 15 | Binned Power Curve ±1σ per farm |
| Fig 16 | TSR Proxy Distribution per farm |

### Run EDA

```bash
# Step 1 — Build main_dataset.csv from raw CARE CSVs
python scripts/run_eda.py --step build --care_root data/CARE

# Step 2 — Generate all 16 EDA figures
python scripts/run_eda.py --step eda --dataset data/main_dataset.csv

# Both steps at once
python scripts/run_eda.py --step all --care_root data/CARE
```

### EDA Module (programmatic)

```python
from eda.dataset_builder import CAREDatasetBuilder
from eda.eda_plots import EDAPlotter

# Build dataset
builder = CAREDatasetBuilder(root="data/CARE")
df = builder.build(save_path="data/main_dataset.csv")

# Generate figures
plotter = EDAPlotter(df, save_dir="figures/EDA", show=True)
plotter.run_all()              # all 16 figures
plotter.figure_power_curve()   # single figure
plotter.figure_violin_plots()
```

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/uzzal2200/A-Physics-Informed-Explainable-Deep-Learning-Digital-Twin-for-Wind-Turbine.git
cd PI-CTBA-Net
pip install -r requirements.txt
```

### 2. Download CARE dataset

```bash
# Download from https://doi.org/10.3390/data9120138
# Then place under:
mkdir -p data/CARE
# Extract Farm_A, Farm_B, Farm_C folders here
```

### 3. Train PI-CTBA-Net

```bash
python scripts/train.py --model pi_ctba_net
```

### 4. Reproduce Table 8

```bash
python scripts/evaluate_all.py
```

### 5. Run explainability

```bash
python scripts/explain.py
```

---

## 🏋️ Training

```bash
# PI-CTBA-Net (with physics loss)
python scripts/train.py --model pi_ctba_net

# Ablation baseline (no physics loss)
python scripts/train.py --model CNN-TB-Att

# Any baseline
python scripts/train.py --model 1D-CNN
python scripts/train.py --model LSTM
python scripts/train.py --model Bi-LSTM
python scripts/train.py --model TCN
python scripts/train.py --model Transformer
python scripts/train.py --model CNN-LSTM
python scripts/train.py --model CNN-BiLSTM

# Custom config / device
python scripts/train.py --model pi_ctba_net --config config/config.yaml --device cpu
```

Checkpoints are saved to `checkpoints/<model_name>_best.pt`.  
Training history is saved to `checkpoints/<model_name>_history.json`.

---

## 📈 Evaluation

```bash
# Evaluate all 9 models and produce Table 8
python scripts/evaluate_all.py

# Output files:
# results/table8_main_results.csv   — main performance table
# results/mcnemar_results.csv       — McNemar's test vs all baselines
```

---

## 🔍 Explainability

```bash
python scripts/explain.py

# Output:
# results/shap_values.npy              — GradientSHAP arrays
# results/attention_weights.npy        — Multi-head attention maps
# figures/shap_global_importance.png   — Feature importance bar chart
# figures/attention_map.png            — Attention weight heatmap
```

**Top-3 anomaly indicators (GradientSHAP):**

| Rank | Feature | Mean \|SHAP\| | Physics term |
|------|---------|--------------|-------------|
| 1 | `gearbox_oil_temp` | 0.0404 | L₃ (Newton cooling) |
| 2 | `ambient_temp` | 0.0279 | L₃ (Newton cooling) |
| 3 | `pitch_angle` | 0.0202 | L₅ (Pitch smoothness) |

---

## 📝 Citation

If you use PI-CTBA-Net in your research, please cite:

```bibtex
@article{debnath2026pictbanet,
  title     = {A Physics-Informed Explainable Deep Learning Digital Twin 
               for Wind Turbine System Identification, Condition Monitoring, 
               and Predictive Maintenance},
  author    = {Debnath, Sajib and Mia, Md Uzzal and Biswas, Arindam Kishor},
  journal   = {Mathematics},
  publisher = {MDPI},
  year      = {2026},
 
}
```

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Made with ❤️ by **Md. Uzzal Mia**   
⭐ Star this repo if you find it useful!

</div>
