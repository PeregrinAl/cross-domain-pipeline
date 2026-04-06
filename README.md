# Cross-Domain Anomaly Recognition MVP

MVP for cross-domain anomaly recognition in non-stationary signals.

The repository implements a small, reproducible experimental pipel:contentReference[oaicite:1]{index=1}y model under domain shift,
2. partial recovery of target quality with source-free domain adaptation,
3. usefulness of fused multi-view representations as a basis for adaptation.

---

## Objective

The goal of this MVP is to show, in one reproducible experiment, that:

- a model trained only on the source domain degrades on a shifted target domain,
- source-free adaptation can recover part of the lost target quality without using source data during adaptation,
- a fused representation built from raw and time-frequency views is a meaningful basis for adaptation.

---

## Current Experimental Setup

### Pipeline

```text
Synthetic signal generation -> Windowing -> Raw/STFT views -> Source-only training -> Source-free adaptation -> Ablation summary
````

### Data protocol

The repository uses a synthetic toy protocol based on generated `.npy` signals.

Each signal-level record contains:

* `path`
* `label`
* `domain`
* `record_id`
* `split`
* `anomaly_intervals`

Each window-level row in `data/processed/manifest.csv` contains:

* `path`
* `label`
* `domain`
* `record_id`
* `split`
* `window_idx`
* `start`
* `end`

### Default window protocol

The current default protocol is:

* `window_size = 2048`
* `stride = 512`
* `window_label_mode = min_fraction`
* `min_anomaly_fraction = 0.05`

This protocol was selected after a dedicated sweep over window size, stride, and labeling rule.

### Domains and splits

* `source/train` — labeled source-domain training records
* `source/val` — labeled source-domain validation records
* `target/test` — labeled target-domain evaluation records
* `target/adapt` — unlabeled target-domain records used for source-free adaptation

### Target-domain shift

The current synthetic target shift includes:

* amplitude / scale shift
* extra noise
* trend drift
* frequency shift
* temporal warp

---

## Implemented Components

### Data

* synthetic signal generation
* anomaly interval storage at signal level
* window-level labeling from true overlap with anomaly intervals
* manifest generation for window-based experiments

### Representations

* `raw_only` — raw temporal windows
* `tfr_only` — STFT-based time-frequency representation
* `fused` — joint representation built from raw and STFT branches

### Models

* `RawEncoder`
* `TFREncoder`
* `FusionEncoder`
* `SourceOnlyClassifier`

### Training and evaluation

* source-only training for `raw_only`, `tfr_only`, and `fused`
* threshold calibration from source validation
* source-free adaptation for `fused`
* ablation summary builder

---

## Main Result

### Ablation summary

| experiment |        stage | source ROC-AUC | target ROC-AUC | source PR-AUC | target PR-AUC | source F1 | target F1 |
| ---------- | -----------: | -------------: | -------------: | ------------: | ------------: | --------: | --------: |
| raw_only   |  source_only |         1.0000 |         0.8976 |        1.0000 |        0.6687 |    1.0000 |    0.1818 |
| tfr_only   |  source_only |         1.0000 |         0.8333 |        1.0000 |        0.4771 |    1.0000 |    0.1818 |
| fused      |  source_only |         1.0000 |         0.9184 |        1.0000 |        0.6933 |    1.0000 |    0.1818 |
| fused      | sfda_minimal |         1.0000 |         0.9792 |        1.0000 |        0.8655 |    1.0000 |    0.6667 |

### Interpretation

The current MVP supports the following conclusions:

* source-only transfer degrades under domain shift,
* fused multi-view representation is the strongest source-only baseline on the target domain,
* source-free adaptation substantially improves the fused model on the target domain,
* the source-target gap becomes much smaller after adaptation.

In the current setup, the fused model improves from:

* target ROC-AUC: `0.9184 -> 0.9792`
* target PR-AUC: `0.6933 -> 0.8655`
* target F1: `0.1818 -> 0.6667`

---

## Repository Structure

```text
cross-domain-pipeline/
├─ configs/
│  └─ base.yaml
├─ data/
│  ├─ raw/
│  │  ├─ records.csv
│  │  └─ *.npy
│  └─ processed/
│     ├─ manifest.csv
│     └─ windows/
├─ experiments/
│  ├─ source_only_training/
│  │  ├─ raw_only/
│  │  ├─ tfr_only/
│  │  └─ fused/
│  ├─ source_free_adaptation/
│  │  └─ fused/
│  └─ ablation_summary/
├─ src/
│  ├─ data/
│  │  ├─ dataset.py
│  │  ├─ dataloader.py
│  │  ├─ transforms.py
│  │  └─ windowing.py
│  ├─ models/
│  ├─ utils/
│  ├─ prepare_dummy_records.py
│  ├─ prepare_data.py
│  ├─ train_source_only.py
│  ├─ adapt_source_free.py
│  └─ build_ablation_summary.py
├─ requirements.txt
└─ README.md
```

---

## Installation

Create and activate a virtual environment, then install dependencies.

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## How to Run

### 1. Generate synthetic records

```bash
python -m src.prepare_dummy_records
```

This creates raw `.npy` signals and `data/raw/records.csv`.

### 2. Build windowed manifest

```bash
python -m src.prepare_data
```

This creates `data/processed/manifest.csv` and window files in `data/processed/windows/`.

### 3. Train source-only baselines

```bash
python -m src.train_source_only --variant raw_only
python -m src.train_source_only --variant tfr_only
python -m src.train_source_only --variant fused
```

Outputs are written to:

```text
experiments/source_only_training/
```

### 4. Run source-free adaptation for fused

```bash
python -m src.adapt_source_free
```

Outputs are written to:

```text
experiments/source_free_adaptation/fused/
```

### 5. Build ablation summary

```bash
python -m src.build_ablation_summary
```

Outputs are written to:

```text
experiments/ablation_summary/
```

---

## Key Output Files

### Source-only

For each variant in `experiments/source_only_training/<variant>/`:

* `best_model.pt`
* `summary.json`
* `source_val_scores.csv`
* `target_test_scores.csv`

### Source-free adaptation

In `experiments/source_free_adaptation/fused/`:

* `adapted_model.pt`
* `summary.json`
* `adapt_history.csv`
* `target_test_scores_before.csv`
* `target_test_scores_after.csv`

### Ablation summary

In `experiments/ablation_summary/`:

* `ablation_summary.csv`
* `target_roc_auc_bar.png`
* `target_pr_auc_bar.png`
* `source_vs_target_roc_auc.png`
* fused before/after comparison plots

---

## Current Scope

This repository currently covers:

* synthetic cross-domain signal protocol,
* window-level anomaly recognition,
* source-only baselines,
* minimal source-free adaptation,
* quantitative comparison across variants.

The next logical extension is event-level evaluation on top of the current window-level pipeline.

---

## Notes

* The project is intentionally kept small and reproducible.
* The current protocol is synthetic and designed for MVP validation rather than final benchmarking.
* The repository is organized around minimal, incremental experiments rather than a large general-purpose framework.

