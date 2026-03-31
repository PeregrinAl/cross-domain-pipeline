
# Cross-Domain Anomaly Recognition MVP

MVP for cross-domain anomaly recognition in non-stationary signals with:

- two signal views:
  - raw temporal window
  - time-frequency representation via STFT
- source-only baseline
- source-free domain adaptation
- comparison of single-view and fused representations

## Objective

The goal of this MVP is to demonstrate, in one reproducible experimental pipeline, that:

1. a source-only anomaly recognition model degrades under domain shift,
2. source-free adaptation on the target domain can partially recover target quality without access to source data during adaptation,
3. multi-view fused representations are a meaningful candidate for adaptation, even if they may be more sensitive to domain shift before adaptation.

## Current Status

Implemented:

- project skeleton and reproducible config-driven pipeline
- data preparation and window segmentation
- raw and STFT representations
- source-only baselines:
  - `raw_only`
  - `tfr_only`
  - `fused`
- target-domain evaluation with calibrated threshold
- minimal source-free adaptation for the fused model
- consistency-filtered adaptation variant
- ablation summary

## Main Experimental Result

### Source-only baselines

| Experiment | Source ROC-AUC | Target ROC-AUC | Source PR-AUC | Target PR-AUC |
|---|---:|---:|---:|---:|
| raw_only | 0.5781 | 0.6588 | 0.6630 | 0.6932 |
| tfr_only | 0.6969 | 0.6531 | 0.7505 | 0.7228 |
| fused | 0.6869 | 0.4700 | 0.7136 | 0.6094 |

### Adaptation results for fused model

| Experiment | Source ROC-AUC | Target ROC-AUC | Source PR-AUC | Target PR-AUC |
|---|---:|---:|---:|---:|
| fused source_only | 0.6869 | 0.4700 | 0.7136 | 0.6094 |
| fused + sfda_minimal | 0.6525 | 0.6200 | 0.7548 | 0.7185 |
| fused + sfda_consistency | 0.6519 | 0.6169 | 0.7541 | 0.7173 |

## Interpretation

The current MVP supports the following conclusions:

- In the source-only setting, the fused two-view model shows good source-domain quality but also the strongest degradation on the target domain.
- Minimal source-free adaptation substantially improves target-domain ranking quality for the fused model:
  - target ROC-AUC improves from **0.4700** to **0.6200**
  - target PR-AUC improves from **0.6094** to **0.7185**
- In the current toy setup, adding cross-view consistency filtering does not provide an additional measurable gain over the simpler adaptation loop.

## Experimental Pipeline

The implemented pipeline is:

`Preprocessing -> Windowing -> Raw/STFT views -> Source-only baseline -> Source-free adaptation -> Ablation summary`

### Source-only baseline

The source-only baseline contains:

- `RawEncoder`: 1D CNN for raw windows
- `TFREncoder`: 2D CNN for STFT inputs
- `FusionEncoder`: concatenation of branch embeddings followed by a fusion head
- `SourceOnlyClassifier`: supervised anomaly classifier trained on source labels

### Source-free adaptation

The minimal adaptation loop:

1. loads the trained fused source-only model,
2. computes a source normal prototype,
3. selects low-score target windows as pseudo-normal samples,
4. fine-tunes the fusion head on target adaptation data,
5. evaluates performance before and after adaptation.

## Project Structure

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
│  ├─ source_only_supervised/
│  │  ├─ raw_only/
│  │  ├─ tfr_only/
│  │  └─ fused/
│  ├─ day5_evaluation/
│  ├─ day6_sfda/
│  ├─ day8_ablation/
│  └─ ablation_inputs/
├─ notebooks/
├─ src/
│  ├─ data/
│  │  ├─ dataset.py
│  │  ├─ dataloader.py
│  │  ├─ transforms.py
│  │  └─ windowing.py
│  ├─ models/
│  │  ├─ raw_encoder.py
│  │  ├─ tfr_encoder.py
│  │  ├─ fusion_model.py
│  │  └─ source_only_classifier.py
│  ├─ utils/
│  │  ├─ metrics.py
│  │  └─ seed.py
│  ├─ prepare_dummy_records.py
│  ├─ prepare_data.py
│  ├─ train.py
│  ├─ train_source_only.py
│  ├─ evaluate_source_only.py
│  ├─ day5_evaluate.py
│  ├─ adapt_source_free.py
│  └─ day8_ablation_summary.py
├─ requirements.txt
└─ README.md
```

## Data Protocol

The current repository uses a reproducible toy protocol based on generated `.npy` signals.

### Splits

* `source/train`: source-domain training signals
* `source/val`: source-domain validation signals
* `target/test`: target-domain evaluation signals
* `target/adapt`: unlabeled target-domain signals used for source-free adaptation

### Processed manifest format

Each row in `data/processed/manifest.csv` contains metadata for one window:

* `path`
* `label`
* `domain`
* `record_id`
* `split`
* `window_idx`
* `start`
* `end`

## Installation

Create a virtual environment and install dependencies.

### Windows

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare dummy raw signals

```bash
python -m src.prepare_dummy_records
```

### 2. Build windowed dataset and manifest

```bash
python -m src.prepare_data
```

### 3. Run smoke test

```bash
python -m src.train
```

### 4. Train source-only baselines

```bash
python -m src.train_source_only --variant raw_only
python -m src.train_source_only --variant tfr_only
python -m src.train_source_only --variant fused
```

### 5. Evaluate Day 5 metrics

```bash
python -m src.day5_evaluate
```

### 6. Run source-free adaptation for fused model

```bash
python -m src.adapt_source_free
```

### 7. Build Day 8 ablation summary

```bash
python -m src.day8_ablation_summary
```

## Main Output Files

### Source-only training

Saved under:

```text
experiments/source_only_supervised/<variant>/
```

Contains:

* `best.pt`
* `history.csv`
* `source_val_scores.csv`
* `target_test_scores.csv`
* `summary.json`

### Day 5 evaluation

Saved under:

```text
experiments/day5_evaluation/
```

Contains:

* `day5_summary.csv`
* score distribution plots
* ROC curves
* PR curves

### Day 6 adaptation

Saved under:

```text
experiments/day6_sfda/fused/
```

Contains:

* adapted checkpoint
* adaptation history
* before/after score files
* `summary.json`

### Day 8 ablation

Saved under:

```text
experiments/day8_ablation/
```

Contains:

* `ablation_summary.csv`
* bar plots for target ROC-AUC and PR-AUC
* source-vs-target scatter plot
* before/after fused plots

## Limitations

This MVP currently has the following limitations:

* It uses a toy synthetic protocol rather than a full real-world benchmark.
* Window labels inherit the label of the parent signal.
* Threshold-based metrics are sensitive to score calibration after adaptation.
* Cross-view consistency filtering has not shown additional benefit in the current setup.
* The current adaptation loop is intentionally minimal and does not yet include more advanced pseudo-label filtering or uncertainty estimation.

## Next Steps

Planned next steps:

* improve target adaptation split size and diversity
* test stronger and more varied domain shifts
* add event-level evaluation
* refine score calibration after adaptation
* optionally test CWT or wavelet multi-scale branch as a stretch goal
* prepare a compact demo and slides

## Glossary

**Anomaly recognition**
Detection of abnormal or non-normal signal behavior.

**Source domain**
The domain on which the model is initially trained.

**Target domain**
A different domain where distribution shift is present and where transfer performance is evaluated.

**Domain shift**
A change in signal statistics, noise, trend, scale, or frequency structure between source and target.

**Source-only model**
A model trained only on source-domain data and applied to the target domain without adaptation.

**Source-free adaptation**
Adaptation on the target domain without access to the original source training samples.

**Non-stationary signal**
A signal whose statistical or spectral properties change over time.

**Windowing**
Splitting a long signal into shorter fixed-length segments.

**Raw view**
The original temporal signal window used directly as model input.

**TFR**
Time-frequency representation. In this project, it is currently implemented as STFT magnitude.

**STFT**
Short-Time Fourier Transform. A time-frequency transform used to represent local spectral content.

**Embedding**
A learned feature vector produced by an encoder.

**Fusion**
Combination of multiple branch embeddings into one joint representation.

**Prototype**
A representative vector, often the mean embedding of normal source samples.

**Pseudo-normal sample**
A target-domain sample selected as likely normal based on model confidence or score filtering.

**Ranking metrics**
Metrics based on score ordering rather than a single threshold, such as ROC-AUC and PR-AUC.

**ROC-AUC**
Area under the ROC curve. Measures ranking quality across thresholds.

**PR-AUC**
Area under the precision-recall curve. Useful for anomaly or imbalanced settings.

**F1 score**
Threshold-based harmonic mean of precision and recall.

**Consistency filtering**
A pseudo-label selection strategy that keeps target samples only when multiple views or branches agree.

## Reproducibility Note

All experiments are intended to be run from a single config file:

```text
configs/base.yaml
```

The repository is structured so that the full experimental story can be reproduced:

1. prepare data,
2. train source-only baselines,
3. evaluate target degradation,
4. run source-free adaptation,
5. summarize ablations.

