# Cross-Domain Anomaly Recognition MVP

A compact research MVP for cross-domain anomaly recognition in non-stationary signals.

The repository implements a reproducible experimental pipeline for studying whether:

1. a source-only anomaly model degrades under domain shift,
2. source-free adaptation can recover part of the lost target-domain quality without access to source data during adaptation,
3. fused multi-view representations are a meaningful basis for adaptation.

The current setup uses synthetic `.npy` signals, PyTorch models, YAML configuration, and a fully scriptable pipeline.

---

## Problem Setting

The MVP studies anomaly recognition under domain shift in non-stationary signals.

Each signal is transformed into overlapping windows. For each window, two complementary views are built:

- raw temporal view,
- time-frequency view based on STFT.

These views are used in three source-only baselines:

- `raw_only`
- `tfr_only`
- `fused`

The fused model is then used as the basis for source-free domain adaptation on unlabeled target-domain windows.

---

## Current Scope

Implemented:

- reproducible synthetic data generation,
- interval-aware anomaly annotation at the record level,
- window-level labeling based on overlap with anomaly intervals,
- dataset and dataloader pipeline,
- raw and STFT representations,
- source-only training for `raw_only`, `tfr_only`, and `fused`,
- source-only evaluation,
- minimal source-free adaptation for the fused model,
- window-size and stride sweep,
- ablation summary generation.

Not implemented yet:

- event-level evaluation,
- stronger adaptation variants beyond the current minimal setup,
- real-world datasets,
- fully realistic temporal distortions with interval remapping.

---

## Main Idea

The implemented pipeline is:

`Raw signal -> Windowing -> Raw/STFT views -> Source-only training -> Source-only evaluation -> Source-free adaptation -> Ablation summary`

The main scientific intention of the MVP is not to claim a production-ready method, but to provide a clean and reproducible experimental baseline for further work on cross-domain anomaly recognition in non-stationary signals.

---

## Data Protocol

The repository uses a synthetic protocol with generated `.npy` signals.

### Splits

- `source/train` — labeled source-domain training signals
- `source/val` — labeled source-domain validation signals
- `target/test` — labeled target-domain evaluation signals
- `target/adapt` — unlabeled target-domain signals used during source-free adaptation

### Record-Level Annotation

Each raw signal record is described in `data/raw/records.csv`.

Important fields:

- `path`
- `label`
- `domain`
- `record_id`
- `split`
- `anomaly_intervals`

`anomaly_intervals` stores anomaly intervals for a record. Normal signals use an empty list.

### Window-Level Annotation

`data/processed/manifest.csv` contains one row per window.

Important fields:

- `path`
- `label`
- `record_label`
- `domain`
- `record_id`
- `split`
- `window_idx`
- `start`
- `end`
- `overlap_samples`
- `overlap_fraction`

A window is labeled as anomalous only if it overlaps an annotated anomaly interval according to the configured labeling rule.

This is more honest than record-level label inheritance, because anomalous records now contain both normal and anomalous windows.

---

## Domain Shift

The current target-domain protocol includes controlled synthetic shift relative to the source domain.

The target signals may differ by:

- amplitude scaling,
- additional noise,
- additional trend drift.

The code already contains placeholders for stronger distortions such as frequency shift and temporal warping, but these are currently disabled to avoid invalidating anomaly interval alignment.

---

## Models

### Raw Branch

`RawEncoder` is a one-dimensional convolutional encoder for raw signal windows.

### Time-Frequency Branch

`TFREncoder` is a two-dimensional convolutional encoder for STFT-based inputs.

### Fusion Branch

`FusionEncoder` combines embeddings from the raw and time-frequency branches.

### Classifier

`SourceOnlyClassifier` is trained in supervised fashion on source-domain labels.

---

## Source-Free Adaptation

The current source-free adaptation pipeline is intentionally minimal.

High-level procedure:

1. load the trained fused source-only model,
2. estimate a source-side normal prototype,
3. score target adaptation windows,
4. select low-score target windows as pseudo-normal candidates,
5. fine-tune part of the fused model on target adaptation data,
6. evaluate before and after adaptation.

This keeps the MVP simple and makes it easier to see whether adaptation helps at all before introducing more complex strategies.

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
│  ├─ source_only_eval/
│  ├─ source_free_adaptation/
│  ├─ ablation_summary/
│  └─ window_sweep/
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
│  ├─ check_dataset.py
│  ├─ train_source_only.py
│  ├─ evaluate_source_only.py
│  ├─ adapt_source_free.py
│  ├─ build_ablation_summary.py
│  └─ window_sweep.py
├─ requirements.txt
├─ README.md
└─ README_ru.md
````

---

## Installation

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows

```bash
py -3.11 -m venv .venv
.\\.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Quick Start

### 1. Generate synthetic raw records

```bash
python -m src.prepare_dummy_records
```

### 2. Build windowed dataset and processed manifest

```bash
python -m src.prepare_data
```

### 3. Check dataset balance and integrity

```bash
python -m src.check_dataset
```

### 4. Train source-only baselines

```bash
python -m src.train_source_only --variant raw_only
python -m src.train_source_only --variant tfr_only
python -m src.train_source_only --variant fused
```

### 5. Evaluate source-only models

```bash
python -m src.evaluate_source_only --variant raw_only
python -m src.evaluate_source_only --variant tfr_only
python -m src.evaluate_source_only --variant fused
```

### 6. Run source-free adaptation for the fused model

```bash
python -m src.adapt_source_free
```

### 7. Build ablation summary

```bash
python -m src.build_ablation_summary
```

### 8. Run window-size / stride sweep

```bash
python -m src.window_sweep
```

---

## Main Output Files

### Raw data

* `data/raw/records.csv`

### Processed data

* `data/processed/manifest.csv`

### Source-only training outputs

* `experiments/source_only_training/raw_only/`
* `experiments/source_only_training/tfr_only/`
* `experiments/source_only_training/fused/`

Typical files:

* `model.pt`
* `summary.json`

### Source-only evaluation outputs

* `experiments/source_only_eval/`

### Adaptation outputs

* `experiments/source_free_adaptation/fused/`

Typical files:

* `summary.json`
* `adapt_history.csv`

### Sweep outputs

* `experiments/window_sweep/window_sweep_summary.csv`

### Ablation outputs

* `experiments/ablation_summary/ablation_summary.csv`

---

## How to Read Results

At the current stage, ranking-based metrics are more reliable than threshold-based metrics.

Most useful metrics:

* ROC-AUC
* PR-AUC

Threshold-based metrics such as F1 may change substantially after adaptation because the score distribution can shift even when ranking quality remains strong.

A good experimental pattern for this MVP is:

* source-only quality is high on the source domain,
* source-only quality degrades on the target domain,
* source-free adaptation partially restores target-domain quality,
* fused representation remains a meaningful adaptation candidate even if it is more sensitive to shift before adaptation.

---

## Current Limitations

The repository is still an MVP and has several important limitations.

* The dataset is synthetic.
* The protocol is still relatively small.
* Event-level metrics are not yet implemented.
* The current target shift is still simplified.
* Threshold calibration remains unstable across some runs.
* Stronger temporal distortions are not yet enabled in the final protocol because interval remapping is not yet implemented.

---

## Next Steps

Planned next steps:

1. strengthen the domain-shift protocol,
2. improve adaptation with better pseudo-label selection and calibration,
3. move from window-level evaluation to event-level evaluation,
4. test on more realistic non-stationary signals,
5. prepare a compact demonstration and slides.

---

## Reproducibility

The repository is designed as a config-driven experimental pipeline.

Main configuration file:

* `configs/base.yaml`

To keep experiments reproducible:

* avoid editing multiple scripts at once,
* change protocol parameters through config where possible,
* store outputs in separate experiment directories,
* compare runs using saved summaries rather than console logs only.

---

## Language

Russian documentation is available in `README_ru.md`.


[1]: https://github.com/PeregrinAl/cross-domain-pipeline "GitHub - PeregrinAl/cross-domain-pipeline · GitHub"
