# Generic Adaptation and Evaluation Pipeline for Nonstationary Signals

This repository implements a reproducible experimental pipeline for cross-domain anomaly recognition in nonstationary signals.

The project is intentionally organized as an **incremental framework**, not as a claim of a new adaptation algorithm. The current emphasis is on building, validating, and extending a generic pipeline that combines:

- raw temporal views,
- time-frequency views,
- source-only training,
- source-free adaptation,
- threshold calibration,
- and clear domain-shift evaluation protocols.

---

### Current empirical takeaway

At the current stage, the repository supports a cautious claim:
representation choice, threshold calibration, and strict evaluation are central parts of the methodology, while source-free adaptation is treated as an optional transfer component whose usefulness depends on the actual domain shift.

Current results show that:
- on synthetic controlled shift, fused source-only is the strongest baseline and SFDA improves target-side metrics;
- on the current Paderborn real-data shift, source-only fused already reaches ceiling-level performance, so adaptation is not the main source of gain;
- therefore the main contribution is not a new adaptation algorithm, but a generic evaluation methodology for nonstationary-signal anomaly recognition.

## Project Status

The repository is currently organized into two stages:

### 1. Synthetic justification stage
This stage is used to justify the representation and adaptation choices.

It includes:
- synthetic signal generation,
- windowing,
- raw/STFT views,
- source-only training,
- minimal source-free adaptation,
- threshold calibration,
- synthetic ablation summary,
- and strict event-level evaluation on synthetic data.

### 2. Real-data vibration stage
This stage tests whether the same framework transfers to real vibration data without changing the core architecture.

The first real-data pilot currently uses the **Paderborn bearing dataset** in a binary setup:
- modality: vibration only,
- task: healthy vs damaged,
- current condition shift: `N15_M07_F10 -> N09_M07_F10`.

At this stage, the real-data part should be treated as a **pilot protocol**, not as a final benchmark.

---

## Current Position in the Roadmap

### Completed
- synthetic cross-domain protocol
- raw/STFT representation branch
- source-only baselines
- minimal source-free adaptation
- threshold calibration
- synthetic ablation comparison
- experiment run separation into dedicated folders
- first real-data vibration pilot setup

### Current focus
The project is currently in the **first real-data vibration evaluation stage**.

The synthetic stage is retained as a justification layer:
- `raw_only`, `tfr_only`, and `fused` are compared on synthetic cross-domain transfer,
- `fused` is used as the main representation in the current real-data stage because it was the strongest synthetic source-only baseline and the only representation currently used for source-free adaptation.

### Planned next steps
- expand the first Paderborn pilot from a smoke-test split to a fuller condition-level run,
- repeat the same protocol for additional operating-condition shifts,
- only after that consider broader extensions such as multi-source, multiclass, or cross-modality transfer.

---

## Objective

The current project aims to show that a generic adaptation and evaluation pipeline for nonstationary signals can be built and tested in a principled way.

More specifically, the repository currently studies whether:

- a model trained only on the source domain degrades on a shifted target domain,
- source-free adaptation can recover part of the lost target quality without using source data during adaptation,
- a fused representation built from raw and STFT views is a strong basis for transfer,
- and the same framework can move from synthetic data to real vibration data with minimal architectural changes.

---

## Synthetic Stage

### Synthetic pipeline

```text
Synthetic signal generation
-> Windowing
-> Raw/STFT views
-> Source-only training
-> Source-free adaptation
-> Ablation summary
-> Event-level evaluation
````

### Synthetic data protocol

The synthetic setup uses generated `.npy` signals and a record-level CSV.

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
* `record_label`
* `domain`
* `record_id`
* `split`
* `window_idx`
* `start`
* `end`

### Default synthetic window protocol

* `window_size = 2048`
* `stride = 512`
* `window_label_mode = min_fraction`
* `min_anomaly_fraction = 0.05`

### Synthetic domains and splits

* `source/train` — labeled source-domain training windows
* `source/val` — labeled source-domain validation windows
* `target/test` — labeled target-domain evaluation windows
* `target/adapt` — unlabeled target-domain windows used for source-free adaptation

### Synthetic target-domain shift

The current synthetic target shift includes:

* amplitude and scale shift,
* extra noise,
* trend drift,
* frequency shift,
* and temporal warp.

---

## Synthetic Baseline Justification

### Representations

* `raw_only` — raw temporal windows
* `tfr_only` — STFT-based time-frequency representation
* `fused` — joint representation built from raw and STFT branches

### Synthetic-stage conclusion

The synthetic stage is used as the justification layer for the current real-data direction.

The working interpretation is:

* source-only transfer degrades under domain shift,
* `fused` is the strongest source-only baseline on the synthetic target,
* source-free adaptation substantially improves the fused model,
* therefore `fused` is used as the main representation in the current real-data stage.

The repository keeps a dedicated saved run for this purpose under the synthetic experiment folders.

---

## Real-Data Vibration Pilot

### Current real-data setup

The first real-data pilot uses the Paderborn bearing dataset.

Current pilot characteristics:

* modality: vibration only,
* task: binary healthy vs damaged,
* source condition: `N15_M07_F10`,
* target condition: `N09_M07_F10`.

### Current real-data goal

The current goal is not yet a final benchmark.

Instead, the goal is to verify that:

* the same preprocessing and windowing pipeline works on real vibration data,
* the same source-only and source-free training scripts can run without architectural changes,
* and experiments can be stored in condition-specific run folders for later comparison.

### Current limitation

The present Paderborn run should be treated as a pilot rather than a definitive result.
The next step is to expand the split so that each selected bearing code contributes multiple measurement files while preserving clean separation across train, validation, adaptation, and test.

---

## Implemented Components

### Data

* synthetic signal generation
* record-level anomaly interval storage
* window-level labeling from record metadata
* processed manifest generation for window-based experiments
* first Paderborn vibration ingestion path via `.mat -> .npy`

### Representations

* `raw_only`
* `tfr_only`
* `fused`

### Models

* `RawEncoder`
* `TFREncoder`
* `FusionEncoder`
* `SourceOnlyClassifier`

### Training and evaluation

* source-only training for `raw_only`, `tfr_only`, and `fused`
* threshold calibration from source validation
* minimal source-free adaptation for `fused`
* synthetic ablation summary builder
* synthetic event-level evaluation
* run-specific output tracking for saved experiments

---

## Experiment Tracking

Experiments are saved in run-specific folders rather than overwritten in a single location.

Typical layout:

```text
experiments/
  synthetic/
    <experiment_name>/
      <run_name>/
        config_snapshot.yaml
        records_snapshot.csv
        source_only_training/
        source_free_adaptation/
  paderborn/
    <experiment_name>/
      <run_name>/
        config_snapshot.yaml
        records_snapshot.csv
        source_only_training/
        source_free_adaptation/
```

Each run stores:

* config snapshot,
* records snapshot,
* score files,
* training history,
* and summary metrics.

This makes synthetic and real-data runs easier to compare and prevents accidental overwriting.

---

## Repository Structure

```text
cross-domain-pipeline/
├─ configs/
│  ├─ base.yaml
│  ├─ paderborn_binary.yaml
│  └─ synthetic_fused_justification.yaml
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ ...
├─ experiments/
│  ├─ synthetic/
│  └─ paderborn/
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
│  ├─ build_ablation_summary.py
│  ├─ evaluate_event_level.py
│  └─ build_event_level_summary.py
├─ tools/
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

## A. Synthetic stage

### 1. Generate synthetic records

```bash
python -m src.prepare_dummy_records
```

### 2. Build the processed manifest

```bash
python -m src.prepare_data --config configs/synthetic_fused_justification.yaml
```

### 3. Train source-only baselines

```bash
python -m src.train_source_only --config configs/synthetic_fused_justification.yaml --variant raw_only
python -m src.train_source_only --config configs/synthetic_fused_justification.yaml --variant tfr_only
python -m src.train_source_only --config configs/synthetic_fused_justification.yaml --variant fused
```

### 4. Run source-free adaptation for fused

```bash
python -m src.adapt_source_free --config configs/synthetic_fused_justification.yaml
```

### 5. Build the synthetic ablation summary

```bash
python -m src.build_ablation_summary
```

### 6. Optional synthetic event-level evaluation

```bash
python -m src.evaluate_event_level --stage source_only --variant fused --min-iou 0.05
python -m src.evaluate_event_level --stage sfda_after --variant fused --min-iou 0.05
python -m src.build_event_level_summary
```

---

## B. Real-data Paderborn stage

### 1. Prepare raw real-data records

Create:

* converted vibration `.npy` files,
* and `data/raw/paderborn_records.csv`.

### 2. Build the processed manifest

```bash
python -m src.prepare_data --config configs/paderborn_binary.yaml
```

### 3. Train the fused source-only model

```bash
python -m src.train_source_only --config configs/paderborn_binary.yaml --variant fused
```

### 4. Run fused source-free adaptation

```bash
python -m src.adapt_source_free --config configs/paderborn_binary.yaml
```

---

## Key Output Files

### Source-only runs

For each saved run and variant:

* `history.csv`
* `summary.json`
* `source_val_scores.csv`
* `target_test_scores.csv`

### Source-free adaptation runs

For each saved fused adaptation run:

* `adapt_history.csv`
* `summary.json`
* `source_val_scores_before.csv`
* `source_val_scores_after.csv`
* `target_test_scores_before.csv`
* `target_test_scores_after.csv`

### Synthetic event-level outputs

When event-level evaluation is used:

* `summary.json`
* `matched_events.csv`
* `missed_events.csv`
* `false_alarm_events.csv`
* consolidated event-level summary tables

---

## Scope and Claims

This repository currently supports the following claims:

* a generic cross-domain anomaly-recognition pipeline can be built in a reproducible form,
* synthetic experiments justify the current focus on `fused`,
* the framework can already be executed on a first real vibration pilot without changing the core architecture.

This repository does **not** currently claim:

* a new adaptation algorithm,
* a final large-scale real-data benchmark,
* or a completed multi-source / multiclass / cross-modality study.

---

## Notes

* The project is intentionally developed through small, saved experimental runs.
* Synthetic experiments are used as a justification stage, not as the final destination.
* The current real-data stage is a pilot designed to support the next round of condition-level vibration experiments.
