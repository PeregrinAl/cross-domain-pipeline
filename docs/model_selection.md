# Evidence-constrained model selection

This project does not use a blind model zoo benchmark.

The benchmark pipeline first builds a lightweight data profile and then restricts the set of candidate models, representations, and adaptation methods according to observable properties of the dataset.

The goal is not to assume that one model family is universally superior. The goal is to avoid physically or methodologically unjustified comparisons while still keeping a fair benchmark among applicable candidates.

## Data profile

The data profile is computed from the record-level CSV or from the processed window manifest.

It includes:

- number of rows, records, and windows;
- domains and splits;
- label balance;
- presence of source and target domains;
- presence of unlabeled target adaptation data;
- number of signal channels;
- signal length statistics;
- stationarity score;
- periodicity score;
- spectral concentration;
- channel correlation score;
- approximate source-target shift score.

The profile is produced with:

```bash
cdp profile-data --config configs/base.yaml --source raw
````

or:

```bash
python -m cross_domain_pipeline.cli profile-data --config configs/base.yaml --source raw
```

## Model applicability filter

The profile is used to form an evidence-constrained candidate set.

This filter does not select the final best model. It only excludes candidates that are not justified for the current data.

For example:

* graph-hybrid models are not recommended for single-channel signals;
* graph-hybrid models require meaningful multichannel relations;
* transformer models are not recommended unless long context and enough windows are available;
* source-free adaptation requires a target/adapt split;
* LLM-related methods are not used as numerical anomaly detectors.

## Model families

The library separates model families into four groups.

### Sanity baselines

These models are cheap and should usually be included:

* statistical thresholding;
* PCA reconstruction;
* Isolation Forest.

They provide a lower-bound reference and help detect cases where a neural model is unnecessary.

### Core anomaly detection models

These are the main MVP models:

* Conv1D autoencoder;
* time-frequency autoencoder;
* raw/time-frequency fusion autoencoder;
* TCN prediction model.

The default direction is reconstruction-based and prediction-based anomaly scoring, not supervised binary classification.

### Conditional advanced models

These models are included only when the data profile supports them:

* graph-hybrid models;
* patch/sequence transformer models;
* contrastive alignment methods.

Graph models require meaningful multichannel structure. Transformer models require enough data and a reason to model long-range dependencies.

### Research-only extensions

These methods are not part of the MVP:

* foundation models;
* diffusion models;
* LLM-based detectors.

They may be useful in future research, but they are too heavy or insufficiently justified for the first stable library version.

## Domain adaptation

Adaptation methods are selected separately from detector models.

The current protocol includes:

* no adaptation;
* threshold recalibration;
* source-free prototype adaptation;
* contrastive alignment, only when source-target shift is visible and target/adapt data exists.

The no-adaptation baseline is always required. Adaptation is only considered useful if it improves target-domain quality under the same evaluation protocol.

## Benchmark plan

A benchmark plan is built with:

```bash
cdp plan-benchmark --config configs/base.yaml --source raw
```

or:

```bash
python -m cross_domain_pipeline.cli plan-benchmark \
  --config configs/base.yaml \
  --source raw \
  --output experiments/benchmark_plan_base.json
```

The output contains:

* recommended model families;
* recommended adaptation methods;
* concrete benchmark candidates;
* excluded models with reasons.

## Academic position

This protocol should be described as:

> evidence-constrained benchmark planning

or:

> domain-informed candidate selection

It should not be described as a hard expert system.

The protocol does not claim to know the best model before experiments. It only prevents unjustified exhaustive search and makes the benchmark more interpretable.

````

# 3. Добавь ссылку в `README.md`

В конец раздела с quickstart или после описания methodology добавь:

```markdown
## Model selection protocol

The project uses an evidence-constrained model selection protocol instead of a blind model zoo benchmark.

See:

```text
docs/model_selection.md
````

The protocol builds a lightweight data profile first and then selects a compact set of justified model families, representations, and adaptation methods.


