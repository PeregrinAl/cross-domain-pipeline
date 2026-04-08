# Claims Scope

## What is currently claimed

This repository currently supports the following cautious claims:

- a generic and reproducible pipeline for cross-domain anomaly recognition in nonstationary signals can be built without changing the core architecture between synthetic and first real-data runs;
- representation choice is a central part of the methodology, and the fused raw plus STFT setup is currently the strongest working representation in the synthetic justification stage;
- source-free adaptation should be treated as an optional transfer component whose usefulness depends on the actual domain shift rather than as the main contribution;
- threshold calibration and strict event-level evaluation are necessary parts of a realistic assessment protocol.

## What is explicitly not claimed

This repository does not currently claim:

- a new source-free domain adaptation algorithm;
- novelty of time-frequency fusion by itself;
- novelty of pseudo-label filtering, prototype methods, curriculum adaptation, uncertainty calibration, or temporal consistency as standalone algorithmic contributions;
- a final large-scale real-data benchmark;
- a universal claim that adaptation always improves transfer.

## What current results already support

At the current stage, the results support the following interpretation:

- on synthetic controlled shift, `fused` is the strongest source-only baseline and minimal source-free adaptation improves target-side metrics;
- on the current Paderborn pilot shift, target-side performance is close to saturation, so this stage should be treated as a pipeline-transfer check rather than as the main comparative benchmark;
- on the current MIMII DUE supervised pilot, `raw_only`, `tfr_only`, and `fused` show different strengths across different metrics, so no universal representation winner should be claimed;
- on the same MIMII DUE pilot, source-free adaptation does not provide a stable and convincing gain across all target metrics;
- therefore the safest contribution is a generic methodology for evaluating anomaly-recognition pipelines on nonstationary signals, including representation choice, optional transfer, threshold handling, and strict evaluation.

## What still needs more evidence

The following points still require additional experiments:

- broader real-data comparison of `raw_only`, `tfr_only`, and `fused` across more than one real shift;
- a broader matrix of real operating-condition and acquisition-condition shifts;
- stronger evidence for when adaptation is actually needed and when source-only transfer is already sufficient;
- event-level evidence on real data;
- validation of the methodology on more than one real signal domain beyond the current pilots.