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

- on synthetic controlled shift, fused source-only is the strongest baseline and source-free adaptation improves target-side metrics;
- on the current Paderborn pilot shift, target-side performance saturates, so this run should be treated mainly as a pipeline-transfer pilot rather than as an informative representation benchmark;
- on the current MIMII DUE supervised pilot, target-side performance does not saturate, and representation comparison becomes informative;
- on the current MIMII DUE supervised pilot, different representations perform best under different metrics, so fused should not be claimed as a universal real-data winner;
- on the current MIMII DUE supervised pilot, source-free adaptation does not provide a consistent real-data improvement across metrics;
- therefore the safest contribution is a generic methodology for evaluating anomaly-recognition pipelines on nonstationary signals, including representation choice, optional transfer, threshold calibration, and strict evaluation.

## What still needs more evidence

The following points still require additional experiments:

- a broader matrix of real operating-condition shifts and machine domains;
- stronger evidence on when adaptation is actually useful and when source-only transfer is already sufficient;
- stronger event-level evidence on real data;
- longer real-data runs for stability analysis beyond short pilot training schedules.