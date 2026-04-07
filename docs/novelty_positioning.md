# Positioning Relative to Related Work

## Purpose of this note

This document fixes the safe positioning of the project relative to the closest papers.

The repository should be positioned as a **generic adaptation and evaluation pipeline for cross-domain anomaly recognition in nonstationary signals**, not as a claim of a new adaptation algorithm. The current project combines multi-view signal representations, source-only baselines, optional source-free adaptation, threshold calibration, and stricter evaluation protocols. Fault diagnosis is treated as one validation domain, not as the only target application.

## Safe focus of this work

The safe focus of the work is a **methodology** for anomaly recognition in nonstationary signals under changing acquisition conditions.

This methodology is centered on:
- informative signal representations;
- source-only baselines as a mandatory reference point;
- optional source-free transfer;
- threshold calibration;
- stricter anomaly evaluation;
- reproducible movement from synthetic justification to real-data validation.

The work should therefore be framed as an **integrated methodological pipeline** rather than a new transfer-learning algorithm.

## Claims that should NOT be used

The project should **not** claim:
- a new source-free adaptation algorithm for time series;
- novelty simply because it combines time-domain and time-frequency representations;
- novelty in pseudo-label filtering, prototype memory, curriculum learning, or privacy-preserving source-free fault diagnosis;
- novelty in uncertainty-aware source-free time-series adaptation by itself;
- novelty in multi-view fusion for cross-condition bearing diagnosis by itself;
- novelty in fair large-scale benchmarking of domain adaptation for time series.

These claims are already too close to existing work.

## Closest and additional relevant papers

### 1. Furqon et al., 2025  
**Time and Frequency Synergy for Source-Free Time-Series Domain Adaptations (TFDA)**  
Link: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0020025524016487)  
Alternative link: [arXiv](https://arxiv.org/abs/2410.17511)

TFDA proposes a dedicated **source-free time-series domain adaptation** method with:
- dual time/frequency branches;
- contrastive learning in time and frequency spaces;
- time-frequency consistency;
- self-distillation;
- uncertainty reduction;
- curriculum learning for noisy pseudo-labels.

**What claim this paper already occupies:**  
Novelty based on combining time and frequency information inside a new SFDA algorithm for time series.

**Difference from this repository:**  
The present project does not propose a new adaptation algorithm. The safer contribution is a broader methodology for anomaly recognition in nonstationary signals, where representation choice, calibration, and stricter evaluation are first-class components.

### 2. Tao et al., 2023  
**Unsupervised Cross-Domain Rolling Bearing Fault Diagnosis Based on Time-Frequency Information Fusion**  
Link: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0016003222008055)

This paper proposes an **unsupervised cross-domain rolling bearing fault diagnosis** method based on **time-frequency information fusion**.

**What claim this paper already occupies:**  
Novelty based on cross-domain bearing diagnosis through time-frequency fusion under changing operating conditions.

**Difference from this repository:**  
The present project is not framed as a bearing-specific diagnosis model. It is framed as a more general methodology for anomaly recognition in nonstationary signals, where vibration fault diagnosis is one validation scenario.

### 3. Li et al., 2023  
**Source-Free Domain Adaptation Framework for Fault Diagnosis of Rotation Machinery under Data Privacy**  
Link: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0951832023003824)

This paper proposes a **source-free fault-diagnosis framework** under **data privacy** constraints, with:
- adaptive entropy thresholds;
- prototype memory;
- confidence-based pseudo-label filtering;
- curriculum learning.

**What claim this paper already occupies:**  
Novelty based on privacy-aware source-free fault diagnosis with sophisticated pseudo-label management and prototype-based target adaptation.

**Difference from this repository:**  
The present project is not centered on privacy-preserving adaptation and should not claim novelty in this target-side mechanism. The safer contribution is methodological integration and evaluation.

### 4. Li et al., 2025  
**Adaptive Multi-View Hypergraph Learning for Cross-Condition Bearing Fault Diagnosis (AMH)**  
Link: [MDPI](https://www.mdpi.com/2504-4990/7/4/147)

This paper proposes an **adaptive multi-view hypergraph learning** framework for **cross-condition bearing fault diagnosis**. It combines multiple views and explicitly models higher-order feature relationships.

**What claim this paper already occupies:**  
Novelty based on multi-view fusion and structured feature interaction for cross-condition bearing diagnosis.

**Difference from this repository:**  
The present project is not centered on a new hypergraph-based diagnosis architecture and is not limited to bearing diagnosis. The safer contribution remains a generic methodology for anomaly recognition in nonstationary signals.

### 5. Ragab et al., 2024  
**Evidentially Calibrated Source-Free Time-Series Domain Adaptation with Temporal Imputation**  
Link: [arXiv](https://arxiv.org/abs/2406.02635)

This paper introduces:
- **MAPU**: temporal imputation for source-free time-series adaptation;
- **E-MAPU**: evidential uncertainty calibration for better-calibrated source-free adaptation.

**What claim this paper already occupies:**  
Novelty based on explicit temporal consistency and evidential uncertainty calibration as a new SFDA mechanism for time series.

**Difference from this repository:**  
The present project does not claim a new temporal-imputation-based or evidentially calibrated adaptation algorithm. If similar components are added later, they should be presented only as optional enhancements inside the broader methodology.

### 6. Patel et al., 2024  
**Efficient Source-Free Time-Series Adaptation via Parameter Subspace Disentanglement**  
Link: [arXiv](https://arxiv.org/abs/2410.02147)  
Alternative link: [OpenReview](https://openreview.net/forum?id=Q5Sawm0nqo)

This paper proposes an **efficient source-free time-series adaptation** framework focused on:
- parameter efficiency;
- sample efficiency;
- source-model reparameterization via Tucker-style decomposition;
- selective fine-tuning of decomposed factors.

**What claim this paper already occupies:**  
Novelty based on efficient, compact, and computation-aware SFDA for time series.

**Difference from this repository:**  
The present project is not focused on computational efficiency, compression, or lightweight adaptation. The core contribution remains methodological integration and evaluation.

### 7. Ragab et al., 2023  
**ADATIME: A Benchmarking Suite for Domain Adaptation on Time Series Data**  
Link: [arXiv](https://arxiv.org/abs/2203.08321)  
Alternative link: [ACM](https://dl.acm.org/doi/10.1145/3587937)  
Code: [GitHub](https://github.com/emadeldeen24/AdaTime)

AdaTime is a **benchmarking suite** for time-series domain adaptation. It standardizes:
- datasets;
- backbones;
- model-selection protocols;
- realistic transfer evaluation settings.

**What claim this paper already occupies:**  
Novelty based on fair and systematic large-scale benchmarking of domain adaptation methods for time series.

**Difference from this repository:**  
The present project is not a benchmarking suite. It is a reproducible anomaly-recognition pipeline focused on methodological integration and staged validation from synthetic to real data.

### 8. Shi et al., 2022  
**Deep Unsupervised Domain Adaptation with Time Series Sensor Data: A Survey**  
Link: [MDPI Sensors](https://www.mdpi.com/1424-8220/22/15/5507)  
Alternative link: [PubMed](https://pubmed.ncbi.nlm.nih.gov/35898010/)

This paper is a **survey** of deep unsupervised domain adaptation for time-series sensor data.

**What claim this paper already occupies:**  
Not a direct method claim, but it removes the possibility of presenting the field as largely unexplored.

**Difference from this repository:**  
The survey provides background and taxonomy, whereas the present work contributes an implemented and experimentally tested pipeline.

## That metology

This is a **generic methodology** for nonstationary-signal anomaly recognition under domain shift, rather than a new transfer algorithm.

The most important points are:
- combining representation choice, source-only baseline, optional transfer, calibration, and stricter evaluation in one reproducible pipeline;
- treating adaptation as **optional rather than universally necessary**;
- explicitly studying **when adaptation helps and when it does not**;
- validating the same methodological structure first on synthetic data and then on real data;
- keeping the framing broader than fault diagnosis alone.

## Novelty formulation

**This work develops and tests a methodology for anomaly recognition in nonstationary signals under changing acquisition conditions. The methodology combines informative multi-view signal representations, source-only baselines, optional source-free transfer, threshold calibration, and stricter event-level evaluation. The contribution is not a new adaptation algorithm, but an integrated and reproducible framework for determining which components improve transfer quality and stability, and under which shifts adaptation is actually beneficial.**

## Practical implication for future work

Future extensions should strengthen the **evidence base** of the methodology rather than move immediately toward a claim of a new algorithm.

The most useful next steps are:
- broader real-shift coverage;
- stronger comparative summaries across shifts;
- one careful enhancement of the transfer stage if needed;
- eventual validation on more than one signal domain.