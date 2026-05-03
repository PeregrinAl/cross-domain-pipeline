from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    tier: str
    purpose: str
    default_representations: tuple[str, ...]
    default_adaptations: tuple[str, ...]
    mvp: bool
    reason: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["default_representations"] = list(self.default_representations)
        data["default_adaptations"] = list(self.default_adaptations)
        return data


@dataclass(frozen=True)
class RepresentationSpec:
    name: str
    kind: str
    requires_multichannel: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AdaptationSpec:
    name: str
    kind: str
    requires_target_domain: bool
    requires_unlabeled_target_adapt: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def get_model_registry() -> dict[str, ModelSpec]:
    return {
        "statistical_threshold": ModelSpec(
            name="statistical_threshold",
            family="statistical_classical",
            tier="sanity_baseline",
            purpose="Fast lower-bound baseline based on simple signal statistics.",
            default_representations=("handcrafted",),
            default_adaptations=("threshold_recalibration",),
            mvp=True,
            reason="Cheap sanity check before neural models.",
        ),
        "pca_reconstruction": ModelSpec(
            name="pca_reconstruction",
            family="statistical_classical",
            tier="sanity_baseline",
            purpose="Classical reconstruction-error anomaly baseline.",
            default_representations=("handcrafted", "raw_only"),
            default_adaptations=("threshold_recalibration",),
            mvp=True,
            reason="Strong lightweight baseline for reconstruction-style anomaly scoring.",
        ),
        "isolation_forest": ModelSpec(
            name="isolation_forest",
            family="statistical_classical",
            tier="sanity_baseline",
            purpose="Unsupervised anomaly detection on handcrafted window features.",
            default_representations=("handcrafted",),
            default_adaptations=("none", "threshold_recalibration"),
            mvp=True,
            reason="Fast non-neural baseline without epochs.",
        ),
        "conv1d_autoencoder": ModelSpec(
            name="conv1d_autoencoder",
            family="reconstruction_deep",
            tier="core",
            purpose="Reconstruction-based anomaly detector on raw temporal windows.",
            default_representations=("raw_only",),
            default_adaptations=("none", "threshold_recalibration"),
            mvp=True,
            reason="Core anomaly detection model for raw signals.",
        ),
        "tfr_autoencoder": ModelSpec(
            name="tfr_autoencoder",
            family="reconstruction_deep",
            tier="core",
            purpose="Reconstruction-based anomaly detector on time-frequency windows.",
            default_representations=("tfr_only",),
            default_adaptations=("none", "threshold_recalibration"),
            mvp=True,
            reason="Core anomaly detection model for STFT/CWT-like representations.",
        ),
        "fusion_autoencoder": ModelSpec(
            name="fusion_autoencoder",
            family="time_frequency_hybrid",
            tier="core",
            purpose="Reconstruction-based anomaly detector using raw and time-frequency views.",
            default_representations=("fused",),
            default_adaptations=("none", "threshold_recalibration", "sfda_prototype"),
            mvp=True,
            reason="Main default candidate for the methodology.",
        ),
        "tcn_prediction": ModelSpec(
            name="tcn_prediction",
            family="prediction_deep",
            tier="conditional",
            purpose="Prediction-error anomaly detector for temporally dependent signals.",
            default_representations=("raw_only",),
            default_adaptations=("none", "threshold_recalibration"),
            mvp=True,
            reason="Useful when windows are long enough and temporal dependency matters.",
        ),
        "graph_hybrid": ModelSpec(
            name="graph_hybrid",
            family="graph_hybrid",
            tier="conditional",
            purpose="Model inter-channel or inter-sensor relations.",
            default_representations=("multichannel_raw", "multichannel_fused"),
            default_adaptations=("none", "threshold_recalibration"),
            mvp=False,
            reason="Only justified for meaningful multichannel sensor structure.",
        ),
        "patch_transformer": ModelSpec(
            name="patch_transformer",
            family="transformer",
            tier="conditional",
            purpose="Model long-range temporal dependencies.",
            default_representations=("raw_only", "fused"),
            default_adaptations=("none", "threshold_recalibration"),
            mvp=False,
            reason="Only justified for long context and enough windows.",
        ),
        "foundation_model": ModelSpec(
            name="foundation_model",
            family="foundation",
            tier="research_only",
            purpose="Optional future zero-shot or transfer baseline.",
            default_representations=("raw_only",),
            default_adaptations=("none",),
            mvp=False,
            reason="Too heavy and unstable for MVP.",
        ),
        "diffusion_model": ModelSpec(
            name="diffusion_model",
            family="diffusion",
            tier="research_only",
            purpose="Optional future generative anomaly model.",
            default_representations=("raw_only",),
            default_adaptations=("none",),
            mvp=False,
            reason="Too heavy for the current library core.",
        ),
        "llm_detector": ModelSpec(
            name="llm_detector",
            family="llm_related",
            tier="research_only",
            purpose="Not a core detector; useful only for reporting.",
            default_representations=("report_text",),
            default_adaptations=("none",),
            mvp=False,
            reason="LLM should not be used as the numerical anomaly detector.",
        ),
    }


def get_representation_registry() -> dict[str, RepresentationSpec]:
    return {
        "handcrafted": RepresentationSpec(
            name="handcrafted",
            kind="tabular_features",
            requires_multichannel=False,
            reason="Cheap statistical and spectral features per window.",
        ),
        "raw_only": RepresentationSpec(
            name="raw_only",
            kind="raw_temporal",
            requires_multichannel=False,
            reason="Existing raw temporal branch.",
        ),
        "tfr_only": RepresentationSpec(
            name="tfr_only",
            kind="time_frequency",
            requires_multichannel=False,
            reason="Existing time-frequency branch.",
        ),
        "fused": RepresentationSpec(
            name="fused",
            kind="raw_plus_time_frequency",
            requires_multichannel=False,
            reason="Existing fused raw/time-frequency branch.",
        ),
        "multichannel_raw": RepresentationSpec(
            name="multichannel_raw",
            kind="multichannel_raw_temporal",
            requires_multichannel=True,
            reason="Only valid when signals preserve several physical channels.",
        ),
        "multichannel_fused": RepresentationSpec(
            name="multichannel_fused",
            kind="multichannel_raw_plus_time_frequency",
            requires_multichannel=True,
            reason="Only valid when several physical channels and TFR views are available.",
        ),
    }


def get_adaptation_registry() -> dict[str, AdaptationSpec]:
    return {
        "none": AdaptationSpec(
            name="none",
            kind="baseline",
            requires_target_domain=False,
            requires_unlabeled_target_adapt=False,
            reason="No adaptation; required baseline.",
        ),
        "threshold_recalibration": AdaptationSpec(
            name="threshold_recalibration",
            kind="calibration",
            requires_target_domain=False,
            requires_unlabeled_target_adapt=False,
            reason="Cheap and usually necessary threshold handling.",
        ),
        "sfda_prototype": AdaptationSpec(
            name="sfda_prototype",
            kind="source_free_domain_adaptation",
            requires_target_domain=True,
            requires_unlabeled_target_adapt=True,
            reason="Matches the current source-free adaptation direction.",
        ),
        "contrastive_alignment": AdaptationSpec(
            name="contrastive_alignment",
            kind="domain_alignment",
            requires_target_domain=True,
            requires_unlabeled_target_adapt=True,
            reason="Conditional method for visible source-target shift.",
        ),
        "adversarial_dann": AdaptationSpec(
            name="adversarial_dann",
            kind="domain_alignment",
            requires_target_domain=True,
            requires_unlabeled_target_adapt=True,
            reason="Later extension; potentially unstable for MVP.",
        ),
    }