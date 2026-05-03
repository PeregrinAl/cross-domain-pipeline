from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .data_profile import DataProfile
from .model_registry import (
    get_adaptation_registry,
    get_model_registry,
    get_representation_registry,
)


@dataclass(frozen=True)
class BenchmarkCandidate:
    model: str
    family: str
    tier: str
    representation: str
    adaptation: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkPlan:
    mode: str
    source_csv: str
    source_type: str
    n_rows: int
    n_records: int
    n_windows: int
    likely_multichannel: bool
    recommended_model_families: list[str]
    recommended_adaptation_methods: list[str]
    candidates: list[BenchmarkCandidate]
    excluded: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(), encoding="utf-8")


def build_benchmark_plan(
    profile: DataProfile,
    mode: str = "compact",
    max_candidates: int = 12,
    include_research_only: bool = False,
) -> BenchmarkPlan:
    if mode not in {"compact", "extended"}:
        raise ValueError("mode must be one of: compact, extended")

    models = get_model_registry()
    representations = get_representation_registry()
    adaptations = get_adaptation_registry()

    allowed_models = set(profile.recommended_model_families)
    allowed_adaptations = set(profile.recommended_adaptation_methods)

    candidates: list[BenchmarkCandidate] = []
    excluded = dict(profile.excluded_model_families)

    for model_name in _ordered_model_names(mode=mode):
        model = models[model_name]

        if model.tier == "research_only" and not include_research_only:
            excluded[model_name] = model.reason
            continue

        if model_name not in allowed_models:
            excluded.setdefault(model_name, "Model was not recommended by the data profile.")
            continue

        for representation_name in model.default_representations:
            representation = representations[representation_name]

            if representation.requires_multichannel and not profile.likely_multichannel:
                excluded[f"{model_name}:{representation_name}"] = (
                    "Representation requires multichannel input, but profile is not multichannel."
                )
                continue

            for adaptation_name in model.default_adaptations:
                adaptation = adaptations[adaptation_name]

                if adaptation_name not in allowed_adaptations:
                    excluded[f"{model_name}:{adaptation_name}"] = (
                        "Adaptation method was not recommended by the data profile."
                    )
                    continue

                if adaptation.requires_target_domain and not profile.has_target_domain:
                    excluded[f"{model_name}:{adaptation_name}"] = (
                        "Adaptation requires target domain."
                    )
                    continue

                if adaptation.requires_unlabeled_target_adapt and not profile.has_unlabeled_target_adapt:
                    excluded[f"{model_name}:{adaptation_name}"] = (
                        "Adaptation requires unlabeled target/adapt split."
                    )
                    continue

                candidates.append(
                    BenchmarkCandidate(
                        model=model.name,
                        family=model.family,
                        tier=model.tier,
                        representation=representation.name,
                        adaptation=adaptation.name,
                        reason=_candidate_reason(model.name, representation.name, adaptation.name),
                    )
                )

                if mode == "compact" and len(candidates) >= max_candidates:
                    return _make_plan(profile, candidates, excluded, mode)

    return _make_plan(profile, candidates, excluded, mode)


def _ordered_model_names(mode: str) -> list[str]:
    base = [
        "statistical_threshold",
        "pca_reconstruction",
        "isolation_forest",
        "conv1d_autoencoder",
        "tfr_autoencoder",
        "fusion_autoencoder",
        "tcn_prediction",
        "graph_hybrid",
        "patch_transformer",
    ]

    if mode == "extended":
        return base + [
            "foundation_model",
            "diffusion_model",
            "llm_detector",
        ]

    return base


def _candidate_reason(model: str, representation: str, adaptation: str) -> str:
    if model in {"statistical_threshold", "pca_reconstruction", "isolation_forest"}:
        return "Cheap sanity baseline."

    if model == "fusion_autoencoder":
        return "Main raw/time-frequency reconstruction candidate."

    if model == "conv1d_autoencoder":
        return "Core raw-signal reconstruction candidate."

    if model == "tfr_autoencoder":
        return "Core time-frequency reconstruction candidate."

    if model == "tcn_prediction":
        return "Prediction-error candidate for temporal dependency."

    if model == "graph_hybrid":
        return "Conditional candidate for meaningful multichannel relations."

    if model == "patch_transformer":
        return "Conditional candidate for long-context dependency."

    return f"{model} with {representation} and {adaptation}."


def _make_plan(
    profile: DataProfile,
    candidates: list[BenchmarkCandidate],
    excluded: dict[str, str],
    mode: str,
) -> BenchmarkPlan:
    return BenchmarkPlan(
        mode=mode,
        source_csv=profile.source_csv,
        source_type=profile.source_type,
        n_rows=profile.n_rows,
        n_records=profile.n_records,
        n_windows=profile.n_windows,
        likely_multichannel=profile.likely_multichannel,
        recommended_model_families=profile.recommended_model_families,
        recommended_adaptation_methods=profile.recommended_adaptation_methods,
        candidates=candidates,
        excluded=excluded,
    )