from __future__ import annotations

from src.preprocessing.methods import (
    BasePreprocessor,
    DomainNormPreprocessor,
    FilterPreprocessor,
)
from src.preprocessing.standard import StandardSignalPreprocessor


def build_preprocessor(name: str, config: dict | None = None):
    config = config or {}

    if name in {"none", None}:
        return None

    if name == "standard":
        return StandardSignalPreprocessor(**config)

    if name == "prep_base":
        return BasePreprocessor(**config)

    if name == "prep_filter":
        return FilterPreprocessor(**config)

    if name == "prep_domain_norm":
        return DomainNormPreprocessor(**config)

    raise ValueError(f"Unknown preprocessing method: {name}")