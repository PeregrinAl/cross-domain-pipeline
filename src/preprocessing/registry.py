from src.preprocessing.methods import (
    BasePreprocessor,
    FilterPreprocessor,
    DomainNormPreprocessor,
)


def build_preprocessor(name: str, config: dict | None = None):
    config = config or {}

    if name == "prep_base":
        return BasePreprocessor(**config)

    if name == "prep_filter":
        return FilterPreprocessor(**config)

    if name == "prep_domain_norm":
        return DomainNormPreprocessor(**config)

    raise ValueError(f"Unknown preprocessing method: {name}")