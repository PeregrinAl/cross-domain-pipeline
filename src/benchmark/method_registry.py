from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    name: str
    group: str
    enabled: bool = True


PREPROCESSING_METHODS = {
    "prep_base": MethodSpec(
        name="prep_base",
        group="preprocessing",
    ),
    "prep_filter": MethodSpec(
        name="prep_filter",
        group="preprocessing",
    ),
    "prep_domain_norm": MethodSpec(
        name="prep_domain_norm",
        group="preprocessing",
    ),
}


REPRESENTATION_METHODS = {
    "raw_time": MethodSpec(
        name="raw_time",
        group="representation",
    ),
    "stft_spectrogram": MethodSpec(
        name="stft_spectrogram",
        group="representation",
    ),
    "cwt_scalogram": MethodSpec(
        name="cwt_scalogram",
        group="representation",
    ),
    "channel_graph": MethodSpec(
        name="channel_graph",
        group="representation",
    ),
    "fused_multiview": MethodSpec(
        name="fused_multiview",
        group="representation",
    ),
}


ADAPTATION_METHODS = {
    "source_only": MethodSpec(
        name="source_only",
        group="adaptation",
    ),
    "da_grl_adversarial": MethodSpec(
        name="da_grl_adversarial",
        group="adaptation",
    ),
    "da_discrepancy": MethodSpec(
        name="da_discrepancy",
        group="adaptation",
    ),
    "da_source_free_ttt": MethodSpec(
        name="da_source_free_ttt",
        group="adaptation",
    ),
}