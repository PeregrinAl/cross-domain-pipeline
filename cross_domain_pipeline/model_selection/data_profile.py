from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross_domain_pipeline.config import PipelineConfig


@dataclass(frozen=True)
class DataProfile:
    source_csv: str
    source_type: str

    n_rows: int
    n_records: int
    n_windows: int

    domains: list[str]
    splits: list[str]
    labels: list[int]
    class_balance: dict[str, int]

    has_labels: bool
    has_anomaly_intervals: bool
    has_source_domain: bool
    has_target_domain: bool
    has_unlabeled_target_adapt: bool

    n_sampled_arrays: int
    signal_ndim_values: list[int]
    n_channels_min: int | None
    n_channels_max: int | None
    length_min: int | None
    length_median: float | None
    length_max: int | None

    stationarity_score: float | None
    periodicity_score: float | None
    spectral_concentration: float | None
    channel_correlation_score: float | None
    source_target_shift_score: float | None

    likely_multichannel: bool
    long_context_required: bool

    recommended_model_families: list[str]
    recommended_adaptation_methods: list[str]
    excluded_model_families: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def profile_from_config(
    config: str | Path | PipelineConfig,
    source: str = "auto",
    max_files: int = 200,
) -> DataProfile:
    cfg = config if isinstance(config, PipelineConfig) else PipelineConfig.from_yaml(config)

    if source not in {"auto", "raw", "manifest"}:
        raise ValueError("source must be one of: auto, raw, manifest")

    data = cfg.data
    config_dir = cfg.path.parent if cfg.path is not None else Path.cwd()

    raw_csv = data.get("raw_records_csv")
    manifest_csv = data.get("manifest_path")

    if source == "raw":
        csv_path = raw_csv
        source_type = "raw_records"
    elif source == "manifest":
        csv_path = manifest_csv
        source_type = "window_manifest"
    else:
        manifest_candidate = _resolve_path(manifest_csv, config_dir) if manifest_csv else None
        if manifest_candidate is not None and manifest_candidate.exists():
            csv_path = manifest_csv
            source_type = "window_manifest"
        else:
            csv_path = raw_csv
            source_type = "raw_records"

    if csv_path is None:
        raise ValueError("Config does not contain data.raw_records_csv or data.manifest_path")

    return profile_from_csv(
        csv_path=csv_path,
        source_type=source_type,
        base_dir=config_dir,
        max_files=max_files,
    )


def profile_from_csv(
    csv_path: str | Path,
    source_type: str,
    base_dir: str | Path | None = None,
    max_files: int = 200,
) -> DataProfile:
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    csv_file = _resolve_path(csv_path, base)

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_file}")

    df = pd.read_csv(csv_file)

    if "path" not in df.columns:
        raise ValueError("CSV must contain column: path")

    domains = _unique_str(df, "domain")
    splits = _unique_str(df, "split")
    labels = _unique_int(df, "label")

    class_balance = {}
    if "label" in df.columns:
        class_balance = {
            str(k): int(v)
            for k, v in df["label"].value_counts(dropna=False).sort_index().items()
        }

    has_anomaly_intervals = "anomaly_intervals" in df.columns
    has_labels = "label" in df.columns
    has_source_domain = any(x.lower() == "source" for x in domains)
    has_target_domain = any(x.lower() == "target" for x in domains)

    has_unlabeled_target_adapt = False
    if {"domain", "split"}.issubset(df.columns):
        mask = (
            df["domain"].astype(str).str.lower().eq("target")
            & df["split"].astype(str).str.lower().eq("adapt")
        )
        has_unlabeled_target_adapt = bool(mask.any())

    sample_df = _sample_rows(df, max_files=max_files)

    array_infos = []
    feature_rows = []
    stationarity_scores = []
    periodicity_scores = []
    spectral_scores = []
    channel_corr_scores = []

    for _, row in sample_df.iterrows():
        signal_path = _resolve_path(row["path"], base)

        if not signal_path.exists():
            continue

        try:
            arr = np.load(signal_path)
        except Exception:
            continue

        channels_first = _to_channels_first(arr)
        n_channels, length = channels_first.shape

        array_infos.append(
            {
                "ndim": int(arr.ndim),
                "n_channels": int(n_channels),
                "length": int(length),
            }
        )

        feature = _basic_features(channels_first)
        feature["domain"] = str(row["domain"]) if "domain" in row else ""
        feature_rows.append(feature)

        stationarity_scores.append(_stationarity_score(channels_first[0]))
        periodicity_scores.append(_periodicity_score(channels_first[0]))
        spectral_scores.append(_spectral_concentration(channels_first[0]))

        if n_channels > 1:
            corr = _channel_correlation_score(channels_first)
            if corr is not None:
                channel_corr_scores.append(corr)

    n_sampled = len(array_infos)

    n_channels_values = [x["n_channels"] for x in array_infos]
    length_values = [x["length"] for x in array_infos]
    ndim_values = sorted(set(x["ndim"] for x in array_infos))

    n_channels_min = min(n_channels_values) if n_channels_values else None
    n_channels_max = max(n_channels_values) if n_channels_values else None
    length_min = min(length_values) if length_values else None
    length_max = max(length_values) if length_values else None
    length_median = float(np.median(length_values)) if length_values else None

    stationarity = _mean_or_none(stationarity_scores)
    periodicity = _mean_or_none(periodicity_scores)
    spectral = _mean_or_none(spectral_scores)
    channel_corr = _mean_or_none(channel_corr_scores)
    shift_score = _source_target_shift_score(feature_rows)

    likely_multichannel = bool(n_channels_max is not None and n_channels_max > 1)
    long_context_required = bool(length_median is not None and length_median >= 4096)

    recommended_models, excluded_models = _recommend_models(
        likely_multichannel=likely_multichannel,
        n_channels_max=n_channels_max,
        channel_correlation_score=channel_corr,
        n_rows=len(df),
        length_median=length_median,
        long_context_required=long_context_required,
    )

    recommended_adaptation = _recommend_adaptation_methods(
        has_target_domain=has_target_domain,
        has_unlabeled_target_adapt=has_unlabeled_target_adapt,
        source_target_shift_score=shift_score,
    )

    return DataProfile(
        source_csv=str(csv_file),
        source_type=source_type,
        n_rows=int(len(df)),
        n_records=int(df["record_id"].nunique()) if "record_id" in df.columns else int(len(df)),
        n_windows=int(len(df)) if source_type == "window_manifest" else 0,
        domains=domains,
        splits=splits,
        labels=labels,
        class_balance=class_balance,
        has_labels=has_labels,
        has_anomaly_intervals=has_anomaly_intervals,
        has_source_domain=has_source_domain,
        has_target_domain=has_target_domain,
        has_unlabeled_target_adapt=has_unlabeled_target_adapt,
        n_sampled_arrays=n_sampled,
        signal_ndim_values=ndim_values,
        n_channels_min=n_channels_min,
        n_channels_max=n_channels_max,
        length_min=length_min,
        length_median=length_median,
        length_max=length_max,
        stationarity_score=stationarity,
        periodicity_score=periodicity,
        spectral_concentration=spectral,
        channel_correlation_score=channel_corr,
        source_target_shift_score=shift_score,
        likely_multichannel=likely_multichannel,
        long_context_required=long_context_required,
        recommended_model_families=recommended_models,
        recommended_adaptation_methods=recommended_adaptation,
        excluded_model_families=excluded_models,
    )


def _resolve_path(path: str | Path, base_dir: Path) -> Path:
    p = Path(path)

    if p.is_absolute():
        return p

    if p.exists():
        return p

    return base_dir / p


def _unique_str(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []

    return sorted(df[column].dropna().astype(str).unique().tolist())


def _unique_int(df: pd.DataFrame, column: str) -> list[int]:
    if column not in df.columns:
        return []

    values = []

    for value in df[column].dropna().unique().tolist():
        try:
            values.append(int(value))
        except Exception:
            pass

    return sorted(values)


def _sample_rows(df: pd.DataFrame, max_files: int) -> pd.DataFrame:
    if len(df) <= max_files:
        return df

    if {"domain", "split"}.issubset(df.columns):
        return (
            df.groupby(["domain", "split"], group_keys=False)
            .apply(lambda x: x.head(max(1, max_files // max(df[["domain", "split"]].drop_duplicates().shape[0], 1))))
            .head(max_files)
        )

    return df.head(max_files)


def _to_channels_first(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)

    if x.ndim == 1:
        return x[None, :]

    if x.ndim == 2:
        if x.shape[0] <= x.shape[1]:
            return x

        return x.T

    flat = x.reshape(x.shape[0], -1)

    if flat.shape[0] <= flat.shape[1]:
        return flat

    return flat.T


def _basic_features(x: np.ndarray) -> dict[str, float]:
    values = x.reshape(-1)

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "ptp": float(np.ptp(values)),
        "abs_mean": float(np.mean(np.abs(values))),
    }


def _stationarity_score(x: np.ndarray) -> float | None:
    signal = _trim_signal(x)

    if signal.size < 32:
        return None

    chunks = np.array_split(signal, 4)
    chunk_stds = np.array([np.std(chunk) for chunk in chunks], dtype=np.float32)
    mean_std = float(np.mean(chunk_stds))

    if mean_std <= 1e-8:
        return 1.0

    variability = float(np.std(chunk_stds) / (mean_std + 1e-8))

    return float(1.0 / (1.0 + variability))


def _periodicity_score(x: np.ndarray) -> float | None:
    signal = _trim_signal(x, max_len=4096)

    if signal.size < 64:
        return None

    signal = signal - np.mean(signal)
    energy = float(np.sum(signal**2))

    if energy <= 1e-8:
        return 0.0

    n = 1
    while n < 2 * signal.size:
        n *= 2

    spectrum = np.fft.rfft(signal, n=n)
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum), n=n)[: signal.size]
    autocorr = autocorr / (autocorr[0] + 1e-8)

    min_lag = max(2, signal.size // 100)
    max_lag = max(min_lag + 1, signal.size // 2)

    return float(np.max(np.abs(autocorr[min_lag:max_lag])))


def _spectral_concentration(x: np.ndarray) -> float | None:
    signal = _trim_signal(x, max_len=4096)

    if signal.size < 32:
        return None

    signal = signal - np.mean(signal)
    spectrum = np.fft.rfft(signal)
    power = np.abs(spectrum) ** 2

    if power.size <= 1:
        return None

    power = power[1:]
    total = float(np.sum(power))

    if total <= 1e-8:
        return 0.0

    return float(np.max(power) / total)


def _channel_correlation_score(x: np.ndarray) -> float | None:
    if x.shape[0] <= 1:
        return None

    signal = x[:, :4096]

    if signal.shape[1] < 16:
        return None

    corr = np.corrcoef(signal)
    corr = np.nan_to_num(corr, nan=0.0)

    mask = ~np.eye(corr.shape[0], dtype=bool)

    return float(np.mean(np.abs(corr[mask])))


def _source_target_shift_score(feature_rows: list[dict[str, Any]]) -> float | None:
    if not feature_rows:
        return None

    df = pd.DataFrame(feature_rows)

    if "domain" not in df.columns:
        return None

    source = df[df["domain"].astype(str).str.lower() == "source"]
    target = df[df["domain"].astype(str).str.lower() == "target"]

    if source.empty or target.empty:
        return None

    feature_cols = ["mean", "std", "rms", "ptp", "abs_mean"]

    source_mean = source[feature_cols].mean()
    target_mean = target[feature_cols].mean()
    pooled_std = df[feature_cols].std().replace(0.0, 1.0)

    distance = np.abs((source_mean - target_mean) / pooled_std)

    return float(distance.mean())


def _recommend_models(
    likely_multichannel: bool,
    n_channels_max: int | None,
    channel_correlation_score: float | None,
    n_rows: int,
    length_median: float | None,
    long_context_required: bool,
) -> tuple[list[str], dict[str, str]]:
    recommended = [
        "statistical_threshold",
        "pca_reconstruction",
        "isolation_forest",
        "conv1d_autoencoder",
        "tfr_autoencoder",
        "fusion_autoencoder",
    ]

    excluded = {}

    if length_median is not None and length_median >= 512:
        recommended.append("tcn_prediction")
    else:
        excluded["tcn_prediction"] = "Signal windows are too short for a useful prediction model."

    graph_allowed = (
        likely_multichannel
        and n_channels_max is not None
        and n_channels_max >= 3
        and channel_correlation_score is not None
        and channel_correlation_score >= 0.10
    )

    if graph_allowed:
        recommended.append("graph_hybrid")
    else:
        excluded["graph_hybrid"] = (
            "No sufficiently strong multichannel relational prior was detected."
        )

    transformer_allowed = (
        long_context_required
        and n_rows >= 50_000
    )

    if transformer_allowed:
        recommended.append("patch_transformer")
    else:
        excluded["patch_transformer"] = (
            "Transformer is not recommended unless long context and enough windows are available."
        )

    excluded["foundation_model"] = "Research-only extension, not part of the MVP."
    excluded["diffusion_model"] = "Research-only extension, not part of the MVP."
    excluded["llm_detector"] = "LLM is suitable for reporting, not as the core detector."

    return recommended, excluded


def _recommend_adaptation_methods(
    has_target_domain: bool,
    has_unlabeled_target_adapt: bool,
    source_target_shift_score: float | None,
) -> list[str]:
    methods = ["none", "threshold_recalibration"]

    if not has_target_domain:
        return methods

    if has_unlabeled_target_adapt:
        methods.append("sfda_prototype")

    if source_target_shift_score is not None and source_target_shift_score >= 0.20:
        methods.append("contrastive_alignment")

    return methods


def _trim_signal(x: np.ndarray, max_len: int | None = None) -> np.ndarray:
    signal = np.asarray(x, dtype=np.float32).reshape(-1)

    if max_len is not None and signal.size > max_len:
        signal = signal[:max_len]

    return signal


def _mean_or_none(values: list[float | None]) -> float | None:
    clean = [float(x) for x in values if x is not None and np.isfinite(x)]

    if not clean:
        return None

    return float(np.mean(clean))