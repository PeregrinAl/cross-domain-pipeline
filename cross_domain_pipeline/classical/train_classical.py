from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from cross_domain_pipeline.classical.features import FEATURE_COLUMNS, build_feature_table


def run_classical_baseline(
    config_path: str | Path,
    method: str,
    source: str = "auto",
    output_dir: str | Path | None = None,
) -> dict:
    config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    csv_path = _select_csv(data_cfg, source)

    base_dir = config_path.parent
    features = build_feature_table(csv_path=csv_path, base_dir=base_dir)

    if features.empty:
        raise RuntimeError("Feature table is empty. Check CSV paths and .npy files.")

    run_dir = Path(output_dir) if output_dir is not None else Path("experiments") / "classical" / method
    run_dir.mkdir(parents=True, exist_ok=True)

    features.to_csv(run_dir / "features.csv", index=False)

    train_df = _select_train_normal(features)
    eval_df = _select_eval_rows(features)

    if train_df.empty:
        raise RuntimeError("No normal source/train rows found for classical baseline.")

    if eval_df.empty:
        raise RuntimeError("No evaluation rows found. Expected source/val or target/test rows.")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    x_eval = scaler.transform(eval_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32))

    if method == "statistical_threshold":
        scores = _score_statistical_threshold(x_train, x_eval)
    elif method == "pca_reconstruction":
        scores = _score_pca_reconstruction(x_train, x_eval)
    elif method == "isolation_forest":
        scores = _score_isolation_forest(x_train, x_eval)
    else:
        raise ValueError(
            "method must be one of: statistical_threshold, pca_reconstruction, isolation_forest"
        )

    scored = eval_df.copy()
    scored["score"] = scores
    scored.to_csv(run_dir / "scores.csv", index=False)

    summary = _compute_summary(scored)
    summary["method"] = method
    summary["csv_path"] = str(csv_path)
    summary["n_train_normal"] = int(len(train_df))
    summary["n_eval"] = int(len(scored))

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return summary


def _select_csv(data_cfg: dict, source: str) -> str:
    if source not in {"auto", "raw", "manifest"}:
        raise ValueError("source must be one of: auto, raw, manifest")

    if source == "raw":
        return data_cfg["raw_records_csv"]

    if source == "manifest":
        return data_cfg.get("manifest_path") or data_cfg["processed_dir"].rstrip("/") + "/manifest.csv"

    manifest_path = data_cfg.get("manifest_path")

    if manifest_path is not None and Path(manifest_path).exists():
        return manifest_path

    processed_manifest = Path(data_cfg.get("processed_dir", "")) / "manifest.csv"

    if processed_manifest.exists():
        return str(processed_manifest)

    return data_cfg["raw_records_csv"]


def _select_train_normal(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["label"].eq(0)

    if "domain" in df.columns:
        mask &= df["domain"].astype(str).str.lower().eq("source")

    if "split" in df.columns:
        mask &= df["split"].astype(str).str.lower().isin(["train", "fit"])

    return df[mask].copy()


def _select_eval_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "split" not in df.columns:
        return df.copy()

    split = df["split"].astype(str).str.lower()

    mask = split.isin(["val", "valid", "validation", "test"])

    return df[mask].copy()


def _score_statistical_threshold(x_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    center = np.median(x_train, axis=0)
    scale = np.median(np.abs(x_train - center), axis=0) + 1e-8

    z = np.abs((x_eval - center) / scale)

    return np.mean(z, axis=1)


def _score_pca_reconstruction(x_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    n_features = x_train.shape[1]
    n_components = max(1, min(5, n_features - 1, x_train.shape[0] - 1))

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(x_train)

    reconstructed = pca.inverse_transform(pca.transform(x_eval))

    return np.mean((x_eval - reconstructed) ** 2, axis=1)


def _score_isolation_forest(x_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train)

    return -model.score_samples(x_eval)


def _compute_summary(scored: pd.DataFrame) -> dict:
    y_true = scored["label"].to_numpy(dtype=int)
    scores = scored["score"].to_numpy(dtype=float)

    summary = {}

    if len(np.unique(y_true)) < 2:
        summary["roc_auc"] = None
        summary["pr_auc"] = None
        summary["best_f1"] = None
        summary["threshold"] = None
        return summary

    summary["roc_auc"] = float(roc_auc_score(y_true, scores))
    summary["pr_auc"] = float(average_precision_score(y_true, scores))

    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    if thresholds.size == 0:
        summary["best_f1"] = None
        summary["threshold"] = None
        return summary

    f1_values = []

    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        f1_values.append(f1_score(y_true, pred, zero_division=0))

    best_idx = int(np.argmax(f1_values))

    summary["best_f1"] = float(f1_values[best_idx])
    summary["threshold"] = float(thresholds[best_idx])

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train classical anomaly detection baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--method",
        required=True,
        choices=["statistical_threshold", "pca_reconstruction", "isolation_forest"],
    )
    parser.add_argument(
        "--source",
        default="auto",
        choices=["auto", "raw", "manifest"],
    )
    parser.add_argument("--output-dir", default=None)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    run_classical_baseline(
        config_path=args.config,
        method=args.method,
        source=args.source,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()