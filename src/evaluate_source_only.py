from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score

from src.data.dataloader import build_dataloader
from src.models.fusion_model import (
    FusionEncoder,
    pairwise_distance_to_prototype,
    cosine_distance_to_prototype,
)
from src.utils.seed import set_seed


def compute_prototype_from_loader(model, loader, embedding_key="fused_embedding", normal_label=0):
    embeddings = []
    labels_all = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch)
            emb = outputs[embedding_key]
            labels = batch["label"]

            embeddings.append(emb)
            labels_all.append(labels)

    embeddings = torch.cat(embeddings, dim=0)
    labels_all = torch.cat(labels_all, dim=0)

    mask = labels_all == normal_label
    if mask.sum() == 0:
        raise ValueError("No normal samples found in prototype loader")

    prototype = embeddings[mask].mean(dim=0)
    return prototype


def score_loader(model, loader, prototype, embedding_key="fused_embedding", distance_type="l2"):
    rows = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch)
            emb = outputs[embedding_key]

            if distance_type == "l2":
                scores = pairwise_distance_to_prototype(emb, prototype)
            elif distance_type == "cosine":
                scores = cosine_distance_to_prototype(emb, prototype)
            else:
                raise ValueError(f"Unsupported distance_type: {distance_type}")

            labels = batch["label"].cpu().numpy()
            paths = batch["path"]
            domains = batch["domain"]
            record_ids = batch["record_id"]
            splits = batch["split"]

            for i in range(len(labels)):
                rows.append(
                    {
                        "label": int(labels[i]),
                        "score": float(scores[i].cpu().item()),
                        "path": paths[i],
                        "domain": domains[i],
                        "record_id": record_ids[i],
                        "split": splits[i],
                    }
                )

    return pd.DataFrame(rows)


def compute_auc_safe(df):
    if df["label"].nunique() < 2:
        return np.nan
    return roc_auc_score(df["label"], df["score"])


def evaluate_variant(config, variant_name, use_raw, use_tfr):
    model = FusionEncoder(
        use_raw=use_raw,
        use_tfr=use_tfr,
        raw_embedding_dim=config["model"]["raw_embedding_dim"],
        tfr_embedding_dim=config["model"]["tfr_embedding_dim"],
        fused_dim=config["model"]["fused_dim"],
        dropout=config["model"]["dropout"],
    )

    if use_raw and use_tfr:
        embedding_key = "fused_embedding"
    elif use_raw:
        embedding_key = "raw_embedding"
    elif use_tfr:
        embedding_key = "tfr_embedding"
    else:
        raise ValueError("At least one branch must be enabled")

    source_train_loader = build_dataloader(
        config=config,
        split="train",
        domain="source",
        shuffle=False,
    )

    source_val_loader = build_dataloader(
        config=config,
        split="val",
        domain="source",
        shuffle=False,
    )

    target_test_loader = build_dataloader(
        config=config,
        split="test",
        domain="target",
        shuffle=False,
    )

    prototype = compute_prototype_from_loader(
        model=model,
        loader=source_train_loader,
        embedding_key=embedding_key,
        normal_label=config["evaluation"]["normal_label"],
    )

    source_val_df = score_loader(
        model=model,
        loader=source_val_loader,
        prototype=prototype,
        embedding_key=embedding_key,
        distance_type=config["evaluation"]["distance"],
    )

    target_test_df = score_loader(
        model=model,
        loader=target_test_loader,
        prototype=prototype,
        embedding_key=embedding_key,
        distance_type=config["evaluation"]["distance"],
    )

    source_val_auc = compute_auc_safe(source_val_df)
    target_test_auc = compute_auc_safe(target_test_df)

    metrics = {
        "variant": variant_name,
        "source_val_auc": source_val_auc,
        "target_test_auc": target_test_auc,
        "source_val_n": len(source_val_df),
        "target_test_n": len(target_test_df),
    }

    return metrics, source_val_df, target_test_df


def main():
    with open("configs/base.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    out_dir = Path("experiments/source_only_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        ("raw_only", True, False),
        ("tfr_only", False, True),
        ("fused", True, True),
    ]

    all_metrics = []

    for variant_name, use_raw, use_tfr in variants:
        metrics, source_val_df, target_test_df = evaluate_variant(
            config=config,
            variant_name=variant_name,
            use_raw=use_raw,
            use_tfr=use_tfr,
        )

        all_metrics.append(metrics)

        source_val_df.to_csv(out_dir / f"{variant_name}_source_val_scores.csv", index=False)
        target_test_df.to_csv(out_dir / f"{variant_name}_target_test_scores.csv", index=False)

        print(f"\nVariant: {variant_name}")
        print(f"source_val_auc: {metrics['source_val_auc']}")
        print(f"target_test_auc: {metrics['target_test_auc']}")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(out_dir / "summary_metrics.csv", index=False)

    print("\nSaved summary to:", out_dir / "summary_metrics.csv")
    print(metrics_df)


if __name__ == "__main__":
    main()