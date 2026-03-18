import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from src.data.dataloader import build_dataloader
from src.models.source_only_classifier import SourceOnlyClassifier
from src.utils.metrics import compute_binary_metrics
from src.utils.seed import set_seed


VARIANTS = {
    "raw_only": (True, False),
    "tfr_only": (False, True),
    "fused": (True, True),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=list(VARIANTS.keys()),
    )
    return parser.parse_args()


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def resolve_device(config_device: str) -> torch.device:
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)


def build_model(config, variant: str):
    use_raw, use_tfr = VARIANTS[variant]

    model = SourceOnlyClassifier(
        use_raw=use_raw,
        use_tfr=use_tfr,
        raw_embedding_dim=config["model"]["raw_embedding_dim"],
        tfr_embedding_dim=config["model"]["tfr_embedding_dim"],
        fused_dim=config["model"]["fused_dim"],
        dropout=config["model"]["dropout"],
        num_classes=2,
    )
    return model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()
        outputs = model(batch)

        logits = outputs["logits"]
        labels = batch["label"]

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_model(model, loader, criterion, device, threshold: float):
    model.eval()

    running_loss = 0.0
    total_samples = 0

    rows = []

    for batch in loader:
        raw_batch = batch
        batch = move_batch_to_device(batch, device)

        outputs = model(batch)

        logits = outputs["logits"]
        probs = outputs["probs"][:, 1]
        labels = batch["label"]

        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        y_true = labels.detach().cpu().numpy()
        y_score = probs.detach().cpu().numpy()

        for i in range(batch_size):
            rows.append(
                {
                    "label": int(y_true[i]),
                    "score": float(y_score[i]),
                    "path": raw_batch["path"][i],
                    "domain": raw_batch["domain"][i],
                    "record_id": raw_batch["record_id"][i],
                    "split": raw_batch["split"][i],
                }
            )

    scores_df = pd.DataFrame(rows)

    metrics = compute_binary_metrics(
        y_true=scores_df["label"].values,
        y_score=scores_df["score"].values,
        threshold=threshold,
    )
    metrics["loss"] = running_loss / max(total_samples, 1)

    return metrics, scores_df


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    device = resolve_device(config["training"].get("device", "auto"))
    print("Using device:", device)

    out_dir = Path(config["outputs"]["source_only_dir"]) / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_dataloader(
        config=config,
        split="train",
        domain="source",
        shuffle=True,
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

    model = build_model(config, args.variant).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    best_val_auc = -np.inf
    best_ckpt_path = out_dir / "best.pt"
    history_rows = []

    epochs = config["training"]["epochs"]
    threshold = config["training"]["threshold"]

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        source_val_metrics, _ = evaluate_model(
            model=model,
            loader=source_val_loader,
            criterion=criterion,
            device=device,
            threshold=threshold,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "source_val_loss": source_val_metrics["loss"],
            "source_val_roc_auc": source_val_metrics["roc_auc"],
            "source_val_pr_auc": source_val_metrics["pr_auc"],
            "source_val_f1": source_val_metrics["f1"],
        }
        history_rows.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={source_val_metrics['loss']:.4f} | "
            f"val_roc_auc={source_val_metrics['roc_auc']:.4f} | "
            f"val_pr_auc={source_val_metrics['pr_auc']:.4f} | "
            f"val_f1={source_val_metrics['f1']:.4f}"
        )

        current_val_auc = source_val_metrics["roc_auc"]
        if not np.isnan(current_val_auc) and current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=source_val_metrics,
            )

    if best_ckpt_path.exists():
        load_checkpoint(best_ckpt_path, model=model, map_location=device)
        print("Loaded best checkpoint from:", best_ckpt_path)
    else:
        print("Best checkpoint was not saved. Using last model state.")

    source_val_metrics, source_val_scores = evaluate_model(
        model=model,
        loader=source_val_loader,
        criterion=criterion,
        device=device,
        threshold=threshold,
    )

    target_test_metrics, target_test_scores = evaluate_model(
        model=model,
        loader=target_test_loader,
        criterion=criterion,
        device=device,
        threshold=threshold,
    )

    summary = {
        "variant": args.variant,
        "source_val": source_val_metrics,
        "target_test": target_test_metrics,
    }

    pd.DataFrame(history_rows).to_csv(out_dir / "history.csv", index=False)
    source_val_scores.to_csv(out_dir / "source_val_scores.csv", index=False)
    target_test_scores.to_csv(out_dir / "target_test_scores.csv", index=False)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nFinal summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()