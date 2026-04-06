import json
from pathlib import Path
import torch.nn.functional as F

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

from src.data.dataloader import build_dataloader, build_dataset
from src.models.source_only_classifier import SourceOnlyClassifier
from src.utils.metrics import compute_binary_metrics
from src.utils.seed import set_seed


def resolve_device(config_device: str) -> torch.device:
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def build_fused_model(config):
    return SourceOnlyClassifier(
        use_raw=True,
        use_tfr=True,
        raw_embedding_dim=config["model"]["raw_embedding_dim"],
        tfr_embedding_dim=config["model"]["tfr_embedding_dim"],
        fused_dim=config["model"]["fused_dim"],
        dropout=config["model"]["dropout"],
        num_classes=2,
    )


def load_source_checkpoint(model, checkpoint_path: Path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint

@torch.no_grad()
def compute_branch_prototypes(model, loader, device, normal_label=0):
    model.eval()

    raw_embeddings = []
    tfr_embeddings = []
    fused_embeddings = []
    labels_all = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)

        raw_embeddings.append(outputs["raw_embedding"])
        tfr_embeddings.append(outputs["tfr_embedding"])
        fused_embeddings.append(outputs["embedding"])
        labels_all.append(batch["label"])

    raw_embeddings = torch.cat(raw_embeddings, dim=0)
    tfr_embeddings = torch.cat(tfr_embeddings, dim=0)
    fused_embeddings = torch.cat(fused_embeddings, dim=0)
    labels_all = torch.cat(labels_all, dim=0)

    mask = labels_all == normal_label
    if mask.sum() == 0:
        raise ValueError("No normal samples found in source train")

    return {
        "raw": raw_embeddings[mask].mean(dim=0),
        "tfr": tfr_embeddings[mask].mean(dim=0),
        "fused": fused_embeddings[mask].mean(dim=0),
    }

@torch.no_grad()
def compute_source_normal_prototype(model, loader, device, normal_label=0):
    model.eval()

    embeddings = []
    labels_all = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)

        embeddings.append(outputs["embedding"])
        labels_all.append(batch["label"])

    embeddings = torch.cat(embeddings, dim=0)
    labels_all = torch.cat(labels_all, dim=0)

    mask = labels_all == normal_label
    if mask.sum() == 0:
        raise ValueError("No normal samples found in source train for prototype computation")

    prototype = embeddings[mask].mean(dim=0)
    return prototype


@torch.no_grad()
def collect_scores(model, loader, device):
    model.eval()

    rows = []

    for batch in loader:
        raw_batch = batch
        batch = move_batch_to_device(batch, device)

        outputs = model(batch)
        scores = outputs["anomaly_score"].detach().cpu().numpy()
        labels = raw_batch["label"].detach().cpu().numpy()

        batch_size = len(scores)
        for i in range(batch_size):
            rows.append(
                {
                    "label": int(labels[i]),
                    "score": float(scores[i]),
                    "path": raw_batch["path"][i],
                    "domain": raw_batch["domain"][i],
                    "record_id": raw_batch["record_id"][i],
                    "split": raw_batch["split"][i],
                }
            )

    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate_split(model, loader, device, threshold):
    scores_df = collect_scores(model, loader, device)

    metrics = compute_binary_metrics(
        y_true=scores_df["label"].values,
        y_score=scores_df["score"].values,
        threshold=threshold,
    )
    return metrics, scores_df


def load_source_only_threshold(config, variant: str = "fused"):
    fallback = float(config["training"]["threshold"])
    summary_path = Path(config["outputs"]["source_only_dir"]) / variant / "summary.json"

    if not summary_path.exists():
        return fallback

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    if "threshold_used" in summary:
        return float(summary["threshold_used"])

    if "best_threshold_source_val_f1" in summary:
        return float(summary["best_threshold_source_val_f1"])

    return fallback


def freeze_for_sfda(model):
    # Сначала замораживаем все параметры модели
    for param in model.parameters():
        param.requires_grad = False

    # Оставляем обучаемой только fusion head
    for param in model.encoder.fusion_head.parameters():
        param.requires_grad = True

    print("Trainable parameters:")

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


@torch.no_grad()
def select_pseudo_normal_indices(
    model,
    dataset,
    device,
    batch_size,
    num_workers,
    pin_memory,
    threshold,
    selection_quantile,
    min_selected,
    branch_prototypes,
    consistency_quantile,
    agreement_margin_quantile,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model.eval()

    fused_scores = []
    raw_dists = []
    tfr_dists = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)

        scores = outputs["anomaly_score"].detach().cpu()
        raw_emb = outputs["raw_embedding"]
        tfr_emb = outputs["tfr_embedding"]

        raw_dist = torch.norm(
            raw_emb - branch_prototypes["raw"].unsqueeze(0),
            p=2,
            dim=1,
        ).detach().cpu()

        tfr_dist = torch.norm(
            tfr_emb - branch_prototypes["tfr"].unsqueeze(0),
            p=2,
            dim=1,
        ).detach().cpu()

        fused_scores.extend(scores.tolist())
        raw_dists.extend(raw_dist.tolist())
        tfr_dists.extend(tfr_dist.tolist())

    fused_scores = np.asarray(fused_scores, dtype=float)
    raw_dists = np.asarray(raw_dists, dtype=float)
    tfr_dists = np.asarray(tfr_dists, dtype=float)

    if len(fused_scores) == 0:
        raise ValueError("Target adapt dataset is empty")

    fused_cutoff = min(
        float(threshold),
        float(np.quantile(fused_scores, selection_quantile)),
    )

    raw_cutoff = float(np.quantile(raw_dists, consistency_quantile))
    tfr_cutoff = float(np.quantile(tfr_dists, consistency_quantile))

    agreement = np.abs(raw_dists - tfr_dists)
    agreement_cutoff = float(np.quantile(agreement, agreement_margin_quantile))

    selected_mask = (
        (fused_scores <= fused_cutoff)
        & (raw_dists <= raw_cutoff)
        & (tfr_dists <= tfr_cutoff)
        & (agreement <= agreement_cutoff)
    )

    selected_indices = np.where(selected_mask)[0].tolist()

    if len(selected_indices) < min_selected:
        ranking = np.argsort(fused_scores + 0.5 * agreement)
        selected_indices = ranking[: min(min_selected, len(ranking))].tolist()

    selection_info = {
        "n_total": int(len(fused_scores)),
        "n_selected": int(len(selected_indices)),
        "fused_score_mean": float(np.mean(fused_scores)),
        "raw_dist_mean": float(np.mean(raw_dists)),
        "tfr_dist_mean": float(np.mean(tfr_dists)),
        "agreement_mean": float(np.mean(agreement)),
        "fused_cutoff": fused_cutoff,
        "raw_cutoff": raw_cutoff,
        "tfr_cutoff": tfr_cutoff,
        "agreement_cutoff": agreement_cutoff,
    }

    return selected_indices, selection_info

def adapt_one_epoch(
    model,
    loader,
    optimizer,
    device,
    source_prototype,
    align_weight,
    pseudo_label_weight,
):
    model.train()

    total_loss_sum = 0.0
    total_samples = 0

    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        outputs = model(batch)
        embeddings = outputs["embedding"]
        logits = outputs["logits"]

        proto = source_prototype.unsqueeze(0).expand_as(embeddings)
        loss_align = mse_loss(embeddings, proto)

        pseudo_labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
        loss_pseudo = ce_loss(logits, pseudo_labels)

        loss = align_weight * loss_align + pseudo_label_weight * loss_pseudo
        loss.backward()
        optimizer.step()

        batch_size = logits.size(0)
        total_loss_sum += loss.item() * batch_size
        total_samples += batch_size

    return total_loss_sum / max(total_samples, 1)


def main():
    with open("configs/base.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = resolve_device(config["training"].get("device", "auto"))

    sfda_variant = config["sfda"]["variant"]
    if sfda_variant != "fused":
        raise ValueError("This minimal SFDA script currently supports only variant='fused'")

    source_ckpt_path = Path(config["outputs"]["source_only_dir"]) / "fused" / "best.pt"
    out_dir = Path(config["outputs"]["sfda_dir"]) / "fused"
    out_dir.mkdir(parents=True, exist_ok=True)

    threshold = load_source_only_threshold(config, variant="fused")
    print("Using evaluation/selection threshold:", threshold)
    print("Using device:", device)

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

    target_adapt_dataset = build_dataset(
        config=config,
        split=config["sfda"]["adapt_split"],
        domain="target",
    )

    model = build_fused_model(config).to(device)
    load_source_checkpoint(model, source_ckpt_path, device)

    before_source_metrics, before_source_scores = evaluate_split(
        model=model,
        loader=source_val_loader,
        device=device,
        threshold=threshold,
    )
    before_target_metrics, before_target_scores = evaluate_split(
        model=model,
        loader=target_test_loader,
        device=device,
        threshold=threshold,
    )

    branch_prototypes = compute_branch_prototypes(
        model=model,
        loader=source_train_loader,
        device=device,
        normal_label=config["evaluation"]["normal_label"],
    )

    source_prototype = branch_prototypes["fused"].detach()

    freeze_for_sfda(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=config["sfda"]["lr"],
        weight_decay=config["sfda"]["weight_decay"],
    )

    best_loss = np.inf
    best_ckpt_path = out_dir / "best_adapted.pt"
    history_rows = []

    for epoch in range(1, config["sfda"]["epochs"] + 1):
        selected_indices, selection_info = select_pseudo_normal_indices(
            model=model,
            dataset=target_adapt_dataset,
            device=device,
            batch_size=config["sfda"]["batch_size"],
            num_workers=config["training"]["num_workers"],
            pin_memory=config["training"]["pin_memory"],
            threshold=threshold,
            selection_quantile=config["sfda"]["selection_quantile"],
            min_selected=config["sfda"]["min_selected"],
            branch_prototypes=branch_prototypes,
            consistency_quantile=config["sfda"]["consistency_quantile"],
            agreement_margin_quantile=config["sfda"]["agreement_margin_quantile"],
        )

        subset = Subset(target_adapt_dataset, selected_indices)
        subset_loader = DataLoader(
            subset,
            batch_size=config["sfda"]["batch_size"],
            shuffle=True,
            num_workers=config["training"]["num_workers"],
            pin_memory=config["training"]["pin_memory"],
        )

        epoch_loss = adapt_one_epoch(
            model=model,
            loader=subset_loader,
            optimizer=optimizer,
            device=device,
            source_prototype=source_prototype,
            align_weight=config["sfda"]["align_weight"],
            pseudo_label_weight=config["sfda"]["pseudo_label_weight"],
        )

        row = {
            "epoch": epoch,
            "adapt_loss": epoch_loss,
            "n_selected": selection_info["n_selected"],
            "n_total": selection_info["n_total"],
            "fused_score_mean": selection_info["fused_score_mean"],
            "raw_dist_mean": selection_info["raw_dist_mean"],
            "tfr_dist_mean": selection_info["tfr_dist_mean"],
            "agreement_mean": selection_info["agreement_mean"],
            "fused_cutoff": selection_info["fused_cutoff"],
            "raw_cutoff": selection_info["raw_cutoff"],
            "tfr_cutoff": selection_info["tfr_cutoff"],
            "agreement_cutoff": selection_info["agreement_cutoff"],
        }
        history_rows.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"adapt_loss={epoch_loss:.6f} | "
            f"selected={selection_info['n_selected']}/{selection_info['n_total']} | "
            f"fused_cutoff={selection_info['fused_cutoff']:.6f} | "
            f"agreement_cutoff={selection_info['agreement_cutoff']:.6f}"
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "adapt_loss": epoch_loss,
                    "selection_info": selection_info,
                },
                best_ckpt_path,
            )

    if best_ckpt_path.exists():
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded best adapted checkpoint from:", best_ckpt_path)

    after_source_metrics, after_source_scores = evaluate_split(
        model=model,
        loader=source_val_loader,
        device=device,
        threshold=threshold,
    )
    after_target_metrics, after_target_scores = evaluate_split(
        model=model,
        loader=target_test_loader,
        device=device,
        threshold=threshold,
    )

    summary = {
        "variant": "fused",
        "threshold_used": threshold,
        "before": {
            "source_val": before_source_metrics,
            "target_test": before_target_metrics,
        },
        "after": {
            "source_val": after_source_metrics,
            "target_test": after_target_metrics,
        },
        "delta": {
            "source_val_roc_auc": after_source_metrics["roc_auc"] - before_source_metrics["roc_auc"],
            "source_val_pr_auc": after_source_metrics["pr_auc"] - before_source_metrics["pr_auc"],
            "source_val_f1": after_source_metrics["f1"] - before_source_metrics["f1"],
            "target_test_roc_auc": after_target_metrics["roc_auc"] - before_target_metrics["roc_auc"],
            "target_test_pr_auc": after_target_metrics["pr_auc"] - before_target_metrics["pr_auc"],
            "target_test_f1": after_target_metrics["f1"] - before_target_metrics["f1"],
        },
    }

    pd.DataFrame(history_rows).to_csv(out_dir / "adapt_history.csv", index=False)

    before_source_scores.to_csv(out_dir / "source_val_scores_before.csv", index=False)
    before_target_scores.to_csv(out_dir / "target_test_scores_before.csv", index=False)
    after_source_scores.to_csv(out_dir / "source_val_scores_after.csv", index=False)
    after_target_scores.to_csv(out_dir / "target_test_scores_after.csv", index=False)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nSFDA summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()