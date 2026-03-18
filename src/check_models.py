import yaml
import torch

from src.data.dataloader import build_dataloader
from src.models.fusion_model import (
    FusionEncoder,
    compute_normal_prototype,
    pairwise_distance_to_prototype,
    cosine_distance_to_prototype,
)
from src.utils.seed import set_seed


def main():
    with open("configs/base.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    loader = build_dataloader(
        config=config,
        split="train",
        domain="source",
        shuffle=False,
    )

    batch = next(iter(loader))

    model = FusionEncoder(
        use_raw=config["model"]["use_raw"],
        use_tfr=config["model"]["use_tfr"],
        raw_embedding_dim=config["model"]["raw_embedding_dim"],
        tfr_embedding_dim=config["model"]["tfr_embedding_dim"],
        fused_dim=config["model"]["fused_dim"],
        dropout=config["model"]["dropout"],
    )

    model.eval()

    with torch.no_grad():
        outputs = model(batch)

    print("Model outputs:")
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")

    fused_embedding = outputs["fused_embedding"]
    labels = batch["label"]

    prototype = compute_normal_prototype(fused_embedding, labels)
    l2_scores = pairwise_distance_to_prototype(fused_embedding, prototype)
    cos_scores = cosine_distance_to_prototype(fused_embedding, prototype)

    print("prototype shape:", prototype.shape)
    print("l2_scores shape:", l2_scores.shape)
    print("cos_scores shape:", cos_scores.shape)
    print("l2_scores:", l2_scores)
    print("cos_scores:", cos_scores)


if __name__ == "__main__":
    main()