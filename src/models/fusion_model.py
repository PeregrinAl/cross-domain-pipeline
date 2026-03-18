from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.raw_encoder import RawEncoder
from src.models.tfr_encoder import TFREncoder


class FusionEncoder(nn.Module):
    def __init__(
        self,
        use_raw: bool = True,
        use_tfr: bool = True,
        raw_embedding_dim: int = 64,
        tfr_embedding_dim: int = 64,
        fused_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        if not use_raw and not use_tfr:
            raise ValueError("At least one branch must be enabled")

        self.use_raw = use_raw
        self.use_tfr = use_tfr

        if self.use_raw:
            self.raw_encoder = RawEncoder(
                in_channels=1,
                embedding_dim=raw_embedding_dim,
                dropout=dropout,
            )

        if self.use_tfr:
            self.tfr_encoder = TFREncoder(
                in_channels=1,
                embedding_dim=tfr_embedding_dim,
                dropout=dropout,
            )

        fusion_input_dim = 0
        if self.use_raw:
            fusion_input_dim += raw_embedding_dim
        if self.use_tfr:
            fusion_input_dim += tfr_embedding_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        embeddings = []

        if self.use_raw:
            if "x_raw" not in batch:
                raise KeyError("Batch does not contain x_raw")
            raw_emb = self.raw_encoder(batch["x_raw"])
            outputs["raw_embedding"] = raw_emb
            embeddings.append(raw_emb)

        if self.use_tfr:
            if "x_tfr" not in batch:
                raise KeyError("Batch does not contain x_tfr")
            tfr_emb = self.tfr_encoder(batch["x_tfr"])
            outputs["tfr_embedding"] = tfr_emb
            embeddings.append(tfr_emb)

        fused_input = torch.cat(embeddings, dim=1)
        fused_embedding = self.fusion_head(fused_input)

        outputs["fused_embedding"] = fused_embedding
        return outputs


def compute_normal_prototype(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    normal_label: int = 0,
) -> torch.Tensor:
    """
    embeddings: [B, D]
    labels: [B]
    returns prototype: [D]
    """
    mask = labels == normal_label
    if mask.sum() == 0:
        raise ValueError("No normal samples in batch to compute prototype")

    prototype = embeddings[mask].mean(dim=0)
    return prototype


def pairwise_distance_to_prototype(
    embeddings: torch.Tensor,
    prototype: torch.Tensor,
    p: int = 2,
) -> torch.Tensor:
    """
    embeddings: [B, D]
    prototype: [D]
    returns distances: [B]
    """
    return torch.norm(embeddings - prototype.unsqueeze(0), p=p, dim=1)


def cosine_distance_to_prototype(
    embeddings: torch.Tensor,
    prototype: torch.Tensor,
) -> torch.Tensor:
    """
    returns cosine distance: 1 - cosine similarity
    """
    proto = prototype.unsqueeze(0).expand_as(embeddings)
    sim = F.cosine_similarity(embeddings, proto, dim=1)
    return 1.0 - sim