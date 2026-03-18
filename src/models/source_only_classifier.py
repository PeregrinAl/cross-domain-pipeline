from typing import Dict

import torch
import torch.nn as nn

from src.models.fusion_model import FusionEncoder


class SourceOnlyClassifier(nn.Module):
    def __init__(
        self,
        use_raw: bool = True,
        use_tfr: bool = True,
        raw_embedding_dim: int = 64,
        tfr_embedding_dim: int = 64,
        fused_dim: int = 64,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()

        self.encoder = FusionEncoder(
            use_raw=use_raw,
            use_tfr=use_tfr,
            raw_embedding_dim=raw_embedding_dim,
            tfr_embedding_dim=tfr_embedding_dim,
            fused_dim=fused_dim,
            dropout=dropout,
        )

        self.use_raw = use_raw
        self.use_tfr = use_tfr

        if use_raw and use_tfr:
            self.embedding_key = "fused_embedding"
            classifier_input_dim = fused_dim
        elif use_raw:
            self.embedding_key = "raw_embedding"
            classifier_input_dim = raw_embedding_dim
        elif use_tfr:
            self.embedding_key = "tfr_embedding"
            classifier_input_dim = tfr_embedding_dim
        else:
            raise ValueError("At least one branch must be enabled")

        self.classifier = nn.Linear(classifier_input_dim, num_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(batch)

        embedding = outputs[self.embedding_key]
        logits = self.classifier(embedding)
        probs = torch.softmax(logits, dim=1)
        anomaly_score = probs[:, 1]

        outputs["embedding"] = embedding
        outputs["logits"] = logits
        outputs["probs"] = probs
        outputs["anomaly_score"] = anomaly_score

        return outputs