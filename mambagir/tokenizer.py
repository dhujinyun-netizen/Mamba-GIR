"""Semantic tokenizer objective from Sec. III-A (before retrieval training)."""

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class SemanticTokenizer(nn.Module):
    """Train a visual encoder on original and straight-through RVQ features.

    The paper specifies head depth but not its block architecture. Here depth=12
    explicitly means eleven Linear/GELU hidden layers plus a classification
    Linear layer. The same classifier supervises the original and all partial
    reconstructions; these implementation choices are recorded in PAPER_MAPPING.
    """

    def __init__(self, encoder: nn.Module, num_classes: int, depth: int = 12,
                 hidden_dim: int | None = None):
        super().__init__()
        if depth < 1 or num_classes < 1:
            raise ValueError('depth and num_classes must be positive')
        self.encoder = encoder
        width = hidden_dim or encoder.feature_dim
        layers = []
        dim = encoder.feature_dim
        for _ in range(depth - 1):
            layers.extend((nn.Linear(dim, width), nn.GELU()))
            dim = width
        layers.append(nn.Linear(dim, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, images: Tensor, labels: Tensor, partial_reconstructions: Tensor,
                rq_weight: float = 1e-7) -> dict[str, Tensor]:
        features = self.encoder.forward_global(images)
        if (partial_reconstructions.ndim != 3 or
                partial_reconstructions.shape[0] != features.shape[0] or
                partial_reconstructions.shape[-1] != features.shape[-1]):
            raise ValueError('partial reconstructions must have shape [B,L,feature_dim]')
        # Forward uses quantized values; backward reaches only the visual feature.
        partials = features[:, None] + (partial_reconstructions - features[:, None]).detach()
        logits = self.head(features)
        rq_logits = self.head(partials)
        levels = partials.shape[1]
        loss_ce = F.cross_entropy(logits, labels)
        loss_ce_rq = F.cross_entropy(
            rq_logits.flatten(0, 1), labels[:, None].expand(-1, levels).reshape(-1),
        ) * levels
        # The manuscript specifies squared L2, not a feature-wise mean squared
        # error. Float32 keeps the norm safe under autocast.
        residual = features[:, None].float() - partial_reconstructions.detach().float()
        loss_quant = residual.square().sum(-1).mean(0).sum()
        return {
            'loss': loss_ce + rq_weight * (loss_ce_rq + loss_quant),
            'loss_ce': loss_ce, 'loss_ce_rq': loss_ce_rq, 'loss_quant': loss_quant,
        }
