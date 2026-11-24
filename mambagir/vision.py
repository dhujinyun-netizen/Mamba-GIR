"""Tokenizer and retrieval share this spatial MambaVision encoder (Sec. III)."""

from pathlib import Path

import torch
from torch import nn, Tensor
from torch.nn import functional as F


VARIANTS = {
    'T': dict(dim=80, in_dim=32, depths=[1, 3, 8, 4], num_heads=[2, 4, 8, 16],
              window_size=[8, 8, 14, 7], layer_scale=None, drop_path_rate=0.2),
    'S': dict(dim=96, in_dim=64, depths=[3, 3, 7, 5], num_heads=[2, 4, 8, 16],
              window_size=[8, 8, 14, 7], layer_scale=None, drop_path_rate=0.2),
    'B': dict(dim=128, in_dim=64, depths=[3, 3, 10, 5], num_heads=[2, 4, 8, 16],
              window_size=[8, 8, 14, 7], layer_scale=1e-5, drop_path_rate=0.3),
    'L_21k': dict(dim=196, in_dim=64, depths=[3, 3, 10, 5], num_heads=[4, 8, 16, 32],
                  window_size=[8, 8, 14, 7], layer_scale=1e-5, drop_path_rate=0.3),
}


class VisualEncoder(nn.Module):
    def __init__(self, variant: str = 'L_21k', backbone: nn.Module | None = None):
        super().__init__()
        if backbone is None:
            if variant not in VARIANTS:
                raise ValueError(f'unknown MambaVision variant {variant!r}; choose {list(VARIANTS)}')
            from ._mambavision import MambaVision
            backbone = MambaVision(**VARIANTS[variant], mlp_ratio=4, num_classes=0,
                                   layer_scale_conv=None)
        self.backbone = backbone
        self.feature_dim = backbone.norm.num_features
        self.variant = variant

    def forward_map(self, images: Tensor) -> Tensor:
        x = self.backbone.patch_embed(images)
        for stage in self.backbone.levels:
            x = stage(x)
        return self.backbone.norm(x)

    def forward_patches(self, images: Tensor) -> Tensor:
        """N=49 spatial tokens before projection into decoder space."""
        return F.adaptive_avg_pool2d(self.forward_map(images), (7, 7)).flatten(2).transpose(1, 2)

    def forward_global(self, images: Tensor) -> Tensor:
        return self.forward_map(images).mean((-2, -1))

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_patches(images)

    def load_pretrained(self, path: str | Path) -> None:
        """Load a trusted pretrained checkpoint; ignore only its discarded classifier.

        NVIDIA's existing training checkpoint contains argparse.Namespace metadata,
        which PyTorch 2.1's weights-only loader cannot read. This explicit legacy
        import is confined to supplied pretrained files; our own artifacts use
        weights_only=True throughout.
        """
        state = torch.load(path, map_location='cpu', weights_only=False)
        for name in ('state_dict', 'model', 'net'):
            if name in state:
                state = state[name]
                break
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        if any(k.startswith('encoder.') for k in state):
            state = {k.removeprefix('encoder.'): v for k, v in state.items() if k.startswith('encoder.')}
        if any(k.startswith('backbone.') for k in state):
            state = {k.removeprefix('backbone.'): v for k, v in state.items()}
        state = {k: v for k, v in state.items() if k not in ('head.weight', 'head.bias')}
        self.backbone.load_state_dict(state, strict=True)
