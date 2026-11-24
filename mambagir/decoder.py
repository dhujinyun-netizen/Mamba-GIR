"""Causal decoder contracts with lazy optional Hugging Face Mamba loading.

The native LoRA wrappers keep all pretrained weights frozen. HF Mamba's CUDA
fast path can consume projection weights directly and skip wrapper forwards;
therefore adapter-enabled mixers use their own ``slow_forward`` method on both
CPU and CUDA. This is a correctness choice with a throughput cost. It changes
only those mixer instances, never the installed transformers package.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Frozen linear map plus a trainable rank-r update, initially zero."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base_layer = base_layer.requires_grad_(False)
        self.lora_A = nn.Parameter(base_layer.weight.new_empty(rank, base_layer.in_features))
        self.lora_B = nn.Parameter(base_layer.weight.new_zeros(base_layer.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

    @property
    def weight(self) -> Tensor:
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, inputs: Tensor) -> Tensor:
        update = F.linear(F.linear(self.dropout(inputs), self.lora_A), self.lora_B)
        return self.base_layer(inputs) + self.scaling * update


class LoRAEmbedding(nn.Module):
    """Frozen lookup plus a low-rank embedding update with dropout on rank features."""

    def __init__(self, base_layer: nn.Embedding, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base_layer = base_layer.requires_grad_(False)
        self.num_embeddings = base_layer.num_embeddings
        self.embedding_dim = base_layer.embedding_dim
        self.lora_A = nn.Parameter(base_layer.weight.new_empty(self.num_embeddings, rank))
        self.lora_B = nn.Parameter(base_layer.weight.new_zeros(self.embedding_dim, rank))
        nn.init.normal_(self.lora_A, std=1 / math.sqrt(rank))
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

    @property
    def weight(self) -> Tensor:
        return self.base_layer.weight

    def forward(self, ids: Tensor) -> Tensor:
        rank_features = F.embedding(ids, self.lora_A)
        update = F.linear(self.dropout(rank_features), self.lora_B)
        return self.base_layer(ids) + self.scaling * update


class MambaDecoder(nn.Module):
    """HF MambaModel with adapter-only or frozen adaptation.

    ``backbone``/``from_backbone`` permits offline construction from an actual
    small HF MambaModel. A path loads pretrained weights with local-only lookup
    by default. Only this constructor imports transformers when loading a path.
    No language-model output head is needed: MambaGIR supplies the RVQ head.
    """

    def __init__(
        self, model_path: str | None = None, adaptation: str = "lora", rank: int = 8,
        alpha: float = 32, dropout: float = .1, local_files_only: bool = True,
        *, backbone: nn.Module | None = None,
    ):
        super().__init__()
        if adaptation not in {"lora", "frozen"}:
            raise ValueError("adaptation must be 'lora' or 'frozen'")
        if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("rank and alpha must be positive, and dropout must be in [0, 1)")
        if backbone is not None and model_path is not None:
            raise ValueError("provide either model_path or backbone, not both")
        if backbone is None:
            if model_path is None:
                raise ValueError("model_path is required unless a backbone is supplied")
            try:
                from transformers import MambaModel
            except ImportError as exc:
                raise ImportError("Loading Mamba weights requires a compatible transformers installation") from exc
            backbone = MambaModel.from_pretrained(model_path, local_files_only=local_files_only)
        if not hasattr(backbone, "config") or not isinstance(getattr(backbone, "embeddings", None), nn.Embedding):
            raise TypeError("backbone must be an unadapted HF MambaModel with embeddings")

        self.backbone = backbone
        self.model_path = model_path
        self.adaptation = adaptation
        self.d_model = int(backbone.config.hidden_size)
        self.vocab_size = int(backbone.embeddings.num_embeddings)
        backbone.requires_grad_(False)
        forced_slow_forward = False
        if adaptation == "lora":
            targets = {"x_proj", "in_proj", "out_proj"}
            replacements = [(name, module) for name, module in backbone.named_modules()
                            if name.rsplit(".", 1)[-1] in targets and isinstance(module, nn.Linear)]
            found = {name.rsplit(".", 1)[-1] for name, _ in replacements}
            if found != targets:
                raise ValueError(f"Mamba backbone is missing requested adapter targets: {sorted(targets - found)}")
            mixers = [module for module in backbone.modules()
                      if callable(getattr(module, "slow_forward", None))
                      and all(hasattr(module, target) for target in targets)]
            if not mixers:
                raise ValueError("Mamba mixers must expose slow_forward to guarantee active LoRA updates")
            for name, module in replacements:
                parent_name, _, child_name = name.rpartition(".")
                parent = backbone.get_submodule(parent_name) if parent_name else backbone
                setattr(parent, child_name, LoRALinear(module, rank, alpha, dropout))
            backbone.embeddings = LoRAEmbedding(backbone.embeddings, rank, alpha, dropout)
            for mixer in mixers:
                mixer.forward = mixer.slow_forward
            forced_slow_forward = True
        self.adapter_config = {
            "adaptation": adaptation, "rank": rank, "alpha": alpha, "dropout": dropout,
            "target_modules": ["x_proj", "in_proj", "out_proj", "embeddings"],
            "implementation": "native_lora", "forced_slow_forward": forced_slow_forward,
        }

    @classmethod
    def from_backbone(cls, backbone: nn.Module, **kwargs) -> MambaDecoder:
        return cls(backbone=backbone, **kwargs)

    def embed_tokens(self, ids: Tensor) -> Tensor:
        return self.backbone.embeddings(ids)

    def forward(self, inputs_embeds: Tensor) -> Tensor:
        output = self.backbone(inputs_embeds=inputs_embeds, use_cache=False, return_dict=True)
        return output.last_hidden_state


class TransformerDecoder(nn.Module):
    """Trainable causal Transformer reference; no claim of 857M size matching.

    Sinusoidal positions and pre-normalized Transformer blocks are explicit
    implementation choices because the paper does not specify this architecture.
    """

    def __init__(self, d_model: int, layers: int = 12, num_heads: int = 8,
                 vocab_size: int = 1025, dropout: float = .1, ffn_ratio: int = 4):
        super().__init__()
        if min(d_model, layers, num_heads, vocab_size, ffn_ratio) < 1 or d_model % num_heads:
            raise ValueError("positive dimensions and d_model divisible by num_heads are required")
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward=int(ffn_ratio * d_model), dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False,
        )

    def embed_tokens(self, ids: Tensor) -> Tensor:
        return self.embeddings(ids)

    def forward(self, inputs_embeds: Tensor) -> Tensor:
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[-1] != self.d_model:
            raise ValueError("inputs_embeds must have shape [batch, sequence, d_model]")
        length = inputs_embeds.shape[1]
        position = torch.arange(length, device=inputs_embeds.device, dtype=torch.float32)[:, None]
        frequency = torch.exp(torch.arange(0, self.d_model, 2, device=inputs_embeds.device,
                                          dtype=torch.float32) * (-math.log(10000.) / self.d_model))
        encoding = inputs_embeds.new_zeros(length, self.d_model)
        encoding[:, 0::2] = torch.sin(position * frequency).to(inputs_embeds.dtype)
        encoding[:, 1::2] = torch.cos(position * frequency[:self.d_model // 2]).to(inputs_embeds.dtype)
        causal_mask = torch.ones(length, length, dtype=torch.bool, device=inputs_embeds.device).triu(1)
        return self.transformer(inputs_embeds + encoding, mask=causal_mask)
