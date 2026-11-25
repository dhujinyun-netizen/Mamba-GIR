"""Paper-aligned codebook-aware prompting and autoregressive RVQ prediction.

Main-mode equations correspond to ``patchprompt``, ``codemem``, ``cavp``,
``head``, ``lgen``, and ``loss`` in TCE_MambaGIR_FINAL.tex. Alternate conditioning
modes are explicit implementations, not assertions of the paper's unreported
architecture details or measured ablation performance.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


CONDITIONING_MODES = frozenset({
    "cavp", "levelwise", "linear", "gated", "patch_pool", "patch_attention", "no_code_memory",
})


class CAVP(nn.Module):
    """Project 49 visual patches into [B, level, query, decoder_dim] prompts.

    ``cavp`` implements both cross-attentions and FFN(LN(T + gamma * CA(T,C,C))).
    Attention and FFN weights are shared across stages; queries/gamma are staged.
    ``levelwise`` omits code attention. ``no_code_memory`` uses this same prompt
    path and additionally removes RVQ vectors from MambaGIR's prediction memory.
    ``linear`` maps a global mean to one token; ``gated`` adds a sigmoid gate.
    Their returned prompt groups repeat that token for the head interface, while
    MambaGIR sends exactly one token to its decoder. ``patch_pool`` adaptively
    pools the ordered patches to L*R slots. ``patch_attention`` attends with one
    bank of L*R queries and reshapes it for the common tensor interface. Its
    attention operator alone is equivalent to levelwise without code attention;
    MambaGIR distinguishes a shared prompt summary from stage-bound summaries.
    """

    def __init__(
        self, visual_dim: int, d_model: int, levels: int = 4, codebook_size: int = 256,
        queries_per_level: int = 4, num_heads: int = 8, ffn_ratio: float = 2,
        dropout: float = .1, gamma_init: float = .1, conditioning: str = "cavp",
    ):
        super().__init__()
        if min(visual_dim, d_model, levels, codebook_size, queries_per_level, num_heads) < 1:
            raise ValueError("CAVP dimensions must be positive")
        if d_model % num_heads or ffn_ratio <= 0:
            raise ValueError("d_model must be divisible by num_heads and ffn_ratio positive")
        if conditioning not in CONDITIONING_MODES:
            raise ValueError(f"unknown conditioning mode: {conditioning}")
        self.visual_dim = visual_dim
        self.d_model = d_model
        self.levels = levels
        self.codebook_size = codebook_size
        self.queries_per_level = queries_per_level
        self.conditioning = conditioning
        self.visual_projection = nn.Linear(visual_dim, d_model)
        if conditioning in {"linear", "gated"}:
            self.global_projection = nn.Linear(d_model, d_model)
            if conditioning == "gated":
                self.global_gate = nn.Linear(d_model, d_model)
        else:
            if conditioning != "patch_pool":
                self.queries = nn.Parameter(torch.empty(levels, queries_per_level, d_model))
                nn.init.normal_(self.queries, std=.02)
                self.patch_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            if conditioning == "cavp":
                self.code_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
                self.gamma = nn.Parameter(torch.full((levels,), float(gamma_init)))
            self.norm = nn.LayerNorm(d_model)
            hidden = int(ffn_ratio * d_model)
            if hidden < 1:
                raise ValueError("ffn_ratio must yield at least one hidden unit")
            self.ffn = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, d_model), nn.Dropout(dropout))

    def project_code_memory(self, memory: Tensor) -> tuple[Tensor, Tensor]:
        """Query-independent K/V projections for eval-mode codebook attention."""
        attention = self.code_attention
        bias = attention.in_proj_bias
        projected = F.linear(memory, attention.in_proj_weight[self.d_model:],
                             None if bias is None else bias[self.d_model:])
        heads = attention.num_heads
        return tuple(part.reshape(self.levels, self.codebook_size, heads, self.d_model // heads)
                     .transpose(1, 2).contiguous() for part in projected.chunk(2, dim=-1))

    def _cached_code_attention(self, query: Tensor, code_kv: tuple[Tensor, Tensor]) -> Tensor:
        attention = self.code_attention
        count, length, _ = query.shape
        heads = attention.num_heads
        bias = attention.in_proj_bias
        projected = F.linear(query, attention.in_proj_weight[:self.d_model],
                             None if bias is None else bias[:self.d_model])
        projected = projected.reshape(count, length, heads, self.d_model // heads).transpose(1, 2)
        keys, values = [part[None].expand(count // self.levels, -1, -1, -1, -1)
                        .reshape(count, heads, self.codebook_size, self.d_model // heads)
                        for part in code_kv]
        attended = F.scaled_dot_product_attention(projected, keys, values)
        attended = attended.transpose(1, 2).contiguous().reshape(count, length, self.d_model)
        return attention.out_proj(attended)

    def forward(self, patches: Tensor, memory: Tensor,
                code_kv: tuple[Tensor, Tensor] | None = None) -> Tensor:
        if patches.ndim != 3 or patches.shape[1:] != (49, self.visual_dim):
            raise ValueError(f"patches must have shape [B, 49, {self.visual_dim}]")
        if tuple(memory.shape) != (self.levels, self.codebook_size, self.d_model):
            raise ValueError("memory shape must be [levels, codebook_size, d_model]")
        batch = patches.shape[0]
        projected = self.visual_projection(patches)
        if self.conditioning in {"linear", "gated"}:
            pooled = projected.mean(dim=1)
            token = self.global_projection(pooled)
            if self.conditioning == "gated":
                token = token * torch.sigmoid(self.global_gate(pooled))
            return token[:, None, None, :].expand(batch, self.levels, self.queries_per_level, self.d_model)
        if self.conditioning == "patch_pool":
            evidence = F.adaptive_avg_pool1d(projected.transpose(1, 2), self.levels * self.queries_per_level)
            evidence = evidence.transpose(1, 2).reshape(batch, self.levels, self.queries_per_level, self.d_model)
        elif self.conditioning == "patch_attention":
            query = self.queries.flatten(0, 1)[None].expand(batch, -1, -1)
            evidence = query + self.patch_attention(query, projected, projected, need_weights=False)[0]
            evidence = evidence.reshape(batch, self.levels, self.queries_per_level, self.d_model)
        else:
            query = self.queries[None].expand(batch, -1, -1, -1).reshape(-1, self.queries_per_level, self.d_model)
            patches_by_level = projected[:, None].expand(-1, self.levels, -1, -1).reshape(-1, 49, self.d_model)
            evidence = query + self.patch_attention(query, patches_by_level, patches_by_level, need_weights=False)[0]
            evidence = evidence.reshape(batch, self.levels, self.queries_per_level, self.d_model)
            if self.conditioning == "cavp":
                query = evidence.reshape(-1, self.queries_per_level, self.d_model)
                if code_kv is None:
                    stage_memory = memory[None].expand(batch, -1, -1, -1).reshape(-1, self.codebook_size, self.d_model)
                    attended = self.code_attention(query, stage_memory, stage_memory, need_weights=False)[0]
                else:
                    if self.training:
                        raise ValueError('cached codebook attention is for inference in eval mode')
                    attended = self._cached_code_attention(query, code_kv)
                evidence = evidence + self.gamma[None, :, None, None] * attended.reshape_as(evidence)
        # Equation (cavp) has no residual around the FFN.
        return self.ffn(self.norm(evidence))


class MambaGIR(nn.Module):
    """Frozen encoder/codebooks with differentiable CAVP, decoder and RVQ heads."""

    def __init__(
        self, encoder: nn.Module, decoder: nn.Module, codebooks: Tensor, num_classes: int,
        queries_per_level: int = 4, num_heads: int = 8, ffn_ratio: float = 2,
        dropout: float = .1, gamma_init: float = .1, alignment_temperature: float = .07,
        lambda_align: float = .05, lambda_cls: float = .1, conditioning: str = "cavp",
        head_prompt_mode: str | None = None,
    ):
        super().__init__()
        if codebooks.ndim != 3 or min(codebooks.shape) < 1 or not codebooks.is_floating_point():
            raise ValueError("codebooks must be a nonempty floating tensor [levels, vocabulary, visual_dim]")
        if codebooks.shape[-1] != encoder.feature_dim:
            raise ValueError("codebook feature dimension must match encoder.feature_dim")
        if num_classes < 1 or alignment_temperature <= 0 or min(lambda_align, lambda_cls) < 0:
            raise ValueError("num_classes and alignment_temperature must be positive; loss weights nonnegative")
        self.levels, self.codebook_size, visual_dim = map(int, codebooks.shape)
        self.d_model = int(decoder.d_model)
        self.num_classes = num_classes
        self.queries_per_level = queries_per_level
        self.conditioning = conditioning
        self.head_prompt_mode = head_prompt_mode or (
            "shared" if conditioning in {"patch_pool", "patch_attention"} else "stage")
        if self.head_prompt_mode not in {"shared", "stage"}:
            raise ValueError("head_prompt_mode must be 'shared' or 'stage'")
        self.sos_id = self.levels * self.codebook_size
        if decoder.vocab_size <= self.sos_id:
            raise ValueError(f"decoder vocabulary needs at least {self.sos_id + 1} tokens for stage codes and SOS")
        self.encoder = encoder.requires_grad_(False)
        self.encoder.eval()
        self.decoder = decoder
        self.register_buffer("codebooks", codebooks.detach().clone())
        self.cavp = CAVP(visual_dim, self.d_model, self.levels, self.codebook_size,
                         queries_per_level, num_heads, ffn_ratio, dropout, gamma_init, conditioning)
        if conditioning != "no_code_memory":
            self.memory_norm = nn.LayerNorm(visual_dim)
            self.memory_projection = nn.Linear(visual_dim, self.d_model, bias=False)
        self.prompt_projection = nn.Linear(self.d_model, self.d_model, bias=False)
        self.output_norm = nn.LayerNorm(self.d_model)
        self.output_projection = nn.Linear(self.d_model, self.d_model, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(self.levels, self.codebook_size))
        self.classifier = nn.Linear(self.d_model, num_classes)
        self.alignment_temperature = alignment_temperature
        self.lambda_align = lambda_align
        self.lambda_cls = lambda_cls

    def train(self, mode: bool = True) -> MambaGIR:
        super().train(mode)
        self.encoder.eval()
        return self

    @property
    def supports_cached_decoding(self):
        return all(callable(getattr(self.decoder, name, None))
                   for name in ('prefill', 'decode_step', 'reorder_cache'))

    def prepare_for_inference(self):
        """Prepare the restored model for inference, merging Mamba's LoRA once."""
        self.eval()
        merge = getattr(self.decoder, 'merge_lora_for_inference', None)
        if callable(merge):
            merge()
        return self

    def _code_memory(self) -> Tensor:
        stage_ids = torch.arange(self.sos_id, device=self.codebooks.device).reshape(self.levels, self.codebook_size)
        memory = self.decoder.embed_tokens(stage_ids)
        if self.conditioning != "no_code_memory":
            memory = self.memory_projection(self.memory_norm(self.codebooks)) + memory
        return memory

    @torch.no_grad()
    def cache_conditioning(self):
        """Prepare fixed memory/K/V after loading weights, under inference precision.

        Keep this cache with the inference runner and rebuild it after changing
        weights, device or precision. It contains no query-dependent tensors.
        """
        if self.training:
            raise ValueError('conditioning caches require eval mode')
        memory = self._code_memory()
        code_kv = self.cavp.project_code_memory(memory) if self.conditioning == 'cavp' else None
        return memory, code_kv

    def encode(self, images: Tensor, *, conditioning_cache=None) -> tuple[Tensor, Tensor]:
        if conditioning_cache is not None and self.training:
            raise ValueError('conditioning caches are for inference in eval mode')
        with torch.no_grad():
            patches = self.encoder.forward_patches(images)
        memory, code_kv = (self._code_memory(), None) if conditioning_cache is None else conditioning_cache
        return self.cavp(patches, memory, code_kv), memory

    def _validate_codes(self, codes: Tensor, batch: int, *, complete: bool) -> None:
        if codes.dtype != torch.long:
            raise TypeError("identifier codes must be torch.long")
        if codes.ndim != 2 or codes.shape[0] != batch:
            raise ValueError("codes must have shape [batch, number_of_levels]")
        if (complete and codes.shape[1] != self.levels) or (not complete and codes.shape[1] >= self.levels):
            raise ValueError("targets need exactly L codes; a generation prefix must have fewer than L")
        if codes.numel() and (codes.min() < 0 or codes.max() >= self.codebook_size):
            raise ValueError("every identifier code must lie in [0, codebook_size)")

    def _decoder_inputs(self, prompts: Tensor, prefix_codes: Tensor) -> tuple[Tensor, int]:
        batch = prompts.shape[0]
        if self.conditioning in {"linear", "gated"}:
            visual_prefix = prompts.mean(dim=(1, 2))[:, None]
        else:
            visual_prefix = prompts.flatten(1, 2)
        sos = torch.full((batch, 1), self.sos_id, device=prompts.device, dtype=torch.long)
        offsets = torch.arange(prefix_codes.shape[1], device=prompts.device) * self.codebook_size
        token_ids = torch.cat((sos, prefix_codes + offsets), dim=1)
        return torch.cat((visual_prefix, self.decoder.embed_tokens(token_ids)), dim=1), visual_prefix.shape[1]

    def _head(self, hidden: Tensor, mean_prompts: Tensor, memory: Tensor, bias: Tensor) -> Tensor:
        projected = self.output_projection(self.output_norm(hidden + self.prompt_projection(mean_prompts)))
        return torch.einsum("bld,lvd->blv", projected, memory) / math.sqrt(self.d_model) + bias

    def _mean_prompts(self, prompts: Tensor) -> Tensor:
        """Define whether the head/alignment binds prompt slots to RVQ levels.

        The decoder always sees the complete visual prefix. A generic patch
        prefix supplies the same summary at every output position; a stage-bound
        prefix supplies the matching group's mean as in the paper's full model.
        """
        if self.head_prompt_mode == "shared":
            return prompts.mean(dim=(1, 2))[:, None].expand(-1, self.levels, -1)
        return prompts.mean(dim=2)

    def next_logits(self, prompts: Tensor, memory: Tensor, prefix_codes: Tensor) -> Tensor:
        if prompts.ndim != 4 or tuple(prompts.shape[1:]) != (self.levels, self.queries_per_level, self.d_model):
            raise ValueError("prompts must have shape [B, levels, queries_per_level, d_model]")
        if tuple(memory.shape) != (self.levels, self.codebook_size, self.d_model):
            raise ValueError("memory must have shape [levels, codebook_size, d_model]")
        self._validate_codes(prefix_codes, prompts.shape[0], complete=False)
        level = prefix_codes.shape[1]
        inputs, _ = self._decoder_inputs(prompts, prefix_codes)
        hidden = self.decoder(inputs)[:, -1:]
        return self._head(hidden, self._mean_prompts(prompts)[:, level:level + 1],
                          memory[level:level + 1], self.output_bias[level:level + 1])[:, 0]

    def prefill_logits(self, prompts: Tensor, memory: Tensor):
        empty = torch.empty(len(prompts), 0, dtype=torch.long, device=prompts.device)
        inputs, _ = self._decoder_inputs(prompts, empty)
        hidden, cache = self.decoder.prefill(inputs)
        logits = self._head(hidden[:, -1:], self._mean_prompts(prompts)[:, :1],
                            memory[:1], self.output_bias[:1])[:, 0]
        return logits, cache

    def next_cached_logits(self, prompts: Tensor, memory: Tensor,
                           previous_codes: Tensor, level: int, cache) -> Tensor:
        if not 1 <= level < self.levels or previous_codes.shape != (len(prompts), 1):
            raise ValueError('cached prediction needs one preceding code and a level in [1,L)')
        tokens = previous_codes + (level - 1) * self.codebook_size
        hidden = self.decoder.decode_step(self.decoder.embed_tokens(tokens), cache)
        return self._head(hidden, self._mean_prompts(prompts)[:, level:level+1],
                          memory[level:level+1], self.output_bias[level:level+1])[:, 0]

    def reorder_cache(self, cache, parents: Tensor):
        self.decoder.reorder_cache(cache, parents)

    def forward(self, images: Tensor, target_codes: Tensor, labels: Tensor | None = None) -> dict[str, Tensor]:
        self._validate_codes(target_codes, images.shape[0], complete=True)
        if labels is not None:
            if labels.dtype != torch.long or tuple(labels.shape) != (images.shape[0],):
                raise ValueError("labels must be torch.long with shape [batch]")
            if labels.numel() and (labels.min() < 0 or labels.max() >= self.num_classes):
                raise ValueError("labels must lie in [0, num_classes)")
        prompts, memory = self.encode(images)
        # SOS predicts c_0; each later state consumes only codes from earlier stages.
        inputs, prefix_length = self._decoder_inputs(prompts, target_codes[:, :-1])
        hidden = self.decoder(inputs)[:, prefix_length:prefix_length + self.levels]
        mean_prompts = self._mean_prompts(prompts)
        logits = self._head(hidden, mean_prompts, memory, self.output_bias)
        loss_gen = F.cross_entropy(logits.flatten(0, 1), target_codes.flatten())
        alignment = torch.einsum("bld,lvd->blv", F.normalize(mean_prompts, dim=-1), F.normalize(memory, dim=-1))
        alignment = alignment / self.alignment_temperature
        loss_align = F.cross_entropy(alignment.flatten(0, 1), target_codes.flatten())
        loss_cls = (F.cross_entropy(self.classifier(prompts.mean(dim=(1, 2))), labels)
                    if labels is not None else logits.new_zeros(()))
        loss = loss_gen + self.lambda_align * loss_align + self.lambda_cls * loss_cls
        return {"loss": loss, "loss_gen": loss_gen, "loss_align": loss_align,
                "loss_cls": loss_cls, "logits": logits}
