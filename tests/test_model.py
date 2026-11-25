"""Math, causal assembly, and real-HF adapter regressions for the generator."""

import math
import subprocess
import sys
import unittest

import torch
from torch import nn
from torch.nn import functional as F


from mambagir import decoder as decoder_module
from mambagir import model as model_module


class PatchEncoder(nn.Module):
    """A small real, stateful encoder; inputs already contain patch features."""

    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.norm = nn.BatchNorm1d(feature_dim)

    def forward_patches(self, images):
        return self.norm(images.transpose(1, 2)).transpose(1, 2)


class CumulativeDecoder(nn.Module):
    """A causal sum has analytically tractable context and stage embeddings."""

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.seen_length = None

    def embed_tokens(self, ids):
        return self.embeddings(ids)

    def forward(self, inputs_embeds):
        self.seen_length = inputs_embeds.shape[1]
        return inputs_embeds.cumsum(dim=1)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(28)
        torch.set_num_threads(1)

    def make_model(self, levels=3, vocab=4, dim=8, conditioning="cavp", decoder=None):
        decoder = decoder or CumulativeDecoder(dim, levels * vocab + 1)
        return model_module.MambaGIR(
            PatchEncoder(dim), decoder, torch.randn(levels, vocab, dim),
            num_classes=3, queries_per_level=2, num_heads=2, dropout=0,
            conditioning=conditioning,
        )

    def test_cavp_matches_paper_without_an_extra_ffn_residual(self):
        cavp = model_module.CAVP(
            2, 2, levels=2, codebook_size=2, queries_per_level=1,
            num_heads=1, ffn_ratio=1, dropout=0,
        )
        with torch.no_grad():
            cavp.visual_projection.weight.copy_(torch.eye(2))
            cavp.visual_projection.bias.zero_()
            cavp.queries.copy_(torch.tensor([[[1., 1.]], [[0., 1.]]]))
            cavp.gamma.copy_(torch.tensor([.25, 2.]))
            for attention in (cavp.patch_attention, cavp.code_attention):
                attention.in_proj_weight.zero_()
                attention.in_proj_weight[4:].copy_(torch.eye(2))
                attention.in_proj_bias.zero_()
                attention.out_proj.weight.copy_(torch.eye(2))
                attention.out_proj.bias.zero_()
            cavp.norm.eps = 0
            for layer in cavp.ffn:
                if isinstance(layer, nn.Linear):
                    layer.weight.copy_(torch.eye(2))
                    layer.bias.zero_()
        patches = torch.tensor([2., 0.]).expand(1, 49, 2)
        memory = torch.tensor([[[0., 6.], [0., 2.]], [[0., 1.], [0., 3.]]])
        expected = torch.tensor([[[[.841344746, -.158655254]],
                                   [[-.158655254, .841344746]]]])
        torch.testing.assert_close(cavp(patches, memory), expected)

    def test_cavp_uses_patch_distribution_beyond_global_average(self):
        cavp = model_module.CAVP(8, 8, levels=3, codebook_size=4,
                                queries_per_level=2, num_heads=2, dropout=0).eval()
        flat = torch.zeros(1, 49, 8)
        varied = flat.clone()
        varied[:, 0] = torch.arange(8.) * 10
        varied[:, 1] = -varied[:, 0]
        memory = torch.randn(3, 4, 8)
        torch.testing.assert_close(flat.mean(1), varied.mean(1))
        self.assertGreater((cavp(flat, memory) - cavp(varied, memory)).abs().max().item(), 1e-4)

    def test_memory_normalizes_codebooks_and_uses_stage_offsets(self):
        decoder = CumulativeDecoder(2, 5)
        with torch.no_grad():
            decoder.embeddings.weight.copy_(torch.tensor([[10., 0.], [20., 0.],
                                                         [30., 0.], [40., 0.], [50., 0.]]))
        books = torch.tensor([[[1., -1.], [-1., 1.]], [[3., 1.], [1., 3.]]])
        model = model_module.MambaGIR(PatchEncoder(2), decoder, books, 2,
                                     queries_per_level=1, num_heads=1, dropout=0)
        model.memory_norm.eps = 0
        with torch.no_grad():
            model.memory_projection.weight.copy_(torch.diag(torch.tensor([2., 3.])))
        _, memory = model.encode(torch.randn(1, 49, 2))
        expected = torch.tensor([[[12., -3.], [18., 3.]], [[32., -3.], [38., 3.]]])
        torch.testing.assert_close(memory, expected)
        books.zero_()
        self.assertGreater(model.codebooks.abs().sum().item(), 0)

    def test_fixed_conditioning_matches_query_dependent_path_for_all_variants(self):
        for conditioning in model_module.CONDITIONING_MODES:
            with self.subTest(conditioning=conditioning):
                model = self.make_model(conditioning=conditioning).eval()
                self.assertTrue(callable(getattr(model, 'cache_conditioning', None)))
                cache = model.cache_conditioning()
                for batch in (1, 3):
                    images = torch.randn(batch, 49, 8)
                    with torch.no_grad():
                        expected_prompts, expected_memory = model.encode(images)
                        prompts, memory = model.encode(images, conditioning_cache=cache)
                    torch.testing.assert_close(prompts, expected_prompts)
                    torch.testing.assert_close(memory, expected_memory, rtol=0, atol=0)
                model.train()
                with self.assertRaisesRegex(ValueError, 'eval|inference'):
                    model.encode(images, conditioning_cache=cache)

    def test_next_logits_matches_normalized_memory_head(self):
        decoder = CumulativeDecoder(2, 5)
        model = model_module.MambaGIR(PatchEncoder(2), decoder, torch.randn(2, 2, 2),
                                     2, queries_per_level=1, num_heads=1, dropout=0)
        with torch.no_grad():
            decoder.embeddings.weight.zero_()
            decoder.embeddings.weight[4].copy_(torch.tensor([1., -1.]))
            model.prompt_projection.weight.copy_(torch.eye(2))
            model.output_projection.weight.copy_(torch.eye(2))
            model.output_bias.copy_(torch.tensor([[.1, .2], [8., 9.]]))
            model.output_norm.eps = 0
        prompts = torch.tensor([[[[1., 0.]], [[0., 2.]]]])
        memory = torch.tensor([[[2., 1.], [-2., 3.]], [[100., 0.], [0., 100.]]])
        actual = model.next_logits(prompts, memory, torch.empty(1, 0, dtype=torch.long))
        torch.testing.assert_close(actual, torch.tensor([[.8071067812, -3.335533906]]))

    def test_previous_codes_use_their_own_stage_embeddings(self):
        decoder = CumulativeDecoder(2, 7)
        model = model_module.MambaGIR(PatchEncoder(2), decoder, torch.randn(3, 2, 2),
                                     2, queries_per_level=1, num_heads=1, dropout=0)
        with torch.no_grad():
            decoder.embeddings.weight.zero_()
            decoder.embeddings.weight[0] = torch.tensor([0., 1.])
            decoder.embeddings.weight[1] = torch.tensor([0., 1.])
            decoder.embeddings.weight[2] = torch.tensor([3., 0.])
            model.prompt_projection.weight.zero_()
            model.output_projection.weight.copy_(torch.eye(2))
            model.output_norm.eps = 0
        prompts = torch.zeros(1, 3, 1, 2)
        memory = torch.eye(2).expand(3, 2, 2)
        actual = model.next_logits(prompts, memory, torch.tensor([[1, 0]]))
        torch.testing.assert_close(actual, torch.tensor([[1 / math.sqrt(2), -1 / math.sqrt(2)]]))

    def test_zero_codes_contribute_to_generation_and_alignment_gradients(self):
        model = self.make_model()
        result = model(torch.randn(2, 49, 8), torch.zeros(2, 3, dtype=torch.long),
                       labels=torch.tensor([0, 2]))
        result["loss"].backward()
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertGreater(result["loss_gen"].item(), 0)
        self.assertGreater(result["loss_align"].item(), 0)
        self.assertTrue(torch.all(model.output_bias.grad[:, 0] < 0))
        self.assertEqual(tuple(result["logits"].shape), (2, 3, 4))
        expected = result["loss_gen"] + .05 * result["loss_align"] + .1 * result["loss_cls"]
        torch.testing.assert_close(result["loss"], expected)

    def test_alignment_uses_cosine_scores_and_mean_ce(self):
        model = self.make_model()
        model.eval()
        images = torch.randn(2, 49, 8)
        targets = torch.tensor([[0, 1, 2], [3, 2, 1]])
        prompts, memory = model.encode(images)
        result = model(images, targets)
        scores = torch.einsum("bld,lvd->blv", F.normalize(prompts.mean(2), dim=-1),
                              F.normalize(memory, dim=-1)) / .07
        expected = F.cross_entropy(scores.flatten(0, 1), targets.flatten())
        torch.testing.assert_close(result["loss_align"], expected)
        torch.testing.assert_close(result["loss_gen"],
                                   F.cross_entropy(result["logits"].flatten(0, 1), targets.flatten()))
        self.assertEqual(result["loss_cls"].item(), 0)

    def test_future_targets_never_change_earlier_teacher_forced_logits(self):
        decoder = decoder_module.TransformerDecoder(8, layers=2, num_heads=2,
                                                    vocab_size=13, dropout=0)
        model = self.make_model(decoder=decoder).eval()
        images = torch.randn(2, 49, 8)
        original = torch.tensor([[0, 1, 2], [3, 2, 1]])
        changed = original.clone()
        changed[:, 1:] = (changed[:, 1:] + 1) % 4
        left = model(images, original)["logits"]
        right = model(images, changed)["logits"]
        torch.testing.assert_close(left[:, :2], right[:, :2])
        self.assertGreater((left[:, 2] - right[:, 2]).abs().max().item(), 1e-6)

    def test_teacher_forcing_agrees_with_each_autoregressive_step(self):
        for mode in ("cavp", "linear", "gated"):
            with self.subTest(mode=mode):
                decoder = decoder_module.TransformerDecoder(8, layers=1, num_heads=2,
                                                            vocab_size=13, dropout=0)
                model = self.make_model(conditioning=mode, decoder=decoder).eval()
                images = torch.randn(2, 49, 8)
                targets = torch.tensor([[0, 1, 2], [3, 2, 1]])
                full = model(images, targets)["logits"]
                prompts, memory = model.encode(images)
                for level in range(3):
                    step = model.next_logits(prompts, memory, targets[:, :level])
                    torch.testing.assert_close(step, full[:, level], atol=1e-6, rtol=1e-5)

    def test_encoder_stays_frozen_and_eval_during_training(self):
        model = self.make_model()
        before = {name: tensor.clone() for name, tensor in model.encoder.named_buffers()}
        model.train()
        self.assertFalse(model.encoder.training)
        result = model(torch.randn(2, 49, 8) + 100, torch.ones(2, 3, dtype=torch.long))
        result["loss"].backward()
        for name, tensor in model.encoder.named_buffers():
            torch.testing.assert_close(tensor, before[name])
        self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
                            for parameter in model.encoder.parameters()))
        self.assertFalse(model.codebooks.requires_grad)
        self.assertGreater(model.cavp.queries.grad.abs().sum().item(), 0)

    def test_all_conditioning_modes_have_working_differentiable_paths(self):
        for mode in ("cavp", "levelwise", "linear", "gated", "patch_pool",
                     "patch_attention", "no_code_memory"):
            with self.subTest(mode=mode):
                model = self.make_model(levels=2, vocab=3, conditioning=mode)
                result = model(torch.randn(2, 49, 8), torch.tensor([[0, 1], [2, 0]]),
                               labels=torch.tensor([0, 1]))
                result["loss"].backward()
                self.assertEqual(tuple(result["logits"].shape), (2, 2, 3))
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for p in model.cavp.parameters()))
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                self.assertEqual(model.decoder.seen_length, 3 if mode in ("linear", "gated") else 6)

    def test_no_code_memory_does_not_consume_rvq_vectors(self):
        model = self.make_model(conditioning="no_code_memory").eval()
        images = torch.randn(1, 49, 8)
        prompts, memory = model.encode(images)
        with torch.no_grad():
            model.codebooks.normal_(mean=100, std=30)
        prompts_after, memory_after = model.encode(images)
        torch.testing.assert_close(prompts_after, prompts)
        torch.testing.assert_close(memory_after, memory)

    def test_generic_prompt_binding_is_shared_while_levelwise_binding_is_staged(self):
        generic = self.make_model(conditioning="patch_attention").eval()
        staged = self.make_model(conditioning="levelwise").eval()
        staged.load_state_dict(generic.state_dict(), strict=True)
        images = torch.randn(2, 49, 8)
        prompts, memory = generic.encode(images)
        staged_prompts, staged_memory = staged.encode(images)
        # Batching attention queries differently alone is not an ablation.
        torch.testing.assert_close(prompts, staged_prompts)
        torch.testing.assert_close(memory, staged_memory)
        prefix = torch.empty(2, 0, dtype=torch.long)
        generic_logits = generic.next_logits(prompts, memory, prefix)
        reordered = generic.next_logits(prompts.flip(1), memory, prefix)
        torch.testing.assert_close(generic_logits, reordered)
        staged_logits = staged.next_logits(prompts, memory, prefix)
        self.assertGreater((staged_logits - generic_logits).abs().max().item(), 1e-6)
        self.assertGreater((staged_logits - staged.next_logits(prompts.flip(1), memory, prefix)).abs().max().item(), 1e-6)

    def test_shared_prompt_binding_has_consistent_training_and_generation(self):
        for mode in ("patch_attention", "patch_pool", "levelwise", "no_code_memory"):
            with self.subTest(mode=mode):
                model = self.make_model(conditioning=mode).eval()
                images = torch.randn(2, 49, 8)
                targets = torch.tensor([[0, 1, 2], [3, 2, 1]])
                teacher = model(images, targets)["logits"]
                prompts, memory = model.encode(images)
                for level in range(3):
                    torch.testing.assert_close(model.next_logits(prompts, memory, targets[:, :level]),
                                               teacher[:, level], atol=1e-6, rtol=1e-5)

    def test_invalid_capacity_codes_and_prefix_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            self.make_model(decoder=CumulativeDecoder(8, 12))
        model = self.make_model()
        images = torch.randn(1, 49, 8)
        for codes in (torch.tensor([[0, 1, 4]]), torch.tensor([[0, -1, 2]]),
                      torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 3)):
            with self.subTest(codes=codes):
                with self.assertRaises((ValueError, TypeError)):
                    model(images, codes)
        prompts, memory = model.encode(images)
        with self.assertRaises(ValueError):
            model.next_logits(prompts, memory, torch.zeros(1, 3, dtype=torch.long))

    def test_model_imports_do_not_import_transformers_timm_or_peft(self):
        # Earlier SciPy/FAISS tests can export MKL_THREADING_LAYER=INTEL. NumPy
        # must initialize MKL before Torch loads libgomp in this fresh process.
        code = ("import sys; import numpy; import mambagir.model, mambagir.decoder; "
                "assert not ({'transformers', 'timm', 'peft'} & set(sys.modules))")
        completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


class RealMambaAdapterTests(unittest.TestCase):
    def setUp(self):
        try:
            from transformers import MambaConfig, MambaModel
        except (ImportError, AttributeError) as exc:
            self.skipTest(f"Compatible optional HF Mamba unavailable: {exc}")
        torch.manual_seed(28)
        torch.set_num_threads(1)
        self.backbone = MambaModel(MambaConfig(
            vocab_size=16, hidden_size=8, state_size=4, num_hidden_layers=2,
            expand=2, conv_kernel=3, time_step_rank=2, use_cache=False,
        ))

    def test_real_mamba_backpropagates_through_every_requested_adapter(self):
        decoder = decoder_module.MambaDecoder.from_backbone(
            self.backbone, rank=2, alpha=4, dropout=0,
        )
        ids = torch.tensor([[0, 3, 7, 12], [8, 2, 4, 15]])
        hidden = decoder(decoder.embed_tokens(ids))
        (hidden * torch.randn_like(hidden)).sum().backward()
        self.assertEqual(tuple(hidden.shape), (2, 4, 8))
        updates = []
        for name, parameter in decoder.named_parameters():
            if parameter.requires_grad:
                self.assertIn("lora_", name)
                self.assertIsNotNone(parameter.grad, name)
                if "lora_B" in name:
                    self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
                    updates.append(name)
            else:
                self.assertIsNone(parameter.grad, name)
        for target in ("x_proj", "in_proj", "out_proj", "embeddings"):
            self.assertTrue(any(target in name for name in updates), target)

    def test_fused_dispatch_cannot_bypass_native_lora(self):
        decoder = decoder_module.MambaDecoder.from_backbone(self.backbone, rank=2, dropout=0)
        for layer in decoder.backbone.layers:
            self.assertEqual(layer.mixer.forward.__func__, layer.mixer.slow_forward.__func__)
        self.assertTrue(decoder.adapter_config["forced_slow_forward"])

    def test_frozen_adaptation_preserves_input_gradient(self):
        decoder = decoder_module.MambaDecoder.from_backbone(self.backbone, adaptation="frozen")
        inputs = torch.randn(2, 4, 8, requires_grad=True)
        (decoder(inputs) * torch.randn(2, 4, 8)).sum().backward()
        self.assertGreater(inputs.grad.abs().sum().item(), 0)
        self.assertTrue(all(not p.requires_grad for p in decoder.parameters()))

    def test_real_mamba_generation_loss_updates_adapters_and_not_encoder(self):
        decoder = decoder_module.MambaDecoder.from_backbone(self.backbone, rank=2, dropout=0)
        model = model_module.MambaGIR(PatchEncoder(8), decoder, torch.randn(3, 4, 8),
                                     3, queries_per_level=2, num_heads=2, dropout=0)
        result = model(torch.randn(2, 49, 8), torch.tensor([[0, 1, 2], [3, 0, 1]]),
                       labels=torch.tensor([0, 2]))
        result["loss"].backward()
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertGreater(decoder.backbone.layers[0].mixer.x_proj.lora_B.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
