"""Nonzero LoRA and recurrent-state equivalence for optimized retrieval."""

import unittest

import torch
from torch import nn

from mambagir.decoder import MambaDecoder, LoRALinear, LoRAEmbedding
from mambagir.model import MambaGIR
from mambagir.retrieval import beam_search


class PatchEncoder(nn.Module):
    feature_dim = 8

    def forward_patches(self, images):
        return images


class InferenceTests(unittest.TestCase):
    def setUp(self):
        try:
            from transformers import MambaConfig, MambaModel
        except (ImportError, AttributeError) as error:
            self.skipTest(str(error))
        torch.manual_seed(712)
        torch.set_num_threads(1)
        self.config = MambaConfig(vocab_size=32, hidden_size=8, state_size=4,
                                  num_hidden_layers=2, expand=2, conv_kernel=4,
                                  time_step_rank=2)
        self.model_class = MambaModel

    def decoder(self, device='cpu'):
        result = MambaDecoder.from_backbone(self.model_class(self.config),
                                           rank=2, alpha=4, dropout=.1).to(device)
        with torch.no_grad():
            for name, parameter in result.named_parameters():
                if 'lora_B' in name:
                    parameter.normal_(std=.03)
        self.assertTrue(callable(getattr(result, 'merge_lora_for_inference', None)),
                        'decoder needs an explicit inference merge operation')
        return result

    def test_nonzero_adapter_merge_preserves_output_and_is_idempotent(self):
        decoder = self.decoder().eval()
        ids = torch.tensor([[0, 5, 7, 9, 2], [3, 2, 8, 1, 6]])
        with torch.no_grad():
            before = decoder(decoder.embed_tokens(ids))
            self.assertIs(decoder.merge_lora_for_inference(), decoder)
            after = decoder(decoder.embed_tokens(ids))
            torch.testing.assert_close(after, before, rtol=3e-5, atol=3e-5)
            self.assertFalse(any(isinstance(m, (LoRALinear, LoRAEmbedding))
                                 for m in decoder.modules()))
            decoder.merge_lora_for_inference()
            torch.testing.assert_close(decoder(decoder.embed_tokens(ids)), after,
                                       rtol=0, atol=0)

    def test_merge_requires_eval_and_prevents_resuming_training(self):
        decoder = self.decoder()
        with self.assertRaisesRegex(RuntimeError, 'eval'):
            decoder.merge_lora_for_inference()
        decoder.eval().merge_lora_for_inference()
        with self.assertRaisesRegex(RuntimeError, 'inference'):
            decoder.train()

    def check_reordered_cache(self, device):
        decoder = self.decoder(device).eval().merge_lora_for_inference()
        ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], device=device)
        parents = torch.tensor([1, 0, 1], device=device)
        tails = torch.tensor([[11, 14], [12, 15], [13, 16]], device=device)
        with torch.no_grad():
            expected_prefix = decoder(decoder.embed_tokens(ids))
            prefix, cache = decoder.prefill(decoder.embed_tokens(ids))
            torch.testing.assert_close(prefix, expected_prefix, rtol=1e-4, atol=1e-4)
            decoder.reorder_cache(cache, parents)
            history = ids.index_select(0, parents)
            # Duplicated parent 1 must acquire independent state for tokens 11/13.
            for column in range(2):
                current = tails[:, column:column+1]
                result = decoder.decode_step(decoder.embed_tokens(current), cache)
                history = torch.cat((history, current), dim=1)
                expected = decoder(decoder.embed_tokens(history))[:, -1:]
                torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_reordered_and_duplicated_cache_matches_full_sequence(self):
        self.check_reordered_cache('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_reordered_cache_matches_full_sequence(self):
        self.check_reordered_cache('cuda')

    def check_beam(self, device, amp=False):
        for conditioning in ('cavp', 'gated'):
            decoder = self.decoder(device)
            model = MambaGIR(PatchEncoder(), decoder, torch.randn(4, 4, 8, device=device),
                             3, queries_per_level=2, num_heads=2, dropout=0,
                             conditioning=conditioning).to(device).eval()
            self.assertTrue(getattr(model, 'supports_cached_decoding', False),
                            'Mamba retrieval needs cached decoding support')
            images = torch.randn(2, 49, 8, device=device)
            with torch.no_grad(), torch.autocast(torch.device(device).type, enabled=amp):
                old_keys, old_scores = beam_search(model, images, 8, use_cache=False)
            model.prepare_for_inference()
            shapes = []
            handle = model.decoder.backbone.register_forward_pre_hook(
                lambda module, args, kwargs: shapes.append(tuple(kwargs['inputs_embeds'].shape[:2])),
                with_kwargs=True)
            try:
                with torch.no_grad(), torch.autocast(torch.device(device).type, enabled=amp):
                    keys, scores = beam_search(model, images, 8)
            finally:
                handle.remove()
            self.assertTrue(torch.equal(keys, old_keys))
            torch.testing.assert_close(scores, old_scores, rtol=3e-3 if amp else 1e-4,
                                       atol=3e-3 if amp else 1e-4)
            # Vocabulary 4 grows the beam from 1 to 4, then to 8 per query.
            self.assertEqual(shapes, [(2, 9 if conditioning == 'cavp' else 2),
                                      (8, 1), (16, 1), (16, 1)])

    def test_cached_batched_beams_preserve_keys_scores_and_process_only_new_tokens(self):
        self.check_beam('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_cached_batched_beams_preserve_keys_scores(self):
        self.check_beam('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_autocast_cached_beams_preserve_keys_scores(self):
        self.check_beam('cuda', amp=True)


if __name__ == '__main__':
    unittest.main()
