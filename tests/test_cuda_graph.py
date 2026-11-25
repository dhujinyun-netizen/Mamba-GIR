"""CUDA replay must preserve independent queries, adapter effects and outputs."""
import unittest

import torch
from torch import nn

from mambagir import retrieval
from mambagir.decoder import MambaDecoder
from mambagir.model import MambaGIR


class PatchEncoder(nn.Module):
    feature_dim = 8

    def forward_patches(self, images):
        return images


class CudaGraphTests(unittest.TestCase):
    def runner_type(self):
        self.assertTrue(hasattr(retrieval, 'CUDAGraphBeamSearch'),
                        'retrieval needs a reusable fixed-shape CUDA graph runner')
        return retrieval.CUDAGraphBeamSearch

    def model(self):
        from transformers import MambaConfig, MambaModel
        torch.manual_seed(712)
        torch.set_num_threads(1)
        config = MambaConfig(vocab_size=32, hidden_size=8, state_size=4,
                             num_hidden_layers=2, expand=2, conv_kernel=4, time_step_rank=2)
        decoder = MambaDecoder.from_backbone(MambaModel(config), rank=2, alpha=4, dropout=.1)
        model = MambaGIR(PatchEncoder(), decoder, torch.randn(4, 4, 8),
                         3, queries_per_level=2, num_heads=2, dropout=0).cuda().eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if 'lora_B' in name:
                    parameter.normal_(std=.03)
        return model

    def test_cpu_input_is_rejected_before_capture(self):
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            self.runner_type()(nn.Identity().eval(), torch.ones(1, 3, 8, 8))

    def check_replay(self, amp):
        runner_type = self.runner_type()
        model = self.model()
        inputs = [torch.randn(2, 49, 8, device='cuda') for _ in range(3)]
        references = []
        with torch.inference_mode(), torch.autocast('cuda', enabled=amp):
            for images in inputs:
                references.append(retrieval.beam_search(model, images, 8, use_cache=False))
        model.prepare_for_inference()
        runner = runner_type(model, inputs[0], beam_size=8, amp=amp)
        retained = None
        # A -> B -> C -> A exposes state leaking between replayed queries.
        for index in (0, 1, 2, 0):
            keys, scores = runner(inputs[index])
            expected_keys, expected_scores = references[index]
            self.assertTrue(torch.equal(keys, expected_keys))
            torch.testing.assert_close(scores, expected_scores,
                                       rtol=3e-3 if amp else 1e-4,
                                       atol=3e-3 if amp else 1e-4)
            if retained is None:
                retained = (keys, scores, keys.clone(), scores.clone())
        # Returned outputs must survive the next replay of the internal buffer.
        torch.testing.assert_close(retained[0], retained[2], rtol=0, atol=0)
        torch.testing.assert_close(retained[1], retained[3], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_nonzero_adapters_batched_beams_and_independent_query_replay(self):
        self.check_replay(False)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_autocast_replay_preserves_keys_and_scores(self):
        self.check_replay(True)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_requires_prepared_model_and_matching_input_layout(self):
        runner_type = self.runner_type()
        model = self.model()
        images = torch.randn(2, 49, 8, device='cuda')
        with self.assertRaisesRegex(ValueError, 'prepare_for_inference'):
            runner_type(model, images)
        model.prepare_for_inference()
        runner = runner_type(model, images)
        for incompatible in (images[:1], images.half(), images.cpu()):
            with self.subTest(shape=incompatible.shape, dtype=incompatible.dtype), \
                    self.assertRaisesRegex(ValueError, 'shape|dtype|device'):
                runner(incompatible)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_benchmark_uses_graph_and_reports_setup_separately(self):
        from mambagir.benchmark import measure_retrieval
        class ImageEncoder(PatchEncoder):
            def forward_patches(self, images):
                pooled = nn.functional.adaptive_avg_pool2d(images.mean(1, keepdim=True), (7, 7))
                return pooled.flatten(2).transpose(1, 2).expand(-1, -1, 8)
        model = self.model()
        model.encoder = ImageEncoder().cuda().eval()
        model.prepare_for_inference()
        images = torch.randn(1, 3, 8, 8, device='cuda')
        keys, _ = retrieval.beam_search(model, images, 8)
        gallery = retrieval.GalleryIndex(keys[0].cpu())
        expected = gallery.resolve(keys, 8)
        report = measure_retrieval(model, images, gallery, beam_size=8,
                                   warmup=1, iterations=2, cuda_graph=True)
        self.assertTrue(report['cuda_graph'])
        self.assertGreater(report['cuda_graph_setup_ms'], 0)
        self.assertEqual(len(report['batch_latency_ms']['samples']), 2)
        self.assertEqual(report['last_gallery_rows'], expected.tolist())


if __name__ == '__main__':
    unittest.main()
