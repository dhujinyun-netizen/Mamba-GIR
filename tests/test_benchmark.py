import contextlib
import importlib.util
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn

from mambagir import runtime
from mambagir.retrieval import GalleryIndex


class TinyRetriever(nn.Module):
    levels = 2
    codebook_size = 2
    queries_per_level = 1
    conditioning = 'cavp'

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.], [-1.]]))
        self.scale = nn.Parameter(torch.ones(1), requires_grad=False)
        self.encode_calls = 0
        self.logit_calls = 0
        self.training_seen = []

    def encode(self, images):
        self.encode_calls += 1
        self.training_seen.append(self.training)
        return images.mean((1, 2, 3)).reshape(-1, 1, 1, 1), images.new_empty(0)

    def next_logits(self, prompts, memory, prefixes):
        self.logit_calls += 1
        return nn.functional.linear(prompts.flatten(1), self.weight) * self.scale


class CountingGallery(GalleryIndex):
    def __init__(self):
        super().__init__(torch.tensor([[0, 0], [0, 0], [1, 1]]))
        self.resolve_calls = 0

    def resolve(self, keys, topk):
        self.resolve_calls += 1
        return super().resolve(keys, topk)


class TinyEncoder(nn.Module):
    feature_dim = 8

    def __init__(self, variant=None):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)

    def forward_patches(self, images):
        return nn.functional.adaptive_avg_pool2d(self.conv(images), (7, 7)).flatten(2).transpose(1, 2)


class BenchmarkTests(unittest.TestCase):
    def benchmark_module(self):
        self.assertIsNotNone(importlib.util.find_spec('mambagir.benchmark'),
                             'the retrieval timing entry point is missing')
        from mambagir import benchmark
        return benchmark

    def test_times_real_retrieval_excludes_warmup_and_preserves_inputs(self):
        benchmark = self.benchmark_module()
        model, gallery = TinyRetriever(), CountingGallery()
        images = torch.ones(2, 3, 4, 4)
        before = images.clone()
        codes_before = gallery.codes.clone()
        with patch('torch.backends.cudnn.version', side_effect=AssertionError('CPU timing must not initialize cuDNN')):
            report = benchmark.measure_retrieval(model, images, gallery, beam_size=2,
                                                 warmup=2, iterations=3, batch_size=2)
        self.assertEqual(model.encode_calls, 5)
        self.assertEqual(model.logit_calls, 10)
        self.assertEqual(gallery.resolve_calls, 5)
        self.assertEqual(model.training_seen, [False] * 5)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(images, before))
        self.assertTrue(torch.equal(gallery.codes, codes_before))
        self.assertIsNone(model.weight.grad)
        self.assertEqual(report['last_gallery_rows'], [[0, 1], [0, 1]])
        self.assertEqual(report['parameter_counts'], {'total': 3, 'trainable': 2})
        self.assertEqual(report['visual_prefix_tokens'], 2)
        self.assertEqual(report['iterations'], 3)
        self.assertEqual(report['warmup'], 2)
        samples = report['batch_latency_ms']['samples']
        self.assertEqual(len(samples), 3)
        for name in ('mean', 'median', 'p95'):
            value = report['batch_latency_ms'][name]
            self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(value, min(samples))
            self.assertLessEqual(value, max(samples))
        self.assertTrue(all(value >= 0 for value in samples))
        self.assertIsNone(report['cuda_memory'])
        self.assertFalse(report['preprocessing_included'])
        self.assertFalse(report['input_transfer_included'])
        self.assertFalse(report['precision']['autocast_enabled'])
        json.dumps(report)

    def test_rejects_invalid_budgets_and_incompatible_batch_without_retrieval(self):
        benchmark = self.benchmark_module()
        model, gallery = TinyRetriever(), CountingGallery()
        images = torch.ones(2, 3, 4, 4)
        for options in ({'iterations': 0}, {'iterations': -1}, {'warmup': -1},
                        {'batch_size': 1}, {'batch_size': 0}, {'beam_size': 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                benchmark.measure_retrieval(model, images, gallery, **options)
        self.assertEqual(model.encode_calls, 0)
        self.assertEqual(gallery.resolve_calls, 0)

    def test_cli_loads_clean_artifact_and_only_requested_query_batch(self):
        benchmark = self.benchmark_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = runtime.load_config()
            config['workers'] = 0
            config['image'] = {'resize': 8, 'crop': 8}
            config['decoder'].update(family='transformer', d_model=8, layers=1, num_heads=2)
            config['model'].update(num_heads=2, queries_per_level=2, dropout=0.)
            config['evaluation'].update(beam_size=2, ks=[1, 2])
            targets = [{'path': str(root / f'target{i}.png'), 'label': i} for i in range(2)]
            queries = [{'path': str(root / f'query{i}.png'), 'label': i % 2} for i in range(3)]
            # The third query is intentionally absent: this CLI measures one batch.
            for i in range(2):
                Image.new('RGB', (8, 8), (20 + i, 30, 40)).save(queries[i]['path'])
            manifest_path = root / 'manifest.json'
            manifest_path.write_text(json.dumps({'targets': targets, 'queries': queries,
                                                 'gallery_indices': [0, 1]}))
            tokenizer = {
                'format_version': 1, 'kind': 'mambagir_tokenizer', 'artifact_id': 'synthetic-export',
                'config': config, 'encoder_state': TinyEncoder().state_dict(),
                'codebooks': torch.arange(32, dtype=torch.float32).reshape(2, 2, 8) / 10,
                'target_codes': torch.tensor([[0, 0], [1, 1]]), 'num_classes': 2,
                'target_paths': [r['path'] for r in targets], 'target_labels': [0, 1],
                'gallery_indices': [0, 1],
            }
            tokenizer_path = root / 'tokenizer.pt'
            torch.save(tokenizer, tokenizer_path)
            with patch('mambagir.runtime.VisualEncoder', TinyEncoder):
                model = runtime.build_generator(config, tokenizer, torch.device('cpu'))
                checkpoint = {
                    'format_version': 1, 'kind': 'mambagir_generator', 'config': config,
                    'tokenizer_path': str(tokenizer_path), 'tokenizer_id': 'synthetic-export',
                    'trainable_state': runtime._trainable_state(model), 'epoch': 0,
                }
                checkpoint_path, output = root / 'generator.pt', root / 'timing.json'
                torch.save(checkpoint, checkpoint_path)
                with contextlib.redirect_stdout(io.StringIO()):
                    benchmark.main(['--checkpoint', str(checkpoint_path), '--manifest', str(manifest_path),
                                    '--output', str(output), '--device', 'cpu', '--warmup', '1',
                                    '--iterations', '2', '--batch-size', '2', '--beam-size', '2'])
            report = json.loads(output.read_text())
            self.assertEqual(report['query_paths'], [r['path'] for r in queries[:2]])
            self.assertEqual(report['decoder_family'], 'transformer')
            self.assertEqual(report['conditioning'], 'cavp')
            self.assertEqual(report['tokenizer_id'], 'synthetic-export')
            self.assertEqual(report['batch_size'], 2)
            self.assertEqual(report['iterations'], 2)
            self.assertEqual(report['resolved_topk'], 2)
            self.assertIn('torch', report['software'])
            self.assertIn('device', report['hardware'])

            tokenizer['artifact_id'] = 'different-export'
            torch.save(tokenizer, tokenizer_path)
            with self.assertRaisesRegex(ValueError, 'tokenizer.*identity'):
                benchmark.benchmark_checkpoint(checkpoint_path, manifest_path, output, device='cpu')
            checkpoint['kind'] = 'IRGen'
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(ValueError, 'clean-version generator'):
                benchmark.benchmark_checkpoint(checkpoint_path, manifest_path, output, device='cpu')


if __name__ == '__main__':
    unittest.main()
