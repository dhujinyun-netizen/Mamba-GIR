import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn

from mambagir.runtime import load_config, learning_rate_factor, train_tokenizer, train_generator, evaluate


class TinySpatialEncoder(nn.Module):
    feature_dim = 8

    def __init__(self, variant=None):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)

    def forward_patches(self, images):
        return nn.functional.adaptive_avg_pool2d(self.conv(images), (7, 7)).flatten(2).transpose(1, 2)

    def forward_global(self, images):
        return self.conv(images).mean((-2, -1))

    def load_pretrained(self, path):
        raise AssertionError('tiny smoke test must not load external weights')


class RuntimeTests(unittest.TestCase):
    def test_schedule_warms_up_then_decays_at_configured_epoch_budget(self):
        self.assertAlmostEqual(learning_rate_factor(0, 100, 10), .1)
        self.assertEqual(learning_rate_factor(9, 100, 10), 1.)
        self.assertEqual(learning_rate_factor(99, 100, 10), 0.)

    def test_cpu_tokenizer_training_generation_and_evaluation_round_trip(self):
        config = load_config()
        config['seed'] = 8
        config['workers'] = 0
        config['image'] = {'resize': 16, 'crop': 14}
        config['rvq'].update(levels=2, codebook_size=2, iterations=2)
        config['tokenizer'].update(epochs=1, batch_size=2, head_depth=2, head_hidden_dim=8,
                                   warmup_epochs=0, allow_random_encoder=True)
        config['generator'].update(epochs=1, batch_size=2, warmup_fraction=0.)
        config['decoder'].update(family='transformer', d_model=8, layers=1, num_heads=2)
        config['model'].update(num_heads=2, queries_per_level=2, dropout=0., conditioning='patch_attention')
        config['evaluation'].update(beam_size=2, ks=[1, 2, 4], batch_size=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets, queries = [], []
            for i, color in enumerate([(255, 0, 0), (220, 0, 0), (0, 255, 0), (0, 220, 0)]):
                path = root / f'target{i}.png'
                Image.new('RGB', (16, 16), color).save(path)
                targets.append({'path': str(path), 'label': i // 2})
            for i, color in enumerate([(240, 0, 0), (0, 240, 0)]):
                path = root / f'query{i}.png'
                Image.new('RGB', (16, 16), color).save(path)
                queries.append({'path': str(path), 'label': i})
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'targets': targets, 'queries': queries, 'gallery_indices': [0, 1, 2, 3]}))
            with patch('mambagir.runtime.VisualEncoder', TinySpatialEncoder):
                tokenizer_path = train_tokenizer(config, manifest, root / 'tokenizer', 'cpu')
                checkpoint = train_generator(config, manifest, tokenizer_path, root / 'generator', 'cpu')
                report = evaluate(checkpoint, manifest, root / 'evaluation', 'cpu')
            self.assertEqual(report['num_queries'], 2)
            self.assertEqual(report['num_gallery'], 4)
            self.assertEqual(set(report['recall']), {'1', '2', '4'})
            self.assertTrue((root / 'evaluation' / 'predictions.pt').is_file())
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            self.assertEqual(saved['config']['model']['head_prompt_mode'], 'shared')
            # A pre-revision generic checkpoint used stage-bound head summaries.
            saved['config']['model']['head_prompt_mode'] = 'stage'
            explicit = root / 'explicit_stage.pt'
            torch.save(saved, explicit)
            del saved['config']['model']['head_prompt_mode']
            legacy = root / 'legacy_generic.pt'
            torch.save(saved, legacy)
            with patch('mambagir.runtime.VisualEncoder', TinySpatialEncoder):
                evaluate(explicit, manifest, root / 'stage_eval', 'cpu')
                evaluate(legacy, manifest, root / 'legacy_eval', 'cpu')
            stage_predictions = torch.load(root / 'stage_eval' / 'predictions.pt', weights_only=True)
            legacy_predictions = torch.load(root / 'legacy_eval' / 'predictions.pt', weights_only=True)
            torch.testing.assert_close(stage_predictions['log_scores'], legacy_predictions['log_scores'],
                                       atol=1e-9, rtol=1e-8)
            self.assertEqual(legacy_predictions['config']['model']['head_prompt_mode'], 'stage')
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            self.assertEqual(saved['epoch'], 1)
            self.assertTrue(all(not n.startswith('encoder.') for n in saved['trainable_state']))
            self.assertGreater(saved['parameter_counts']['trainable'], 0)
            # A different tokenizer with the same shapes and manifest must not
            # silently reinterpret the generator's identifiers.
            altered = torch.load(tokenizer_path, map_location='cpu', weights_only=True)
            altered['artifact_id'] = 'a-different-training-export'
            altered['codebooks'] = altered['codebooks'] + 1
            alternate_path = root / 'other_tokenizer.pt'
            torch.save(altered, alternate_path)
            with patch('mambagir.runtime.VisualEncoder', TinySpatialEncoder):
                with self.assertRaisesRegex(ValueError, 'tokenizer.*identity'):
                    evaluate(checkpoint, manifest, root / 'wrong_evaluation', 'cpu', tokenizer_path=alternate_path)


if __name__ == '__main__':
    unittest.main()
