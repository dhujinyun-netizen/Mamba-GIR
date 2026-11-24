import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from mambagir.data import Manifest


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest_path = self.root / 'manifest.json'
        self.manifest_path.write_text(json.dumps({
            'targets': [{'path': f'g{i}.png', 'label': label}
                        for i, label in enumerate([10, 20, 10, 10, 20])],
            'queries': [{'path': f'q{i}.png', 'label': 10} for i in range(3)],
            # Deliberately not target row order: gallery rows control bucket order.
            'gallery_indices': [3, 1, 2, 0, 4],
        }))
        manifest = Manifest.load(self.manifest_path)
        self.tokenizer = {
            'kind': 'mambagir_tokenizer', 'format_version': 1, 'artifact_id': 'fixture-v1',
            'target_codes': torch.tensor([[0], [1], [1], [0], [2]]),
            'target_paths': [row.path for row in manifest.targets],
            'target_labels': [row.label for row in manifest.targets],
            'gallery_indices': manifest.gallery_indices,
        }
        self.predictions = {
            'kind': 'mambagir_predictions', 'format_version': 1, 'tokenizer_id': 'fixture-v1',
            'raw_keys': torch.tensor([[[9], [0]], [[1], [2]], [[2], [9]]]),
            'log_scores': torch.tensor([[-.1, -.2], [-.1, -.2], [-.1, -.2]]),
            'ranks': torch.tensor([[0, 3, -1, -1], [1, 2, 4, -1], [4, -1, -1, -1]]),
            'query_paths': [row.path for row in manifest.queries],
            'query_labels': [row.label for row in manifest.queries],
            'gallery_paths': [manifest.targets[i].path for i in manifest.gallery_indices],
            'gallery_labels': [manifest.targets[i].label for i in manifest.gallery_indices],
            'metrics': {'recall': {'1': 1 / 3, '2': 2 / 3}},
        }
        self.tokenizer_path = self.root / 'tokenizer.pt'
        self.predictions_path = self.root / 'predictions.pt'
        self.output = self.root / 'diagnostics'

    def run_cli(self, *extra):
        torch.save(self.tokenizer, self.tokenizer_path)
        torch.save(self.predictions, self.predictions_path)
        return subprocess.run([
            sys.executable, '-m', 'mambagir.diagnostics',
            '--predictions', str(self.predictions_path),
            '--tokenizer', str(self.tokenizer_path), '--manifest', str(self.manifest_path),
            '--output-dir', str(self.output), *extra,
        ], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)

    def read_csv(self, name):
        with (self.output / name).open(newline='', encoding='utf-8') as handle:
            return list(csv.DictReader(handle))

    def test_offline_cli_exports_auditable_raw_and_collision_failure_cases_without_images(self):
        completed = self.run_cli()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads((self.output / 'diagnostics.json').read_text())
        self.assertAlmostEqual(report['recall']['1'], 1 / 3)
        self.assertAlmostEqual(report['recall']['2'], 2 / 3)
        self.assertAlmostEqual(report['raw_key_hit']['1'], 1 / 3)
        self.assertAlmostEqual(report['raw_key_hit']['2'], 2 / 3)
        self.assertEqual(report['tokenizer_id'], 'fixture-v1')
        raw = self.read_csv('raw_candidates.csv')
        self.assertEqual(len(raw), 6)
        self.assertEqual([row['raw_rank'] for row in raw[:2]], ['1', '2'])
        self.assertEqual(json.loads(raw[0]['members']), [])
        self.assertEqual(raw[1]['relevant_key'], 'True')
        members = json.loads(raw[1]['members'])
        self.assertEqual([row['gallery_row'] for row in members], [0, 3])
        self.assertEqual([row['target_row'] for row in members], [3, 0])
        self.assertEqual([row['label'] for row in members], [0, 0])
        self.assertEqual(members[0]['path'], str(self.root / 'g3.png'))
        cases = self.read_csv('failure_cases.csv')
        self.assertEqual([row['raw_top1_outcome'] for row in cases], [
            'invalid_key', 'mixed_label_bucket_first_member_miss', 'valid_no_relevant_member',
        ])
        self.assertEqual(cases[1]['query_path'], str(self.root / 'q1.png'))
        self.assertEqual(json.loads(cases[1]['raw_key']), [1])
        self.assertEqual(cases[1]['raw_rank'], '1')
        self.assertEqual([row['label'] for row in json.loads(cases[1]['members'])], [1, 0])
        queries = self.read_csv('queries.csv')
        self.assertEqual([row['first_relevant_raw_key_rank'] for row in queries], ['2', '1', ''])
        self.assertEqual([row['first_relevant_image_rank'] for row in queries], ['1', '2', ''])
        buckets = self.read_csv('gallery_buckets.csv')
        self.assertEqual([row['collision_type'] for row in buckets], ['same_label', 'mixed_label', 'singleton'])

    def test_offline_cli_rejects_a_different_tokenizer_even_with_matching_shapes(self):
        self.tokenizer['artifact_id'] = 'another-run'
        completed = self.run_cli()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('tokenizer identity', completed.stderr.lower())
        self.assertFalse(self.output.exists())

    def test_offline_cli_requires_saved_identity_for_legacy_predictions(self):
        del self.predictions['tokenizer_id']
        completed = self.run_cli()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('tokenizer_id', completed.stderr)

    def test_offline_cli_rejects_a_manifest_with_changed_gallery_order(self):
        self.tokenizer['gallery_indices'] = [0, 1, 2, 3, 4]
        completed = self.run_cli()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('gallery_indices', completed.stderr)

    def test_offline_cli_rejects_query_label_mismatch(self):
        self.predictions['query_labels'] = [1, 0, 0]
        completed = self.run_cli()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('query_labels', completed.stderr)

    def test_offline_cli_rejects_saved_ranks_that_reorder_a_collision_bucket(self):
        self.predictions['ranks'][1, :2] = torch.tensor([2, 1])
        completed = self.run_cli()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('ranks', completed.stderr)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
