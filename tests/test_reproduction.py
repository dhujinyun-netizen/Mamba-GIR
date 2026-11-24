import copy
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from mambagir.data import Manifest
from mambagir.runtime import _load_tokenizer, load_config


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        try:
            from mambagir.reproduction import bind_bundle, export_bundle
        except ImportError as error:
            self.fail(f'Reproduction export is unavailable: {error}')
        self.export_bundle, self.bind_bundle = export_bundle, bind_bundle
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / 'original_data'
        self.dataset.mkdir()
        self.manifest_path = self.root / 'source_manifest.json'
        self.manifest_path.write_text(json.dumps({
            'targets': [
                {'path': str(self.dataset / 'images' / name), 'label': label}
                for name, label in [('train.jpg', 2), ('a.jpg', 0), ('b.jpg', 1),
                                    ('c.jpg', 1), ('d.jpg', 0)]
            ],
            'queries': [
                {'path': str(self.dataset / 'images/query_a.jpg'), 'label': 0},
                {'path': str(self.dataset / 'images/query_b.jpg'), 'label': 1},
            ],
            'gallery_indices': [3, 1, 4, 2],
        }))
        manifest = Manifest.load(self.manifest_path)
        self.artifact = {
            'kind': 'mambagir_tokenizer', 'format_version': 1,
            'artifact_id': 'fixture-tokenizer-export', 'config': load_config(),
            'encoder_state': {'weight': torch.arange(6.).reshape(2, 3)},
            'codebooks': torch.arange(12.).reshape(2, 2, 3),
            'target_codes': torch.tensor([[1, 1], [0, 1], [1, 0], [1, 0], [1, 0]]),
            'target_paths': [r.path for r in manifest.targets],
            'target_labels': [r.label for r in manifest.targets],
            'gallery_indices': [3, 1, 4, 2], 'num_classes': 3,
            'manifest_path': str(self.manifest_path), 'epoch': 200,
            'parameter_counts': {'total': 6, 'trainable': 6},
        }
        self.artifact['config']['rvq'].update(levels=2, codebook_size=2)
        self.tokenizer_path = self.root / 'tokenizer.pt'
        torch.save(self.artifact, self.tokenizer_path)
        self.bundle = self.root / 'bundle'

    def export(self, **kwargs):
        return self.export_bundle(self.manifest_path, self.tokenizer_path,
                                  self.dataset, self.bundle, **kwargs)

    def test_export_keeps_all_collision_members_in_gallery_order(self):
        self.export()
        manifest = json.loads((self.bundle / 'manifest.json').read_text())
        self.assertEqual([row['path'] for row in manifest['targets']],
                         ['images/train.jpg', 'images/a.jpg', 'images/b.jpg',
                          'images/c.jpg', 'images/d.jpg'])
        self.assertEqual([row['label'] for row in manifest['targets']], [2, 0, 1, 1, 0])
        self.assertEqual(manifest['gallery_indices'], [3, 1, 4, 2])
        mapping = json.loads((self.bundle / 'identifier_to_gallery.json').read_text())
        self.assertEqual(mapping['tokenizer_artifact_id'], 'fixture-tokenizer-export')
        self.assertEqual(mapping['entries'], [
            {'identifier': [1, 0], 'members': [
                {'gallery_row': 0, 'target_row': 3, 'path': 'images/c.jpg', 'label': 1},
                {'gallery_row': 2, 'target_row': 4, 'path': 'images/d.jpg', 'label': 0},
                {'gallery_row': 3, 'target_row': 2, 'path': 'images/b.jpg', 'label': 1},
            ]},
            {'identifier': [0, 1], 'members': [
                {'gallery_row': 1, 'target_row': 1, 'path': 'images/a.jpg', 'label': 0},
            ]},
        ])
        provenance = json.loads((self.bundle / 'provenance.json').read_text())
        self.assertEqual(provenance['tokenizer_artifact_id'], self.artifact['artifact_id'])
        self.assertEqual(provenance['effective_config'], self.artifact['config'])
        self.assertEqual(provenance['codebooks'], {'shape': [2, 2, 3], 'dtype': 'torch.float32'})
        self.assertEqual(provenance['gallery_indices'], [3, 1, 4, 2])

    def test_portable_manifest_loads_at_a_new_dataset_root(self):
        self.export()
        relocated = self.root / 'new_machine' / 'dataset'
        relocated.mkdir(parents=True)
        shutil.copyfile(self.bundle / 'manifest.json', relocated / 'manifest.json')
        loaded = Manifest.load(relocated / 'manifest.json')
        self.assertEqual(loaded.targets[3].path, str(relocated / 'images/c.jpg'))
        self.assertEqual(loaded.queries[1].path, str(relocated / 'images/query_b.jpg'))
        self.assertEqual(loaded.gallery_indices, [3, 1, 4, 2])

    def test_export_rejects_wrong_labels_target_order_and_gallery_order(self):
        changes = {
            'target_labels': [2, 1, 0, 1, 0],
            'target_paths': list(reversed(self.artifact['target_paths'])),
            'gallery_indices': [1, 3, 4, 2],
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                wrong = copy.deepcopy(self.artifact)
                wrong[field] = value
                torch.save(wrong, self.tokenizer_path)
                with self.assertRaisesRegex(ValueError, field):
                    self.export()
                self.assertFalse(self.bundle.exists())

    def test_export_rejects_codes_without_a_valid_codebook_row(self):
        for codes in (torch.zeros(4, 2, dtype=torch.long),
                      torch.zeros(5, 3, dtype=torch.long),
                      torch.full((5, 2), 2, dtype=torch.long),
                      torch.zeros(5, 2, dtype=torch.float32)):
            with self.subTest(shape=tuple(codes.shape), dtype=codes.dtype):
                wrong = copy.deepcopy(self.artifact)
                wrong['target_codes'] = codes
                torch.save(wrong, self.tokenizer_path)
                with self.assertRaisesRegex(ValueError, 'target_codes'):
                    self.export()
                self.assertFalse(self.bundle.exists())

    def test_export_rejects_images_outside_dataset_root(self):
        with self.assertRaisesRegex(ValueError, 'dataset root'):
            self.export_bundle(self.manifest_path, self.tokenizer_path,
                               self.dataset / 'wrong_subdirectory', self.bundle)
        self.assertFalse(self.bundle.exists())

    def test_bind_changes_only_path_metadata_and_preserves_generator_identity(self):
        self.export()
        moved_bundle = self.root / 'new_machine' / 'bundle'
        moved_bundle.parent.mkdir()
        shutil.move(self.bundle, moved_bundle)
        relocated = self.root / 'new_machine' / 'dataset'
        shutil.move(self.dataset, relocated)
        local = self.root / 'new_machine' / 'run'
        self.bind_bundle(moved_bundle, self.tokenizer_path, relocated, local)
        manifest = Manifest.load(local / 'manifest.json')
        bound = _load_tokenizer(local / 'tokenizer.pt', manifest)
        self.assertEqual(manifest.targets[3].path, str(relocated / 'images/c.jpg'))
        self.assertEqual(bound['artifact_id'], 'fixture-tokenizer-export')
        self.assertEqual(bound['manifest_path'], str(local / 'manifest.json'))
        for field in ('config', 'target_labels', 'gallery_indices', 'epoch', 'num_classes'):
            self.assertEqual(bound[field], self.artifact[field])
        for field in ('codebooks', 'target_codes'):
            self.assertTrue(torch.equal(bound[field], self.artifact[field]))
        self.assertTrue(torch.equal(bound['encoder_state']['weight'],
                                    self.artifact['encoder_state']['weight']))
        original = torch.load(self.tokenizer_path, map_location='cpu', weights_only=True)
        self.assertEqual(original['target_paths'], self.artifact['target_paths'])

    def test_bind_rejects_another_tokenizer_and_changed_config(self):
        self.export()
        for field in ('artifact_id', 'config'):
            with self.subTest(field=field):
                wrong = copy.deepcopy(self.artifact)
                if field == 'artifact_id':
                    wrong[field] = 'another-export'
                else:
                    wrong[field]['seed'] += 1
                torch.save(wrong, self.tokenizer_path)
                local = self.root / f'wrong_{field}'
                with self.assertRaisesRegex(ValueError, 'identity|config'):
                    self.bind_bundle(self.bundle, self.tokenizer_path, self.dataset, local)
                self.assertFalse(local.exists())

    def test_bind_rejects_manifest_reordering_and_lost_collision_members(self):
        self.export()
        manifest_path = self.bundle / 'manifest.json'
        payload = json.loads(manifest_path.read_text())
        payload['targets'][1], payload['targets'][4] = payload['targets'][4], payload['targets'][1]
        manifest_path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'target_paths'):
            self.bind_bundle(self.bundle, self.tokenizer_path, self.dataset, self.root / 'wrong_order')
        payload['targets'][1], payload['targets'][4] = payload['targets'][4], payload['targets'][1]
        manifest_path.write_text(json.dumps(payload))
        mapping_path = self.bundle / 'identifier_to_gallery.json'
        mapping = json.loads(mapping_path.read_text())
        mapping['entries'][0]['members'].pop()
        mapping_path.write_text(json.dumps(mapping))
        with self.assertRaisesRegex(ValueError, 'mapping'):
            self.bind_bundle(self.bundle, self.tokenizer_path, self.dataset, self.root / 'wrong_mapping')

    def test_cli_exports_without_models_or_dataset_images(self):
        completed = subprocess.run([
            sys.executable, '-m', 'mambagir.reproduction', 'export',
            '--manifest', str(self.manifest_path), '--tokenizer', str(self.tokenizer_path),
            '--dataset-root', str(self.dataset), '--output', str(self.bundle),
        ], capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)['manifest'], str(self.bundle / 'manifest.json'))

    def test_main_run_plan_covers_four_datasets_three_seeds_without_running(self):
        project = Path(__file__).resolve().parents[1]
        output = self.root / 'planned runs'
        environment = dict(os.environ, PYTHON=sys.executable, RUN_ROOT=str(output),
                           CUB_ROOT='/unavailable data/cub', CARS_ROOT='/unavailable data/cars',
                           AIRCRAFT_ROOT='/unavailable data/aircraft', INSHOP_ROOT='/unavailable data/inshop',
                           VISION_CHECKPOINT='/unavailable weights/vision.pt',
                           MAMBA_MODEL='/unavailable weights/mamba-790m')
        completed = subprocess.run(['bash', str(project / 'scripts/reproduce.sh'), 'plan'],
                                   env=environment, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        commands = [shlex.split(line) for line in completed.stdout.splitlines() if line.strip()]
        python_commands = [line for line in commands if line[0] == sys.executable]
        for command in ('train-tokenizer', 'train', 'evaluate', 'export'):
            selected = [line for line in python_commands if line[3] == command]
            self.assertEqual(len(selected), 12, command)
            self.assertEqual(len({line[line.index('--output') + 1] for line in selected}), 12)
        training = [line for line in python_commands if line[3] == 'train']
        self.assertEqual(sorted(line[line.index('--seed') + 1] for line in training),
                         ['3407'] * 4 + ['3408'] * 4 + ['3409'] * 4)
        prepared = [line for line in python_commands if line[3] == 'prepare-data']
        self.assertEqual({line[line.index('--dataset') + 1] for line in prepared},
                         {'cub', 'cars', 'aircraft', 'inshop'})
        self.assertIn('/unavailable data/cub', prepared[0])
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
