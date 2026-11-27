import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch


class ReleasedInferenceTest(unittest.TestCase):
    def test_replay_uses_relative_metadata_and_preserves_raw_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifests').mkdir()
            (root / 'predictions').mkdir()
            targets = [{'path': f'gallery/{label}.jpg', 'label': label} for label in range(2)]
            queries = [{'path': f'query/{label}.jpg', 'label': label} for label in range(2)]
            manifest = {'targets': targets, 'queries': queries, 'gallery_indices': [0, 1]}
            (root / 'manifests/cub.json').write_text(json.dumps(manifest))
            (root / 'inference.json').write_text(json.dumps({'datasets': {'cub': {'beam_size': 8}}}))
            codes = torch.tensor([[0] * 4, [1] * 4])
            torch.save({
                'format_version': 1, 'artifact_id': 'paired', 'kind': 'mambagir_fixed_identifiers',
                'target_paths': [row['path'] for row in targets], 'target_labels': [0, 1],
                'gallery_indices': [0, 1], 'target_codes': codes,
            }, root / 'cub_identifiers.pt')
            keys = torch.tensor([[[2] * 4, [0] * 4] + [[2] * 4] * 6,
                                 [[0] * 4, [1] * 4] + [[2] * 4] * 6])
            torch.save({'keys': keys}, root / 'predictions/cub.pt')
            process = subprocess.run(
                [sys.executable, '-m', 'mambagir.infer', 'cub', '--artifacts', str(root),
                 '--replay', '--output-dir', str(root / 'result')],
                capture_output=True, text=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            metrics = json.loads((root / 'result/metrics.json').read_text())
            self.assertEqual(metrics['recall']['1'], 0.5)
            self.assertEqual(metrics['raw_key_hit']['1'], 0)
            self.assertEqual(metrics['raw_key_hit']['8'], 1)


if __name__ == '__main__':
    unittest.main()
