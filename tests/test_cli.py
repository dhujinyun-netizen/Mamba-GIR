import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from mambagir.__main__ import main
from mambagir.runtime import summarize_reports


class CliTests(unittest.TestCase):
    def test_four_dataset_average_is_unweighted_and_can_be_summarized(self):
        from mambagir.runtime import average_datasets
        reports = {}
        for i, name in enumerate(('cub', 'cars', 'aircraft', 'inshop')):
            reports[name] = {'seed': 4, 'num_queries': 10 if i < 3 else 10000,
                             'num_gallery': 100, 'beam_size': 8, 'identifier_length': 4,
                             'recall': {'1': 0. if i < 3 else 1.}}
        result = average_datasets(reports)
        self.assertEqual(result['recall']['1'], .25)
        reports['cub']['seed'] = 5
        with self.assertRaises(ValueError):
            average_datasets(reports)

    def test_cli_help_is_available_without_loading_pretrained_models(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), self.assertRaises(SystemExit) as result:
            main(['--help'])
        self.assertEqual(result.exception.code, 0)
        self.assertIn('train-tokenizer', stream.getvalue())

    def test_independent_run_mean_and_sample_deviation(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for seed, score in enumerate([.8, .9, 1.]):
                path = Path(directory) / f'{seed}.json'
                path.write_text(json.dumps({'seed': seed, 'num_queries': 10, 'num_gallery': 20,
                    'beam_size': 8, 'identifier_length': 4, 'recall': {'1': score}}))
                paths.append(path)
            report = summarize_reports(paths)
        self.assertAlmostEqual(report['recall']['1']['mean'], .9)
        self.assertAlmostEqual(report['recall']['1']['std'], .1)


if __name__ == '__main__':
    unittest.main()
