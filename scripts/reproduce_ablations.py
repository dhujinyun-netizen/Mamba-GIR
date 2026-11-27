"""Train and evaluate the CAVP controls with the main-run tokenizer for each seed."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('cub', 'cars', 'aircraft', 'inshop')
VARIANTS = {
    'linear': {'conditioning': 'linear', 'lambda_align': 0},
    'gated': {'conditioning': 'gated', 'lambda_align': 0},
    'gated_align': {'conditioning': 'gated'},
    'patch_pool': {'conditioning': 'patch_pool', 'head_prompt_mode': 'shared'},
    'patch_attention': {'conditioning': 'patch_attention', 'head_prompt_mode': 'shared'},
    'levelwise': {'conditioning': 'levelwise'},
    'no_code_memory': {'conditioning': 'no_code_memory'},
    'no_alignment': {'lambda_align': 0},
    'no_classification': {'lambda_cls': 0},
    'full_cavp': {},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True, help='output of scripts/reproduce.sh')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mamba-model', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=REPO_ROOT / 'configs/paper.json')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seeds', type=int, nargs='+', default=[3407, 3408, 3409])
    parser.add_argument('--variants', choices=VARIANTS, nargs='+', default=list(VARIANTS))
    parser.add_argument('--execute', action='store_true', help='start training and evaluation')
    args = parser.parse_args()
    run_root, out = args.run_root.resolve(), args.output.resolve()
    decoder = args.mamba_model.resolve()
    base = json.loads(args.config.read_text())
    if args.execute:
        required = [run_root / 'manifests' / f'{dataset}.json' for dataset in DATASETS]
        required.extend(run_root / dataset / f'seed{seed}' / 'tokenizer/tokenizer.pt'
                        for dataset in DATASETS for seed in args.seeds)
        for path in required:
            if not path.is_file():
                parser.error(f'missing main-run input: {path}')
        out.mkdir(parents=True, exist_ok=False)

    def run(*arguments):
        command = [sys.executable, '-m', 'mambagir', *map(str, arguments)]
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=REPO_ROOT, check=True)

    for variant in args.variants:
        variant_root = out / variant
        config = {**base, 'model': {**base['model'], 'conditioning': 'cavp',
                                   'head_prompt_mode': 'stage', **VARIANTS[variant]}}
        config_file = variant_root / 'config.json'
        if args.execute:
            variant_root.mkdir()
            config_file.write_text(json.dumps(config, indent=2) + '\n')
        print(f'{variant}: {json.dumps(config["model"])}', flush=True)
        macro_reports = []
        for seed in args.seeds:
            reports = []
            for dataset in DATASETS:
                manifest = run_root / 'manifests' / f'{dataset}.json'
                tokenizer = run_root / dataset / f'seed{seed}' / 'tokenizer/tokenizer.pt'
                work = variant_root / dataset / f'seed{seed}'
                run('train', '--config', config_file, '--manifest', manifest, '--tokenizer', tokenizer,
                    '--mamba-model', decoder, '--output', work / 'generator', '--seed', seed,
                    '--device', args.device)
                run('evaluate', '--checkpoint', work / 'generator/last.pt', '--manifest', manifest,
                    '--output', work / 'evaluation', '--beam-size', 8, '--ks', 1, 4, 8,
                    '--device', args.device)
                reports.extend((f'--{dataset}', work / 'evaluation/metrics.json'))
            report = variant_root / f'average_seed{seed}.json'
            run('average', *reports, '--output', report)
            macro_reports.append(report)
        for dataset in DATASETS:
            metrics = [variant_root / dataset / f'seed{seed}' / 'evaluation/metrics.json' for seed in args.seeds]
            run('summarize', *metrics, '--output', variant_root / dataset / 'summary.json')
        run('summarize', *macro_reports, '--output', variant_root / 'average_summary.json')


if __name__ == '__main__':
    main()
