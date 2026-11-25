"""Run with python -m mambagir; paths are explicit and relative to your cwd."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Mamba-GIR paper implementation')
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare-data', help='build a manifest from official dataset metadata')
    prepare.add_argument('--dataset', choices=['cub', 'cars', 'aircraft', 'inshop'], required=True)
    prepare.add_argument('--root', required=True)
    prepare.add_argument('--output', required=True)

    for name in ('train-tokenizer', 'train'):
        command = commands.add_parser(name)
        command.add_argument('--config', default=None, help='defaults to configs/paper.json')
        command.add_argument('--manifest', required=True)
        command.add_argument('--output', required=True)
        command.add_argument('--device', default='cuda:0')
        command.add_argument('--seed', type=int)
        command.add_argument('--epochs', type=int, help='override for a smoke run or configured budget')
        command.add_argument('--batch-size', type=int)
        command.add_argument('--workers', type=int)
        if name == 'train-tokenizer':
            command.add_argument('--vision-checkpoint', help='local pretrained MambaVision weights')
        else:
            command.add_argument('--tokenizer', required=True, help='tokenizer.pt from the first stage')
            command.add_argument('--mamba-model', help='local Hugging Face Mamba model directory')

    test = commands.add_parser('evaluate')
    test.add_argument('--checkpoint', required=True)
    test.add_argument('--manifest', required=True)
    test.add_argument('--output', required=True)
    test.add_argument('--device', default='cuda:0')
    test.add_argument('--beam-size', type=int)
    test.add_argument('--cuda-graph', action='store_true', help='replay fixed-shape Mamba inference on CUDA')
    test.add_argument('--ks', nargs='+', type=int)
    test.add_argument('--tokenizer', help='override tokenizer artifact path if moved')
    test.add_argument('--mamba-model', help='override pretrained decoder path if moved')

    summary = commands.add_parser('summarize', help='mean/std across independent seeds of one dataset')
    summary.add_argument('reports', nargs='+', help='metrics.json from each evaluation run')
    summary.add_argument('--output')
    average = commands.add_parser('average', help='unweighted four-dataset average for one matched seed')
    for dataset in ('cub', 'cars', 'aircraft', 'inshop'):
        average.add_argument('--' + dataset, required=True, help='dataset metrics.json')
    average.add_argument('--output')
    args = parser.parse_args(argv)

    if args.command == 'prepare-data':
        from .data import prepare_manifest
        manifest = prepare_manifest(args.dataset, args.root)
        manifest.save(args.output)
        print(json.dumps({'targets': len(manifest.targets), 'gallery': len(manifest.gallery_indices),
                          'queries': len(manifest.queries), 'classes': manifest.num_classes,
                          'manifest': str(Path(args.output).resolve())}))
        return
    from .runtime import load_config, train_tokenizer, train_generator, evaluate, summarize_reports, average_datasets
    if args.command in ('train-tokenizer', 'train'):
        config = load_config(args.config)
        section = 'tokenizer' if args.command == 'train-tokenizer' else 'generator'
        for key in ('seed', 'workers'):
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        for key in ('epochs', 'batch_size'):
            if getattr(args, key) is not None:
                config[section][key] = getattr(args, key)
        if args.command == 'train-tokenizer':
            if args.vision_checkpoint:
                config['vision']['checkpoint'] = str(Path(args.vision_checkpoint).resolve())
            result = train_tokenizer(config, args.manifest, args.output, args.device)
        else:
            if args.mamba_model:
                config['decoder']['model_path'] = str(Path(args.mamba_model).resolve())
            result = train_generator(config, args.manifest, args.tokenizer, args.output, args.device)
        print(f'Saved: {result}')
    elif args.command == 'evaluate':
        evaluate(args.checkpoint, args.manifest, args.output, args.device, beam_size=args.beam_size,
                 ks=args.ks, tokenizer_path=args.tokenizer, decoder_path=args.mamba_model,
                 cuda_graph=args.cuda_graph)
    elif args.command == 'summarize':
        result = json.dumps(summarize_reports(args.reports), indent=2)
        print(result)
        if args.output:
            Path(args.output).write_text(result + '\n')
    elif args.command == 'average':
        reports = {name: json.loads(Path(getattr(args, name)).read_text())
                   for name in ('cub', 'cars', 'aircraft', 'inshop')}
        result = json.dumps(average_datasets(reports), indent=2)
        print(result)
        if args.output:
            Path(args.output).write_text(result + '\n')


if __name__ == '__main__':
    main()
