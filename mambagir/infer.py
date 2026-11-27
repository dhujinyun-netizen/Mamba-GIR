"""Evaluate released Mamba-GIR checkpoints or replay their saved raw beams."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import ImageDataset, Record
from .reproduction import _manifest_at_root
from .retrieval import CUDAGraphBeamSearch, GalleryIndex, retrieval_report
from .runtime import (_generator_checkpoint_config, _load_tokenizer, _restore_trainable,
                      _save, build_generator)


def load_model(checkpoint_path, tokenizer, decoder_path, device):
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=True, mmap=True)
    if checkpoint.get('tokenizer_id') != tokenizer['artifact_id']:
        raise ValueError('checkpoint and identifier artifact do not match')
    config = _generator_checkpoint_config(checkpoint)
    config['decoder']['model_path'] = str(decoder_path.resolve(strict=True))
    model = build_generator(config, tokenizer, 'cpu', encoder_state=checkpoint['encoder_state'])
    _restore_trainable(model, checkpoint['trainable_state'])
    return model.prepare_for_inference().requires_grad_(False).to(device), config


@torch.inference_mode()
def predict(model, records, config, settings, device):
    batch_size = min(settings['batch_size'], len(records))
    loader = DataLoader(
        ImageDataset(records, **config['image']), batch_size=batch_size, num_workers=2,
        pin_memory=config.get('loader_pin_memory', False), multiprocessing_context='spawn',
    )
    amp = settings['precision'] == 'BF16'
    runner = None
    all_keys, all_scores = [], []
    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        count = len(images)
        if count < batch_size:
            images = torch.cat((images, images[:1].expand(batch_size - count, -1, -1, -1)))
        if runner is None:
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                runner = CUDAGraphBeamSearch(model, images, settings['beam_size'], amp=amp)
        keys, scores = runner(images)
        if not torch.isfinite(scores[:count]).all():
            raise FloatingPointError('nonfinite beam scores')
        all_keys.append(keys[:count].cpu())
        all_scores.append(scores[:count].cpu())
    return torch.cat(all_keys), torch.cat(all_scores)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=('inshop', 'cub', 'cars', 'aircraft'))
    parser.add_argument('--artifacts', required=True, type=Path, help='extracted release directory')
    parser.add_argument('--data-root', type=Path, help='root of the unpacked images for this dataset')
    parser.add_argument('--mamba-model', type=Path, help='local pretrained Mamba-790M HF directory')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--image', nargs='+', type=Path, help='query images instead of the benchmark split')
    mode.add_argument('--replay', action='store_true', help='recompute metrics from saved beams on CPU')
    args = parser.parse_args(argv)
    if not args.replay and (args.data_root is None or args.mamba_model is None):
        parser.error('inference requires --data-root and --mamba-model')

    artifacts = args.artifacts.resolve(strict=True)
    data_root = artifacts if args.replay else args.data_root.resolve(strict=True)
    manifest_file = artifacts / 'manifests' / f'{args.dataset}.json'
    payload = json.loads(manifest_file.read_text(encoding='utf-8'))
    manifest = _manifest_at_root(payload, data_root)
    tokenizer = _load_tokenizer(artifacts / f'{args.dataset}_identifiers.pt', manifest,
                               dataset_root=data_root)
    settings = json.loads((artifacts / 'inference.json').read_text(encoding='utf-8'))['datasets'][args.dataset]
    queries = manifest.queries
    gallery_rows = [manifest.targets[index] for index in manifest.gallery_indices]
    gallery = GalleryIndex(tokenizer['target_codes'][manifest.gallery_indices])

    if args.replay:
        saved = torch.load(artifacts / 'predictions' / f'{args.dataset}.pt', map_location='cpu', weights_only=True)
        keys = saved['keys']
    else:
        device = torch.device(args.device)
        if device.type != 'cuda':
            parser.error('--device must be a CUDA device for model inference')
        if device.index is not None:
            torch.cuda.set_device(device)
        torch.set_num_threads(4)
        torch.manual_seed(settings['seed'])
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if args.image:
            queries = [Record(str(path.resolve(strict=True)), 0) for path in args.image]
        model, config = load_model(artifacts / f'{args.dataset}_weights.pt', tokenizer,
                                   args.mamba_model, device)
        keys, scores = predict(model, queries, config, settings, device)

    ranks = gallery.resolve(keys, settings['beam_size'])
    if args.image:
        result = [
            {'image': query.path, 'raw_keys': raw_keys, 'log_scores': log_scores,
             'results': [gallery_rows[index].path for index in indices if index >= 0]}
            for query, raw_keys, log_scores, indices in
            zip(queries, keys.tolist(), scores.tolist(), ranks.tolist())
        ]
        output_name = 'retrieval.json'
    else:
        result = retrieval_report(keys, ranks, gallery, [row.label for row in queries],
                                  [row.label for row in gallery_rows])
        result.update(dataset=args.dataset, precision=settings.get('precision'))
        output_name = 'metrics.json'
    encoded = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / output_name).write_text(encoded + '\n', encoding='utf-8')
        if not args.replay and not args.image:
            _save(args.output_dir / 'predictions.pt',
                  {'keys': keys, 'log_scores': scores, 'image_indices': ranks})
    print(encoded)


if __name__ == '__main__':
    main()
