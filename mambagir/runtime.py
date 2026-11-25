"""Executable paper training/evaluation pipeline; no imports from legacy scripts."""

import copy
import json
import math
import random
import uuid
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ImageDataset, Manifest, sample_positive_indices
from .retrieval import CUDAGraphBeamSearch, GalleryIndex, beam_search, retrieval_report
from .rvq import ResidualVectorQuantizer
from .tokenizer import SemanticTokenizer
from .vision import VisualEncoder


def load_config(path=None):
    default = Path(__file__).resolve().parents[1] / 'configs' / 'paper.json'
    return json.loads(Path(path or default).read_text())


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def learning_rate_factor(step, total_steps, warmup_steps):
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    return .5 * (1 + math.cos(math.pi * min(1., max(0., progress))))


def _set_learning_rate(optimizer, step, total_steps, warmup_steps):
    factor = learning_rate_factor(step, total_steps, warmup_steps)
    for group in optimizer.param_groups:
        group['lr'] = group['initial_lr'] * factor


def _loader(records, config, batch_size, train, device):
    dataset = ImageDataset(records, train=train, **config['image'])
    return DataLoader(dataset, batch_size=batch_size, shuffle=train,
                      num_workers=config['workers'], pin_memory=device.type == 'cuda')


@torch.no_grad()
def extract_features(encoder, loader, device):
    was_training = encoder.training
    encoder.eval()
    features = [encoder.forward_global(batch['image'].to(device)).float().cpu() for batch in loader]
    encoder.train(was_training)
    return torch.cat(features)


def parameter_counts(model):
    """parameters() removes shared-parameter aliases; buffers are not parameters."""
    return {'total': sum(p.numel() for p in model.parameters()),
            'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)}


def _save(path, state):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def _log(path, entry):
    print(json.dumps(entry, ensure_ascii=False), flush=True)
    with Path(path).open('a') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def _cpu_state(module):
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def _manifest_identity(manifest):
    return {'target_paths': [r.path for r in manifest.targets],
            'target_labels': [r.label for r in manifest.targets],
            'gallery_indices': list(manifest.gallery_indices)}


def _check_identity(artifact, manifest):
    for key, expected in _manifest_identity(manifest).items():
        if artifact.get(key) != expected:
            raise ValueError(f'{key} differs from tokenizer artifact; rebuild identifiers for this manifest')


def train_tokenizer(config, manifest_path, output_dir, device='cuda'):
    """Alternate encoder optimization and residual-code refresh, Sec. III-A."""
    config = copy.deepcopy(config)
    if config['vision']['checkpoint']:
        config['vision']['checkpoint'] = str(Path(config['vision']['checkpoint']).expanduser().resolve())
    seed_everything(config['seed'])
    device = torch.device(device)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest.load(manifest_path)
    settings = config['tokenizer']
    if settings['epochs'] < 1:
        raise ValueError('tokenizer epochs must be positive')
    encoder = VisualEncoder(config['vision']['variant'])
    pretrained = config['vision']['checkpoint']
    if pretrained:
        encoder.load_pretrained(pretrained)
    elif not settings.get('allow_random_encoder', False):
        raise ValueError('set vision.checkpoint to the pretrained MambaVision checkpoint')
    model = SemanticTokenizer(encoder, manifest.num_classes, settings['head_depth'],
                              settings['head_hidden_dim']).to(device)
    train_loader = _loader(manifest.targets, config, settings['batch_size'], True, device)
    feature_loader = _loader(manifest.targets, config, settings['batch_size'], False, device)
    optimizer = torch.optim.AdamW([
        {'params': encoder.parameters(), 'lr': settings['encoder_lr'], 'initial_lr': settings['encoder_lr']},
        {'params': model.head.parameters(), 'lr': settings['head_lr'], 'initial_lr': settings['head_lr']},
    ], betas=tuple(settings['betas']), weight_decay=settings['weight_decay'])
    amp = config['amp'] and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    quantizer = ResidualVectorQuantizer(**config['rvq'], seed=config['seed']).to(device)
    total = settings['epochs'] * len(train_loader)
    warmup = settings['warmup_epochs'] * len(train_loader)
    features = extract_features(encoder, feature_loader, device)
    for epoch in range(settings['epochs']):
        quantizer.fit(features.to(device))
        codes = quantizer.encode(features.to(device))
        partials = quantizer.reconstruct(codes, cumulative=True).detach().cpu()
        model.train()
        totals = {}
        count = 0
        for step, batch in enumerate(train_loader):
            _set_learning_rate(optimizer, epoch * len(train_loader) + step, total, warmup)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                losses = model(batch['image'].to(device), batch['label'].to(device),
                               partials[batch['index']].to(device), settings['rq_weight'])
            scaler.scale(losses['loss']).backward()
            scaler.step(optimizer)
            scaler.update()
            n = len(batch['image'])
            count += n
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.) + value.detach().float().item() * n
        features = extract_features(encoder, feature_loader, device)
        _log(output_dir / 'train.jsonl', {'epoch': epoch + 1, **{k: v/count for k, v in totals.items()}})
        if (epoch + 1) % settings['save_every'] == 0:
            _save(output_dir / 'tokenizer_training.pt', {
                'epoch': epoch + 1, 'state': _cpu_state(model),
                'optimizer': optimizer.state_dict(), 'config': config,
            })
    # The exported codes correspond to the FINAL encoder, not the preceding epoch.
    quantizer.fit(features.to(device))
    final_codes = quantizer.encode(features.to(device)).cpu()
    path = output_dir / 'tokenizer.pt'
    _save(path, {
        'format_version': 1, 'kind': 'mambagir_tokenizer', 'artifact_id': uuid.uuid4().hex, 'config': config,
        'encoder_state': _cpu_state(encoder), 'codebooks': quantizer.codebooks.detach().cpu(),
        'target_codes': final_codes, 'num_classes': manifest.num_classes,
        'manifest_path': str(Path(manifest_path).resolve()), **_manifest_identity(manifest),
        'parameter_counts': parameter_counts(model), 'epoch': settings['epochs'],
    })
    return path


def _load_tokenizer(path, manifest):
    artifact = torch.load(path, map_location='cpu', weights_only=True)
    if (artifact.get('kind') != 'mambagir_tokenizer' or artifact.get('format_version') != 1
            or not artifact.get('artifact_id')):
        raise ValueError('expected a clean-version tokenizer artifact (legacy .pkl is not interchangeable)')
    _check_identity(artifact, manifest)
    return artifact


def build_generator(config, tokenizer, device):
    from .decoder import MambaDecoder, TransformerDecoder
    from .model import MambaGIR
    encoder = VisualEncoder(tokenizer['config']['vision']['variant'])
    encoder.load_state_dict(tokenizer['encoder_state'], strict=True)
    decoder_config = config['decoder']
    if decoder_config['family'] == 'mamba':
        if not decoder_config['model_path']:
            raise ValueError('set decoder.model_path to a local pretrained Hugging Face Mamba directory')
        decoder = MambaDecoder(decoder_config['model_path'], adaptation=decoder_config['adaptation'],
                               rank=decoder_config['rank'], alpha=decoder_config['alpha'],
                               dropout=decoder_config['dropout'], local_files_only=True)
    elif decoder_config['family'] == 'transformer':
        levels, vocab, _ = tokenizer['codebooks'].shape
        decoder = TransformerDecoder(d_model=decoder_config['d_model'], layers=decoder_config['layers'],
                                     num_heads=decoder_config['num_heads'], vocab_size=levels*vocab+1)
    else:
        raise ValueError('decoder.family must be mamba or transformer')
    return MambaGIR(encoder, decoder, tokenizer['codebooks'], tokenizer['num_classes'],
                    **config['model']).to(device)


def _trainable_state(model):
    names = {name for name, value in model.named_parameters() if value.requires_grad}
    return {name: value.detach().cpu() for name, value in model.state_dict().items() if name in names}


def _restore_trainable(model, state):
    expected = set(_trainable_state(model))
    if set(state) != expected:
        raise ValueError(f'trainable checkpoint architecture differs: missing={expected-set(state)}, extra={set(state)-expected}')
    model.load_state_dict(state, strict=False)  # all omitted weights come from the frozen tokenizer/decoder


def _generator_checkpoint_config(checkpoint):
    """Preserve the stage binding used by pre-revision format-1 checkpoints."""
    config = copy.deepcopy(checkpoint['config'])
    config['model'].setdefault('head_prompt_mode', 'stage')
    return config


def train_generator(config, manifest_path, tokenizer_path, output_dir, device='cuda'):
    config = copy.deepcopy(config)
    if config['decoder']['model_path']:
        config['decoder']['model_path'] = str(Path(config['decoder']['model_path']).expanduser().resolve())
    seed_everything(config['seed'])
    device = torch.device(device)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest.load(manifest_path)
    tokenizer = _load_tokenizer(tokenizer_path, manifest)
    model = build_generator(config, tokenizer, device)
    config['model']['head_prompt_mode'] = model.head_prompt_mode
    settings = config['generator']
    if settings['epochs'] < 1:
        raise ValueError('generator epochs must be positive')
    loader = _loader(manifest.targets, config, settings['batch_size'], True, device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=settings['lr'], betas=tuple(settings['betas']),
                                  weight_decay=settings['weight_decay'])
    for group in optimizer.param_groups:
        group['initial_lr'] = settings['lr']
    amp = config['amp'] and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    steps = settings['epochs'] * len(loader)
    warmup = int(settings['warmup_fraction'] * steps)
    best_loss = float('inf')
    counts = parameter_counts(model)
    counts['decoder_total'] = sum(p.numel() for p in model.decoder.parameters())
    counts['decoder_trainable'] = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print(json.dumps({'parameter_counts': counts}), flush=True)
    labels = [r.label for r in manifest.targets]
    for epoch in range(settings['epochs']):
        positives = sample_positive_indices(labels, config['seed'], epoch)
        model.train()
        totals = {}
        seen = 0
        for step, batch in enumerate(loader):
            _set_learning_rate(optimizer, epoch * len(loader) + step, steps, warmup)
            targets = tokenizer['target_codes'][positives[batch['index']]].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                result = model(batch['image'].to(device), targets, batch['label'].to(device))
            scaler.scale(result['loss']).backward()
            scaler.step(optimizer)
            scaler.update()
            n = len(batch['image'])
            seen += n
            for key in ('loss', 'loss_gen', 'loss_align', 'loss_cls'):
                totals[key] = totals.get(key, 0.) + result[key].detach().float().item() * n
        means = {k: value / seen for k, value in totals.items()}
        _log(output_dir / 'train.jsonl', {'epoch': epoch + 1, **means})
        checkpoint = {
            'format_version': 1, 'kind': 'mambagir_generator', 'config': config,
            'tokenizer_path': str(Path(tokenizer_path).resolve()), 'trainable_state': _trainable_state(model),
            'tokenizer_id': tokenizer['artifact_id'],
            'epoch': epoch + 1, 'train_loss': means['loss'], 'parameter_counts': counts,
        }
        if means['loss'] < best_loss:
            best_loss = means['loss']
            _save(output_dir / 'best.pt', checkpoint)
        if (epoch + 1) % settings['save_every'] == 0 or epoch + 1 == settings['epochs']:
            _save(output_dir / 'last.pt', checkpoint)
    return output_dir / 'last.pt'


@torch.no_grad()
def evaluate(checkpoint_path, manifest_path, output_dir, device='cuda', *,
             beam_size=None, ks=None, tokenizer_path=None, decoder_path=None, cuda_graph=False):
    device = torch.device(device)
    if cuda_graph and device.type != 'cuda':
        raise ValueError('CUDA graph evaluation requires a CUDA device')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint.get('kind') != 'mambagir_generator' or checkpoint.get('format_version') != 1:
        raise ValueError('expected a clean-version generator checkpoint')
    config = _generator_checkpoint_config(checkpoint)
    seed_everything(config['seed'])
    if decoder_path is not None:
        config['decoder']['model_path'] = str(Path(decoder_path).resolve())
    manifest = Manifest.load(manifest_path)
    tokenizer_path = tokenizer_path or checkpoint['tokenizer_path']
    tokenizer = _load_tokenizer(tokenizer_path, manifest)
    if tokenizer['artifact_id'] != checkpoint.get('tokenizer_id'):
        raise ValueError('tokenizer artifact identity differs from the generator training checkpoint')
    model = build_generator(config, tokenizer, device)
    _restore_trainable(model, checkpoint['trainable_state'])
    restored_parameter_counts = parameter_counts(model)
    model.prepare_for_inference()
    settings = config['evaluation']
    beam_size = settings['beam_size'] if beam_size is None else beam_size
    ks = settings['ks'] if ks is None else ks
    if not ks or min(ks) < 1:
        raise ValueError('ks must contain positive cutoffs')
    loader = _loader(manifest.queries, config, settings['batch_size'], False, device)
    gallery = GalleryIndex(tokenizer['target_codes'][manifest.gallery_indices])
    all_keys, all_scores = [], []
    amp = config['amp'] and device.type == 'cuda'
    runner, graph_queries = None, 0
    for batch in loader:
        images = batch['image'].to(device)
        if cuda_graph and runner is None:
            runner = CUDAGraphBeamSearch(model, images, beam_size, amp=amp)
        with torch.cuda.amp.autocast(enabled=amp):
            if runner is not None and images.shape == runner.images.shape:
                keys, scores = runner(images)
                graph_queries += len(images)
            else:
                # A smaller final batch uses the same cached eager decoder.
                keys, scores = beam_search(model, images, beam_size)
        all_keys.append(keys.cpu())
        all_scores.append(scores.cpu())
    keys, scores = torch.cat(all_keys), torch.cat(all_scores)
    ranks = gallery.resolve(keys, max(ks))
    report = retrieval_report(keys, ranks, gallery, [r.label for r in manifest.queries],
                              [manifest.targets[i].label for i in manifest.gallery_indices], ks)
    report.update(seed=config['seed'], checkpoint=str(Path(checkpoint_path).resolve()),
                  checkpoint_epoch=checkpoint['epoch'], parameter_counts=restored_parameter_counts,
                  inference_parameter_counts=parameter_counts(model),
                  cached_decoding=model.supports_cached_decoding,
                  cuda_graph=bool(cuda_graph), cuda_graph_queries=graph_queries)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    _save(output_dir / 'predictions.pt', {
        'format_version': 1, 'kind': 'mambagir_predictions',
        'tokenizer_id': tokenizer['artifact_id'],
        'raw_keys': keys, 'log_scores': scores, 'ranks': ranks, 'config': config,
        'metrics': report, 'query_paths': [r.path for r in manifest.queries],
        'query_labels': [r.label for r in manifest.queries],
        'gallery_paths': [manifest.targets[i].path for i in manifest.gallery_indices],
        'gallery_labels': [manifest.targets[i].label for i in manifest.gallery_indices],
    })
    print(json.dumps(report, indent=2), flush=True)
    return report


def summarize_reports(paths):
    """Mean and sample standard deviation over independent runs of one dataset."""
    reports = [json.loads(Path(path).read_text()) for path in paths]
    if len(reports) < 2:
        raise ValueError('at least two independent runs are needed for standard deviation')
    if len({r['seed'] for r in reports}) != len(reports):
        raise ValueError('reports must use distinct seeds')
    for report in reports[1:]:
        for key in ('num_queries', 'num_gallery', 'beam_size', 'identifier_length'):
            if report[key] != reports[0][key]:
                raise ValueError(f'cannot aggregate runs with different {key}')
        if set(report['recall']) != set(reports[0]['recall']):
            raise ValueError('Recall cutoffs differ between runs')
    return {'runs': len(reports), 'seeds': [r['seed'] for r in reports],
            'recall': {k: {'mean': float(np.mean([r['recall'][k] for r in reports])),
                           'std': float(np.std([r['recall'][k] for r in reports], ddof=1))}
                       for k in reports[0]['recall']}}


def average_datasets(reports):
    """Paper dataset average: each of the four benchmarks has weight 1/4.

    Average one matched seed first, then summarize these averages across seeds.
    This defines uncertainty without averaging dataset standard deviations.
    """
    if set(reports) != {'cub', 'cars', 'aircraft', 'inshop'}:
        raise ValueError('provide exactly cub, cars, aircraft and inshop reports')
    first = reports['cub']
    for report in reports.values():
        for key in ('seed', 'beam_size', 'identifier_length'):
            if report[key] != first[key]:
                raise ValueError(f'dataset reports must use the same {key}')
        if set(report['recall']) != set(first['recall']):
            raise ValueError('Recall cutoffs differ between datasets')
    return {
        'aggregation': 'unweighted_four_datasets',
        'seed': first['seed'], 'beam_size': first['beam_size'], 'identifier_length': first['identifier_length'],
        'num_queries': sum(r['num_queries'] for r in reports.values()),
        'num_gallery': sum(r['num_gallery'] for r in reports.values()),
        'recall': {k: sum(r['recall'][k] for r in reports.values()) / 4 for k in first['recall']},
        'datasets': {name: report['recall'] for name, report in reports.items()},
    }
