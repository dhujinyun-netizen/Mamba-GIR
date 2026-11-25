"""Time resident query tensors through generation and CPU gallery resolution.

This measures one repeated query batch. Artifact loading, image preprocessing,
and input transfer are outside the timing interval; generated-key transfer is
inside it. No FLOP estimate or baseline comparison is inferred from timing.
"""

import argparse
import copy
import json
import math
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from itertools import chain
from numbers import Integral
from pathlib import Path

import torch

from . import runtime
from .data import ImageDataset, Manifest
from .retrieval import CUDAGraphBeamSearch, GalleryIndex, beam_search


def _validate_settings(warmup, iterations, batch_size, beam_size):
    for name, value, minimum in (('warmup', warmup, 0), ('iterations', iterations, 1),
                                  ('batch_size', batch_size, 1), ('beam_size', beam_size, 1)):
        if not isinstance(value, Integral) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')


def measure_retrieval(model, images, gallery, beam_size=8, warmup=5, iterations=20,
                      *, batch_size=None, topk=None, amp=False, cuda_graph=False):
    """Measure a preloaded model and query batch; leave tensors and mode intact.

    Each sample covers beam_search, generated-key device-to-host transfer, and
    gallery.resolve. CUDA synchronization brackets every measured iteration.
    The per-query values are batch time / batch size, not single-query latency.
    """
    if images.ndim != 4 or min(images.shape) < 1 or not images.is_floating_point():
        raise ValueError('images must be a nonempty floating batch [B,C,H,W]')
    batch_size = len(images) if batch_size is None else batch_size
    _validate_settings(warmup, iterations, batch_size, beam_size)
    if batch_size != len(images):
        raise ValueError('batch_size must match the resident query tensor')
    topk = beam_size if topk is None else topk
    if not isinstance(topk, Integral) or topk < 1:
        raise ValueError('topk must be a positive integer')
    if gallery.codes.shape[1] != model.levels:
        raise ValueError('gallery identifier length differs from the model')
    device = images.device
    if device.type not in {'cpu', 'cuda'}:
        raise ValueError('benchmark device must be cpu or cuda')
    if cuda_graph and device.type != 'cuda':
        raise ValueError('CUDA graph timing requires a CUDA device')
    if any(value.device != device for value in chain(model.parameters(), model.buffers())):
        raise ValueError('model and query batch must already reside on the same device')
    autocast_enabled = bool(amp and device.type == 'cuda')
    runner = None

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def retrieve():
        with torch.inference_mode(), torch.autocast(device.type, enabled=autocast_enabled):
            keys, _ = runner(images) if runner is not None else beam_search(model, images, beam_size)
        return gallery.resolve(keys, topk), keys.shape[1]

    was_training = model.training
    model.eval()
    memory = None
    samples = []
    try:
        if cuda_graph:
            runner = CUDAGraphBeamSearch(model, images, beam_size, amp=autocast_enabled)
        for _ in range(warmup):
            retrieve()
        synchronize()
        if device.type == 'cuda':
            memory = {'baseline_allocated_bytes': torch.cuda.memory_allocated(device),
                      'baseline_reserved_bytes': torch.cuda.memory_reserved(device)}
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(iterations):
            synchronize()
            start = time.perf_counter()
            ranks, generated_width = retrieve()
            synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        if memory is not None:
            memory.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            for name in ('allocated', 'reserved'):
                memory[f'peak_{name}_growth_bytes'] = (
                    memory[f'peak_{name}_bytes'] - memory[f'baseline_{name}_bytes'])
            memory['scope'] = (
                'PyTorch allocator on this device: baseline is after model/input residency and warmup; '
                'reserved baseline includes cached warmup blocks. Peaks cover measured iterations; '
                'growth is above that baseline, not total working memory or system-wide GPU usage.')
    finally:
        model.train(was_training)

    ordered = sorted(samples)
    latency = {'mean': statistics.mean(samples), 'median': statistics.median(samples),
               'p95': ordered[math.ceil(.95 * len(ordered)) - 1], 'samples': samples,
               'percentile_method': 'nearest_rank'}
    levels = model.levels
    prompts_per_level = getattr(model, 'queries_per_level', None)
    conditioning = getattr(model, 'conditioning', None)
    prefix_tokens = (1 if conditioning in {'linear', 'gated'} else
                     levels * prompts_per_level if prompts_per_level is not None else None)
    hardware = {'device': str(device), 'cpu': platform.processor() or platform.machine(),
                'logical_cpu_count': os.cpu_count()}
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        hardware.update(gpu_name=properties.name, gpu_total_memory_bytes=properties.total_memory,
                        gpu_compute_capability=[properties.major, properties.minor])
    return {
        'format_version': 1, 'kind': 'mambagir_timing',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Repeated resident query batch: beam_search plus CPU gallery.resolve, including key transfer.',
        'preprocessing_included': False, 'input_transfer_included': False,
        'artifact_loading_included': False, 'query_tensor_resident_before_warmup': True,
        'model_resident_before_warmup': True,
        'warmup': warmup, 'iterations': iterations, 'batch_size': batch_size,
        'beam_size': beam_size, 'generated_beam_width': generated_width, 'resolved_topk': topk,
        'identifier_length': levels, 'codebook_size': getattr(model, 'codebook_size', None),
        'queries_per_level': prompts_per_level, 'visual_prefix_tokens': prefix_tokens,
        'conditioning': conditioning, 'model_class': type(model).__name__,
        'cached_decoding': bool(getattr(model, 'supports_cached_decoding', False)),
        'lora_merged_for_inference': bool(getattr(getattr(model, 'decoder', None), 'inference_merged', False)),
        'cuda_graph': bool(cuda_graph),
        'cuda_graph_setup_ms': runner.setup_ms if runner is not None else None,
        'num_gallery': len(gallery.codes), 'image_shape': list(images.shape),
        'parameter_counts': runtime.parameter_counts(model), 'batch_latency_ms': latency,
        'amortized_per_query_ms': {name: latency[name] / batch_size for name in ('mean', 'median', 'p95')},
        'cuda_memory': memory, 'last_gallery_rows': ranks.tolist(),
        'precision': {'input_dtype': str(images.dtype),
                      'parameter_dtypes': sorted({str(p.dtype) for p in model.parameters()}),
                      'autocast_enabled': autocast_enabled,
                      'autocast_dtype': str(torch.get_autocast_gpu_dtype()) if autocast_enabled else None},
        'hardware': hardware,
        'software': {'python': platform.python_version(), 'torch': str(torch.__version__),
                     'cuda_build': torch.version.cuda,
                     'cudnn': torch.backends.cudnn.version() if device.type == 'cuda' else None,
                     'platform': platform.platform(), 'torch_threads': torch.get_num_threads(),
                     'torch_interop_threads': torch.get_num_interop_threads()},
    }


def benchmark_checkpoint(checkpoint_path, manifest_path, output, device='cuda', *,
                         tokenizer_path=None, decoder_path=None, warmup=5, iterations=20,
                         batch_size=1, beam_size=None, cuda_graph=False):
    """Load a clean artifact and time the first requested batch, without training."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint.get('kind') != 'mambagir_generator' or checkpoint.get('format_version') != 1:
        raise ValueError('expected a clean-version generator checkpoint; legacy IRGen artifacts are unsupported')
    config = runtime._generator_checkpoint_config(checkpoint)
    beam_size = config['evaluation']['beam_size'] if beam_size is None else beam_size
    _validate_settings(warmup, iterations, batch_size, beam_size)
    device = torch.device(device)
    if decoder_path is not None:
        config['decoder']['model_path'] = str(Path(decoder_path).expanduser().resolve())
    manifest = Manifest.load(manifest_path)
    if batch_size > len(manifest.queries):
        raise ValueError('batch_size exceeds the available evaluation queries')
    tokenizer_path = tokenizer_path or checkpoint['tokenizer_path']
    output = Path(output).expanduser().resolve()
    if output in {Path(path).expanduser().resolve() for path in
                  (checkpoint_path, manifest_path, tokenizer_path)}:
        raise ValueError('output must not overwrite an input artifact or manifest')
    tokenizer = runtime._load_tokenizer(tokenizer_path, manifest)
    if tokenizer['artifact_id'] != checkpoint.get('tokenizer_id'):
        raise ValueError('tokenizer artifact identity differs from the generator training checkpoint')
    runtime.seed_everything(config['seed'])
    model = runtime.build_generator(config, tokenizer, device)
    runtime._restore_trainable(model, checkpoint['trainable_state'])
    restored_parameter_counts = runtime.parameter_counts(model)
    model.prepare_for_inference()
    dataset = ImageDataset(manifest.queries[:batch_size], train=False, **config['image'])
    images = torch.stack([dataset[index]['image'] for index in range(batch_size)]).to(device)
    gallery = GalleryIndex(tokenizer['target_codes'][manifest.gallery_indices])
    report = measure_retrieval(model, images, gallery, beam_size, warmup, iterations,
                               batch_size=batch_size, topk=max(config['evaluation']['ks']), amp=config['amp'],
                               cuda_graph=cuda_graph)
    report.update(checkpoint=str(Path(checkpoint_path).expanduser().resolve()),
                  parameter_counts=restored_parameter_counts,
                  inference_parameter_counts=runtime.parameter_counts(model),
                  checkpoint_epoch=checkpoint.get('epoch'), tokenizer=str(Path(tokenizer_path).expanduser().resolve()),
                  tokenizer_id=tokenizer['artifact_id'], manifest=str(Path(manifest_path).expanduser().resolve()),
                  query_paths=[r.path for r in manifest.queries[:batch_size]], seed=config['seed'],
                  decoder_family=config['decoder']['family'], decoder_config=config['decoder'],
                  model_config=config['model'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True, help='JSON timing report path')
    parser.add_argument('--device', required=True, help='cpu or a CUDA device such as cuda:0')
    parser.add_argument('--tokenizer', dest='tokenizer_path')
    parser.add_argument('--mamba-model', dest='decoder_path')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--beam-size', type=int)
    parser.add_argument('--cuda-graph', action='store_true', help='capture Mamba inference before timing')
    arguments = vars(parser.parse_args(argv))
    arguments['checkpoint_path'] = arguments.pop('checkpoint')
    arguments['manifest_path'] = arguments.pop('manifest')
    try:
        report = benchmark_checkpoint(**arguments)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == '__main__':
    main()
