"""Paper Sec. III-B/III-D and VI-A: raw key generation, resolution and metrics."""

from collections import Counter, defaultdict
from itertools import chain
import time
from typing import Iterable

import torch
from torch import Tensor
from torch.nn import functional as F


def gallery_prefix_tree(codes: Tensor, device) -> list[Tensor]:
    """Gallery-only transitions; invalid extensions retain probability zero."""
    codes = torch.as_tensor(codes, dtype=torch.long, device='cpu')
    if codes.ndim != 2 or not len(codes) or bool((codes < 0).any()):
        raise ValueError('expected nonnegative gallery codes [N,L]')
    vocabulary = max(256, int(codes.max())+1)
    prefixes = {(): 0}
    result = []
    for level in range(codes.shape[1]):
        following = {}
        transition = torch.full((len(prefixes), vocabulary), -1, dtype=torch.long)
        for row in codes.tolist():
            previous, current = tuple(row[:level]), tuple(row[:level+1])
            if current not in following:
                following[current] = len(following)
            transition[prefixes[previous], row[level]] = following[current]
        result.append(transition.to(device))
        prefixes = following
    return result


@torch.no_grad()
def beam_search(model, images: Tensor, beam_size: int = 8, *,
                use_cache: bool = True, conditioning_cache=None,
                prefix_tree=None) -> tuple[Tensor, Tensor]:
    """Generate unconstrained fixed-length keys, ranked by summed log probability.

    Returns keys [B, K, L] and log scores [B, K]. Gallery validity is deliberately
    checked afterwards, so invalid-key statistics measure the raw generator.
    Mamba evaluation prefills once and carries recurrent states in selected
    parent order. ``use_cache=False`` retains full-prefix decoding as a reference;
    decoders without a cache interface also use that path.
    """
    if beam_size < 1:
        raise ValueError('beam_size must be positive')
    prompts, memory = (model.encode(images) if conditioning_cache is None else
                       model.encode(images, conditioning_cache=conditioning_cache))
    batch = images.shape[0]
    keys = torch.empty(batch, 1, 0, dtype=torch.long, device=images.device)
    scores = torch.zeros(batch, 1, device=images.device)
    nodes = torch.zeros(batch, 1, dtype=torch.long, device=images.device)
    cached = (use_cache and getattr(model, 'supports_cached_decoding', False)
              and not getattr(model, 'training', False))
    cache = None
    for level in range(model.levels):
        width = keys.shape[1]
        expanded = prompts.unsqueeze(1).expand(batch, width, *prompts.shape[1:])
        prompt_batch = expanded.reshape(batch * width, *prompts.shape[1:])
        if cached and level == 0:
            logits, cache = model.prefill_logits(prompts, memory)
        elif cached:
            logits = model.next_cached_logits(prompt_batch, memory,
                                              keys[:, :, -1:].reshape(batch * width, 1), level, cache)
        else:
            logits = model.next_logits(prompt_batch, memory, keys.reshape(batch * width, -1))
        log_probs = F.log_softmax(logits.float(), dim=-1).reshape(batch, width, -1)
        vocab = log_probs.shape[-1]
        if prefix_tree is not None:
            transitions = prefix_tree[level][nodes, :vocab]
            log_probs = log_probs.masked_fill(transitions < 0, -torch.inf)
        candidates = (scores.unsqueeze(-1) + log_probs).flatten(1)
        scores, selected = candidates.topk(min(beam_size, candidates.shape[1]), dim=1)
        parents, tokens = selected // vocab, selected % vocab
        if prefix_tree is not None:
            nodes = transitions.gather(1, parents[..., None].expand(-1, -1, vocab))
            nodes = nodes.gather(2, tokens[..., None]).squeeze(-1).clamp_min(0)
        history = keys.gather(1, parents.unsqueeze(-1).expand(-1, -1, keys.shape[-1]))
        keys = torch.cat((history, tokens.unsqueeze(-1)), dim=-1)
        if cached and level + 1 < model.levels:
            offsets = torch.arange(batch, device=parents.device)[:, None] * width
            model.reorder_cache(cache, (parents + offsets).flatten())
    return keys, scores


@torch.no_grad()
def repair_nonfinite_beams(model, images, keys, scores, *, prefix_tree=None):
    """Retry numerically invalid rows in BF16, preserving every finite row.

    Negative-infinity padding for small constrained galleries is legitimate.
    Retry decisions depend only on numerical validity, never query labels.
    """
    bad = (torch.isnan(scores) | torch.isposinf(scores)).any(dim=1)
    count = int(bad.sum())
    if not count:
        return keys, scores, 0
    with torch.autocast(images.device.type, dtype=torch.bfloat16,
                        enabled=images.device.type == 'cuda', cache_enabled=False):
        replacement, values = beam_search(model, images, keys.shape[1], prefix_tree=prefix_tree)
    if bool((torch.isnan(values[bad]) | torch.isposinf(values[bad])).any()):
        raise FloatingPointError('BF16 retry also produced nonfinite beam scores')
    return (torch.where(bad[:, None, None], replacement, keys),
            torch.where(bad[:, None], values, scores), count)


class CUDAGraphBeamSearch:
    """Replay fixed-shape Mamba retrieval after ``prepare_for_inference()``.

    The graph includes encoding, CAVP and all beam steps. Gallery resolution
    remains outside. Construct after loading/moving the model and call serially
    with the captured input shape, device and dtype. Recreate the runner after
    changing weights, precision or the beam budget. Returned tensors own their
    storage, so subsequent queries do not overwrite earlier predictions.
    """

    @torch.inference_mode()
    def __init__(self, model, example_images: Tensor, beam_size: int = 8, *,
                 amp: bool = False, warmup: int = 3, prefix_tree=None):
        if example_images.device.type != 'cuda':
            raise ValueError('CUDA graph inference requires CUDA query tensors')
        if (model.training or not getattr(model, 'supports_cached_decoding', False)
                or not getattr(getattr(model, 'decoder', None), 'inference_merged', False)):
            raise ValueError('call model.prepare_for_inference() before CUDA graph capture')
        if beam_size < 1 or warmup < 1:
            raise ValueError('beam_size and graph warmup must be positive')
        if not example_images.is_floating_point() or not example_images.numel():
            raise ValueError('query tensors must be nonempty floating tensors')
        if any(t.device != example_images.device for t in chain(model.parameters(), model.buffers())):
            raise ValueError('model and query tensors must use the same CUDA device')
        self.model = model
        self.beam_size = beam_size
        self.amp = amp
        self.images = example_images.clone()
        started = time.perf_counter()
        with torch.cuda.device(example_images.device), torch.autocast('cuda', enabled=amp, cache_enabled=False):
            self.conditioning_cache = model.cache_conditioning()
            stream = torch.cuda.Stream(device=example_images.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    beam_search(model, self.images, beam_size, conditioning_cache=self.conditioning_cache,
                                prefix_tree=prefix_tree)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize(example_images.device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.keys, self.scores = beam_search(model, self.images, beam_size,
                                                    conditioning_cache=self.conditioning_cache,
                                                    prefix_tree=prefix_tree)
            torch.cuda.synchronize(example_images.device)
        self.setup_ms = (time.perf_counter() - started) * 1000

    @torch.inference_mode()
    def __call__(self, images: Tensor) -> tuple[Tensor, Tensor]:
        if (images.shape != self.images.shape or images.dtype != self.images.dtype
                or images.device != self.images.device):
            raise ValueError('CUDA replay requires the captured input shape, dtype and device')
        self.images.copy_(images)
        self.graph.replay()
        return self.keys.clone(), self.scores.clone()


class GalleryIndex:
    """Pi(key), preserving gallery row order within collisions; no reranking."""

    def __init__(self, codes: Tensor):
        codes = torch.as_tensor(codes, dtype=torch.long, device='cpu')
        if codes.ndim != 2 or not codes.shape[0] or not codes.shape[1]:
            raise ValueError('gallery codes must have nonempty shape [N, L]')
        self.codes = codes
        self.members: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for row, code in enumerate(codes.tolist()):
            self.members[tuple(code)].append(row)

    def resolve(self, keys: Tensor, topk: int) -> Tensor:
        """Return [B, topk] gallery row IDs, padding absent results with -1."""
        if topk < 1 or keys.ndim != 3 or keys.shape[-1] != self.codes.shape[1]:
            raise ValueError('expected keys [B,K,L] and positive topk')
        ranks = torch.full((len(keys), topk), -1, dtype=torch.long)
        for query, candidates in enumerate(keys.detach().cpu().tolist()):
            result = []
            seen = set()
            for key in candidates:
                key = tuple(key)
                if key in seen:
                    continue
                seen.add(key)
                result.extend(self.members.get(key, ()))
                if len(result) >= topk:
                    break
            result = result[:topk]
            ranks[query, :len(result)] = torch.tensor(result, dtype=torch.long)
        return ranks


def query_diagnostics(keys: Tensor, gallery: GalleryIndex,
                      query_labels: Iterable[int], gallery_labels: Iterable[int]) -> list[dict]:
    """Per-query raw-key and expanded-image ranks (one-based; None = absent).

    A raw key is relevant if ANY member of its gallery bucket has the query
    label. Invalid and repeated keys retain their slots in raw ranks. Image
    ranks instead follow resolve(): skip invalid keys, expand a repeated key
    only once, and retain gallery row order within each bucket.

    The label oracle moves a relevant image to the front of its existing bucket
    only. It cannot change predicted keys, key order, or preceding bucket sizes.
    It is an evaluation diagnostic, not an implemented retrieval strategy.
    """
    keys = keys.detach().cpu()
    query_labels, gallery_labels = list(query_labels), list(gallery_labels)
    if (keys.ndim != 3 or not len(keys) or not keys.shape[1]
            or keys.shape[2] != gallery.codes.shape[1]):
        raise ValueError('expected nonempty raw keys [N, B, L] matching gallery code length')
    if len(query_labels) != len(keys) or len(gallery_labels) != len(gallery.codes):
        raise ValueError('query and gallery labels must align with keys and gallery rows')
    # First within-bucket position for each label avoids rescanning large buckets
    # for every query, while retaining the exact stable gallery ordering.
    label_positions = {}
    for code, members in gallery.members.items():
        positions = {}
        for offset, member in enumerate(members):
            positions.setdefault(int(gallery_labels[member]), offset)
        label_positions[code] = positions
    result = []
    for candidates, label in zip(keys.tolist(), query_labels):
        label = int(label)
        first_raw = first_image = oracle_image = None
        image_offset = 0
        seen = set()
        for raw_rank, values in enumerate(candidates, 1):
            code = tuple(values)
            members = gallery.members.get(code, ())
            relevant_offset = label_positions.get(code, {}).get(label)
            if raw_rank == 1:
                if not members:
                    outcome = 'invalid_key'
                elif relevant_offset is None:
                    outcome = 'valid_no_relevant_member'
                elif relevant_offset:
                    outcome = 'mixed_label_bucket_first_member_miss'
                else:
                    outcome = 'first_member_hit'
            if relevant_offset is not None and first_raw is None:
                first_raw = raw_rank
            if code in seen:
                continue
            seen.add(code)
            if relevant_offset is not None and first_image is None:
                first_image = image_offset + relevant_offset + 1
                oracle_image = image_offset + 1
            image_offset += len(members)
        result.append({
            'first_relevant_raw_key_rank': first_raw,
            'first_relevant_image_rank': first_image,
            'within_bucket_oracle_first_relevant_image_rank': oracle_image,
            'raw_top1_outcome': outcome,
        })
    return result


def gallery_collision_statistics(gallery: GalleryIndex, gallery_labels: Iterable[int]) -> dict:
    """Gallery occupancy statistics; these counts do not attribute query errors."""
    gallery_labels = list(gallery_labels)
    if len(gallery_labels) != len(gallery.codes):
        raise ValueError('gallery labels must align with gallery code rows')
    sizes = Counter(len(rows) for rows in gallery.members.values())
    same_label = mixed_label = same_items = mixed_items = 0
    for rows in gallery.members.values():
        if len(rows) <= 1:
            continue
        if len({int(gallery_labels[row]) for row in rows}) == 1:
            same_label += 1
            same_items += len(rows)
        else:
            mixed_label += 1
            mixed_items += len(rows)
    colliding_items = same_items + mixed_items
    return {
        'bucket_size_histogram': {str(size): count for size, count in sorted(sizes.items())},
        'mean_bucket_size': len(gallery.codes) / len(gallery.members),
        'max_bucket_size': max(sizes),
        'items_in_colliding_buckets': colliding_items,
        'item_weighted_collision_fraction': colliding_items / len(gallery.codes),
        'same_label_colliding_keys': same_label,
        'mixed_label_colliding_keys': mixed_label,
        'items_in_same_label_collisions': same_items,
        'items_in_mixed_label_collisions': mixed_items,
    }


def retrieval_report(keys: Tensor, ranks: Tensor, gallery: GalleryIndex,
                     query_labels: Iterable[int], gallery_labels: Iterable[int],
                     ks: Iterable[int] = (1, 4, 8)) -> dict:
    """Primary image Recall@K plus raw-key, collision, and ordering diagnostics.

    Pass image ranks from gallery.resolve(). Existing recall/reuse/validity
    fields retain their meanings. Raw-key cutoffs include requested ks that fit
    within the beam and the full beam width; no unavailable raw ranks are
    implied. All fractions use all queries unless the field names its condition.
    """
    keys, ranks = keys.detach().cpu(), ranks.detach().cpu()
    query_labels = torch.as_tensor(list(query_labels))
    gallery_labels = torch.as_tensor(list(gallery_labels))
    ks = tuple(ks)
    if not len(keys) or keys.ndim != 3 or keys.shape[1] == 0:
        raise ValueError('evaluation requires nonempty raw candidates')
    if len(query_labels) != len(keys) or len(ranks) != len(keys):
        raise ValueError('query labels, generated keys and ranks must align')
    if len(gallery_labels) != len(gallery.codes):
        raise ValueError('gallery labels must align with gallery code rows')
    if any(k < 1 or k > ranks.shape[1] for k in ks):
        raise ValueError('resolve at least max(ks) gallery positions before evaluation')
    if ((ranks < -1) | (ranks >= len(gallery.codes))).any():
        raise ValueError('rank IDs must be gallery row numbers or -1')
    relevant = (ranks >= 0) & (gallery_labels[ranks.clamp_min(0)] == query_labels[:, None])
    raw = [[tuple(code) for code in row] for row in keys.tolist()]
    validity = [[code in gallery.members for code in row] for row in raw]
    unique = len({row[0] for row in raw})
    n = len(raw)
    diagnostics = query_diagnostics(keys, gallery, query_labels, gallery_labels)
    first_raw = [row['first_relevant_raw_key_rank'] for row in diagnostics]
    raw_cutoffs = sorted({k for k in ks if k <= keys.shape[1]} | {keys.shape[1]})
    rank_counts = Counter(rank for rank in first_raw if rank is not None)
    rank_counts = {str(rank): count for rank, count in sorted(rank_counts.items())}
    rank_counts['not_found'] = sum(rank is None for rank in first_raw)
    outcomes = Counter(row['raw_top1_outcome'] for row in diagnostics)
    oracle_hits = {
        str(k): [row['within_bucket_oracle_first_relevant_image_rank'] is not None
                 and row['within_bucket_oracle_first_relevant_image_rank'] <= k
                 for row in diagnostics]
        for k in ks
    }
    actual_hits = {str(k): relevant[:, :k].any(1).tolist() for k in ks}
    rescued = {str(k): sum(oracle and not actual for oracle, actual
                          in zip(oracle_hits[str(k)], actual_hits[str(k)])) for k in ks}
    misses = {str(k): n - sum(actual_hits[str(k)]) for k in ks}
    return {
        'num_queries': n,
        'num_gallery': len(gallery.codes),
        'beam_size': keys.shape[1],
        'identifier_length': keys.shape[2],
        'recall': {str(k): relevant[:, :k].any(1).float().mean().item() for k in ks},
        'unique_top1_keys': unique,
        'reuse_rate': 1 - unique / n,
        'top1_invalid_rate': sum(not row[0] for row in validity) / n,
        'beam_valid_coverage': sum(any(row) for row in validity) / n,
        'unique_gallery_keys': len(gallery.members),
        'colliding_gallery_keys': sum(len(rows) > 1 for rows in gallery.members.values()),
        'raw_key_hit': {str(k): sum(rank is not None and rank <= k for rank in first_raw) / n
                        for k in raw_cutoffs},
        'raw_key_mrr_at_beam': sum(1 / rank for rank in first_raw if rank is not None) / n,
        'raw_first_relevant_key_rank_counts': rank_counts,
        'raw_top1_outcomes': {
            name: {'count': outcomes[name], 'fraction': outcomes[name] / n}
            for name in ('invalid_key', 'valid_no_relevant_member',
                         'mixed_label_bucket_first_member_miss', 'first_member_hit')
        },
        'gallery_collision': gallery_collision_statistics(gallery, gallery_labels),
        'within_bucket_oracle': {
            'recall': {str(k): sum(oracle_hits[str(k)]) / n for k in ks},
            'recall_gap': {str(k): rescued[str(k)] / n for k in ks},
            'rescued_queries': rescued,
            'observed_misses': misses,
            'conditional_rescue_fraction_among_misses': {
                str(k): rescued[str(k)] / misses[str(k)] if misses[str(k)] else None for k in ks
            },
            'scope': ('Query-label oracle reorders members only within each predicted bucket; '
                      'generated key order, invalid-key filtering, duplicate-key removal and '
                      'bucket sizes stay fixed. This is not a deployed reranker or a causal '
                      'estimate of total collision burden.'),
        },
        'ranking_policy': ('Keep supplied raw beam order; skip invalid keys; expand each '
                           'distinct valid key once; retain gallery row order within its '
                           'bucket; truncate to K images and pad shortfalls with -1.'),
    }
