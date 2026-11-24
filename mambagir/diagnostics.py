"""Offline, auditable diagnostics for saved clean-version retrieval predictions.

Example (no image loading, model inference, or training):
    python -m mambagir.diagnostics --predictions runs/eval/predictions.pt \
        --tokenizer runs/tokenizer/tokenizer.pt --manifest data/cub.json \
        --output-dir runs/eval/diagnostics

CSV ranks are one-based; gallery_row and target_row are zero-based. Empty rank
cells mean no relevant result in the saved beam. Labels use Manifest's canonical
target-derived IDs. JSON-encoded member lists preserve the actual bucket order.
"""

import argparse
import csv
import json
from pathlib import Path

import torch

from .data import Manifest
from .retrieval import GalleryIndex, query_diagnostics, retrieval_report


def _integer_tensor(payload, name, ndim):
    value = payload.get(name)
    if (not isinstance(value, torch.Tensor) or value.ndim != ndim
            or value.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
        raise ValueError(f'{name} must be an integer tensor with {ndim} dimensions')
    return value.detach().cpu()


def _load_inputs(predictions_path, tokenizer_path, manifest_path):
    manifest = Manifest.load(manifest_path)
    tokenizer = torch.load(tokenizer_path, map_location='cpu', weights_only=True)
    predictions = torch.load(predictions_path, map_location='cpu', weights_only=True)
    if (not isinstance(tokenizer, dict) or tokenizer.get('kind') != 'mambagir_tokenizer'
            or tokenizer.get('format_version') != 1 or not tokenizer.get('artifact_id')):
        raise ValueError('expected a clean-version tokenizer artifact with artifact_id')
    if not isinstance(predictions, dict) or not predictions.get('tokenizer_id'):
        raise ValueError('predictions require tokenizer_id; rerun evaluation to save identity metadata')
    if (predictions.get('kind') != 'mambagir_predictions'
            or predictions.get('format_version') != 1):
        raise ValueError('expected a clean-version predictions artifact with format_version=1')
    if predictions['tokenizer_id'] != tokenizer['artifact_id']:
        raise ValueError('prediction tokenizer identity differs from the supplied tokenizer artifact')
    identity = {
        'target_paths': [r.path for r in manifest.targets],
        'target_labels': [r.label for r in manifest.targets],
        'gallery_indices': manifest.gallery_indices,
    }
    for name, expected in identity.items():
        if tokenizer.get(name) != expected:
            raise ValueError(f'{name} differs between tokenizer artifact and manifest')
    gallery_records = [manifest.targets[i] for i in manifest.gallery_indices]
    prediction_identity = {
        'query_paths': [r.path for r in manifest.queries],
        'query_labels': [r.label for r in manifest.queries],
        'gallery_paths': [r.path for r in gallery_records],
        'gallery_labels': [r.label for r in gallery_records],
    }
    for name, expected in prediction_identity.items():
        if predictions.get(name) != expected:
            raise ValueError(f'{name} differs between predictions and manifest (or metadata is missing)')
    target_codes = _integer_tensor(tokenizer, 'target_codes', 2)
    if len(target_codes) != len(manifest.targets):
        raise ValueError('target_codes must align with manifest target rows')
    gallery = GalleryIndex(target_codes[manifest.gallery_indices])
    keys = _integer_tensor(predictions, 'raw_keys', 3)
    if (len(keys) != len(manifest.queries) or not len(keys) or not keys.shape[1]
            or keys.shape[2] != target_codes.shape[1]):
        raise ValueError('raw_keys must have nonempty shape [queries, beam, tokenizer code length]')
    scores = predictions.get('log_scores')
    if not isinstance(scores, torch.Tensor) or scores.shape != keys.shape[:2]:
        raise ValueError('log_scores must align with raw_keys [queries, beam]')
    ranks = _integer_tensor(predictions, 'ranks', 2)
    if (len(ranks) != len(keys) or not ranks.shape[1]
            or not torch.equal(ranks, gallery.resolve(keys, ranks.shape[1]))):
        raise ValueError('saved ranks do not match stable gallery resolution of raw_keys')
    return predictions, manifest, gallery


def _write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_diagnostics(predictions_path, tokenizer_path, manifest_path, output_dir, ks=None):
    """Verify saved evidence and export aggregate metrics plus all raw candidates.

    Legacy prediction dumps without tokenizer identity and label metadata are
    rejected rather than guessed. The four CSVs contain all queries, all raw
    candidates, raw-top-1 failure cases, and every gallery bucket respectively.
    A raw-top-1 failure can still be an image R@1 success after invalid filtering.
    """
    predictions, manifest, gallery = _load_inputs(predictions_path, tokenizer_path, manifest_path)
    keys = predictions['raw_keys']
    if ks is None:
        saved_cutoffs = predictions.get('metrics', {}).get('recall', {})
        ks = [int(k) for k in saved_cutoffs] if saved_cutoffs else [1, predictions['ranks'].shape[1]]
    ks = sorted(set(ks))
    if not ks or any(not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError('ks must contain positive integer image cutoffs')
    query_labels = predictions['query_labels']
    gallery_labels = predictions['gallery_labels']
    report = retrieval_report(keys, gallery.resolve(keys, max(ks)), gallery,
                              query_labels, gallery_labels, ks)
    per_query = query_diagnostics(keys, gallery, query_labels, gallery_labels)
    report.update({
        'tokenizer_id': predictions['tokenizer_id'],
        'inputs': {
            'predictions': str(Path(predictions_path).resolve()),
            'tokenizer': str(Path(tokenizer_path).resolve()),
            'manifest': str(Path(manifest_path).resolve()),
        },
        'diagnostic_definitions': {
            'raw_key_hit': ('Fraction of queries with a relevant key in the first K raw beam slots; '
                            'a key is relevant if any bucket member matches the query label. '
                            'Invalid and repeated keys retain raw slots. This is not image Recall@K.'),
            'raw_key_mrr_at_beam': ('Mean reciprocal first relevant raw-key rank within the saved '
                                    'beam, with zero for no relevant key; not full-vocabulary MRR.'),
            'item_weighted_collision_fraction': ('Gallery images belonging to buckets of size >1 '
                                                  'divided by all gallery images.'),
            'failure_cases': ('One raw-rank-1 row per query whose first raw bucket is invalid, '
                              'contains no relevant member, or starts with an irrelevant member. '
                              'This list is not a partition of post-filter image R@1 failures.'),
            'row_order': ('Gallery rows follow manifest.gallery_indices in its recorded order, '
                          'not sorted target row IDs or filenames. Bucket members preserve this order.'),
            'missing_ranks': 'Null in JSON and empty in CSV mean not found within the saved beam.',
            'labels': 'Canonical integer labels assigned by Manifest from the target label set.',
        },
    })
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'diagnostics.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def members_for(code):
        return [{
            'gallery_row': row,
            'target_row': manifest.gallery_indices[row],
            'path': predictions['gallery_paths'][row],
            'label': gallery_labels[row],
        } for row in gallery.members.get(code, ())]

    def query_rows():
        for query, detail in enumerate(per_query):
            yield {'query_index': query, 'query_path': predictions['query_paths'][query],
                   'query_label': query_labels[query], **detail}

    query_fields = ['query_index', 'query_path', 'query_label', 'first_relevant_raw_key_rank',
                    'first_relevant_image_rank', 'within_bucket_oracle_first_relevant_image_rank',
                    'raw_top1_outcome']
    _write_csv(output_dir / 'queries.csv', query_fields, query_rows())

    candidate_fields = ['query_index', 'query_path', 'query_label', 'raw_rank', 'raw_key',
                        'log_score', 'valid', 'relevant_key', 'first_member_relevant', 'bucket_size',
                        'members', 'raw_top1_outcome', 'first_relevant_raw_key_rank',
                        'first_relevant_image_rank', 'within_bucket_oracle_first_relevant_image_rank']

    def candidate_rows(failures_only=False):
        for query, candidates in enumerate(keys.tolist()):
            detail = per_query[query]
            if failures_only and detail['raw_top1_outcome'] == 'first_member_hit':
                continue
            for raw_rank, values in enumerate(candidates, 1):
                if failures_only and raw_rank > 1:
                    break
                members = members_for(tuple(values))
                label = query_labels[query]
                yield {
                    'query_index': query, 'query_path': predictions['query_paths'][query],
                    'query_label': label, 'raw_rank': raw_rank, 'raw_key': json.dumps(values),
                    'log_score': predictions['log_scores'][query, raw_rank - 1].item(),
                    'valid': bool(members), 'relevant_key': any(r['label'] == label for r in members),
                    'first_member_relevant': bool(members) and members[0]['label'] == label,
                    'bucket_size': len(members), 'members': json.dumps(members, ensure_ascii=False),
                    **detail,
                }

    _write_csv(output_dir / 'raw_candidates.csv', candidate_fields, candidate_rows())
    _write_csv(output_dir / 'failure_cases.csv', candidate_fields, candidate_rows(failures_only=True))

    def bucket_rows():
        for code, rows in gallery.members.items():
            labels = sorted({gallery_labels[row] for row in rows})
            collision_type = ('singleton' if len(rows) == 1 else
                              'same_label' if len(labels) == 1 else 'mixed_label')
            yield {'key': json.dumps(code), 'bucket_size': len(rows),
                   'collision_type': collision_type, 'labels': json.dumps(labels),
                   'members': json.dumps(members_for(code), ensure_ascii=False)}

    _write_csv(output_dir / 'gallery_buckets.csv',
               ['key', 'bucket_size', 'collision_type', 'labels', 'members'], bucket_rows())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--predictions', required=True, help='Saved predictions.pt from evaluation')
    parser.add_argument('--tokenizer', required=True, help='The exact tokenizer.pt used by the generator')
    parser.add_argument('--manifest', required=True, help='Matching target/query/gallery manifest JSON')
    parser.add_argument('--output-dir', required=True, help='Destination for JSON and CSV diagnostics')
    parser.add_argument('--ks', type=int, nargs='+', help='Image Recall cutoffs (default: saved report cutoffs)')
    args = parser.parse_args(argv)
    try:
        report = export_diagnostics(args.predictions, args.tokenizer, args.manifest, args.output_dir, args.ks)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({'num_queries': report['num_queries'], 'tokenizer_id': report['tokenizer_id'],
                      'output_dir': str(Path(args.output_dir).expanduser().resolve())}, indent=2))


if __name__ == '__main__':
    main()
