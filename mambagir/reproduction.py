"""Export fixed-gallery metadata and bind the same tokenizer after relocation.

Run ``python -m mambagir.reproduction --help``. No model is instantiated, no
image is read, and no training or identifier reassignment occurs here.
"""

import argparse
import json
from pathlib import Path, PurePosixPath

import torch

from .data import Manifest, Record
from .retrieval import GalleryIndex
from .runtime import _load_tokenizer


def _relative_manifest(manifest, dataset_root):
    def records(rows):
        result = []
        for row in rows:
            try:
                relative = Path(row.path).relative_to(dataset_root)
            except ValueError as error:
                raise ValueError(f'image is outside the dataset root: {row.path}') from error
            result.append({'path': relative.as_posix(), 'label': row.label})
        return result

    return {'targets': records(manifest.targets), 'queries': records(manifest.queries),
            'gallery_indices': list(manifest.gallery_indices)}


def _manifest_at_root(payload, dataset_root):
    def records(name):
        result = []
        for row in payload[name]:
            relative = PurePosixPath(row['path'])
            if relative.is_absolute() or '..' in relative.parts or not relative.parts:
                raise ValueError('manifest paths must be relative to the dataset root')
            image = dataset_root.joinpath(*relative.parts).resolve()
            if not image.is_relative_to(dataset_root):
                raise ValueError('manifest image resolves outside the dataset root')
            result.append(Record(str(image), row['label']))
        return result

    manifest = Manifest(records('targets'), records('queries'), payload['gallery_indices'])
    if _relative_manifest(manifest, dataset_root) != payload:
        raise ValueError('portable manifest must preserve canonical labels, paths and gallery order')
    return manifest


def _codebook_metadata(artifact):
    codebooks, codes = artifact['codebooks'], artifact['target_codes']
    if (not isinstance(codebooks, torch.Tensor) or codebooks.ndim != 3
            or min(codebooks.shape) < 1 or not codebooks.is_floating_point()):
        raise ValueError('codebooks must be a floating tensor [levels, vocabulary, feature_dim]')
    if (not isinstance(codes, torch.Tensor) or codes.dtype != torch.long
            or tuple(codes.shape) != (len(artifact['target_paths']), codebooks.shape[0])
            or (codes < 0).any() or (codes >= codebooks.shape[1]).any()):
        raise ValueError('target_codes must have one valid long identifier per target row')
    rvq = artifact['config']['rvq']
    if (rvq['levels'], rvq['codebook_size']) != tuple(codebooks.shape[:2]):
        raise ValueError('codebooks shape differs from tokenizer config')
    return {'shape': list(codebooks.shape), 'dtype': str(codebooks.dtype)}


def _gallery_mapping(payload, artifact):
    indices = payload['gallery_indices']
    gallery = GalleryIndex(artifact['target_codes'][indices])
    entries = []
    for identifier, rows in gallery.members.items():
        entries.append({'identifier': list(identifier), 'members': [
            {'gallery_row': row, 'target_row': indices[row],
             **payload['targets'][indices[row]]} for row in rows
        ]})
    return {'format_version': 1, 'tokenizer_artifact_id': artifact['artifact_id'],
            'entries': entries}


def _output_paths(output_dir, names):
    output_dir = Path(output_dir).expanduser().resolve()
    paths = {name: output_dir / filename for name, filename in names.items()}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f'output already exists: {path}; choose a new directory')
    output_dir.mkdir(parents=True, exist_ok=True)
    return paths


def _write_json(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write('\n')


def export_bundle(manifest_path, tokenizer_path, dataset_root, output_dir):
    """Export relative records and all key members from a matching tokenizer.

    The exported manifest uses the existing Manifest schema. Put it directly at
    a dataset root to load it there, or use bind_bundle to produce local paths.
    The existing tokenizer UUID identifies its frozen codebooks and config.
    """
    dataset_root = Path(dataset_root).expanduser().resolve()
    manifest = Manifest.load(manifest_path)
    artifact = _load_tokenizer(tokenizer_path, manifest)
    codebooks = _codebook_metadata(artifact)
    payload = _relative_manifest(manifest, dataset_root)
    mapping = _gallery_mapping(payload, artifact)
    provenance = {
        'format_version': 1, 'kind': 'mambagir_reproduction',
        'tokenizer_artifact_id': artifact['artifact_id'],
        'tokenizer_format_version': artifact['format_version'],
        'tokenizer_epoch': artifact['epoch'],
        'effective_config': artifact['config'], 'codebooks': codebooks,
        'source_dataset_root': str(dataset_root),
        'gallery_indices': list(manifest.gallery_indices),
        'num_targets': len(manifest.targets), 'num_queries': len(manifest.queries),
        'source_metadata': 'manifest records and tokenizer training-export metadata supplied by the caller',
        'derived_metadata': 'dataset-relative paths and identifier groups computed by this exporter',
        'path_base': 'dataset_root',
        'row_order': 'target/query input order; gallery_indices order; keys by first gallery occurrence',
    }
    paths = _output_paths(output_dir, {
        'manifest': 'manifest.json', 'mapping': 'identifier_to_gallery.json',
        'provenance': 'provenance.json',
    })
    for name, value in (('manifest', payload), ('mapping', mapping), ('provenance', provenance)):
        _write_json(paths[name], value)
    return paths


def bind_bundle(bundle_dir, tokenizer_path, dataset_root, output_dir):
    """Copy the same tokenizer with local dataset paths, retaining its UUID.

    Existing runtime identity checks intentionally compare absolute paths. This
    explicit binding checks the exported source identity and mappings before
    changing only target_paths and manifest_path in a new tokenizer file.
    The source dataset directory need not exist on the receiving machine.
    """
    bundle_dir = Path(bundle_dir).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    payload = json.loads((bundle_dir / 'manifest.json').read_text(encoding='utf-8'))
    provenance = json.loads((bundle_dir / 'provenance.json').read_text(encoding='utf-8'))
    mapping = json.loads((bundle_dir / 'identifier_to_gallery.json').read_text(encoding='utf-8'))
    if provenance.get('kind') != 'mambagir_reproduction' or provenance.get('format_version') != 1:
        raise ValueError('expected a version 1 reproduction bundle')
    source = _manifest_at_root(payload, Path(provenance['source_dataset_root']))
    artifact = _load_tokenizer(tokenizer_path, source)
    if artifact['artifact_id'] != provenance['tokenizer_artifact_id']:
        raise ValueError('tokenizer artifact identity differs from the exported bundle')
    if artifact['config'] != provenance['effective_config']:
        raise ValueError('tokenizer config differs from the exported bundle')
    if _codebook_metadata(artifact) != provenance['codebooks']:
        raise ValueError('tokenizer codebook metadata differs from the exported bundle')
    if (payload['gallery_indices'] != provenance['gallery_indices']
            or len(source.targets) != provenance['num_targets']
            or len(source.queries) != provenance['num_queries']):
        raise ValueError('manifest metadata differs from the exported bundle')
    if _gallery_mapping(payload, artifact) != mapping:
        raise ValueError('identifier mapping differs from the tokenizer or gallery row order')
    local = _manifest_at_root(payload, dataset_root)
    paths = _output_paths(output_dir, {'manifest': 'manifest.json', 'tokenizer': 'tokenizer.pt'})
    artifact['target_paths'] = [row.path for row in local.targets]
    artifact['manifest_path'] = str(paths['manifest'])
    local.save(paths['manifest'])
    torch.save(artifact, paths['tokenizer'])
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export', help='export dataset-relative manifest, key mapping and provenance')
    export.add_argument('--manifest', required=True)
    bind = commands.add_parser('bind', help='bind an exported bundle and the same tokenizer to a local dataset root')
    bind.add_argument('--bundle', required=True)
    for command in (export, bind):
        command.add_argument('--tokenizer', required=True)
        command.add_argument('--dataset-root', required=True)
        command.add_argument('--output', required=True, help='directory with no existing output artifacts')
    args = parser.parse_args(argv)
    if args.command == 'export':
        paths = export_bundle(args.manifest, args.tokenizer, args.dataset_root, args.output)
    else:
        paths = bind_bundle(args.bundle, args.tokenizer, args.dataset_root, args.output)
    print(json.dumps({name: str(path) for name, path in paths.items()}))


if __name__ == '__main__':
    main()
