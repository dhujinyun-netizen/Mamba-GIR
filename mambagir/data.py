"""Official fixed-gallery data roles and image preprocessing.

Manifests contain absolute paths in memory. Labels are mapped using only targets;
evaluation queries must have a relevant image in the searchable gallery. File
existence is checked when an image is loaded, so metadata can be prepared offline.
"""

from dataclasses import asdict, dataclass
import json
from numbers import Integral
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Record:
    path: str
    label: int


@dataclass
class Manifest:
    targets: list[Record]
    queries: list[Record]
    gallery_indices: list[int]

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Validate roles and canonicalize paths and target-derived class IDs."""
        if not self.targets:
            raise ValueError("A manifest must contain target images")
        if any(not isinstance(r.label, Integral) for r in [*self.targets, *self.queries]):
            raise ValueError("Record labels must be integers")
        mapping = {label: i for i, label in enumerate(sorted({r.label for r in self.targets}))}
        unknown = {r.label for r in self.queries} - mapping.keys()
        if unknown:
            raise ValueError(f"Query labels are absent from the target pool: {sorted(unknown)}")

        def canonical(records):
            return [Record(str(Path(r.path).expanduser().resolve()), mapping[r.label])
                    for r in records]

        targets, queries = canonical(self.targets), canonical(self.queries)
        target_paths = {r.path for r in targets}
        query_paths = {r.path for r in queries}
        if len(target_paths) != len(targets) or len(query_paths) != len(queries):
            raise ValueError("Duplicate image paths within a split are not allowed")
        if target_paths & query_paths:
            raise ValueError("Evaluation queries must not overlap the target pool")
        if not self.gallery_indices:
            raise ValueError("The searchable gallery must not be empty")
        if any(not isinstance(i, Integral) or i < 0 or i >= len(targets)
               for i in self.gallery_indices):
            raise ValueError("gallery_indices must refer to valid target rows")
        if len(set(self.gallery_indices)) != len(self.gallery_indices):
            raise ValueError("gallery_indices must not contain duplicate rows")
        gallery_labels = {targets[i].label for i in self.gallery_indices}
        if {r.label for r in queries} - gallery_labels:
            raise ValueError("Every query label must have a relevant searchable gallery image")
        self.targets, self.queries = targets, queries
        self.gallery_indices = [int(i) for i in self.gallery_indices]

    @property
    def num_classes(self):
        return len({r.label for r in self.targets})

    @classmethod
    def load(cls, path):
        path = Path(path).expanduser().resolve()
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        def records(name):
            result = []
            for row in payload[name]:
                image = Path(row["path"]).expanduser()
                if not image.is_absolute():
                    image = path.parent / image
                result.append(Record(str(image), row["label"]))
            return result

        return cls(records("targets"), records("queries"), payload["gallery_indices"])

    def save(self, path):
        self.validate()
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump({"targets": [asdict(r) for r in self.targets],
                       "queries": [asdict(r) for r in self.queries],
                       "gallery_indices": self.gallery_indices}, handle, indent=2)
            handle.write("\n")


def _find_file(candidates):
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Required metadata not found; checked: " +
                            ", ".join(str(p) for p in candidates))


def _keyed_rows(path):
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            key, value = line.strip().split(maxsplit=1)
            key = int(key)
            if key in rows:
                raise ValueError(f"Duplicate metadata row {key} in {path}")
            rows[key] = value
    return rows


def _manifest_from_named_labels(targets, queries, gallery_indices):
    """Map string labels from target metadata without fitting on queries."""
    labels = {label: i for i, label in enumerate(sorted({label for _, label in targets}))}
    unknown = {label for _, label in queries} - labels.keys()
    if unknown:
        raise ValueError(f"Query labels are absent from the target pool: {sorted(unknown)}")
    return Manifest([Record(str(path), labels[label]) for path, label in targets],
                    [Record(str(path), labels[label]) for path, label in queries],
                    gallery_indices)


def _prepare_cub(root):
    image_list = _find_file([base / "images.txt" for base in
                             (root, root / "CUB_200_2011", root / "cub")])
    base = image_list.parent
    images = _keyed_rows(image_list)
    labels = _keyed_rows(base / "image_class_labels.txt")
    splits = _keyed_rows(base / "train_test_split.txt")
    if images.keys() != labels.keys() or images.keys() != splits.keys():
        raise ValueError("CUB image, label, and split metadata must have matching IDs")
    targets, queries = [], []
    for row, name in images.items():
        if splits[row] not in ("0", "1"):
            raise ValueError(f"Unknown CUB split flag: {splits[row]}")
        record = Record(str(base / "images" / name), int(labels[row]))
        (targets if splits[row] == "1" else queries).append(record)
    return Manifest(targets, queries, list(range(len(targets))))


def _prepare_cars(root):
    try:
        from scipy.io import loadmat
    except ImportError as error:
        raise ImportError("Reading official Cars .mat annotations requires scipy") from error
    metadata_dirs = (root, root / "devkit", root / "devkit" / "devkit")
    train_annotations = _find_file([p / "cars_train_annos.mat" for p in metadata_dirs])
    test_annotations = _find_file([p / "cars_test_annos_withlabels.mat" for p in metadata_dirs])

    def read(path, split):
        raw = loadmat(path, squeeze_me=True, struct_as_record=False)
        if "annotations" not in raw:
            raise ValueError(f"No annotations in {path}")
        image_dirs = (root / split, root / split / split,
                      root / "archive" / split, root / "archive" / split / split)
        default_dir = next((p for p in image_dirs if p.is_dir()), image_dirs[0])
        records = []
        for row in np.atleast_1d(raw["annotations"]).reshape(-1):
            if not hasattr(row, "class") or not hasattr(row, "fname"):
                raise ValueError(f"Cars annotations must include class and fname: {path}")
            name = str(row.fname)
            image = next((p / name for p in image_dirs if (p / name).is_file()), default_dir / name)
            records.append(Record(str(image), int(getattr(row, "class"))))
        return records

    targets = read(train_annotations, "cars_train")
    return Manifest(targets, read(test_annotations, "cars_test"), list(range(len(targets))))


def _prepare_aircraft(root):
    train_file = _find_file([base / "images_variant_trainval.txt" for base in
                             (root, root / "data", root / "fgvc-aircraft-2013b" / "data")])
    base = train_file.parent

    def read(path):
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                image_id, variant = line.strip().split(maxsplit=1)
                records.append((base / "images" / f"{image_id}.jpg", variant))
        return records

    targets = read(train_file)
    return _manifest_from_named_labels(targets, read(base / "images_variant_test.txt"),
                                       list(range(len(targets))))


def _prepare_inshop(root):
    partition = _find_file([root / "Eval" / "list_eval_partition.txt",
                            root / "list_eval_partition.txt"])
    lines = [line.strip() for line in partition.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    declared_count = int(lines.pop(0)) if lines and lines[0].isdigit() else None
    if lines and lines[0].split()[0] == "image_name":
        lines.pop(0)
    if declared_count is not None and declared_count != len(lines):
        raise ValueError("InShop partition count does not match its rows")
    splits = {"train": [], "gallery": [], "query": []}
    for line in lines:
        name, item_id, split = line.split()
        if split not in splits:
            raise ValueError(f"Unknown InShop split: {split}")
        splits[split].append((root / name, item_id))
    targets = splits["train"] + splits["gallery"]
    return _manifest_from_named_labels(targets, splits["query"],
                                       list(range(len(splits["train"]), len(targets))))


def prepare_manifest(dataset: str, root: str) -> Manifest:
    """Read official splits; never include evaluation queries in the target pool."""
    dataset = dataset.lower().replace("-", "").replace("_", "")
    loaders = {"cub": _prepare_cub, "cub200": _prepare_cub, "cub2002011": _prepare_cub,
               "cars": _prepare_cars, "car": _prepare_cars, "cars196": _prepare_cars,
               "aircraft": _prepare_aircraft, "fgvcaircraft": _prepare_aircraft,
               "fgvcaircraft2013b": _prepare_aircraft, "inshop": _prepare_inshop}
    if dataset not in loaders:
        raise ValueError(f"Unsupported dataset: {dataset}; use cub, cars, aircraft, or inshop")
    return loaders[dataset](Path(root).expanduser().resolve())


class ImageDataset(Dataset):
    def __init__(self, records: Sequence[Record], train=False, resize=256, crop=224):
        if crop <= 0 or resize < crop:
            raise ValueError("Image sizes must satisfy resize >= crop > 0")
        self.records = list(records)
        self.train, self.resize, self.crop = train, int(resize), int(crop)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record.path) as source:
            image = source.convert("RGB").resize((self.resize, self.resize), Image.Resampling.BILINEAR)
        margin = self.resize - self.crop
        if self.train:
            top, left = torch.randint(margin + 1, (2,)).tolist()
        else:
            top = left = margin // 2
        image = image.crop((left, top, left + self.crop, top + self.crop))
        if self.train and torch.rand(()).item() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        tensor = torch.from_numpy(np.array(image, dtype=np.float32, copy=True))
        tensor = tensor.permute(2, 0, 1).contiguous().div_(255.0)
        return {"image": (tensor - self.mean) / self.std,
                "label": torch.tensor(record.label, dtype=torch.long), "index": int(index)}


def sample_positive_indices(labels, seed, epoch) -> torch.LongTensor:
    """Independent uniform draws per source, excluding itself unless singleton."""
    labels = torch.as_tensor(labels).detach().cpu()
    if labels.ndim != 1:
        raise ValueError("labels must have shape [N]")
    generator = torch.Generator().manual_seed((int(seed) + int(epoch)) % (2**63))
    result = torch.empty(labels.numel(), dtype=torch.long)
    groups = {}
    for index, label in enumerate(labels.tolist()):
        groups.setdefault(label, []).append(index)
    for indices in groups.values():
        rows = torch.tensor(indices, dtype=torch.long)
        count = len(indices)
        if count == 1:
            result[rows] = rows
        else:
            choices = torch.randint(count - 1, (count,), generator=generator)
            choices += choices >= torch.arange(count)
            result[rows] = rows[choices]
    return result
