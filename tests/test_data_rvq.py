"""Data-role and residual-quantization contracts, using only temporary data."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mambagir.data import (
    ImageDataset,
    Manifest,
    Record,
    prepare_manifest,
    sample_positive_indices,
)
from mambagir.rvq import ResidualVectorQuantizer


class ManifestTests(unittest.TestCase):
    def test_load_resolves_paths_and_maps_labels_from_targets_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "manifest.json"
            source.write_text(json.dumps({
                "targets": [
                    {"path": "t0.jpg", "label": 90},
                    {"path": "t1.jpg", "label": 4},
                    {"path": "t2.jpg", "label": 90},
                ],
                "queries": [
                    {"path": "q0.jpg", "label": 4},
                    {"path": "q1.jpg", "label": 90},
                ],
                "gallery_indices": [2, 1],
            }))
            manifest = Manifest.load(source)
            self.assertEqual([r.label for r in manifest.targets], [1, 0, 1])
            self.assertEqual([r.label for r in manifest.queries], [0, 1])
            self.assertEqual(manifest.gallery_indices, [2, 1])
            self.assertEqual(manifest.num_classes, 2)
            self.assertEqual(manifest.targets[0].path, str(root / "t0.jpg"))
            saved = root / "nested" / "roundtrip.json"
            manifest.save(saved)
            loaded = Manifest.load(saved)
            self.assertEqual(loaded.targets, manifest.targets)
            self.assertEqual(loaded.queries, manifest.queries)
            self.assertEqual(loaded.gallery_indices, [2, 1])

    def test_unknown_query_class_is_rejected_instead_of_fitted(self):
        with self.assertRaises(ValueError):
            Manifest([Record("target.jpg", 1)], [Record("query.jpg", 99)], [0])

    def test_query_requires_relevant_searchable_gallery_image(self):
        with self.assertRaises(ValueError):
            Manifest(
                [Record("train.jpg", 1), Record("gallery.jpg", 2)],
                [Record("query.jpg", 1)],
                [1],
            )

    def test_path_alias_cannot_leak_a_query_into_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                Manifest(
                    [Record(str(root / "image.jpg"), 0)],
                    [Record(str(root / "subdir" / ".." / "image.jpg"), 0)],
                    [0],
                )

    def test_invalid_gallery_rows_are_rejected(self):
        for indices in ([-1], [1], [0, 0], []):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                Manifest([Record("target.jpg", 0)], [Record("query.jpg", 0)], indices)


class OfficialSplitTests(unittest.TestCase):
    def test_cub_uses_official_image_split_with_shared_species(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images.txt").write_text(
                "1 species_b/train.jpg\n2 species_b/test.jpg\n"
                "3 species_a/train.jpg\n4 species_a/test.jpg\n"
            )
            (root / "image_class_labels.txt").write_text("1 20\n2 20\n3 10\n4 10\n")
            (root / "train_test_split.txt").write_text("1 1\n2 0\n3 1\n4 0\n")
            manifest = prepare_manifest("cub", str(root))
            self.assertEqual([r.label for r in manifest.targets], [1, 0])
            self.assertEqual([r.label for r in manifest.queries], [1, 0])
            self.assertEqual(manifest.gallery_indices, [0, 1])
            self.assertEqual(manifest.targets[0].path, str(root / "images/species_b/train.jpg"))
            self.assertEqual(manifest.queries[0].path, str(root / "images/species_b/test.jpg"))

    def test_aircraft_preserves_variant_names_with_spaces(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            (data / "images_variant_trainval.txt").write_text(
                "0000001 Boeing 737-800\n0000002 Airbus A320\n0000003 Airbus A320\n"
            )
            (data / "images_variant_test.txt").write_text(
                "0000004 Airbus A320\n0000005 Boeing 737-800\n"
            )
            manifest = prepare_manifest("aircraft", str(root))
            self.assertEqual([r.label for r in manifest.targets], [1, 0, 0])
            self.assertEqual([r.label for r in manifest.queries], [0, 1])
            self.assertEqual(manifest.gallery_indices, [0, 1, 2])
            self.assertEqual(manifest.targets[0].path, str(data / "images/0000001.jpg"))

    def test_aircraft_query_only_variant_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images_variant_trainval.txt").write_text("0000001 A320\n")
            (root / "images_variant_test.txt").write_text("0000002 A380\n")
            with self.assertRaises(ValueError):
                prepare_manifest("aircraft", str(root))

    def test_inshop_targets_include_train_then_gallery_but_only_gallery_is_searchable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Eval").mkdir()
            (root / "Eval/list_eval_partition.txt").write_text(
                "5\nimage_name item_id evaluation_status\n"
                "img/gallery2.jpg id_00000009 gallery\n"
                "img/train.jpg id_00000003 train\n"
                "img/query.jpg id_00000009 query\n"
                "img/gallery1.jpg id_00000009 gallery\n"
                "img/train2.jpg id_00000003 train\n"
            )
            manifest = prepare_manifest("inshop", str(root))
            self.assertEqual([Path(r.path).name for r in manifest.targets],
                             ["train.jpg", "train2.jpg", "gallery2.jpg", "gallery1.jpg"])
            self.assertEqual(manifest.gallery_indices, [2, 3])
            self.assertEqual([r.label for r in manifest.targets], [0, 0, 1, 1])
            self.assertEqual([r.label for r in manifest.queries], [1])
            self.assertNotIn(manifest.queries[0].path, [r.path for r in manifest.targets])

    def test_cars_reads_official_mat_annotations(self):
        try:
            from scipy.io import savemat
        except ImportError:
            self.skipTest("Cars annotations require optional scipy")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "devkit").mkdir()
            (root / "cars_train").mkdir()
            (root / "cars_test").mkdir()
            dtype = [("bbox_x1", "O"), ("bbox_y1", "O"), ("bbox_x2", "O"),
                     ("bbox_y2", "O"), ("class", "O"), ("fname", "O")]
            train = np.array([(1, 1, 8, 8, 7, "00001.jpg"),
                              (1, 1, 8, 8, 2, "00002.jpg")], dtype=dtype)
            query = np.array([(1, 1, 8, 8, 2, "00003.jpg"),
                              (1, 1, 8, 8, 7, "00004.jpg")], dtype=dtype)
            savemat(root / "devkit/cars_train_annos.mat", {"annotations": train})
            savemat(root / "cars_test_annos_withlabels.mat", {"annotations": query})
            manifest = prepare_manifest("cars", str(root))
            self.assertEqual([r.label for r in manifest.targets], [1, 0])
            self.assertEqual([r.label for r in manifest.queries], [0, 1])
            self.assertEqual(manifest.targets[0].path, str(root / "cars_train/00001.jpg"))
            self.assertEqual(manifest.queries[0].path, str(root / "cars_test/00003.jpg"))


class ImageAndPositiveTests(unittest.TestCase):
    def test_eval_image_rgb_shape_and_imagenet_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.new("L", (17, 11), color=255).save(path)
            sample = ImageDataset([Record(str(path), 7)], resize=10, crop=8)[0]
            self.assertEqual(tuple(sample["image"].shape), (3, 8, 8))
            self.assertEqual(sample["image"].dtype, torch.float32)
            torch.testing.assert_close(sample["image"][:, 0, 0],
                                       torch.tensor([2.2489083, 2.4285714, 2.64]))
            self.assertEqual(sample["label"].dtype, torch.long)
            self.assertEqual(sample["label"].item(), 7)
            self.assertEqual(sample["index"], 0)

    def test_evaluation_uses_center_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            array = np.zeros((6, 6, 3), dtype=np.uint8)
            array[1:5, 1:5] = 255
            Image.fromarray(array).save(path)
            image = ImageDataset([Record(str(path), 0)], resize=6, crop=4)[0]["image"]
            self.assertTrue(torch.all(image > 2.0))

    def test_train_augmentation_uses_torch_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            array = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
            Image.fromarray(array).save(path)
            dataset = ImageDataset([Record(str(path), 0)], train=True, resize=8, crop=4)
            torch.manual_seed(42)
            first = dataset[0]["image"]
            torch.manual_seed(42)
            torch.testing.assert_close(first, dataset[0]["image"])
            self.assertEqual(tuple(first.shape), (3, 4, 4))

    def test_positive_sampling_excludes_self_and_handles_singletons(self):
        labels = torch.tensor([10, 10, 10, 20, 20, 99])
        positives = sample_positive_indices(labels, seed=3407, epoch=0)
        self.assertEqual(positives.dtype, torch.long)
        torch.testing.assert_close(labels[positives], labels)
        self.assertTrue(torch.all(positives[:5] != torch.arange(5)))
        self.assertEqual(positives[5].item(), 5)
        torch.testing.assert_close(positives, sample_positive_indices(labels, 3407, 0))

    def test_positive_draws_are_independent_not_a_group_permutation(self):
        labels = [0, 0, 0, 0]
        draws = torch.stack([sample_positive_indices(labels, 21, epoch) for epoch in range(300)])
        self.assertTrue(any(len(set(row.tolist())) < 4 for row in draws))
        self.assertGreater(len(set(tuple(row.tolist()) for row in draws)), 20)
        for source in range(4):
            counts = torch.bincount(draws[:, source], minlength=4)
            self.assertEqual(counts[source].item(), 0)
            self.assertTrue(torch.all(counts[torch.arange(4) != source] > 60))


class RVQTests(unittest.TestCase):
    def test_encoding_quantizes_residual_instead_of_original_feature(self):
        rvq = ResidualVectorQuantizer(levels=2, codebook_size=2)
        rvq.codebooks = torch.tensor([[[0.0], [10.0]], [[0.0], [3.0]]])
        codes = rvq.encode(torch.tensor([[10.0], [13.0], [3.0], [0.0]]))
        torch.testing.assert_close(codes, torch.tensor([[1, 0], [1, 1], [0, 1], [0, 0]]))
        self.assertEqual(codes.dtype, torch.long)

    def test_reconstruction_sums_stage_specific_code_vectors(self):
        rvq = ResidualVectorQuantizer(levels=2, codebook_size=2)
        rvq.codebooks = torch.tensor([[[0.0], [10.0]], [[0.0], [3.0]]])
        codes = torch.tensor([[1, 1], [0, 1]])
        torch.testing.assert_close(rvq.reconstruct(codes), torch.tensor([[13.0], [3.0]]))
        torch.testing.assert_close(rvq.reconstruct(codes, cumulative=True),
                                   torch.tensor([[[10.0], [13.0]], [[0.0], [3.0]]]))

    def test_fit_learns_true_residuals_and_does_not_change_caller_features(self):
        features = torch.tensor([[0.0], [2.0], [10.0], [12.0]], requires_grad=True)
        original = features.detach().clone()
        rvq = ResidualVectorQuantizer(levels=2, codebook_size=2, seed=5, iterations=20)
        self.assertIs(rvq.fit(features), rvq)
        torch.testing.assert_close(features.detach(), original)
        torch.testing.assert_close(rvq.codebooks[0, :, 0].sort().values, torch.tensor([1.0, 11.0]))
        torch.testing.assert_close(rvq.codebooks[1, :, 0].sort().values, torch.tensor([-1.0, 1.0]))
        torch.testing.assert_close(rvq.reconstruct(rvq.encode(features)), original)
        self.assertFalse(rvq.codebooks.requires_grad)
        self.assertIn("codebooks", dict(rvq.named_buffers()))

    def test_fit_is_seeded_and_empty_clusters_stay_finite(self):
        features = torch.tensor([[3.0, 4.0]]).repeat(4, 1)
        a = ResidualVectorQuantizer(levels=3, codebook_size=3, seed=7, iterations=5).fit(features)
        b = ResidualVectorQuantizer(levels=3, codebook_size=3, seed=7, iterations=5).fit(features)
        torch.testing.assert_close(a.codebooks, b.codebooks)
        self.assertTrue(torch.isfinite(a.codebooks).all())
        torch.testing.assert_close(a.reconstruct(a.encode(features)), features)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_fit_keeps_every_residual_and_codebook_on_input_device(self):
        # Inspect the actual k-means input, while running the unmodified algorithm.
        devices = []

        class ObservedRVQ(ResidualVectorQuantizer):
            def _kmeans(self, features, generator):
                devices.append(features.device)
                return super()._kmeans(features, generator)

        features = torch.tensor([[0.0], [2.0], [10.0], [12.0]], device="cuda:0")
        rvq = ObservedRVQ(levels=2, codebook_size=2, seed=5).fit(features)
        self.assertEqual(devices, [features.device, features.device])
        self.assertEqual(rvq.codebooks.device, features.device)
        self.assertEqual(rvq.encode(features).device, features.device)
        torch.testing.assert_close(rvq.reconstruct(rvq.encode(features)), features)

    def test_checkpoint_restores_fitted_codebooks_without_refit(self):
        rvq = ResidualVectorQuantizer(levels=2, codebook_size=2, seed=5)
        features = torch.tensor([[0.0], [2.0], [10.0], [12.0]])
        rvq.fit(features)
        restored = ResidualVectorQuantizer(levels=2, codebook_size=2)
        restored.load_state_dict(rvq.state_dict())
        torch.testing.assert_close(restored.encode(features), rvq.encode(features))

    def test_fit_rejects_insufficient_or_invalid_features(self):
        for features in (torch.randn(2, 3), torch.empty(0, 3),
                         torch.ones(3), torch.tensor([[1.0], [float("nan")], [2.0]])):
            with self.subTest(shape=tuple(features.shape)), self.assertRaises(ValueError):
                ResidualVectorQuantizer(levels=2, codebook_size=3).fit(features)

    def test_reconstruct_rejects_codes_outside_stage_vocabulary(self):
        rvq = ResidualVectorQuantizer(levels=2, codebook_size=2)
        rvq.codebooks = torch.zeros(2, 2, 1)
        for codes in (torch.tensor([[-1, 0]]), torch.tensor([[0, 2]]), torch.tensor([[0]])):
            with self.subTest(codes=codes), self.assertRaises(ValueError):
                rvq.reconstruct(codes)


if __name__ == "__main__":
    unittest.main()
