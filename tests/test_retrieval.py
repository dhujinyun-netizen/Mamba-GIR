import unittest

import torch

from mambagir.retrieval import GalleryIndex, beam_search, retrieval_report


class ProbabilityModel:
    levels = 4
    codebook_size = 2

    def encode(self, images):
        return images[:, :1].reshape(-1, 1, 1, 1), torch.empty(0)

    def next_logits(self, prompts, memory, prefixes):
        result = []
        for query, seq in zip(prompts[:, 0, 0, 0], prefixes.tolist()):
            if not seq:
                p = [0.6, 0.4] if query == 0 else [0.2, 0.8]
            elif seq == [0]:
                p = [0.51, 0.49]
            elif seq == [1]:
                p = [0.99, 0.01]
            elif seq == [0, 0]:
                p = [0.1, 0.9]
            elif seq == [1, 0]:
                p = [0.9, 0.1]
            else:
                p = [0.7, 0.3]
            result.append(p)
        return torch.tensor(result).log()


class RetrievalTests(unittest.TestCase):
    def test_beam_parent_reordering_keeps_full_history(self):
        keys, scores = beam_search(ProbabilityModel(), torch.tensor([[0.], [1.]]), 2)
        self.assertEqual(keys[0].tolist(), [[1, 0, 0, 0], [0, 0, 1, 0]])
        self.assertEqual(keys[1].tolist(), [[1, 0, 0, 0], [1, 0, 0, 1]])
        self.assertAlmostEqual(scores[0, 0].exp().item(), .4 * .99 * .9 * .7, places=6)

    def test_invalid_keys_skipped_collisions_expanded_and_shortfalls_not_image_zero(self):
        gallery = GalleryIndex(torch.tensor([[0, 1], [0, 1], [1, 0]]))
        keys = torch.tensor([[[1, 1], [0, 1]], [[1, 1], [1, 1]]])
        ranks = gallery.resolve(keys, topk=4)
        self.assertEqual(ranks.tolist(), [[0, 1, -1, -1], [-1, -1, -1, -1]])

    def test_recall_is_query_hit_rate_and_validity_uses_raw_keys(self):
        gallery = GalleryIndex(torch.tensor([[0, 1], [0, 1], [1, 0]]))
        keys = torch.tensor([[[1, 1], [0, 1]], [[1, 0], [1, 1]]])
        ranks = gallery.resolve(keys, 4)
        report = retrieval_report(keys, ranks, gallery, [0, 1], [0, 7, 1], ks=[1, 2, 4])
        self.assertEqual(report['recall'], {'1': 1.0, '2': 1.0, '4': 1.0})
        self.assertEqual(report['top1_invalid_rate'], .5)
        self.assertEqual(report['beam_valid_coverage'], 1.)
        self.assertEqual(report['unique_top1_keys'], 2)
        self.assertEqual(report['reuse_rate'], 0.)

    def test_missing_results_do_not_match_last_gallery_label(self):
        gallery = GalleryIndex(torch.tensor([[0]]))
        keys = torch.tensor([[[1]]])
        ranks = gallery.resolve(keys, 2)
        report = retrieval_report(keys, ranks, gallery, [3], [3], ks=[1, 2])
        self.assertEqual(report['recall'], {'1': 0., '2': 0.})

    def test_raw_key_ranks_retain_invalid_slots_and_missing_keys_have_zero_reciprocal_rank(self):
        gallery = GalleryIndex(torch.tensor([[0], [1]]))
        keys = torch.tensor([[[9], [0]], [[1], [9]]])
        report = retrieval_report(keys, gallery.resolve(keys, 4), gallery,
                                  [0, 0], [0, 1], ks=[1, 2, 4])
        self.assertEqual(report['recall'], {'1': .5, '2': .5, '4': .5})
        self.assertIn('raw_key_hit', report)
        self.assertEqual(report['raw_key_hit'], {'1': 0., '2': .5})
        self.assertEqual(report['raw_key_mrr_at_beam'], .25)
        self.assertEqual(report['raw_first_relevant_key_rank_counts'], {'2': 1, 'not_found': 1})
        self.assertEqual(report['raw_top1_outcomes']['invalid_key']['count'], 1)
        self.assertEqual(report['raw_top1_outcomes']['valid_no_relevant_member']['count'], 1)

    def test_gallery_collision_statistics_separate_same_label_and_mixed_label_buckets(self):
        gallery = GalleryIndex(torch.tensor([[0], [0], [1], [1], [1], [2]]))
        keys = torch.tensor([[[0]]])
        report = retrieval_report(keys, gallery.resolve(keys, 1), gallery,
                                  [4], [4, 4, 4, 7, 7, 9], ks=[1])
        self.assertEqual(report['unique_gallery_keys'], 3)
        self.assertEqual(report['colliding_gallery_keys'], 2)
        self.assertIn('gallery_collision', report)
        stats = report['gallery_collision']
        self.assertEqual(stats['bucket_size_histogram'], {'1': 1, '2': 1, '3': 1})
        self.assertEqual(stats['items_in_colliding_buckets'], 5)
        self.assertAlmostEqual(stats['item_weighted_collision_fraction'], 5 / 6)
        self.assertEqual(stats['same_label_colliding_keys'], 1)
        self.assertEqual(stats['mixed_label_colliding_keys'], 1)
        self.assertEqual(stats['max_bucket_size'], 3)
        self.assertEqual(stats['mean_bucket_size'], 2.)

    def test_relevant_raw_key_can_miss_image_top1_due_to_stable_bucket_order(self):
        gallery = GalleryIndex(torch.tensor([[0], [0], [1]]))
        keys = torch.tensor([[[0], [1]]])
        report = retrieval_report(keys, gallery.resolve(keys, 2), gallery,
                                  [5], [8, 5, 5], ks=[1, 2])
        self.assertEqual(report['recall'], {'1': 0., '2': 1.})
        self.assertIn('raw_key_hit', report)
        self.assertEqual(report['raw_key_hit'], {'1': 1., '2': 1.})
        self.assertEqual(report['raw_key_mrr_at_beam'], 1.)
        self.assertEqual(report['raw_top1_outcomes']['mixed_label_bucket_first_member_miss']['count'], 1)
        oracle = report['within_bucket_oracle']
        self.assertEqual(oracle['recall'], {'1': 1., '2': 1.})
        self.assertEqual(oracle['recall_gap'], {'1': 1., '2': 0.})
        self.assertEqual(oracle['conditional_rescue_fraction_among_misses'], {'1': 1., '2': None})

    def test_tie_order_oracle_cannot_reorder_generated_keys_or_remove_duplicate_raw_slots(self):
        gallery = GalleryIndex(torch.tensor([[0], [0], [1], [1]]))
        keys = torch.tensor([[[0], [0], [1]]])
        report = retrieval_report(keys, gallery.resolve(keys, 4), gallery,
                                  [5], [8, 8, 8, 5], ks=[1, 2, 3, 4])
        self.assertEqual(report['recall'], {'1': 0., '2': 0., '3': 0., '4': 1.})
        self.assertIn('raw_key_hit', report)
        self.assertEqual(report['raw_key_hit'], {'1': 0., '2': 0., '3': 1.})
        self.assertAlmostEqual(report['raw_key_mrr_at_beam'], 1 / 3)
        oracle = report['within_bucket_oracle']
        self.assertEqual(oracle['recall'], {'1': 0., '2': 0., '3': 1., '4': 1.})
        self.assertEqual(oracle['recall_gap'], {'1': 0., '2': 0., '3': 1., '4': 0.})

    def test_raw_key_relevance_uses_any_bucket_member_without_changing_gallery_order(self):
        gallery = GalleryIndex(torch.tensor([[2], [1], [2], [1]]))
        keys = torch.tensor([[[2], [1]]])
        ranks = gallery.resolve(keys, 4)
        self.assertEqual(ranks.tolist(), [[0, 2, 1, 3]])
        report = retrieval_report(keys, ranks, gallery, [5], [5, 7, 8, 5], ks=[1, 4])
        self.assertIn('raw_top1_outcomes', report)
        self.assertEqual(report['raw_top1_outcomes']['first_member_hit']['count'], 1)
        self.assertEqual(report['within_bucket_oracle']['recall_gap'], {'1': 0., '4': 0.})


if __name__ == '__main__':
    unittest.main()
