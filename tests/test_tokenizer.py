import unittest

import torch
from torch import nn

from mambagir.tokenizer import SemanticTokenizer


class TinyEncoder(nn.Module):
    feature_dim = 4

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 4)

    def forward_global(self, images):
        return self.projection(images.mean((-2, -1)))


class TokenizerTests(unittest.TestCase):
    def test_semantic_and_partial_losses_train_encoder_without_training_codes(self):
        torch.manual_seed(4)
        tokenizer = SemanticTokenizer(TinyEncoder(), 3, depth=12, hidden_dim=8)
        images = torch.randn(2, 3, 4, 4)
        labels = torch.tensor([0, 2])
        reconstructed = torch.randn(2, 4, 4, requires_grad=True)
        out = tokenizer(images, labels, reconstructed, rq_weight=0.2)
        self.assertTrue(torch.isfinite(out['loss']))
        self.assertTrue(torch.allclose(out['loss'], out['loss_ce'] + .2 * (out['loss_ce_rq'] + out['loss_quant'])))
        out['loss'].backward()
        self.assertGreater(tokenizer.encoder.projection.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(reconstructed.grad)

    def test_partial_penalty_is_squared_l2_summed_across_levels(self):
        encoder = TinyEncoder()
        nn.init.zeros_(encoder.projection.weight)
        nn.init.zeros_(encoder.projection.bias)
        tokenizer = SemanticTokenizer(encoder, 2, depth=2, hidden_dim=4)
        out = tokenizer(torch.zeros(1, 3, 2, 2), torch.tensor([0]), torch.tensor([[[1., 1., 1., 1.], [2., 2., 2., 2.]]]))
        self.assertEqual(out['loss_quant'].item(), 20.0)


if __name__ == '__main__':
    unittest.main()
