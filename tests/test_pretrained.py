"""Opt-in real pretrained GPU integration; no training run or artifacts written."""

import os
import unittest

import torch


@unittest.skipUnless(os.environ.get('MAMBAGIR_ENCODER_CHECKPOINT') and
                     os.environ.get('MAMBAGIR_DECODER_PATH') and torch.cuda.is_available(),
                     'set both MAMBAGIR pretrained paths and use a CUDA environment')
class PretrainedTests(unittest.TestCase):
    def test_actual_backbones_cavp_gradient_and_generation(self):
        from mambagir.decoder import MambaDecoder
        from mambagir.model import MambaGIR
        from mambagir.retrieval import beam_search
        from mambagir.runtime import parameter_counts
        from mambagir.vision import VisualEncoder

        torch.manual_seed(3407)
        encoder = VisualEncoder(os.environ.get('MAMBAGIR_ENCODER_VARIANT', 'B'))
        encoder.load_pretrained(os.environ['MAMBAGIR_ENCODER_CHECKPOINT'])
        decoder = MambaDecoder(os.environ['MAMBAGIR_DECODER_PATH'])
        codebooks = torch.randn(4, 256, encoder.feature_dim)
        model = MambaGIR(encoder, decoder, codebooks, num_classes=200).cuda().train()
        images = torch.randn(1, 3, 224, 224, device='cuda')
        targets = torch.tensor([[0, 1, 2, 3]], device='cuda')
        with torch.cuda.amp.autocast():
            result = model(images, targets, torch.tensor([0], device='cuda'))
        self.assertTrue(torch.isfinite(result['loss']))
        self.assertEqual(tuple(result['logits'].shape), (1, 4, 256))
        result['loss'].backward()
        self.assertIsNone(model.codebooks.grad)
        self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))
        for target in ('x_proj', 'in_proj', 'out_proj', 'embeddings'):
            gradients = [p.grad for name, p in decoder.named_parameters() if target in name and 'lora_B' in name]
            self.assertTrue(gradients)
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.float().abs().sum().item() for g in gradients), 0)
        model.eval()
        with torch.cuda.amp.autocast():
            keys, scores = beam_search(model, images, 2)
        self.assertEqual(tuple(keys.shape), (1, 2, 4))
        self.assertTrue(torch.isfinite(scores).all())
        print('Actual full-model parameter count:', parameter_counts(model))


if __name__ == '__main__':
    unittest.main()
