"""Synthetic Dual-EBM tests; no dataset or downloaded backbone required."""
import unittest
import torch
from torch import nn
from dual_ebm.model import DualEnergyHead
from dual_ebm.losses import compute_dual_ebm_loss
from dual_ebm.svd import SVDResidualLinear
from dual_ebm.schedule import lr_multiplier


class TestCore(unittest.TestCase):
    def test_forward_and_backprop(self):
        model = DualEnergyHead(32, hidden_dim=16)
        logits, er, ef, nr, nf = model(torch.randn(4, 32), True)
        self.assertEqual(tuple(logits.shape), (4, 2))
        self.assertTrue(torch.allclose(logits[:, 0], -er[:, 0]))
        loss = compute_dual_ebm_loss(logits, torch.tensor([0, 1, 0, 1]),
                                     er, ef, nr, nf)["total"]
        loss.backward()
        self.assertIsNotNone(model.noise_std.grad)
        self.assertEqual(len(model(torch.randn(1, 32), False)), 3)

    def test_svd_reconstruction_and_gradient(self):
        layer = nn.Linear(12, 12)
        low = SVDResidualLinear(layer, 0.25)
        x = torch.randn(2, 12)
        self.assertTrue(torch.allclose(low(x), layer(x), rtol=1e-5, atol=1e-5))
        low(x).sum().backward()
        self.assertIsNotNone(low.S_residual.grad)

    def test_schedule(self):
        self.assertEqual(lr_multiplier(0, 10), 1.0)
        self.assertEqual(lr_multiplier(10, 10), 0.5)
        self.assertTrue(all(0 < lr_multiplier(i, 10) <= 1 for i in range(200)))


if __name__ == "__main__":
    unittest.main()
