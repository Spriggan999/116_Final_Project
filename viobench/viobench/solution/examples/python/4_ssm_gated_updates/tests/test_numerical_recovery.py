"""Regression checks for unsafe backward passes before Adam/EMA updates."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
import legacy
from gated_training import train_student
from performance import _allow_amp


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        self.register_buffer('target', torch.tensor(1.))

    def forward(self, **inputs):
        return self.weight.square()

    def update_target(self, decay):
        self.target.lerp_(self.weight.detach(), 1-decay)


def toy_objective(model, prediction, *args):
    return prediction, prediction, {'total_loss': prediction.detach()}


class NumericalRecoveryTests(unittest.TestCase):
    def setUp(self):
        model = ToyModel()
        self.row = dict(model=model, optimizer=torch.optim.AdamW(model.parameters(), lr=.01))

    def train(self):
        return train_student('innovation', self.row, ({}, {}), 'cpu')

    def test_normal_update_does_not_retry(self):
        with patch('gated_training.objective', side_effect=toy_objective) as objective:
            self.train()
        self.assertEqual(objective.call_count, 1)
        self.assertEqual(self.row['loss_terms']['optimizer_update_skipped'], 0)
        self.assertLess(self.row['model'].weight.item(), 1.)

    def test_bad_amp_backward_retries_fp32_before_one_update(self):
        modes = []
        def gradient(grad):
            modes.append(_allow_amp.get())
            return torch.full_like(grad, float('nan')) if modes[-1] else grad
        self.row['model'].weight.register_hook(gradient)
        with patch('gated_training.objective', side_effect=toy_objective):
            self.train()
        self.assertEqual(modes, [True, False])
        self.assertEqual(self.row['loss_terms']['numerical_retry'], 1)
        self.assertEqual(self.row['optimizer'].state[self.row['model'].weight]['step'].item(), 1)
        self.assertTrue(torch.isfinite(self.row['model'].target))

    def test_persistent_invalid_backward_preserves_optimizer_weights_and_ema(self):
        # Start with initialized Adam moments, so the test also checks their preservation.
        with patch('gated_training.objective', side_effect=toy_objective):
            self.train()
        model = self.row['model']
        before = copy.deepcopy(model.state_dict())
        moments = copy.deepcopy(self.row['optimizer'].state[model.weight])
        model.weight.register_hook(lambda grad: torch.full_like(grad, float('inf')))
        with patch('gated_training.objective', side_effect=toy_objective):
            loss = self.train()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(self.row['loss_terms']['optimizer_update_skipped'], 1)
        self.assertIsNone(model.weight.grad)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        for key, value in self.row['optimizer'].state[model.weight].items():
            torch.testing.assert_close(value, moments[key], rtol=0, atol=0)

    def test_nonfinite_forward_retry_and_persistent_failure_diagnostics(self):
        def objective(model, prediction, *args):
            value = prediction if not _allow_amp.get() else prediction * float('nan')
            return value, value, {'embedding_prediction_raw': value.detach()}
        with patch('gated_training.objective', side_effect=objective):
            self.train()
        self.assertEqual(self.row['loss_terms']['numerical_retry'], 1)
        before = self.row['model'].weight.detach().clone()
        def bad_objective(model, prediction, *args):
            value = prediction * float('nan')
            return value, value, {'embedding_prediction_raw': value.detach()}
        with patch('gated_training.objective', side_effect=bad_objective):
            with self.assertRaisesRegex(ValueError, 'after FP32 retry.*embedding_prediction_raw'):
                self.train()
        torch.testing.assert_close(self.row['model'].weight, before, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
