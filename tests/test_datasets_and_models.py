"""Unit tests for synthetic datasets, ASCII rendering, and dynamic CNN/MLP builders."""

from __future__ import annotations

import unittest

import numpy as np

from ki_studio.config import ModelMode, TrainingConfig
from ki_studio.core.ascii_art import render_ascii
from ki_studio.core.datasets import compute_happiness_percent, generate_smiley_sample
from ki_studio.core.training import (
    build_classifier_model,
    build_optimizer,
    compute_conv_filters,
)


class TestDatasetsAndModels(unittest.TestCase):
    def test_happiness_anchors(self) -> None:
        expected = [0.0, 25.0, 50.0, 75.0, 100.0]
        for idx, score in enumerate(expected):
            probs = np.zeros((5,), dtype=np.float32)
            probs[idx] = 1.0
            self.assertAlmostEqual(compute_happiness_percent(probs), score, places=4)

    def test_generate_smiley_sample_all_classes(self) -> None:
        rng = np.random.default_rng(123)
        for mood_class in range(5):
            sample = generate_smiley_sample(mood_class, rng)
            self.assertEqual(sample.shape, (48, 48))
            self.assertGreaterEqual(float(sample.min()), 0.0)
            self.assertLessEqual(float(sample.max()), 1.0)
            self.assertGreater(float(sample.sum()), 1.0)

        with self.assertRaises(ValueError):
            generate_smiley_sample(5, rng)

    def test_render_ascii_direct_and_downscaled(self) -> None:
        img = np.zeros((48, 48), dtype=np.float32)
        img[12:36, 18:30] = 1.0
        art = render_ascii(img, max_size=24)
        self.assertTrue(len(art) > 0)
        self.assertLessEqual(len(art.splitlines()), 24)

    def test_conv_filter_schedule_and_optimizers(self) -> None:
        self.assertEqual(compute_conv_filters(0, 32), [])
        self.assertEqual(compute_conv_filters(1, 32), [32])
        self.assertEqual(compute_conv_filters(3, 32), [32, 64, 64])
        self.assertEqual(compute_conv_filters(2, 64), [64, 128])

        for name in ("adam", "rmsprop", "sgd"):
            opt = build_optimizer(name, 0.002)
            self.assertIsNotNone(opt)

    def test_dynamic_model_builder_cnn_and_mlp(self) -> None:
        logs: list[str] = []
        cnn_cfg = TrainingConfig.from_dict(
            ModelMode.DIGITS, {"conv_layers": 2, "dense_layers": 1, "hidden_units": 32}
        )
        cnn = build_classifier_model(ModelMode.DIGITS, cnn_cfg, log_fn=lambda t, _: logs.append(t))
        self.assertEqual(tuple(cnn.input_shape), (None, 28, 28))
        self.assertEqual(tuple(cnn.output_shape), (None, 10))

        mlp_cfg = TrainingConfig.from_dict(
            ModelMode.SMILEY, {"conv_layers": 0, "dense_layers": 2, "hidden_units": 16}
        )
        mlp = build_classifier_model(ModelMode.SMILEY, mlp_cfg, log_fn=lambda t, _: logs.append(t))
        self.assertEqual(tuple(mlp.input_shape), (None, 48, 48))
        self.assertEqual(tuple(mlp.output_shape), (None, 5))


if __name__ == "__main__":
    unittest.main()
