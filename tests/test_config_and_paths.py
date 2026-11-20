"""Unit tests for configuration validation, task aliases, and path resolution."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ki_studio.config import ModelMode, TrainingConfig, TrainingTask
from ki_studio.paths import StudioPaths

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestTrainingConfig(unittest.TestCase):
    def test_defaults_by_mode(self) -> None:
        digits_cfg = TrainingConfig.from_dict(ModelMode.DIGITS)
        smiley_cfg = TrainingConfig.from_dict(ModelMode.SMILEY)

        self.assertEqual(digits_cfg.conv_layers, 2)
        self.assertEqual(digits_cfg.epochs, 4)
        self.assertEqual(smiley_cfg.conv_layers, 3)
        self.assertEqual(smiley_cfg.epochs, 5)

    def test_clamping_and_fallbacks(self) -> None:
        cfg = TrainingConfig.from_dict(
            ModelMode.DIGITS,
            {
                "conv_layers": 99,
                "base_filters": 2,
                "dense_layers": 0,
                "hidden_units": 999,
                "activation": "invalid_act",
                "optimizer": "invalid_opt",
                "epochs": 100,
                "learning_rate": 5.0,
                "batch_size": 1,
                "dropout": 0.99,
                "data_fraction": 0.0001,
                "samples_per_class": 10,
            },
        )
        self.assertEqual(cfg.conv_layers, 3)
        self.assertEqual(cfg.base_filters, 8)
        self.assertEqual(cfg.dense_layers, 1)
        self.assertEqual(cfg.hidden_units, 256)
        self.assertEqual(cfg.activation, "relu")
        self.assertEqual(cfg.optimizer, "adam")
        self.assertEqual(cfg.epochs, 20)
        self.assertAlmostEqual(cfg.learning_rate, 0.2)
        self.assertEqual(cfg.batch_size, 8)
        self.assertAlmostEqual(cfg.dropout, 0.65)
        self.assertAlmostEqual(cfg.data_fraction, 0.02)
        self.assertEqual(cfg.samples_per_class, 100)
        self.assertIn("Zahlen-Training", cfg.preview(ModelMode.DIGITS))

    def test_task_resolution_and_aliases(self) -> None:
        self.assertIs(TrainingTask.resolve("train_digits"), TrainingTask.TRAIN_DIGITS)
        self.assertIs(TrainingTask.resolve("neurales_netz.py"), TrainingTask.TRAIN_DIGITS)
        self.assertIs(TrainingTask.resolve("train_smiley"), TrainingTask.TRAIN_SMILEY)
        self.assertIs(TrainingTask.resolve("smiley_netz.py"), TrainingTask.TRAIN_SMILEY)
        self.assertIs(TrainingTask.resolve("quick_test"), TrainingTask.QUICK_TEST)
        self.assertIs(TrainingTask.resolve("vorhersage_nutzen.py"), TrainingTask.QUICK_TEST)
        self.assertIsNone(TrainingTask.resolve("unknown_task"))


class TestStudioPaths(unittest.TestCase):
    def test_factory_fallback_and_custom_override_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            app_dir = Path(tmp_dir)
            paths = StudioPaths(
                bundle_dir=REPO_ROOT,
                app_dir=app_dir,
                web_dir=REPO_ROOT / "src" / "ki_studio" / "web",
            )

            model_path, meta_path, is_custom = paths.resolve_model_paths(ModelMode.DIGITS)
            self.assertFalse(is_custom)
            self.assertTrue(model_path.exists())
            self.assertEqual(model_path.name, "zahlen_netz.keras")
            self.assertTrue(meta_path.exists())

            info_before = paths.get_storage_info()
            self.assertFalse(info_before["exists"])
            self.assertFalse(info_before["folder_exists"])

            custom_model, custom_meta = paths.prepare_training_paths(ModelMode.DIGITS)
            custom_model.write_bytes(b"dummy-keras")
            custom_meta.write_text("{}", encoding="utf-8")

            resolved_model, _, is_custom_now = paths.resolve_model_paths(ModelMode.DIGITS)
            self.assertTrue(is_custom_now)
            self.assertEqual(resolved_model, custom_model)

            info_after = paths.get_storage_info()
            self.assertTrue(info_after["exists"])
            self.assertTrue(info_after["custom_zahlen"])
            self.assertFalse(info_after["custom_smiley"])

            ok, _ = paths.delete_custom_models()
            self.assertTrue(ok)
            self.assertFalse(paths.data_dir.exists())

            paths.prepare_training_paths(ModelMode.SMILEY)
            self.assertTrue(paths.data_dir.exists())
            ok_full, _ = paths.schedule_full_cleanup()
            self.assertTrue(ok_full)
            self.assertFalse(paths.data_dir.exists())


if __name__ == "__main__":
    unittest.main()
