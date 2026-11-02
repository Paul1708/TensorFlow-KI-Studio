"""Unit tests for image decoding, background normalization, and centering."""

from __future__ import annotations

import base64
from pathlib import Path
import unittest

import numpy as np

from ki_studio.core.preprocessing import (
    decode_image_bytes,
    encode_grayscale_png_b64,
    load_image_file,
    prepare_digit_image,
    prepare_smiley_image,
)

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"


class TestPreprocessing(unittest.TestCase):
    def test_empty_images_detected(self) -> None:
        white = np.full((64, 64, 3), 255, dtype=np.uint8)
        black = np.zeros((64, 64), dtype=np.float32)

        digit_w, empty_w = prepare_digit_image(white)
        digit_b, empty_b = prepare_digit_image(black)
        smiley_w, empty_sw = prepare_smiley_image(white)

        self.assertTrue(empty_w)
        self.assertTrue(empty_b)
        self.assertTrue(empty_sw)
        self.assertEqual(digit_w.shape, (28, 28))
        self.assertEqual(smiley_w.shape, (48, 48))

    def test_digit_white_background_inversion_and_center_of_mass(self) -> None:
        canvas = np.full((120, 120, 4), 255, dtype=np.uint8)
        # Draw an off-center vertical stroke near top-left
        canvas[12:55, 22:30, :3] = 0

        prepared, is_empty = prepare_digit_image(canvas)
        self.assertFalse(is_empty)
        self.assertEqual(prepared.shape, (28, 28))
        self.assertEqual(prepared.dtype, np.float32)
        self.assertGreaterEqual(float(prepared.min()), 0.0)
        self.assertLessEqual(float(prepared.max()), 1.0)

        total = float(prepared.sum())
        gy, gx = np.indices((28, 28), dtype=np.float32)
        com_y = float((gy * prepared).sum()) / total
        com_x = float((gx * prepared).sum()) / total
        self.assertAlmostEqual(com_y, 13.5, delta=1.5)
        self.assertAlmostEqual(com_x, 13.5, delta=1.5)

    def test_invalid_shape_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            prepare_digit_image(np.zeros((10,), dtype=np.float32))

    def test_png_roundtrip_and_sample_loading(self) -> None:
        synthetic = np.zeros((28, 28), dtype=np.float32)
        synthetic[6:22, 12:16] = 1.0
        b64 = encode_grayscale_png_b64(synthetic, target_size=56)
        decoded = decode_image_bytes(base64.b64decode(b64))
        self.assertEqual(decoded.shape[:2], (56, 56))

        sample_digit = load_image_file(SAMPLES_DIR / "beispiel_zahl_8.png")
        d28, d_empty = prepare_digit_image(sample_digit)
        self.assertFalse(d_empty)
        self.assertEqual(d28.shape, (28, 28))

        sample_smiley = load_image_file(SAMPLES_DIR / "smiley_gluecklich.png")
        s48, s_empty = prepare_smiley_image(sample_smiley)
        self.assertFalse(s_empty)
        self.assertEqual(s48.shape, (48, 48))


if __name__ == "__main__":
    unittest.main()
