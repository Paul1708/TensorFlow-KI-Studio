"""ASCII terminal renderer for normalized grayscale images."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

_ASCII_PALETTE = " .:-=+*#%@"


def render_ascii(image_2d: np.ndarray, max_size: int | None = None) -> str:
    """
    Render a normalized 2D grayscale image (`0.0..1.0`) as double-width ASCII art.

    When `max_size` is provided and smaller than the image dimensions, the image is
    area-downsampled first so large inputs stay readable in a terminal.
    """
    img = np.asarray(image_2d, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale array, got shape {img.shape}")

    if max_size is not None and (img.shape[0] > max_size or img.shape[1] > max_size):
        img = (
            tf.image.resize(img[:, :, np.newaxis], [max_size, max_size], method="area")
            .numpy()[:, :, 0]
        )
        peak = float(np.max(img, initial=0.0))
        if peak > 0.0:
            img = img / peak

    max_idx = len(_ASCII_PALETTE) - 1
    rows: list[str] = []
    for row in img:
        if float(np.max(row, initial=0.0)) > 0.05:
            chars = "".join(
                _ASCII_PALETTE[min(max(0, int(pixel * max_idx)), max_idx)] * 2
                for pixel in row
            )
            rows.append("   " + chars)
    return "\n".join(rows)
