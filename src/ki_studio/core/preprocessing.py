"""Shared image decoding, normalization, centering, and PNG encoding utilities."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import tensorflow as tf

DIGIT_CANVAS_SIZE = 28
DIGIT_INNER_BOX = 20

SMILEY_CANVAS_SIZE = 48
SMILEY_INNER_BOX = 40


def load_image_file(image_path: str | Path) -> np.ndarray:
    """Decode an image file (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`) using TensorFlow I/O."""
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Die Bilddatei wurde nicht gefunden:\n  {path}")
    if not path.is_file():
        raise ValueError(f"Der angegebene Pfad ist keine Datei:\n  {path}")

    try:
        raw_bytes = path.read_bytes()
        return decode_image_bytes(raw_bytes)
    except Exception as exc:
        raise ValueError(
            f"Die Datei '{path.name}' konnte nicht als Bild gelesen werden.\n"
            f"Unterstützte Formate: PNG, JPG/JPEG, BMP, GIF.\n(Technischer Grund: {exc})"
        ) from exc


def decode_image_bytes(raw_bytes: bytes) -> np.ndarray:
    """Decode encoded image bytes into an RGB/RGBA/grayscale NumPy array."""
    tensor = tf.io.decode_image(raw_bytes, expand_animations=False)
    return tensor.numpy()


def encode_grayscale_png_b64(gray_2d: np.ndarray, target_size: int = 112) -> str:
    """Nearest-neighbor upscale a normalized 2D array (`0..1`) and return a Base64 PNG string."""
    resized = tf.image.resize(
        np.clip(gray_2d, 0.0, 1.0)[:, :, np.newaxis],
        size=[target_size, target_size],
        method="nearest",
    ).numpy()
    uint8_gray = (resized * 255.0).astype(np.uint8)
    png_bytes = tf.io.encode_png(uint8_gray).numpy()
    return base64.b64encode(png_bytes).decode("ascii")


def _to_grayscale(raw_pixels: np.ndarray) -> np.ndarray:
    """Convert 2D, grayscale, RGB, or RGBA pixel arrays into normalized `float32` grayscale."""
    raw_arr = np.asarray(raw_pixels)
    arr = np.asarray(raw_arr, dtype=np.float32)
    if np.issubdtype(raw_arr.dtype, np.integer) or float(np.max(arr, initial=0.0)) > 1.0:
        arr = arr / 255.0

    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    if arr.ndim == 3 and arr.shape[-1] == 4:
        alpha = arr[..., 3:4]
        rgb = arr[..., :3] * alpha + (1.0 - alpha)
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    raise ValueError(f"Ungültige Bildform: {arr.shape}")


def _extract_and_resize_foreground(
    raw_pixels: np.ndarray,
    *,
    canvas_size: int,
    inner_box: int,
    min_side_for_dilation: int,
    max_density_for_dilation: float,
    dilation_divisor: float,
) -> tuple[np.ndarray, bool]:
    """Invert light backgrounds, crop tight foreground bounds, and fit into `canvas_size`."""
    gray = _to_grayscale(raw_pixels)
    if gray.ndim != 2 or gray.shape[0] < 2 or gray.shape[1] < 2:
        return np.zeros((canvas_size, canvas_size), dtype=np.float32), True

    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    if float(np.median(border)) > 0.5:
        gray = 1.0 - gray

    border_inv = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    floor = float(np.percentile(border_inv, 85))
    if float(np.max(gray)) - floor < 0.05:
        floor = float(np.percentile(gray, 25))

    gray = np.clip(gray - floor, 0.0, None)
    peak = float(np.max(gray))
    if peak < 0.05:
        return np.zeros((canvas_size, canvas_size), dtype=np.float32), True

    gray = gray / peak
    gray = np.where(gray < 0.12, 0.0, gray)

    ys, xs = np.where(gray > 0.25)
    if len(ys) == 0:
        ys, xs = np.where(gray > 0.12)
    if len(ys) == 0:
        return np.zeros((canvas_size, canvas_size), dtype=np.float32), True

    y0 = max(0, int(ys.min()) - 1)
    y1 = min(gray.shape[0], int(ys.max()) + 2)
    x0 = max(0, int(xs.min()) - 1)
    x1 = min(gray.shape[1], int(xs.max()) + 2)
    crop = gray[y0:y1, x0:x1]

    height, width = crop.shape
    max_side = max(height, width)

    density = float(np.sum(crop > 0.2)) / float(max_side * max_side)
    if max_side > min_side_for_dilation and density < max_density_for_dilation:
        kernel = max(3, int(round(max_side / dilation_divisor)))
        tensor_4d = crop[np.newaxis, :, :, np.newaxis]
        crop = tf.nn.max_pool2d(tensor_4d, ksize=kernel, strides=1, padding="SAME").numpy()[0, :, :, 0]

    new_h = max(1, int(round(float(inner_box) * height / max_side)))
    new_w = max(1, int(round(float(inner_box) * width / max_side)))
    method = "area" if max_side > inner_box else "bilinear"

    resized = tf.image.resize(crop[:, :, np.newaxis], size=[new_h, new_w], method=method).numpy()[:, :, 0]
    max_resized = float(np.max(resized))
    if max_resized > 0.0:
        resized = np.clip(resized / max_resized, 0.0, 1.0)

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    start_y = (canvas_size - new_h) // 2
    start_x = (canvas_size - new_w) // 2
    canvas[start_y : start_y + new_h, start_x : start_x + new_w] = resized
    return canvas, False


def _center_by_mass(canvas: np.ndarray) -> np.ndarray:
    """Shift foreground pixels inside a 28x28 canvas so their Center of Mass sits at (13.5, 13.5)."""
    size = canvas.shape[0]
    total = float(np.sum(canvas))
    if total <= 0.0:
        return canvas

    grid_y, grid_x = np.indices((size, size), dtype=np.float32)
    com_y = float(np.sum(grid_y * canvas)) / total
    com_x = float(np.sum(grid_x * canvas)) / total

    active_y, active_x = np.where(canvas > 0.05)
    if len(active_y) == 0:
        return canvas

    target_center = (size - 1) / 2.0
    shift_y = int(round(target_center - com_y))
    shift_x = int(round(target_center - com_x))

    shift_y = max(-int(active_y.min()), min(size - 1 - int(active_y.max()), shift_y))
    shift_x = max(-int(active_x.min()), min(size - 1 - int(active_x.max()), shift_x))

    if shift_y == 0 and shift_x == 0:
        return canvas

    centered = np.zeros_like(canvas)
    src_y0 = max(0, -shift_y)
    src_y1 = min(size, size - shift_y)
    src_x0 = max(0, -shift_x)
    src_x1 = min(size, size - shift_x)

    dst_y0 = max(0, shift_y)
    dst_y1 = min(size, size + shift_y)
    dst_x0 = max(0, shift_x)
    dst_x1 = min(size, size + shift_x)

    centered[dst_y0:dst_y1, dst_x0:dst_x1] = canvas[src_y0:src_y1, src_x0:src_x1]
    return centered


def prepare_digit_image(raw_pixels: np.ndarray) -> tuple[np.ndarray, bool]:
    """Preprocess an arbitrary digit image into a `(28, 28)` MNIST-compatible tensor and emptiness flag."""
    canvas, is_empty = _extract_and_resize_foreground(
        raw_pixels,
        canvas_size=DIGIT_CANVAS_SIZE,
        inner_box=DIGIT_INNER_BOX,
        min_side_for_dilation=40,
        max_density_for_dilation=0.06,
        dilation_divisor=18.0,
    )
    if is_empty:
        return canvas, True
    return _center_by_mass(canvas), False


def prepare_smiley_image(raw_pixels: np.ndarray) -> tuple[np.ndarray, bool]:
    """Preprocess an arbitrary smiley image into a geometrically centered `(48, 48)` tensor."""
    return _extract_and_resize_foreground(
        raw_pixels,
        canvas_size=SMILEY_CANVAS_SIZE,
        inner_box=SMILEY_INNER_BOX,
        min_side_for_dilation=72,
        max_density_for_dilation=0.045,
        dilation_divisor=28.0,
    )
