"""Dataset loaders and synthetic hand-drawn smiley generator."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable

import numpy as np
import tensorflow as tf

from ki_studio.config import DEFAULT_SEED
from ki_studio.paths import DEFAULT_PATHS, StudioPaths

DIGIT_IMAGE_SIZE = 28
SMILEY_IMAGE_SIZE = 48

MOOD_NAMES: tuple[str, ...] = (
    "Sehr traurig",
    "Etwas traurig",
    "Neutral",
    "Glücklich",
    "Überglücklich",
)

MOOD_SYMBOLS: tuple[str, ...] = (
    ":'(",
    ":-(",
    ":-|",
    ":-)",
    ":-D",
)

HAPPINESS_ANCHORS = np.array([0.0, 25.0, 50.0, 75.0, 100.0], dtype=np.float32)

LogFn = Callable[[str, bool], None]


def _noop_log(_text: str, _overwrite: bool = False) -> None:
    return


def compute_happiness_percent(probabilities: np.ndarray) -> float:
    """Compute a continuous `0..100` happiness score from the 5 mood probabilities."""
    return float(np.sum(np.asarray(probabilities, dtype=np.float32) * HAPPINESS_ANCHORS))


def load_mnist_offline(
    paths: StudioPaths = DEFAULT_PATHS,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Load MNIST from `assets/mnist.npz`, falling back to Keras datasets if absent."""
    for candidate in (paths.assets_dir / "mnist.npz", paths.app_dir / "assets" / "mnist.npz"):
        if candidate.exists():
            with np.load(candidate) as data:
                return (data["x_train"], data["y_train"]), (data["x_test"], data["y_test"])
    return tf.keras.datasets.mnist.load_data()


def load_digits_dataset(
    data_fraction: float = 1.0,
    paths: StudioPaths = DEFAULT_PATHS,
    log_fn: LogFn = _noop_log,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Load and normalize MNIST images, optionally subsampling a deterministic fraction."""
    log_fn(
        "1. Lade integrierte Trainings- und Testbilder (MNIST-Datensatz, 100 % offline)...",
        False,
    )
    (x_train, y_train), (x_test, y_test) = load_mnist_offline(paths)

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    fraction = float(np.clip(data_fraction, 0.02, 1.0))
    if fraction < 0.999:
        count = max(500, int(round(x_train.shape[0] * fraction)))
        rng = np.random.default_rng(DEFAULT_SEED)
        selection = rng.choice(x_train.shape[0], size=count, replace=False)
        x_train = x_train[selection]
        y_train = y_train[selection]

    log_fn(
        f"   -> Trainingsbilder bereit:  {x_train.shape[0]} Bilder "
        f"({x_train.shape[1]}x{x_train.shape[2]} Pixel, {fraction * 100:.0f} % Datensatz)",
        False,
    )
    log_fn(f"   -> Testbilder geladen:      {x_test.shape[0]} Bilder (zum Prüfen am Ende)\n", False)
    return (x_train, y_train), (x_test, y_test)


def _paint_points(canvas: np.ndarray, xs: np.ndarray, ys: np.ndarray, radius: float) -> None:
    """Stamp anti-aliased circular brush strokes along `(xs, ys)` onto a 2D canvas."""
    height, width = canvas.shape
    r_ceil = int(math.ceil(radius + 0.6))
    ix_arr = np.round(xs).astype(np.int32)
    iy_arr = np.round(ys).astype(np.int32)

    dy, dx = np.ogrid[-r_ceil : r_ceil + 1, -r_ceil : r_ceil + 1]
    distance = np.sqrt(dx * dx + dy * dy).astype(np.float32)
    stamp = np.clip(radius + 0.5 - distance, 0.0, 1.0)

    for ix, iy in zip(ix_arr, iy_arr, strict=True):
        y0 = max(0, iy - r_ceil)
        y1 = min(height, iy + r_ceil + 1)
        x0 = max(0, ix - r_ceil)
        x1 = min(width, ix + r_ceil + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        sy0 = y0 - (iy - r_ceil)
        sy1 = sy0 + (y1 - y0)
        sx0 = x0 - (ix - r_ceil)
        sx1 = sx0 + (x1 - x0)
        target = canvas[y0:y1, x0:x1]
        np.maximum(target, stamp[sy0:sy1, sx0:sx1], out=target)


def generate_smiley_sample(mood_class: int, rng: np.random.Generator) -> np.ndarray:
    """Synthesize a single hand-drawn style `(48, 48)` smiley normalized to `[0.0, 1.0]`."""
    if not 0 <= mood_class <= 4:
        raise ValueError(f"mood_class must be in 0..4, got {mood_class}")

    size = 64
    canvas = np.zeros((size, size), dtype=np.float32)
    center_x = 32.0 + float(rng.uniform(-1.5, 1.5))
    center_y = 32.0 + float(rng.uniform(-1.5, 1.5))
    brush = float(rng.uniform(0.95, 2.15))

    angle = float(rng.uniform(-0.16, 0.16))
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    def rotate(px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx = px - center_x
        dy = py - center_y
        return center_x + dx * cos_a - dy * sin_a, center_y + dx * sin_a + dy * cos_a

    rx = float(rng.uniform(21.0, 26.5))
    ry = float(rng.uniform(21.0, 26.5))

    if rng.random() < 0.72:
        start_theta = float(rng.uniform(0.0, 6.28))
        span = float(rng.uniform(5.95, 6.45))
        theta = np.linspace(start_theta, start_theta + span, 90, dtype=np.float32)
        wobble = rng.uniform(0.0, 1.1) * np.sin(2.0 * theta + rng.uniform(0, 6.28))
        kx, ky = rotate(
            center_x + (rx + wobble) * np.cos(theta),
            center_y + (ry + wobble) * np.sin(theta),
        )
        _paint_points(canvas, kx, ky, brush)

    eye_offset_x = float(rng.uniform(0.30, 0.50)) * rx
    eye_base_y = center_y - float(rng.uniform(0.18, 0.42)) * ry
    eye_style = rng.random()

    for side in (-1.0, 1.0):
        ax = center_x + side * eye_offset_x + float(rng.uniform(-1.0, 1.0))
        ay = eye_base_y + float(rng.uniform(-1.2, 1.2))
        if eye_style < 0.48:
            ex, ey = rotate(np.array([ax]), np.array([ay]))
            _paint_points(canvas, ex, ey, brush * float(rng.uniform(1.0, 1.8)))
        elif eye_style < 0.78:
            length = float(rng.uniform(2.2, 4.8))
            ys_eye = np.linspace(ay - length, ay + length, 9, dtype=np.float32)
            xs_eye = np.full_like(ys_eye, ax)
            ex, ey = rotate(xs_eye, ys_eye)
            _paint_points(canvas, ex, ey, brush * 0.95)
        elif eye_style < 0.92:
            eye_r = float(rng.uniform(2.2, 3.6))
            eye_theta = np.linspace(0, 6.3, 18, dtype=np.float32)
            ex, ey = rotate(ax + eye_r * np.cos(eye_theta), ay + eye_r * np.sin(eye_theta))
            _paint_points(canvas, ex, ey, max(0.8, brush * 0.75))
        else:
            t_eye = np.linspace(-1.0, 1.0, 11, dtype=np.float32)
            eye_w = float(rng.uniform(2.2, 3.6))
            ex, ey = rotate(ax + t_eye * eye_w, ay - 2.0 * (1.0 - t_eye * t_eye))
            _paint_points(canvas, ex, ey, brush * 0.85)

    if rng.random() < 0.18:
        nx, ny = rotate(
            np.array([center_x + rng.uniform(-1.0, 1.0)]),
            np.array([center_y + rng.uniform(-2.0, 1.5)]),
        )
        _paint_points(canvas, nx, ny, brush * 0.9)

    mouth_half_w = float(rng.uniform(0.36, 0.66)) * rx
    mouth_base_y = center_y + float(rng.uniform(0.20, 0.44)) * ry
    t = np.linspace(-1.0, 1.0, 40, dtype=np.float32)

    open_laugh = False
    if mood_class == 0:
        bend = -float(rng.uniform(0.55, 0.95)) * mouth_half_w
        mouth_corners_y = mouth_base_y - bend * 0.55
    elif mood_class == 1:
        bend = -float(rng.uniform(0.20, 0.42)) * mouth_half_w
        mouth_corners_y = mouth_base_y - bend * 0.45
    elif mood_class == 2:
        bend = float(rng.uniform(-0.05, 0.05)) * mouth_half_w
        mouth_corners_y = mouth_base_y
    elif mood_class == 3:
        bend = float(rng.uniform(0.22, 0.48)) * mouth_half_w
        mouth_corners_y = mouth_base_y - bend * 0.35
    else:
        mouth_half_w = float(rng.uniform(0.48, 0.72)) * rx
        bend = float(rng.uniform(0.62, 1.05)) * mouth_half_w
        mouth_corners_y = mouth_base_y - bend * 0.40
        open_laugh = rng.random() < 0.48

    mx = center_x + t * mouth_half_w + float(rng.uniform(-1.0, 1.0))
    my = mouth_corners_y + bend * (1.0 - t * t)

    mx_rot, my_rot = rotate(mx, my)
    _paint_points(canvas, mx_rot, my_rot, brush * float(rng.uniform(0.95, 1.2)))

    if open_laugh:
        upper_my = mouth_corners_y + float(rng.uniform(-0.06, 0.08)) * mouth_half_w * (1.0 - t * t)
        ox_rot, oy_rot = rotate(mx, upper_my)
        _paint_points(canvas, ox_rot, oy_rot, brush * 0.95)
        if rng.random() < 0.35:
            for fill_factor in (0.3, 0.6, 0.85):
                fy = upper_my + fill_factor * (my - upper_my)
                fx_rot, fy_rot = rotate(mx, fy)
                _paint_points(canvas, fx_rot, fy_rot, brush * 1.1)

    ys, xs = np.where(canvas > 0.25)
    y0 = max(0, int(ys.min()) - 1)
    y1 = min(size, int(ys.max()) + 2)
    x0 = max(0, int(xs.min()) - 1)
    x1 = min(size, int(xs.max()) + 2)
    crop = canvas[y0:y1, x0:x1]
    h, w = crop.shape
    max_s = max(h, w)
    nh = max(1, int(round(40.0 * h / max_s)))
    nb = max(1, int(round(40.0 * w / max_s)))
    resized = tf.image.resize(crop[:, :, np.newaxis], [nh, nb], method="area").numpy()[:, :, 0]
    m_val = float(np.max(resized))
    if m_val > 0.0:
        resized = resized / m_val

    out = np.zeros((SMILEY_IMAGE_SIZE, SMILEY_IMAGE_SIZE), dtype=np.float32)
    sy = (SMILEY_IMAGE_SIZE - nh) // 2
    sx = (SMILEY_IMAGE_SIZE - nb) // 2
    out[sy : sy + nh, sx : sx + nb] = resized
    return out


def generate_smiley_dataset(
    samples_per_class_train: int = 2400,
    samples_per_class_test: int = 400,
    log_fn: LogFn = _noop_log,
    stop_event: threading.Event | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Generate balanced synthetic training and evaluation datasets across all 5 mood levels."""
    total_train = samples_per_class_train * 5
    total_test = samples_per_class_test * 5
    log_fn(f"1. Erzeuge {total_train} Trainings-Smileys und {total_test} Test-Smileys...", False)

    rng = np.random.default_rng(DEFAULT_SEED)

    x_train = np.zeros((total_train, SMILEY_IMAGE_SIZE, SMILEY_IMAGE_SIZE), dtype=np.float32)
    y_train = np.zeros((total_train,), dtype=np.int32)
    idx = 0
    last_log = 0.0
    for mood_class in range(5):
        for _ in range(samples_per_class_train):
            if stop_event is not None and (idx % 100 == 0) and stop_event.is_set():
                return (x_train[:1], y_train[:1]), (x_train[:1], y_train[:1])
            x_train[idx] = generate_smiley_sample(mood_class, rng)
            y_train[idx] = mood_class
            idx += 1
            now = time.monotonic()
            if now - last_log >= 0.25:
                last_log = now
                pct = int(round((idx / max(1, total_train)) * 100))
                log_fn(f"   -> Zeichne Trainings-Smileys: {idx}/{total_train} ({pct} %)...", True)

    order = rng.permutation(total_train)
    x_train = x_train[order]
    y_train = y_train[order]

    x_test = np.zeros((total_test, SMILEY_IMAGE_SIZE, SMILEY_IMAGE_SIZE), dtype=np.float32)
    y_test = np.zeros((total_test,), dtype=np.int32)
    idx = 0
    for mood_class in range(5):
        for _ in range(samples_per_class_test):
            if stop_event is not None and (idx % 100 == 0) and stop_event.is_set():
                return (x_train[:1], y_train[:1]), (x_test[:1], y_test[:1])
            x_test[idx] = generate_smiley_sample(mood_class, rng)
            y_test[idx] = mood_class
            idx += 1

    log_fn(
        f"   -> Fertig! Bildgröße: {SMILEY_IMAGE_SIZE}x{SMILEY_IMAGE_SIZE} Pixel, "
        "5 Glücks-Stufen.",
        True,
    )
    log_fn("", False)
    return (x_train, y_train), (x_test, y_test)
