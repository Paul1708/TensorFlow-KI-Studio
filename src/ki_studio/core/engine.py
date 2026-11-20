"""Real-time layer-by-layer neural network introspection and background training engine."""

from __future__ import annotations

import json
import logging
import random
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from ki_studio.config import ModelMode, TrainingConfig, TrainingTask
from ki_studio.core.datasets import (
    MOOD_NAMES,
    MOOD_SYMBOLS,
    compute_happiness_percent,
    generate_smiley_sample,
    load_mnist_offline,
)
from ki_studio.core.preprocessing import (
    decode_image_bytes,
    encode_grayscale_png_b64,
    load_image_file,
    prepare_digit_image,
    prepare_smiley_image,
)
from ki_studio.core.training import (
    run_quick_digit_test,
    train_digits_model,
    train_smiley_model,
)
from ki_studio.paths import DEFAULT_PATHS, StudioPaths

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, str], None]


def _normalize_magnitude_01(vector: np.ndarray) -> np.ndarray:
    """Normalize activation magnitudes onto `0.0..1.0` across ReLU, ELU, Tanh, and Sigmoid."""
    magnitude = np.abs(vector)
    peak = float(np.max(magnitude, initial=0.0))
    if peak <= 1e-8:
        return np.zeros_like(magnitude, dtype=np.float32)
    return np.clip(magnitude / peak, 0.0, 1.0)


def _compute_conv_links(model: tf.keras.Model) -> dict[str, list[list[list[float]]]]:
    """Precompute top-k filter-to-filter coupling weights between consecutive Conv2D layers."""
    conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    links: dict[str, list[list[list[float]]]] = {}
    for idx in range(1, len(conv_layers)):
        kernel = conv_layers[idx].get_weights()[0]  # (KH, KW, C_in, C_out)
        strength = np.mean(np.abs(kernel), axis=(0, 1))
        max_w = max(float(np.max(strength)), 1e-6)
        top_k = min(3, strength.shape[0])
        per_target: list[list[list[float]]] = []
        for c_out in range(strength.shape[1]):
            col = strength[:, c_out]
            top_in = np.argsort(col)[-top_k:][::-1]
            per_target.append([[int(ci), round(float(col[ci] / max_w), 3)] for ci in top_in])
        links[f"conv{idx}_to_conv{idx + 1}"] = per_target
    return links


class BrainEngine:
    """Loads digit and smiley Keras models and computes live layer-by-layer synapse flows."""

    def __init__(
        self,
        paths: StudioPaths = DEFAULT_PATHS,
        progress_cb: ProgressCallback | None = None,
    ) -> None:
        self.paths = paths
        self.lock = threading.Lock()
        self.digits_model: tf.keras.Model | None = None
        self.smiley_model: tf.keras.Model | None = None
        self.digits_meta: dict[str, Any] | None = None
        self.smiley_meta: dict[str, Any] | None = None
        self._mnist_test: tuple[np.ndarray, np.ndarray] | None = None

        self._digits_static_links: dict[str, list[list[list[float]]]] = {}
        self._smiley_static_links: dict[str, list[list[list[float]]]] = {}

        self.train_lock = threading.Lock()
        self.train_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.train_active = False
        self.train_script = ""
        self.train_lines: list[str] = [
            "Bereit! Passe links deine gewünschten Architektur- und Lernparameter an "
            "und klicke auf 'Training starten'."
        ]
        self.train_metrics: list[dict[str, Any]] = []

        self.load_models(progress_cb=progress_cb)

    def load_models(
        self,
        only_mode: ModelMode | str | None = None,
        progress_cb: ProgressCallback | None = None,
    ) -> None:
        """Load active models (user-trained overrides if present, else bundled factory models)."""
        target = ModelMode.parse(only_mode) if only_mode is not None else None
        with self.lock:
            if target in (None, ModelMode.DIGITS):
                if progress_cb is not None:
                    progress_cb(68, "Lade Zahlen-Erkenner (Neuronales Netz 0–9)...")
                d_path, d_meta_path, d_custom = self.paths.resolve_model_paths(ModelMode.DIGITS)
                if d_path.exists():
                    self.digits_model = tf.keras.models.load_model(d_path)
                    self._digits_static_links = _compute_conv_links(self.digits_model)
                    self.digits_meta = self._read_model_metadata(
                        self.digits_model, ModelMode.DIGITS, d_meta_path, d_custom
                    )

            if target in (None, ModelMode.SMILEY):
                if progress_cb is not None:
                    progress_cb(88, "Lade Smiley-Glücks-Netz (5 Gefühls-Stufen)...")
                s_path, s_meta_path, s_custom = self.paths.resolve_model_paths(ModelMode.SMILEY)
                if s_path.exists():
                    self.smiley_model = tf.keras.models.load_model(s_path)
                    self._smiley_static_links = _compute_conv_links(self.smiley_model)
                    self.smiley_meta = self._read_model_metadata(
                        self.smiley_model, ModelMode.SMILEY, s_meta_path, s_custom
                    )

    def _read_model_metadata(
        self,
        model: tf.keras.Model,
        mode: ModelMode,
        meta_file: Path | None = None,
        is_custom: bool = False,
    ) -> dict[str, Any]:
        """Extract layer architecture metadata directly from the model and matching JSON file."""
        conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
        drop_layers = [
            layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)
        ]

        hidden_layers = dense_layers[:-1] if len(dense_layers) >= 2 else []
        output_layer = dense_layers[-1] if dense_layers else None

        conv_counts = [int(layer.filters) for layer in conv_layers]
        conv_names = [layer.name for layer in conv_layers]
        hidden_counts = [int(layer.units) for layer in hidden_layers]
        hidden_names = [layer.name for layer in hidden_layers]
        output_units = (
            int(output_layer.units)
            if output_layer is not None
            else (10 if mode is ModelMode.DIGITS else 5)
        )

        act_name = "relu"
        ref_layer = hidden_layers[0] if hidden_layers else (conv_layers[0] if conv_layers else None)
        if ref_layer is not None and hasattr(ref_layer, "activation"):
            act_name = getattr(ref_layer.activation, "__name__", "relu")

        dropout_rate = round(float(drop_layers[0].rate), 3) if drop_layers else 0.0
        total_nodes = sum(conv_counts) + sum(hidden_counts) + output_units

        meta: dict[str, Any] = {
            "mode": mode.value,
            "is_custom": bool(is_custom),
            "source_label": (
                "Eigenes Training (KI_Studio_Daten)"
                if is_custom
                else "Werkseinstellung (integriert)"
            ),
            "arch_type": "CNN" if conv_counts else "MLP",
            "conv_layers": conv_counts,
            "conv_names": conv_names,
            "base_filters": conv_counts[0] if conv_counts else 32,
            "hidden_layers": hidden_counts,
            "hidden_names": hidden_names,
            "dense_layers": max(1, len(hidden_counts)),
            "hidden_units": hidden_counts[0] if hidden_counts else 64,
            "output_units": output_units,
            "total_nodes": total_nodes,
            "total_params": int(model.count_params()),
            "activation": act_name,
            "dropout": dropout_rate,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 4 if mode is ModelMode.DIGITS else 5,
            "test_accuracy": None,
            "test_loss": None,
            "trained_at": None,
            "history": None,
        }

        if meta_file is None:
            _, meta_file, _ = self.paths.resolve_model_paths(mode)
        if meta_file.exists():
            try:
                saved = json.loads(meta_file.read_text(encoding="utf-8"))
                if (
                    saved.get("conv_layers") == conv_counts
                    and saved.get("hidden_layers") == hidden_counts
                ):
                    for field in (
                        "optimizer",
                        "learning_rate",
                        "batch_size",
                        "epochs",
                        "data_fraction",
                        "samples_per_class",
                        "train_samples",
                        "test_accuracy",
                        "test_loss",
                        "trained_at",
                        "history",
                    ):
                        if field in saved:
                            meta[field] = saved[field]
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Could not read model metadata %s: %s", meta_file, exc)

        return meta

    def get_status(self) -> dict[str, Any]:
        """Return readiness, loaded model metadata, training status, and data directory info."""
        with self.lock:
            d_ready = self.digits_model is not None
            s_ready = self.smiley_model is not None
            d_meta = self.digits_meta
            s_meta = self.smiley_meta
        return {
            "tf_version": tf.__version__,
            "zahlen_bereit": d_ready,
            "smiley_bereit": s_ready,
            "zahlen_meta": d_meta,
            "smiley_meta": s_meta,
            "train_aktiv": self.train_active,
            "train_skript": self.train_script,
            "data_info": self.paths.get_storage_info(),
        }

    def get_training_status(self) -> dict[str, Any]:
        """Return a thread-safe snapshot of active training logs and epoch metrics."""
        with self.train_lock:
            return {
                "active": self.train_active,
                "script": self.train_script,
                "lines": list(self.train_lines),
                "metrics": list(self.train_metrics),
            }

    @staticmethod
    def decode_image_bytes(raw_bytes: bytes) -> np.ndarray:
        """Decode uploaded raw image bytes into a NumPy pixel array."""
        return decode_image_bytes(raw_bytes)

    def analyze_random_sample(self, mode: ModelMode | str) -> dict[str, Any]:
        """Select a random MNIST digit or synthesized smiley and inspect its full forward pass."""
        parsed_mode = ModelMode.parse(mode)
        if parsed_mode is ModelMode.DIGITS:
            with self.lock:
                if self._mnist_test is None:
                    _, (x_test, y_test) = load_mnist_offline(self.paths)
                    self._mnist_test = (x_test.astype("float32") / 255.0, y_test)
                x_test, y_test = self._mnist_test
            idx = random.randint(0, len(x_test) - 1)
            image = x_test[idx]
            result = self.analyze_preprocessed_image(ModelMode.DIGITS, image, is_empty=False)
            result["source_label"] = f"MNIST-Testbild #{idx} (Echte Zahl: {int(y_test[idx])})"
            result["canvas_png_b64"] = encode_grayscale_png_b64(1.0 - image, target_size=280)
            return result

        rng = np.random.default_rng()
        mood_class = int(rng.integers(0, 5))
        image = generate_smiley_sample(mood_class, rng)
        result = self.analyze_preprocessed_image(ModelMode.SMILEY, image, is_empty=False)
        result["source_label"] = (
            f"Zufalls-Smiley ({MOOD_NAMES[mood_class]} {MOOD_SYMBOLS[mood_class]})"
        )
        result["canvas_png_b64"] = encode_grayscale_png_b64(1.0 - image, target_size=280)
        return result

    def analyze_raw_image(
        self,
        mode: ModelMode | str,
        raw_pixels: np.ndarray,
        return_canvas_image: bool = False,
    ) -> dict[str, Any]:
        """Preprocess an arbitrary raw image and analyze the active model's activations."""
        parsed_mode = ModelMode.parse(mode)
        if parsed_mode is ModelMode.DIGITS:
            prepared, is_empty = prepare_digit_image(raw_pixels)
        else:
            prepared, is_empty = prepare_smiley_image(raw_pixels)

        result = self.analyze_preprocessed_image(parsed_mode, prepared, is_empty=is_empty)
        if return_canvas_image and not is_empty:
            result["canvas_png_b64"] = encode_grayscale_png_b64(1.0 - prepared, target_size=280)
        return result

    def analyze_sample_file(
        self, mode: ModelMode | str, file_path: Path, filename: str
    ) -> dict[str, Any]:
        """Load an image file from disk and inspect its forward pass."""
        pixels = load_image_file(file_path)
        result = self.analyze_raw_image(mode, pixels, return_canvas_image=True)
        result["source_label"] = f"Beispielbild: {filename}"
        return result

    def analyze_preprocessed_image(
        self,
        mode: ModelMode | str,
        image: np.ndarray,
        is_empty: bool,
    ) -> dict[str, Any]:
        """Execute a dynamic layer-by-layer forward pass and return activations and synapses."""
        parsed_mode = ModelMode.parse(mode)
        is_digits = parsed_mode is ModelMode.DIGITS
        labels = [str(i) for i in range(10)] if is_digits else list(MOOD_NAMES)
        symbols = [str(i) for i in range(10)] if is_digits else list(MOOD_SYMBOLS)

        resized_14 = tf.image.resize(image[:, :, np.newaxis], size=[14, 14], method="area")
        input_14 = resized_14.numpy()[:, :, 0]
        input_grid = np.round(np.clip(input_14, 0.0, 1.0) * 100).astype(int).tolist()

        with self.lock:
            model = self.digits_model if is_digits else self.smiley_model
            static_links = self._digits_static_links if is_digits else self._smiley_static_links
            model_meta = self.digits_meta if is_digits else self.smiley_meta

            if model is None:
                return {
                    "empty": True,
                    "model_missing": True,
                    "mode": parsed_mode.value,
                    "preview_b64": encode_grayscale_png_b64(image, 112),
                }

            if model_meta is None:
                model_meta = self._read_model_metadata(model, parsed_mode)

            if is_empty:
                return self._empty_response(
                    parsed_mode, labels, symbols, image, input_grid, model_meta
                )

            trace = self._run_layer_trace(model, image)

        return self._assemble_analysis_payload(
            mode=parsed_mode,
            labels=labels,
            symbols=symbols,
            image=image,
            input_grid=input_grid,
            model_meta=model_meta,
            static_links=static_links,
            trace=trace,
        )

    def _run_layer_trace(self, model: tf.keras.Model, image: np.ndarray) -> dict[str, Any]:
        """Run one forward pass and collect per-layer activations, feature maps, and logits."""
        all_dense = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
        if len(all_dense) < 2:
            raise ValueError(
                "Loaded model must contain at least one hidden Dense layer and one output layer."
            )

        output_layer = all_dense[-1]
        x = tf.constant(image[np.newaxis, :, :], dtype=tf.float32)

        conv_layers_out: list[dict[str, Any]] = []
        last_feature_map: np.ndarray | None = None
        hidden_data: list[tuple[tf.keras.layers.Dense, np.ndarray, np.ndarray]] = []
        w_out: np.ndarray | None = None
        b_out: np.ndarray | None = None
        logits: np.ndarray | None = None

        conv_idx = 0
        for layer in model.layers:
            if layer is output_layer:
                w_out, b_out = layer.get_weights()
                pre_out = x.numpy()[0]
                logits = pre_out @ w_out + b_out
                x = layer(x, training=False)
            elif isinstance(layer, tf.keras.layers.Conv2D):
                conv_idx += 1
                x = layer(x, training=False)
                fmap = x.numpy()[0]
                last_feature_map = fmap
                conv_layers_out.append(
                    self._summarize_conv_layer(layer.name, fmap, is_first=(conv_idx == 1))
                )
            elif isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = layer(x, training=False)
                last_feature_map = x.numpy()[0]
            elif isinstance(layer, tf.keras.layers.Dense):
                w_h, _ = layer.get_weights()
                x = layer(x, training=False)
                hidden_data.append((layer, w_h, x.numpy()[0]))
            else:
                x = layer(x, training=False)

        if not hidden_data or w_out is None or b_out is None or logits is None:
            raise ValueError("Forward trace did not produce expected Dense/output weights.")

        return {
            "conv_layers": conv_layers_out,
            "last_feature_map": last_feature_map,
            "hidden_data": hidden_data,
            "w_out": w_out,
            "b_out": b_out,
            "logits": logits,
            "probabilities": x.numpy()[0],
        }

    @staticmethod
    def _summarize_conv_layer(
        layer_name: str, fmap: np.ndarray, *, is_first: bool
    ) -> dict[str, Any]:
        fmap_mag = np.abs(fmap)
        means = np.mean(fmap_mag, axis=(0, 1))
        peaks = np.max(fmap_mag, axis=(0, 1))
        raw_strength = 0.55 * means + 0.45 * peaks
        norm_strength = _normalize_magnitude_01(raw_strength)

        mini_8 = tf.image.resize(fmap_mag, size=[8, 8], method="area").numpy()
        max_per_filter = np.maximum(np.max(mini_8, axis=(0, 1), keepdims=True), 1e-6)
        mini_norm = np.round((mini_8 / max_per_filter) * 100).astype(int)
        mini_maps = np.transpose(mini_norm, (2, 0, 1)).tolist()

        top_maps: list[dict[str, Any]] = []
        if is_first:
            top_count = min(4, int(fmap.shape[-1]))
            top_indices = np.argsort(raw_strength)[-top_count:][::-1]
            for f_idx in top_indices:
                single = fmap_mag[:, :, int(f_idx)]
                m_val = max(float(np.max(single)), 1e-6)
                top_maps.append(
                    {
                        "filter_index": int(f_idx),
                        "strength": round(float(norm_strength[f_idx]) * 100.0, 1),
                        "png_b64": encode_grayscale_png_b64(single / m_val, 64),
                    }
                )

        return {
            "name": layer_name,
            "count": int(fmap.shape[-1]),
            "activations": np.round(norm_strength, 4).tolist(),
            "raw_means": np.round(means, 3).tolist(),
            "mini_maps": mini_maps,
            "top_maps": top_maps,
        }

    def _assemble_analysis_payload(
        self,
        *,
        mode: ModelMode,
        labels: list[str],
        symbols: list[str],
        image: np.ndarray,
        input_grid: list[list[int]],
        model_meta: dict[str, Any],
        static_links: dict[str, list[list[list[float]]]],
        trace: dict[str, Any],
    ) -> dict[str, Any]:
        conv_layers_out: list[dict[str, Any]] = trace["conv_layers"]
        hidden_data: list[tuple[tf.keras.layers.Dense, np.ndarray, np.ndarray]] = trace[
            "hidden_data"
        ]
        last_feature_map: np.ndarray | None = trace["last_feature_map"]
        w_out: np.ndarray = trace["w_out"]
        b_out: np.ndarray = trace["b_out"]
        logits: np.ndarray = trace["logits"]
        probs: np.ndarray = trace["probabilities"]

        act_label = str(model_meta.get("activation", "relu")).upper()
        hidden_layers_out: list[dict[str, Any]] = []
        for idx_h, (layer_h, _, act_h) in enumerate(hidden_data):
            title = (
                f"Denkschicht ({act_h.shape[0]} {act_label})"
                if len(hidden_data) == 1
                else f"Denkschicht {idx_h + 1} ({act_h.shape[0]} {act_label})"
            )
            hidden_layers_out.append(
                {
                    "name": title,
                    "layer_name": layer_h.name,
                    "activation": act_label,
                    "count": int(act_h.shape[0]),
                    "raw": np.round(act_h, 4).tolist(),
                    "norm": np.round(_normalize_magnitude_01(act_h), 4).tolist(),
                    "mini_maps": [],
                }
            )

        conv_to_hidden: list[list[list[float]]] = []
        input_to_hidden: list[list[list[float]]] = []
        _, w_h0, act_h0 = hidden_data[0]

        if conv_layers_out and last_feature_map is not None:
            ph, pw, pc = last_feature_map.shape
            if ph * pw * pc == w_h0.shape[0]:
                w_4d = w_h0.reshape(ph, pw, pc, act_h0.shape[0])
                flow_matrix = np.einsum("yxc,yxcj->cj", last_feature_map, w_4d)
                max_flow = max(float(np.max(np.abs(flow_matrix))), 1e-6)
                top_k = min(3, flow_matrix.shape[0])
                for j in range(act_h0.shape[0]):
                    col = flow_matrix[:, j]
                    top_c = np.argsort(np.abs(col))[-top_k:][::-1]
                    conv_to_hidden.append(
                        [[int(c), round(float(col[c] / max_flow), 3)] for c in top_c]
                    )
        elif not conv_layers_out:
            in_h, in_w = image.shape
            if in_h * in_w == w_h0.shape[0]:
                pixel_flow = image[:, :, np.newaxis] * w_h0.reshape(in_h, in_w, act_h0.shape[0])
                pf_14 = tf.image.resize(pixel_flow, size=[14, 14], method="area").numpy()
                max_pf = max(float(np.max(np.abs(pf_14))), 1e-6)
                for j in range(act_h0.shape[0]):
                    field = pf_14[:, :, j]
                    flat = np.abs(field).reshape(-1)
                    top_idx = np.argsort(flat)[-3:][::-1]
                    links: list[list[float]] = []
                    for flat_i in top_idx:
                        if flat[flat_i] <= 1e-6:
                            continue
                        r14, c14 = divmod(int(flat_i), 14)
                        links.append([r14, c14, round(float(field[r14, c14] / max_pf), 3)])
                    input_to_hidden.append(links)

                pf_8 = tf.image.resize(np.abs(pixel_flow), size=[8, 8], method="area").numpy()
                max_p8 = np.maximum(np.max(pf_8, axis=(0, 1), keepdims=True), 1e-6)
                mini_mlp = np.round((pf_8 / max_p8) * 100).astype(int)
                hidden_layers_out[0]["mini_maps"] = np.transpose(mini_mlp, (2, 0, 1)).tolist()

        hidden_to_hidden: list[list[list[list[float]]]] = []
        for d_idx in range(len(hidden_data) - 1):
            act_prev = hidden_data[d_idx][2]
            w_next = hidden_data[d_idx + 1][1]
            flow_hh = act_prev[:, np.newaxis] * w_next
            max_hh = max(float(np.max(np.abs(flow_hh))), 1e-6)
            top_k = min(4, flow_hh.shape[0])
            layer_links: list[list[list[float]]] = []
            for j in range(flow_hh.shape[1]):
                col = flow_hh[:, j]
                top_i = np.argsort(np.abs(col))[-top_k:][::-1]
                layer_links.append([[int(i), round(float(col[i] / max_hh), 3)] for i in top_i])
            hidden_to_hidden.append(layer_links)

        last_hidden_act = hidden_data[-1][2]
        contributions = last_hidden_act[:, np.newaxis] * w_out

        z_max = float(np.max(logits))
        exp_shifted = np.exp(logits - z_max)
        exp_sum = float(np.sum(exp_shifted))

        outputs = [
            {
                "index": k,
                "label": labels[k],
                "symbol": symbols[k],
                "bias": round(float(b_out[k]), 4),
                "neuron_sum": round(float(np.sum(contributions[:, k])), 4),
                "logit": round(float(logits[k]), 4),
                "exp_rel": round(float(exp_shifted[k]), 6),
                "prob": round(float(probs[k]) * 100.0, 3),
            }
            for k in range(len(labels))
        ]

        winner_idx = int(np.argmax(probs))
        return {
            "empty": False,
            "model_missing": False,
            "mode": mode.value,
            "model_meta": model_meta,
            "preview_b64": encode_grayscale_png_b64(image, 112),
            "input_grid": input_grid,
            "prediction": {
                "index": winner_idx,
                "label": labels[winner_idx],
                "symbol": symbols[winner_idx],
                "confidence": round(float(probs[winner_idx]) * 100.0, 2),
                "happiness": (
                    round(compute_happiness_percent(probs), 1)
                    if mode is ModelMode.SMILEY
                    else None
                ),
            },
            "conv_layers": conv_layers_out,
            "hidden_layers": hidden_layers_out,
            "hidden_layer": hidden_layers_out[-1],
            "outputs": outputs,
            "softmax_meta": {
                "z_max": round(z_max, 4),
                "exp_sum": round(exp_sum, 6),
            },
            "synapses": {
                "contributions": np.round(contributions, 4).tolist(),
                "weights": np.round(w_out, 4).tolist(),
                "conv_to_hidden": conv_to_hidden,
                "input_to_hidden": input_to_hidden,
                "hidden_to_hidden": hidden_to_hidden,
                **static_links,
            },
        }

    def _empty_response(
        self,
        mode: ModelMode,
        labels: list[str],
        symbols: list[str],
        image: np.ndarray,
        input_grid: list[list[int]],
        model_meta: dict[str, Any],
    ) -> dict[str, Any]:
        """Build an idle zero-activation payload matching the currently loaded model topology."""
        conv_counts = model_meta.get("conv_layers", [])
        conv_names = model_meta.get(
            "conv_names", [f"Filter_{i + 1}" for i in range(len(conv_counts))]
        )
        hidden_counts = model_meta.get("hidden_layers", [64])
        hidden_names = model_meta.get(
            "hidden_names", [f"Denkschicht_{i + 1}" for i in range(len(hidden_counts))]
        )
        act_label = str(model_meta.get("activation", "relu")).upper()

        hidden_layers = []
        for idx_h, h_cnt in enumerate(hidden_counts):
            title = (
                f"Denkschicht ({h_cnt} {act_label})"
                if len(hidden_counts) == 1
                else f"Denkschicht {idx_h + 1} ({h_cnt} {act_label})"
            )
            hidden_layers.append(
                {
                    "name": title,
                    "layer_name": (
                        hidden_names[idx_h]
                        if idx_h < len(hidden_names)
                        else f"Denkschicht_{idx_h + 1}"
                    ),
                    "activation": act_label,
                    "count": int(h_cnt),
                    "raw": [0.0] * int(h_cnt),
                    "norm": [0.0] * int(h_cnt),
                    "mini_maps": [],
                }
            )

        last_hidden = int(hidden_counts[-1]) if hidden_counts else 64
        return {
            "empty": True,
            "model_missing": False,
            "mode": mode.value,
            "model_meta": model_meta,
            "preview_b64": encode_grayscale_png_b64(image, 112),
            "input_grid": input_grid,
            "prediction": None,
            "conv_layers": [
                {
                    "name": name,
                    "count": int(cnt),
                    "activations": [0.0] * int(cnt),
                    "raw_means": [0.0] * int(cnt),
                    "mini_maps": [],
                    "top_maps": [],
                }
                for name, cnt in zip(conv_names, conv_counts, strict=False)
            ],
            "hidden_layers": hidden_layers,
            "hidden_layer": hidden_layers[-1],
            "outputs": [
                {
                    "index": k,
                    "label": labels[k],
                    "symbol": symbols[k],
                    "bias": 0.0,
                    "neuron_sum": 0.0,
                    "logit": 0.0,
                    "exp_rel": 0.0,
                    "prob": 0.0,
                }
                for k in range(len(labels))
            ],
            "softmax_meta": {"z_max": 0.0, "exp_sum": 1.0},
            "synapses": {
                "contributions": [[0.0] * len(labels) for _ in range(last_hidden)],
                "weights": [[0.0] * len(labels) for _ in range(last_hidden)],
                "conv_to_hidden": [],
                "input_to_hidden": [],
                "hidden_to_hidden": [],
            },
        }

    def start_training(
        self,
        task_name: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Validate task parameters and start an asynchronous training or evaluation worker."""
        task = TrainingTask.resolve(task_name)
        if task is None:
            return False, "Unbekannte Trainings-Aktion."

        target_mode = task.target_mode
        config: TrainingConfig | None = None
        if target_mode is not None:
            config = TrainingConfig.from_dict(target_mode, params)
            preview = config.preview(target_mode)
        else:
            preview = "Zufalls-Test aus dem integrierten MNIST-Testset"

        with self.train_lock:
            if self.train_active:
                return False, "Es läuft bereits ein Training!"
            self.train_active = True
            self.train_script = task.value
            self.stop_event = threading.Event()
            stop_ev = self.stop_event
            self.train_metrics = []
            self.train_lines = [f">>> Starte: {preview}\n"]

        log_state = {"last_was_cr": False}

        def _log_fn(line: str, overwrite_last: bool = False) -> None:
            with self.train_lock:
                if overwrite_last:
                    if line.strip():
                        if log_state["last_was_cr"] and self.train_lines:
                            self.train_lines[-1] = line
                        else:
                            self.train_lines.append(line)
                        log_state["last_was_cr"] = True
                    return

                if log_state["last_was_cr"] and line == "":
                    log_state["last_was_cr"] = False
                    return
                self.train_lines.append(line)
                log_state["last_was_cr"] = False
                if len(self.train_lines) > 450:
                    self.train_lines = self.train_lines[-450:]

        def _metric_fn(metric: dict[str, Any]) -> None:
            with self.train_lock:
                epoch = metric.get("epoch")
                for idx_m, old in enumerate(self.train_metrics):
                    if old.get("epoch") == epoch:
                        self.train_metrics[idx_m] = metric
                        return
                self.train_metrics.append(metric)

        def _worker() -> None:
            try:
                if task is TrainingTask.TRAIN_DIGITS:
                    success = train_digits_model(
                        config=config,
                        paths=self.paths,
                        log_fn=_log_fn,
                        metric_fn=_metric_fn,
                        stop_event=stop_ev,
                    )
                elif task is TrainingTask.TRAIN_SMILEY:
                    success = train_smiley_model(
                        config=config,
                        paths=self.paths,
                        log_fn=_log_fn,
                        metric_fn=_metric_fn,
                        stop_event=stop_ev,
                    )
                else:
                    success = run_quick_digit_test(paths=self.paths, log_fn=_log_fn)

                if stop_ev.is_set():
                    with self.train_lock:
                        self.train_lines.append(
                            "\n>>> Vom Benutzer gestoppt (keine unfertigen Dateien angelegt)."
                        )
                elif success:
                    if target_mode is not None:
                        self.load_models(only_mode=target_mode)
                    with self.train_lock:
                        if target_mode is not None:
                            self.train_lines.append(
                                "\n>>> Fertig! Neues Gehirn wurde im Ordner 'KI_Studio_Daten' "
                                "gespeichert und sofort live geladen!"
                            )
                        else:
                            self.train_lines.append("\n>>> Zufalls-Test erfolgreich abgeschlossen!")
                else:
                    with self.train_lock:
                        self.train_lines.append("\n>>> Vorgang wurde beendet.")
            except Exception as exc:
                logger.exception("Training worker failed for task %s", task.value)
                with self.train_lock:
                    self.train_lines.append(f"\nFEHLER: {exc}")
            finally:
                with self.train_lock:
                    self.train_thread = None
                    self.train_active = False

        worker = threading.Thread(target=_worker, daemon=True)
        with self.train_lock:
            self.train_thread = worker
        worker.start()
        return True, "Gestartet"

    def stop_training(self) -> None:
        """Signal any running training worker to halt after the current batch."""
        with self.train_lock:
            if self.train_active:
                self.stop_event.set()

    def reset_user_data(self) -> tuple[bool, str]:
        """Stop training if running, delete custom models, and reload factory weights."""
        running_thread: threading.Thread | None = None
        with self.train_lock:
            if self.train_active:
                self.stop_event.set()
                running_thread = self.train_thread

        if running_thread is not None:
            running_thread.join(timeout=3.0)

        ok, msg = self.paths.delete_custom_models()
        if ok:
            self.load_models()
            with self.train_lock:
                self.train_metrics = []
                self.train_lines.append(f"\n>>> {msg}")
        return ok, msg
