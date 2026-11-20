"""Configuration models, enumerations, and runtime environment setup."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

VALID_ACTIVATIONS: tuple[str, ...] = ("relu", "elu", "tanh", "sigmoid")
VALID_OPTIMIZERS: tuple[str, ...] = ("adam", "rmsprop", "sgd")

DEFAULT_SEED = 42


class ModelMode(str, Enum):
    """Supported neural-network domain modes (wire keys preserved for web UI compatibility)."""

    DIGITS = "zahlen"
    SMILEY = "smiley"

    @classmethod
    def parse(cls, value: ModelMode | str) -> ModelMode:
        """Convert a mode string or enum into a validated `ModelMode`."""
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Unsupported model mode: {value!r}")


class TrainingTask(str, Enum):
    """Background task identifiers accepted by `/api/train/start`."""

    TRAIN_DIGITS = "train_digits"
    TRAIN_SMILEY = "train_smiley"
    QUICK_TEST = "quick_test"

    @classmethod
    def resolve(cls, raw_name: str) -> TrainingTask | None:
        """Resolve canonical task names and legacy script aliases."""
        normalized = raw_name.strip()
        aliases = {
            "train_digits": cls.TRAIN_DIGITS,
            "neurales_netz.py": cls.TRAIN_DIGITS,
            "train_smiley": cls.TRAIN_SMILEY,
            "smiley_netz.py": cls.TRAIN_SMILEY,
            "quick_test": cls.QUICK_TEST,
            "vorhersage_nutzen.py": cls.QUICK_TEST,
        }
        return aliases.get(normalized)

    @property
    def target_mode(self) -> ModelMode | None:
        """Return the model mode updated after this task finishes, if any."""
        if self is TrainingTask.TRAIN_DIGITS:
            return ModelMode.DIGITS
        if self is TrainingTask.TRAIN_SMILEY:
            return ModelMode.SMILEY
        return None


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _clamp_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if parsed != parsed:  # NaN guard
        parsed = default
    return max(minimum, min(maximum, parsed))


@dataclass(frozen=True)
class TrainingConfig:
    """Validated hyperparameter configuration shared across CLI and live lab training."""

    conv_layers: int = 2
    base_filters: int = 32
    dense_layers: int = 1
    hidden_units: int = 64
    activation: str = "relu"
    optimizer: str = "adam"
    epochs: int = 4
    learning_rate: float = 0.001
    batch_size: int = 64
    dropout: float = 0.3
    data_fraction: float = 1.0
    samples_per_class: int = 2400

    @classmethod
    def from_dict(
        cls,
        mode: ModelMode | str,
        raw: Mapping[str, Any] | None = None,
    ) -> TrainingConfig:
        """Validate and clamp a raw parameter mapping for the target model mode."""
        parsed_mode = ModelMode.parse(mode)
        params = dict(raw or {})

        default_conv = 2 if parsed_mode is ModelMode.DIGITS else 3
        default_epochs = 4 if parsed_mode is ModelMode.DIGITS else 5

        activation = str(params.get("activation", "relu")).lower().strip()
        if activation not in VALID_ACTIVATIONS:
            activation = "relu"

        optimizer = str(params.get("optimizer", "adam")).lower().strip()
        if optimizer not in VALID_OPTIMIZERS:
            optimizer = "adam"

        lr_raw = params.get("learning_rate", params.get("lr", 0.001))

        return cls(
            conv_layers=_clamp_int(params.get("conv_layers"), default_conv, 0, 3),
            base_filters=_clamp_int(params.get("base_filters"), 32, 8, 64),
            dense_layers=_clamp_int(params.get("dense_layers"), 1, 1, 2),
            hidden_units=_clamp_int(params.get("hidden_units"), 64, 8, 256),
            activation=activation,
            optimizer=optimizer,
            epochs=_clamp_int(params.get("epochs"), default_epochs, 1, 20),
            learning_rate=_clamp_float(lr_raw, 0.001, 1e-5, 0.2),
            batch_size=_clamp_int(params.get("batch_size"), 64, 8, 512),
            dropout=_clamp_float(params.get("dropout"), 0.3, 0.0, 0.65),
            data_fraction=_clamp_float(params.get("data_fraction"), 1.0, 0.02, 1.0),
            samples_per_class=_clamp_int(params.get("samples_per_class"), 2400, 100, 4000),
        )

    def preview(self, mode: ModelMode | str) -> str:
        """Return a human-readable summary string for training logs."""
        parsed_mode = ModelMode.parse(mode)
        title = "Zahlen-Training" if parsed_mode is ModelMode.DIGITS else "Smiley-Training"
        data_desc = (
            f"Daten={self.data_fraction * 100:.0f}%"
            if parsed_mode is ModelMode.DIGITS
            else f"Smileys={self.samples_per_class * 5}"
        )
        return (
            f"{title} (Filter-Ebenen={self.conv_layers}, "
            f"Denkschichten={self.dense_layers}x{self.hidden_units} {self.activation.upper()}, "
            f"Epochen={self.epochs}, {self.optimizer.upper()} lr={self.learning_rate:g}, "
            f"{data_desc})"
        )


def configure_runtime_env(assets_dir: Path | None = None) -> None:
    """Configure TensorFlow log verbosity and local Keras home prior to importing TensorFlow."""
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if assets_dir is not None and (assets_dir / "keras.json").exists():
        os.environ.setdefault("KERAS_HOME", str(assets_dir))


def configure_stdio_utf8() -> None:
    """Ensure stdout/stderr are non-null and UTF-8 encoded (important under pythonw.exe)."""
    import io

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            setattr(sys, stream_name, io.StringIO())
        elif hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except OSError:
                pass


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging cleanly when running interactively or as a CLI tool."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
