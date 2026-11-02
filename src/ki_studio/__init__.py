"""TensorFlow KI-Studio: Real-time neural network visualizer and training lab."""

from __future__ import annotations

import os

# Set quiet TensorFlow defaults before any submodule imports `tensorflow`.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

__version__ = "1.0.0"

__all__ = ["__version__"]
