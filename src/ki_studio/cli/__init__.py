"""Command-line interfaces and standalone Tkinter drawing testers."""

from __future__ import annotations

import argparse

from ki_studio.config import VALID_ACTIVATIONS, VALID_OPTIMIZERS

__all__ = [
    "app",
    "build_training_parser",
    "predict_digits",
    "predict_smiley",
    "quick_test",
    "tk_canvas",
    "train_digits",
    "train_smiley",
]


def build_training_parser(
    *,
    prog: str,
    description: str,
    default_conv_layers: int,
    default_epochs: int,
) -> argparse.ArgumentParser:
    """Create an `ArgumentParser` populated with shared neural-network training flags."""
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument("--conv-layers", type=int, default=default_conv_layers, choices=[0, 1, 2, 3])
    parser.add_argument("--base-filters", type=int, default=32)
    parser.add_argument("--dense-layers", type=int, default=1, choices=[1, 2])
    parser.add_argument("--hidden-units", type=int, default=64)
    parser.add_argument("--activation", type=str, default="relu", choices=list(VALID_ACTIVATIONS))
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="adam", choices=list(VALID_OPTIMIZERS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    return parser
