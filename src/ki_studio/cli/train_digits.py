"""CLI command to train a custom MNIST digit (0-9) classifier."""

from __future__ import annotations

import sys

from ki_studio.cli import build_training_parser
from ki_studio.config import ModelMode, TrainingConfig, configure_stdio_utf8
from ki_studio.core.training import train_digits_model
from ki_studio.paths import DEFAULT_PATHS


def main(argv: list[str] | None = None) -> None:
    configure_stdio_utf8()
    parser = build_training_parser(
        prog="ki-studio-train-digits",
        description="Zahlen-Netz (0-9) mit TensorFlow offline trainieren",
        default_conv_layers=2,
        default_epochs=4,
    )
    parser.add_argument("--data-fraction", type=float, default=1.0)
    args = parser.parse_args(argv)

    config = TrainingConfig.from_dict(ModelMode.DIGITS, vars(args))
    if not train_digits_model(config=config, paths=DEFAULT_PATHS):
        sys.exit(1)


if __name__ == "__main__":
    main()
