"""CLI command to train a custom 5-level smiley mood classifier."""

from __future__ import annotations

import sys

from ki_studio.cli import build_training_parser
from ki_studio.config import ModelMode, TrainingConfig, configure_stdio_utf8
from ki_studio.core.training import train_smiley_model
from ki_studio.paths import DEFAULT_PATHS


def main(argv: list[str] | None = None) -> None:
    configure_stdio_utf8()
    parser = build_training_parser(
        prog="ki-studio-train-smiley",
        description="Smiley-Glücks-Netz mit TensorFlow trainieren",
        default_conv_layers=3,
        default_epochs=5,
    )
    parser.add_argument("--samples-per-class", type=int, default=2400)
    args = parser.parse_args(argv)

    config = TrainingConfig.from_dict(ModelMode.SMILEY, vars(args))
    if not train_smiley_model(config=config, paths=DEFAULT_PATHS):
        sys.exit(1)


if __name__ == "__main__":
    main()
