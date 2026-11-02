"""CLI command for running a quick random MNIST sample prediction."""

from __future__ import annotations

import sys

from ki_studio.config import configure_stdio_utf8
from ki_studio.core.training import run_quick_digit_test
from ki_studio.paths import DEFAULT_PATHS


def main() -> None:
    configure_stdio_utf8()
    if not run_quick_digit_test(paths=DEFAULT_PATHS):
        sys.exit(1)


if __name__ == "__main__":
    main()
