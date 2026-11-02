"""Main desktop Studio launcher CLI."""

from __future__ import annotations

import argparse
import ctypes
import logging
import os

from ki_studio.config import configure_logging, configure_stdio_utf8
from ki_studio.paths import DEFAULT_PATHS
from ki_studio.server.desktop import run_event_loop, start_server_async


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ki-studio",
        description="Launch TensorFlow KI-Studio desktop visualization & training lab.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Start the local HTTP server without opening a desktop app window.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind host for the local HTTP server (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Bind port for the local HTTP server (default: 0 for ephemeral port).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entrypoint for starting TensorFlow KI-Studio."""
    configure_stdio_utf8()
    args = parse_args(argv)
    configure_logging(logging.DEBUG if args.debug else logging.WARNING)

    open_window = not args.no_window
    try:
        server = start_server_async(
            open_window=open_window,
            host=args.host,
            port=args.port,
            paths=DEFAULT_PATHS,
        )
        run_event_loop(server)
    except Exception as exc:
        if open_window and os.name == "nt":
            try:
                ctypes.windll.user32.MessageBoxW(
                    0,
                    f"Fehler beim Starten von TensorFlow KI-Studio:\n\n{exc}",
                    "TensorFlow KI-Studio",
                    0x10,
                )
            except OSError:
                pass
        raise


if __name__ == "__main__":
    main()
