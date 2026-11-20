"""Fast, stdlib-only local HTTP server for instant UI boot and live inference."""

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from ki_studio.paths import DEFAULT_PATHS, StudioPaths

logger = logging.getLogger(__name__)

MAX_REQUEST_BYTES = 20 * 1024 * 1024  # 20 MiB upper bound for uploaded images

_STATIC_MIME_TYPES: dict[str, str] = {
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


def _open_folder_in_file_manager(target: Path) -> None:
    """Open a local directory using the platform's native file manager."""
    if os.name == "nt":
        os.startfile(str(target))  # type: ignore[attr-defined]
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.Popen([opener, str(target)])


class StudioHTTPServer(ThreadingHTTPServer):
    """Localhost-bound threaded HTTP server managing boot progress and client heartbeats."""

    def __init__(
        self,
        engine: Any = None,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        paths: StudioPaths = DEFAULT_PATHS,
    ) -> None:
        self.paths = paths
        self.boot_lock = threading.Lock()
        self.engine: Any = engine
        self.engine_ready_event = threading.Event()
        if engine is not None:
            self.engine_ready_event.set()
            self.boot_status: dict[str, Any] = {
                "ready": True,
                "percent": 100,
                "step": "KI-Gehirne bereit!",
                "error": None,
            }
        else:
            self.boot_status = {
                "ready": False,
                "percent": 18,
                "step": "Initialisiere lokale KI-Umgebung...",
                "error": None,
            }

        self.client_connected = False
        self.last_heartbeat = time.time()
        self.close_requested_at: float | None = None
        self.should_exit = False
        super().__init__((host, port), StudioRequestHandler)

    def update_boot_status(
        self,
        percent: int,
        step: str,
        *,
        ready: bool = False,
        error: str | None = None,
    ) -> None:
        """Update thread-safe boot progress read by `/api/boot`."""
        with self.boot_lock:
            self.boot_status = {
                "ready": bool(ready),
                "percent": int(percent),
                "step": step,
                "error": error,
            }

    def set_engine(self, engine: Any) -> None:
        """Attach the initialized `BrainEngine` and mark boot complete."""
        self.engine = engine
        self.update_boot_status(100, "KI-Gehirne bereit!", ready=True)
        self.engine_ready_event.set()

    def wait_for_engine(self, timeout: float = 45.0) -> Any:
        """Wait until background TensorFlow initialization completes or times out."""
        if self.engine is not None:
            return self.engine
        self.engine_ready_event.wait(timeout=timeout)
        return self.engine


class StudioRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler serving static UI assets and `/api/*` inference/training endpoints."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        logger.debug("%s - %s", self.address_string(), format % args)

    def _send_bytes(
        self, content: bytes, content_type: str, cache_control: str = "no-store"
    ) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _record_heartbeat(self) -> None:
        srv: StudioHTTPServer = self.server  # type: ignore[assignment]
        srv.client_connected = True
        srv.last_heartbeat = time.time()
        srv.close_requested_at = None

    def _serve_static_asset(self, rel_path: str) -> None:
        srv: StudioHTTPServer = self.server  # type: ignore[assignment]
        static_root = srv.paths.static_dir.resolve()
        clean_rel = unquote(rel_path).lstrip("/")
        candidate = (static_root / clean_rel).resolve()

        if not candidate.is_relative_to(static_root):
            self.send_error(403, "Forbidden")
            return
        if not candidate.is_file():
            self.send_error(404, "Not Found")
            return

        content_type = _STATIC_MIME_TYPES.get(candidate.suffix.lower(), "application/octet-stream")
        self._send_bytes(candidate.read_bytes(), content_type)

    def _read_json_body(self) -> tuple[dict[str, Any] | None, int | None, str | None]:
        raw_len = self.headers.get("Content-Length")
        if raw_len is None or raw_len.strip() == "":
            length = 0
        else:
            try:
                length = int(raw_len)
            except ValueError:
                return None, 400, "Invalid Content-Length header."
            if length < 0:
                return None, 400, "Negative Content-Length header."

        if length > MAX_REQUEST_BYTES:
            return None, 413, "Request payload exceeds 20 MiB limit."

        if length == 0:
            return {}, None, None

        raw_body = self.rfile.read(length)
        try:
            decoded = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, 400, "Invalid JSON payload."

        if not isinstance(decoded, dict):
            return None, 400, "JSON payload must be an object."
        return decoded, None, None

    def do_GET(self) -> None:
        srv: StudioHTTPServer = self.server  # type: ignore[assignment]
        path = self.path.split("?", 1)[0]

        if path in ("/", "/index.html"):
            self._record_heartbeat()
            index_file = srv.paths.web_dir / "index.html"
            if not index_file.is_file():
                self.send_error(404, "UI index not found")
                return
            self._send_bytes(index_file.read_bytes(), "text/html; charset=utf-8")
            return

        if path.startswith("/static/"):
            self._serve_static_asset(path[len("/static/") :])
            return

        if path == "/favicon.ico":
            ico_file = srv.paths.assets_dir / "app_icon.ico"
            if ico_file.is_file():
                self._send_bytes(
                    ico_file.read_bytes(),
                    "image/x-icon",
                    cache_control="public, max-age=86400",
                )
                return

        if path in ("/icon-256.png", "/assets/icon_256.png"):
            png_file = srv.paths.assets_dir / "icon_256.png"
            if png_file.is_file():
                self._send_bytes(
                    png_file.read_bytes(),
                    "image/png",
                    cache_control="public, max-age=86400",
                )
                return

        if path == "/manifest.json":
            manifest = {
                "name": "TensorFlow KI-Studio",
                "short_name": "KI-Studio",
                "start_url": "/",
                "display": "standalone",
                "background_color": "#060a14",
                "theme_color": "#090e1a",
                "icons": [
                    {
                        "src": "/icon-256.png",
                        "sizes": "256x256",
                        "type": "image/png",
                        "purpose": "any",
                    },
                    {
                        "src": "/favicon.ico",
                        "sizes": "256x256 64x64 32x32",
                        "type": "image/x-icon",
                    },
                ],
            }
            raw = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
            self._send_bytes(raw, "application/manifest+json; charset=utf-8")
            return

        if path == "/api/boot":
            self._record_heartbeat()
            with srv.boot_lock:
                data = dict(srv.boot_status)
            self._send_json(data)
            return

        if path == "/api/status":
            self._record_heartbeat()
            if srv.engine is None:
                with srv.boot_lock:
                    boot_state = dict(srv.boot_status)
                self._send_json(
                    {
                        "booting": True,
                        "boot": boot_state,
                        "tf_version": "lädt...",
                        "zahlen_bereit": False,
                        "smiley_bereit": False,
                        "zahlen_meta": None,
                        "smiley_meta": None,
                        "train_aktiv": False,
                        "train_skript": "",
                        "data_info": srv.paths.get_storage_info(),
                    }
                )
                return
            status_data = srv.engine.get_status()
            status_data["booting"] = False
            self._send_json(status_data)
            return

        if path == "/api/data/status":
            self._record_heartbeat()
            self._send_json(srv.paths.get_storage_info())
            return

        if path == "/api/train/status":
            self._record_heartbeat()
            if srv.engine is None:
                self._send_json({"active": False, "script": "", "lines": [], "metrics": []})
                return
            self._send_json(srv.engine.get_training_status())
            return

        self.send_error(404)

    def do_POST(self) -> None:
        srv: StudioHTTPServer = self.server  # type: ignore[assignment]
        path = self.path.split("?", 1)[0]

        payload, err_status, err_msg = self._read_json_body()
        if err_status is not None:
            self._send_json({"error": err_msg}, status=err_status)
            return
        assert payload is not None

        if path == "/api/heartbeat":
            self._record_heartbeat()
            self._send_json({"ok": True})
            return

        if path == "/api/close":
            srv.close_requested_at = time.time()
            self._send_json({"ok": True})
            return

        self._record_heartbeat()

        if path == "/api/data/open":
            try:
                target = srv.paths.data_dir if srv.paths.data_dir.exists() else srv.paths.app_dir
                _open_folder_in_file_manager(target)
                self._send_json({"ok": True, "opened": str(target)})
            except OSError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)
            return

        if path == "/api/data/cleanup_and_exit":
            if srv.engine is not None:
                srv.engine.stop_training()
            ok, msg = srv.paths.schedule_full_cleanup()
            self._send_json({"ok": ok, "message": msg}, status=200 if ok else 500)
            if ok:
                srv.should_exit = True
            return

        engine = srv.wait_for_engine(timeout=45.0)
        if engine is None:
            self._send_json({"error": "KI-Engine wird noch geladen..."}, status=503)
            return

        if path == "/api/analyze":
            try:
                mode = payload.get("mode", "zahlen")
                b64_img = str(payload.get("image_b64", ""))
                if "," in b64_img:
                    b64_img = b64_img.split(",", 1)[1]
                raw_bytes = base64.b64decode(b64_img)
                pixels = engine.decode_image_bytes(raw_bytes)
                return_canvas = bool(payload.get("return_canvas", False))
                result = engine.analyze_raw_image(mode, pixels, return_canvas_image=return_canvas)
                self._send_json(result)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

        if path == "/api/random":
            mode = payload.get("mode", "zahlen")
            try:
                result = engine.analyze_random_sample(mode)
                self._send_json(result)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

        if path == "/api/sample_file":
            mode = payload.get("mode", "zahlen")
            filename = str(payload.get("filename", "")).strip()
            samples_root = srv.paths.samples_dir.resolve()
            file_path = (samples_root / filename).resolve()
            if (
                not filename
                or not file_path.is_relative_to(samples_root)
                or not file_path.is_file()
            ):
                self._send_json({"error": "Beispielbild nicht gefunden."}, status=404)
                return
            try:
                result = engine.analyze_sample_file(mode, file_path, filename)
                self._send_json(result)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

        if path == "/api/train/start":
            task_name = str(payload.get("script", ""))
            params = payload.get("params")
            if not isinstance(params, dict):
                params = None
            ok, msg = engine.start_training(task_name, params=params)
            self._send_json({"ok": ok, "message": msg})
            return

        if path == "/api/train/stop":
            engine.stop_training()
            self._send_json({"ok": True})
            return

        if path == "/api/data/reset":
            ok, msg = engine.reset_user_data()
            self._send_json(
                {
                    "ok": ok,
                    "message": msg,
                    "data_info": srv.paths.get_storage_info(),
                },
                status=200 if ok else 500,
            )
            return

        self.send_error(404)
