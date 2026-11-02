"""Desktop app-window launcher, Win32 taskbar icon setup, and server lifecycle loop."""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
import subprocess
import threading
import time
import webbrowser

from ki_studio.paths import DEFAULT_PATHS, StudioPaths
from ki_studio.server.http_server import StudioHTTPServer

logger = logging.getLogger(__name__)

_WINDOW_TITLE_FRAGMENT = "TensorFlow KI-Studio – Live Gehirn-Visualisierung"


def find_app_browser() -> str | None:
    """Locate Microsoft Edge or Google Chrome on Windows for frameless `--app` window mode."""
    candidates: list[Path] = []

    if os.name == "nt":
        try:
            import winreg

            for browser_exe in ("msedge.exe", "chrome.exe"):
                for root_key in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                    try:
                        with winreg.OpenKey(
                            root_key,
                            rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{browser_exe}",
                        ) as key:
                            val, _ = winreg.QueryValueEx(key, "")
                            if val:
                                candidates.append(Path(val))
                    except OSError:
                        continue
        except OSError as exc:
            logger.debug("Registry browser lookup skipped: %s", exc)

    prog_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    prog = os.environ.get("ProgramFiles", r"C:\Program Files")
    local_app = os.environ.get("LOCALAPPDATA", "")

    candidates.extend(
        [
            Path(prog_x86) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
            Path(prog) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
            Path(prog) / "Google" / "Chrome" / "Application" / "chrome.exe",
            Path(prog_x86) / "Google" / "Chrome" / "Application" / "chrome.exe",
        ]
    )
    if local_app:
        candidates.append(Path(local_app) / "Google" / "Chrome" / "Application" / "chrome.exe")

    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def _apply_win32_taskbar_icon(assets_dir: Path) -> None:
    """Attach `app_icon.ico` (`ICON_BIG` & `ICON_SMALL`) to the opened Edge/Chrome app window."""
    if os.name != "nt":
        return
    ico_path = assets_dir / "app_icon.ico"
    if not ico_path.is_file():
        return

    try:
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        user32.LoadImageW.argtypes = [
            wintypes.HINSTANCE,
            wintypes.LPCWSTR,
            wintypes.UINT,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        user32.LoadImageW.restype = wintypes.HANDLE
        user32.SendMessageTimeoutW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
            wintypes.UINT,
            wintypes.UINT,
            ctypes.POINTER(ctypes.c_size_t),
        ]

        lr_fromfile = 0x0010
        image_icon = 1
        wm_seticon = 0x0080

        h_big = user32.LoadImageW(None, str(ico_path), image_icon, 64, 64, lr_fromfile)
        h_small = user32.LoadImageW(None, str(ico_path), image_icon, 32, 32, lr_fromfile)
        if not h_big and not h_small:
            return

        enum_proc_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        for delay_sec in (0.15, 0.35, 0.7, 1.3, 2.2, 3.5):
            time.sleep(delay_sec)
            matches: list[int] = []

            @enum_proc_type
            def _enum_cb(hwnd: int, _lparam: int) -> bool:
                if user32.IsWindowVisible(hwnd):
                    length = user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buf = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buf, length + 1)
                        if _WINDOW_TITLE_FRAGMENT in buf.value:
                            matches.append(hwnd)
                return True

            user32.EnumWindows(_enum_cb, 0)
            for hwnd in matches:
                dummy = ctypes.c_size_t(0)
                if h_big:
                    user32.SendMessageTimeoutW(hwnd, wm_seticon, 1, h_big, 0x0002, 200, ctypes.byref(dummy))
                if h_small:
                    user32.SendMessageTimeoutW(hwnd, wm_seticon, 0, h_small, 0x0002, 200, ctypes.byref(dummy))
    except Exception as exc:
        logger.debug("Could not apply Win32 window icon: %s", exc)


def open_app_window(url: str, assets_dir: Path) -> None:
    """Open the Studio URL in a frameless Edge/Chrome `--app` window or the default browser."""
    browser_exe = find_app_browser()
    if browser_exe:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            subprocess.Popen(
                [browser_exe, f"--app={url}", "--window-size=1460,920"],
                creationflags=flags,
            )
            threading.Thread(
                target=_apply_win32_taskbar_icon,
                args=(assets_dir,),
                daemon=True,
            ).start()
            return
        except OSError as exc:
            logger.warning("Failed to launch app browser %s: %s", browser_exe, exc)

    webbrowser.open(url)


def _load_engine_in_background(server: StudioHTTPServer) -> None:
    """Import TensorFlow and initialize `BrainEngine` asynchronously while the boot screen shows."""
    try:
        server.paths.write_data_info_file()
        server.update_boot_status(42, "Lade TensorFlow C++-Rechenkerne...")
        from ki_studio.core.engine import BrainEngine

        engine = BrainEngine(
            paths=server.paths,
            progress_cb=lambda pct, msg: server.update_boot_status(pct, msg),
        )
        server.set_engine(engine)
    except ModuleNotFoundError as mod_err:
        missing = getattr(mod_err, "name", "") or str(mod_err)
        hint = (
            f"Python-Paket '{missing}' fehlt! Bitte führe einmalig im Terminal aus:  pip install tensorflow"
        )
        logger.error(hint)
        server.update_boot_status(100, hint, ready=False, error=hint)
    except Exception as exc:
        logger.exception("Failed to initialize BrainEngine")
        server.update_boot_status(100, f"Fehler beim Laden der KI-Engine: {exc}", ready=False, error=str(exc))


def _signal_ready_event() -> None:
    """Signal the native Windows splash launcher that the local HTTP server is accepting connections."""
    ready_evt = os.environ.get("KI_STUDIO_READY_EVENT", "").strip()
    if not ready_evt or os.name != "nt":
        return
    try:
        handle = ctypes.windll.kernel32.OpenEventW(0x0002, False, ready_evt)
        if handle:
            ctypes.windll.kernel32.SetEvent(handle)
            ctypes.windll.kernel32.CloseHandle(handle)
    except OSError as exc:
        logger.debug("Could not signal ready event %s: %s", ready_evt, exc)


def start_server_async(
    *,
    open_window: bool = True,
    host: str = "127.0.0.1",
    port: int = 0,
    paths: StudioPaths = DEFAULT_PATHS,
) -> StudioHTTPServer:
    """Start the local HTTP server immediately and begin loading TensorFlow in the background."""
    server = StudioHTTPServer(engine=None, host=host, port=port, paths=paths)
    bound_port = server.server_address[1]
    url = f"http://{host}:{bound_port}/"

    threading.Thread(target=server.serve_forever, daemon=True).start()
    threading.Thread(target=_load_engine_in_background, args=(server,), daemon=True).start()

    if open_window:
        open_app_window(url, paths.assets_dir)

    _signal_ready_event()
    print(f"KI_STUDIO_READY {url}", flush=True)
    return server


def run_event_loop(server: StudioHTTPServer) -> None:
    """Keep the server running while the UI window sends heartbeats, then shut down cleanly."""
    start_time = time.time()
    try:
        while not server.should_exit:
            time.sleep(0.35)
            now = time.time()

            if not server.client_connected:
                if now - start_time > 60.0:
                    break
                continue

            if server.close_requested_at is not None and (now - server.close_requested_at) >= 2.2:
                break

            if now - server.last_heartbeat > 300.0:
                break

        if server.should_exit:
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        if server.engine is not None:
            server.engine.stop_training()
        server.shutdown()
        server.server_close()
