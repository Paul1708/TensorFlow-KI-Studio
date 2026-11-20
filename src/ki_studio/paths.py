"""Centralized filesystem paths and transparent user-data management."""

from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ki_studio.config import ModelMode, configure_runtime_env

logger = logging.getLogger(__name__)

DATA_DIR_NAME = "KI_Studio_Daten"
CUSTOM_MODELS_DIR_NAME = "Eigene_Modelle"
RUNTIME_DIR_NAME = "Laufzeit"
DELETE_FLAG_NAME = "_delete_all_on_exit.flag"
INFO_FILE_NAME = "INFO_KI_STUDIO_DATEN.txt"


def _model_filenames(mode: ModelMode) -> tuple[str, str]:
    if mode is ModelMode.DIGITS:
        return "zahlen_netz.keras", "zahlen_netz_meta.json"
    return "smiley_netz.keras", "smiley_netz_meta.json"


@dataclass(frozen=True)
class StudioPaths:
    """Resolved directories for bundled resources, static web assets, and user training data."""

    bundle_dir: Path
    app_dir: Path
    web_dir: Path
    is_exe: bool = False

    @classmethod
    def detect(cls) -> StudioPaths:
        """Detect repository, frozen bundle, and user-data locations across dev and EXE modes."""
        is_exe = bool(getattr(sys, "frozen", False)) or os.environ.get("KI_STUDIO_IS_EXE") == "1"

        bundle_override = os.environ.get("KI_STUDIO_BUNDLE_DIR", "").strip()
        meipass = getattr(sys, "_MEIPASS", None)
        package_root = Path(__file__).resolve().parents[2]
        if bundle_override:
            bundle_dir = Path(bundle_override).resolve()
        elif meipass:
            bundle_dir = Path(meipass).resolve()
        else:
            bundle_dir = package_root

        app_override = os.environ.get("KI_STUDIO_APP_DIR", "").strip()
        if app_override:
            app_dir = Path(app_override).resolve()
        elif bundle_dir.name == RUNTIME_DIR_NAME and bundle_dir.parent.name == DATA_DIR_NAME:
            app_dir = bundle_dir.parent.parent
        elif getattr(sys, "frozen", False):
            exe_parent = Path(sys.executable).resolve().parent
            if exe_parent.name == RUNTIME_DIR_NAME and exe_parent.parent.name == DATA_DIR_NAME:
                app_dir = exe_parent.parent.parent
            else:
                app_dir = exe_parent
        else:
            app_dir = bundle_dir

        web_dir = Path(__file__).resolve().parent / "web"
        return cls(bundle_dir=bundle_dir, app_dir=app_dir, web_dir=web_dir, is_exe=is_exe)

    @property
    def assets_dir(self) -> Path:
        return self.bundle_dir / "assets"

    @property
    def models_dir(self) -> Path:
        candidate = self.bundle_dir / "models"
        if candidate.exists():
            return candidate
        return self.bundle_dir

    @property
    def samples_dir(self) -> Path:
        candidate = self.bundle_dir / "samples"
        if candidate.exists():
            return candidate
        return self.bundle_dir / "beispiel_bilder"

    @property
    def static_dir(self) -> Path:
        return self.web_dir / "static"

    @property
    def data_dir(self) -> Path:
        return self.app_dir / DATA_DIR_NAME

    @property
    def custom_models_dir(self) -> Path:
        return self.data_dir / CUSTOM_MODELS_DIR_NAME

    @property
    def runtime_dir(self) -> Path:
        return self.data_dir / RUNTIME_DIR_NAME

    @property
    def delete_on_exit_flag(self) -> Path:
        return self.data_dir / DELETE_FLAG_NAME

    def resolve_model_paths(self, mode: ModelMode | str) -> tuple[Path, Path, bool]:
        """
        Resolve `(model_path, meta_path, is_custom)` for a model mode.

        Custom user-trained models in `KI_Studio_Daten/Eigene_Modelle` take precedence over
        the bundled factory models in `models/`.
        """
        parsed_mode = ModelMode.parse(mode)
        model_name, meta_name = _model_filenames(parsed_mode)

        for custom_base in (self.custom_models_dir, self.data_dir):
            custom_model = custom_base / model_name
            custom_meta = custom_base / meta_name
            if custom_model.exists():
                return custom_model, custom_meta, True

        factory_model = self.models_dir / model_name
        factory_meta = self.models_dir / meta_name
        if not factory_meta.exists():
            factory_meta = self.assets_dir / meta_name
        return factory_model, factory_meta, False

    def write_data_info_file(self) -> None:
        """Write an explanatory README note inside `KI_Studio_Daten` if that folder exists."""
        if not self.data_dir.exists():
            return
        info_file = self.data_dir / INFO_FILE_NAME
        if info_file.exists():
            return
        try:
            info_file.write_text(
                "TensorFlow KI-Studio – Sichtbarer Daten- & Schnellstart-Ordner\n"
                "==============================================================\n"
                "Die Anwendung legt KEINE versteckten Dateien auf deinem PC an.\n"
                "Alles liegt transparent hier in diesem einzigen Ordner direkt neben der App:\n\n"
                "  1. 'Laufzeit'       -> Schlanke App-Ressourcen "
                "(UI, Werks-Modelle & Offline-MNIST)\n"
                "                         für sofortigen Start im gepackten EXE-Modus.\n"
                "  2. 'Eigene_Modelle' -> Nur vorhanden, wenn du im KI-Labor selbst ein Netz\n"
                "                         trainiert hast.\n\n"
                "Du kannst eigene Modelle jederzeit in der App per 1-Klick löschen oder diesen\n"
                "gesamten Ordner 'KI_Studio_Daten' entfernen.\n",
                encoding="utf-8",
            )
        except OSError as exc:
            logger.debug("Could not write info file %s: %s", info_file, exc)

    def prepare_training_paths(self, mode: ModelMode | str) -> tuple[Path, Path]:
        """Ensure `KI_Studio_Daten/Eigene_Modelle` exists and return `(model_path, meta_path)`."""
        parsed_mode = ModelMode.parse(mode)
        model_name, meta_name = _model_filenames(parsed_mode)
        self.custom_models_dir.mkdir(parents=True, exist_ok=True)
        self.write_data_info_file()
        return self.custom_models_dir / model_name, self.custom_models_dir / meta_name

    def get_storage_info(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of custom user models and runtime cache state."""
        folder_exists = self.data_dir.exists() and self.data_dir.is_dir()
        custom_files: list[dict[str, Any]] = []
        custom_bytes = 0

        if self.custom_models_dir.exists() and self.custom_models_dir.is_dir():
            for entry in sorted(self.custom_models_dir.iterdir()):
                if entry.is_file():
                    size = entry.stat().st_size
                    custom_bytes += size
                    custom_files.append({"name": entry.name, "size_kb": round(size / 1024.0, 1)})

        if folder_exists:
            for name in (
                "zahlen_netz.keras",
                "zahlen_netz_meta.json",
                "smiley_netz.keras",
                "smiley_netz_meta.json",
            ):
                legacy_file = self.data_dir / name
                if legacy_file.exists() and legacy_file.is_file():
                    size = legacy_file.stat().st_size
                    custom_bytes += size
                    custom_files.append(
                        {"name": legacy_file.name, "size_kb": round(size / 1024.0, 1)}
                    )

        runtime_exists = self.runtime_dir.exists() and self.runtime_dir.is_dir()
        return {
            "is_exe": self.is_exe,
            "app_dir": str(self.app_dir),
            "data_dir": str(self.data_dir),
            "models_dir": str(self.custom_models_dir),
            "data_dir_name": DATA_DIR_NAME,
            "exists": len(custom_files) > 0,
            "folder_exists": folder_exists,
            "runtime_cache_exists": runtime_exists,
            "files": custom_files,
            "total_bytes": custom_bytes,
            "total_kb": round(custom_bytes / 1024.0, 1),
            "total_mb": round(custom_bytes / (1024.0 * 1024.0), 2),
            "custom_zahlen": (self.custom_models_dir / "zahlen_netz.keras").exists()
            or (self.data_dir / "zahlen_netz.keras").exists(),
            "custom_smiley": (self.custom_models_dir / "smiley_netz.keras").exists()
            or (self.data_dir / "smiley_netz.keras").exists(),
        }

    def _is_safe_data_dir(self) -> bool:
        return self.data_dir.name == DATA_DIR_NAME and self.data_dir.parent == self.app_dir

    def delete_custom_models(self) -> tuple[bool, str]:
        """Remove all custom trained models so factory weights become active again."""
        try:
            if not self._is_safe_data_dir():
                return False, "Sicherheitsprüfung des Datenpfads fehlgeschlagen."

            if self.custom_models_dir.exists():
                shutil.rmtree(self.custom_models_dir)

            for name in (
                "zahlen_netz.keras",
                "zahlen_netz_meta.json",
                "smiley_netz.keras",
                "smiley_netz_meta.json",
            ):
                legacy_file = self.data_dir / name
                if legacy_file.exists():
                    legacy_file.unlink()

            if self.data_dir.exists() and not self.runtime_dir.exists():
                shutil.rmtree(self.data_dir, ignore_errors=True)

            return True, (
                "Alle eigenen Trainingsdaten wurden gelöscht. Werks-Gehirne sind wieder aktiv!"
            )
        except OSError as exc:
            return False, f"Fehler beim Löschen der eigenen Modelle: {exc}"

    def schedule_full_cleanup(self) -> tuple[bool, str]:
        """Delete `KI_Studio_Daten` immediately or mark it for cleanup on EXE exit."""
        try:
            if not self._is_safe_data_dir():
                return False, "Sicherheitsprüfung des Datenpfads fehlgeschlagen."
            if not self.data_dir.exists():
                return True, "Es ist kein Datenordner vorhanden."
            if not self.runtime_dir.exists():
                shutil.rmtree(self.data_dir, ignore_errors=True)
                return True, "Datenordner wurde vollständig entfernt."

            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.delete_on_exit_flag.write_text("DELETE_ALL_ON_EXIT\n", encoding="utf-8")
            return True, (
                "App wird geschlossen und der gesamte Ordner 'KI_Studio_Daten' "
                "wird restlos entfernt."
            )
        except OSError as exc:
            return False, f"Fehler beim Vorbereiten der Komplettlöschung: {exc}"


DEFAULT_PATHS = StudioPaths.detect()
configure_runtime_env(DEFAULT_PATHS.assets_dir)
