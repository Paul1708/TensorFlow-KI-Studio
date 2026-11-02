"""Standalone smiley mood predictor (CLI image evaluation or interactive drawing window)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import tensorflow as tf

from ki_studio.cli.tk_canvas import InteractiveDrawingCanvas, array_to_tk_photo
from ki_studio.config import ModelMode, configure_stdio_utf8
from ki_studio.core.ascii_art import render_ascii
from ki_studio.core.datasets import MOOD_NAMES, MOOD_SYMBOLS, compute_happiness_percent
from ki_studio.core.preprocessing import load_image_file, prepare_smiley_image
from ki_studio.paths import DEFAULT_PATHS, StudioPaths


def load_smiley_model(paths: StudioPaths = DEFAULT_PATHS) -> tf.keras.Model:
    """Load the active smiley model (custom user model if present, else bundled factory weights)."""
    model_path, _, _ = paths.resolve_model_paths(ModelMode.SMILEY)
    if not model_path.exists():
        print("Hinweis: 'smiley_netz.keras' wurde noch nicht gefunden.")
        print("Bitte führe zuerst 'ki-studio-train-smiley' aus, um das Smiley-Netz zu trainieren!")
        sys.exit(1)
    return tf.keras.models.load_model(model_path)


def predict_smiley(
    model: tf.keras.Model, image_48: np.ndarray
) -> tuple[int, float, float, np.ndarray]:
    """Run inference on a `(48, 48)` smiley and return `(class_idx, conf_pct, happiness_pct, probs)`."""
    probs = model.predict(np.expand_dims(image_48, axis=0), verbose=0)[0]
    mood_idx = int(np.argmax(probs))
    confidence = float(probs[mood_idx]) * 100.0
    happiness = compute_happiness_percent(probs)
    return mood_idx, confidence, happiness, probs


def _mood_color(happiness: float) -> str:
    if happiness < 20.0:
        return "#c53030"
    if happiness < 40.0:
        return "#dd6b20"
    if happiness < 60.0:
        return "#b7791f"
    if happiness < 82.0:
        return "#2f855a"
    return "#276749"


def recognize_smiley_in_terminal(model: tf.keras.Model, image_path: str | Path) -> None:
    """Evaluate a smiley image file and print mood, happiness bar, and class probabilities."""
    resolved = Path(image_path).expanduser().resolve()
    print(f"Lese Smiley-Bilddatei:\n  {resolved}\n")

    raw_pixels = load_image_file(resolved)
    image_48, is_empty = prepare_smiley_image(raw_pixels)
    if is_empty:
        print("Hinweis: Auf dem Bild konnte kein Smiley erkannt werden (das Bild wirkt leer).")
        return

    idx, confidence, happiness, probs = predict_smiley(model, image_48)
    mood = MOOD_NAMES[idx]
    symbol = MOOD_SYMBOLS[idx]

    print("So sieht das Neuronale Netz deinen Smiley (48x48):")
    print(render_ascii(image_48, max_size=24))

    bar_width = 30
    filled = int(round((happiness / 100.0) * bar_width))
    bar = "█" * filled + "░" * (bar_width - filled)

    print("\n" + "=" * 58)
    print(f" STIMMUNG:      >>> {mood} {symbol} <<<  (Sicherheit: {confidence:.1f} %)")
    print(f" GLÜCKS-LEVEL:  [{bar}] {happiness:.1f} %")
    print("=" * 58 + "\n")

    print("Detail-Wahrscheinlichkeiten der 5 Gefühls-Stufen:")
    for level in range(5):
        pct = float(probs[level]) * 100.0
        blocks = "█" * int(round(pct / 4.0))
        marker = " <-- Treffer!" if level == idx else ""
        label = f"{MOOD_NAMES[level]} ({MOOD_SYMBOLS[level]})"
        print(f"  {label:<21}: {pct:6.2f} % | {blocks}{marker}")


def launch_smiley_window(model: tf.keras.Model, paths: StudioPaths = DEFAULT_PATHS) -> None:
    """Open an interactive Tkinter window for drawing smileys or opening image files."""
    import tkinter as tk

    refs: list[Any] = []
    widgets: dict[str, Any] = {}

    def build_panel(parent: Any) -> None:
        mood_lbl = tk.Label(
            parent,
            text="Stimmung: –",
            font=("Segoe UI", 18, "bold"),
            bg="white",
            fg="#2b6cb0",
            width=22,
            anchor="w",
        )
        mood_lbl.pack(anchor="w")

        happy_lbl = tk.Label(
            parent,
            text="Glücks-Level: Bitte Smiley zeichnen",
            font=("Segoe UI", 11, "bold"),
            bg="white",
            fg="#4a5568",
        )
        happy_lbl.pack(anchor="w", pady=(4, 6))

        bar_width, bar_height = 290, 20
        meter = tk.Canvas(
            parent,
            width=bar_width,
            height=bar_height,
            bg="#edf2f7",
            highlightthickness=1,
            highlightbackground="#cbd5e0",
        )
        meter.pack(anchor="w", pady=(0, 12))

        preview_frame = tk.Frame(parent, bg="white")
        preview_frame.pack(fill="x", pady=(0, 12))
        preview_lbl = tk.Label(preview_frame, bg="#000000", width=96, height=96)
        preview_lbl.pack(side="left", padx=(0, 12))
        tk.Label(
            preview_frame,
            text="48x48 KI-Ansicht\n(automatisch ausgeschnitten\n& zentriert)",
            font=("Segoe UI", 9),
            bg="white",
            fg="#718096",
            justify="left",
        ).pack(side="left")

        bars_frame = tk.Frame(parent, bg="white")
        bars_frame.pack(fill="x")
        bar_labels: list[Any] = []
        for level in range(5):
            name = f"{MOOD_NAMES[level]} {MOOD_SYMBOLS[level]}"
            lbl = tk.Label(
                bars_frame,
                text=f"{name:<18}:   0.0 %",
                font=("Consolas", 9),
                bg="white",
                fg="#4a5568",
                anchor="w",
            )
            lbl.pack(fill="x", pady=1)
            bar_labels.append(lbl)

        widgets.update(
            mood=mood_lbl,
            happy=happy_lbl,
            meter=meter,
            bar_width=bar_width,
            bar_height=bar_height,
            preview=preview_lbl,
            bars=bar_labels,
        )

    def update_meter(happiness: float | None, color: str) -> None:
        meter = widgets["meter"]
        w, h = widgets["bar_width"], widgets["bar_height"]
        meter.delete("all")
        if happiness is None:
            return
        fill_w = max(2, int(round((happiness / 100.0) * w)))
        meter.create_rectangle(0, 0, fill_w, h, fill=color, outline="")
        meter.create_line(w // 2, 0, w // 2, h, fill="#a0aec0", dash=(2, 2))

    def evaluate(raw_pixels: np.ndarray, source_name: str) -> np.ndarray:
        image_48, is_empty = prepare_smiley_image(raw_pixels)
        photo = array_to_tk_photo(image_48, 96)
        refs.clear()
        refs.append(photo)
        widgets["preview"].configure(image=photo)

        bar_labels = widgets["bars"]
        if is_empty:
            widgets["mood"].configure(text="Stimmung: –", fg="#718096")
            widgets["happy"].configure(text="Glücks-Level: Bitte Smiley zeichnen", fg="#718096")
            update_meter(None, "#cbd5e0")
            for s in range(5):
                name = f"{MOOD_NAMES[s]} {MOOD_SYMBOLS[s]}"
                bar_labels[s].configure(text=f"{name:<18}:   0.0 %", fg="#4a5568", font=("Consolas", 9))
            return image_48

        idx, confidence, happiness, probs = predict_smiley(model, image_48)
        color = _mood_color(happiness)
        widgets["mood"].configure(text=f"{MOOD_NAMES[idx]}  {MOOD_SYMBOLS[idx]}", fg=color)
        widgets["happy"].configure(
            text=f"Glücks-Level: {happiness:.1f} %   (Sicherheit: {confidence:.0f} % | {source_name})",
            fg=color,
        )
        update_meter(happiness, color)

        for s, value in enumerate(probs):
            pct = float(value) * 100.0
            blocks = "█" * int(round(pct / 10.0))
            name = f"{MOOD_NAMES[s]} {MOOD_SYMBOLS[s]}"
            if s == idx:
                bar_labels[s].configure(
                    text=f"{name:<18}: {pct:5.1f} % {blocks}", fg=color, font=("Consolas", 9, "bold")
                )
            else:
                bar_labels[s].configure(
                    text=f"{name:<18}: {pct:5.1f} % {blocks}", fg="#4a5568", font=("Consolas", 9)
                )
        return image_48

    initial_dir = paths.samples_dir if paths.samples_dir.exists() else paths.app_dir
    window = InteractiveDrawingCanvas(
        title="KI Smiley-Glücks-Erkenner (TensorFlow)",
        headline="Wie glücklich ist dein Smiley?",
        subtitle="Zeichne einen Smiley mit der Maus (mit oder ohne Kopfkreis) ODER lade eine Bilddatei:",
        brush_radius=6,
        initial_dir=initial_dir,
        default_save_name="mein_smiley.png",
        on_evaluate=evaluate,
        build_results_panel=build_panel,
    )
    window.run()


def main(argv: list[str] | None = None) -> None:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser(
        prog="ki-studio-smiley",
        description="Classify smiley mood (0-100% happiness) via CLI file or interactive drawing window.",
    )
    parser.add_argument("image", nargs="*", help="Optional path to a smiley image file.")
    args = parser.parse_args(argv)

    model = load_smiley_model(DEFAULT_PATHS)
    if args.image:
        image_path = " ".join(args.image).strip().strip('"').strip("'")
        try:
            recognize_smiley_in_terminal(model, image_path)
        except Exception as exc:
            print(f"\nFEHLER: {exc}")
            sys.exit(1)
        return

    print("Öffne interaktives Smiley-Glücks-Fenster...")
    try:
        launch_smiley_window(model, DEFAULT_PATHS)
    except Exception as gui_err:
        print(f"(Fenster konnte nicht geöffnet werden: {gui_err})")
        entered = input("Bitte gib den Pfad zu deinem Smiley-Bild ein (.png/.jpg): ").strip().strip('"').strip("'")
        if entered:
            recognize_smiley_in_terminal(model, entered)


if __name__ == "__main__":
    main()
