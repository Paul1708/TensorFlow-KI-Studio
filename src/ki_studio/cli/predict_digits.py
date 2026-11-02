"""Standalone digit predictor (CLI image evaluation or interactive Tkinter drawing window)."""

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
from ki_studio.core.preprocessing import load_image_file, prepare_digit_image
from ki_studio.paths import DEFAULT_PATHS, StudioPaths


def load_digit_model(paths: StudioPaths = DEFAULT_PATHS) -> tf.keras.Model:
    """Load the active digit model (custom user model if present, else bundled factory weights)."""
    model_path, _, _ = paths.resolve_model_paths(ModelMode.DIGITS)
    if not model_path.exists():
        print("Hinweis: 'zahlen_netz.keras' wurde noch nicht gefunden.")
        print("Bitte führe zuerst 'ki-studio-train-digits' aus, um das Netz zu trainieren!")
        sys.exit(1)
    return tf.keras.models.load_model(model_path)


def predict_digit(model: tf.keras.Model, image_28: np.ndarray) -> tuple[int, float, np.ndarray]:
    """Run inference on a normalized `(28, 28)` image and return `(label, confidence_pct, probs)`."""
    probs = model.predict(np.expand_dims(image_28, axis=0), verbose=0)[0]
    predicted = int(np.argmax(probs))
    confidence = float(probs[predicted]) * 100.0
    return predicted, confidence, probs


def recognize_file_in_terminal(model: tf.keras.Model, image_path: str | Path) -> None:
    """Evaluate a digit image file and print ASCII art and class probabilities."""
    resolved = Path(image_path).expanduser().resolve()
    print(f"Lese Bilddatei:\n  {resolved}\n")

    raw_pixels = load_image_file(resolved)
    image_28, is_empty = prepare_digit_image(raw_pixels)
    if is_empty:
        print("Hinweis: Auf dem Bild konnte keine Zahl erkannt werden (das Bild wirkt leer).")
        return

    predicted, confidence, probs = predict_digit(model, image_28)
    print("So sieht das Neuronale Netz dein Bild (auf 28x28 zentriert):")
    print(render_ascii(image_28))
    print("\n" + "=" * 50)
    print(f" ERGEBNIS: Die KI erkennt eine  >>> {predicted} <<<  ({confidence:.2f} %)")
    print("=" * 50 + "\n")

    print("Wahrscheinlichkeiten für alle Zahlen (0 bis 9):")
    for digit, value in enumerate(probs):
        pct = float(value) * 100.0
        bar = "█" * int(round(pct / 4.0))
        marker = " <-- Treffer!" if digit == predicted else ""
        print(f"  [{digit}]: {pct:6.2f} % | {bar}{marker}")


def launch_digit_window(model: tf.keras.Model, paths: StudioPaths = DEFAULT_PATHS) -> None:
    """Open an interactive Tkinter window for drawing digits or opening image files."""
    import tkinter as tk

    refs: list[Any] = []
    widgets: dict[str, Any] = {}

    def build_panel(parent: Any) -> None:
        result_lbl = tk.Label(
            parent,
            text="Erkannte Zahl: –",
            font=("Segoe UI", 20, "bold"),
            bg="white",
            fg="#2b6cb0",
            width=18,
            anchor="w",
        )
        result_lbl.pack(anchor="w")

        conf_lbl = tk.Label(
            parent,
            text="Sicherheit: Bitte Bild wählen oder zeichnen",
            font=("Segoe UI", 10),
            bg="white",
            fg="#4a5568",
        )
        conf_lbl.pack(anchor="w", pady=(2, 10))

        preview_frame = tk.Frame(parent, bg="white")
        preview_frame.pack(fill="x", pady=(0, 10))
        preview_lbl = tk.Label(preview_frame, bg="#000000", width=84, height=84)
        preview_lbl.pack(side="left", padx=(0, 12))
        tk.Label(
            preview_frame,
            text="28x28 KI-Ansicht\n(automatisch zentriert\n& invertiert)",
            font=("Segoe UI", 9),
            bg="white",
            fg="#718096",
            justify="left",
        ).pack(side="left")

        bars_frame = tk.Frame(parent, bg="white")
        bars_frame.pack(fill="x")
        bar_labels: list[Any] = []
        for digit in range(10):
            lbl = tk.Label(
                bars_frame,
                text=f"{digit}:   0.0 %",
                font=("Consolas", 9),
                bg="white",
                fg="#2d3748",
                anchor="w",
            )
            lbl.pack(fill="x")
            bar_labels.append(lbl)

        widgets["result"] = result_lbl
        widgets["conf"] = conf_lbl
        widgets["preview"] = preview_lbl
        widgets["bars"] = bar_labels

    def evaluate(raw_pixels: np.ndarray, source_name: str) -> np.ndarray:
        image_28, is_empty = prepare_digit_image(raw_pixels)
        photo = array_to_tk_photo(image_28, 84)
        refs.clear()
        refs.append(photo)
        widgets["preview"].configure(image=photo)

        bar_labels = widgets["bars"]
        if is_empty:
            widgets["result"].configure(text="Erkannte Zahl: –", fg="#718096")
            widgets["conf"].configure(text="Das Bild ist noch leer.")
            for d in range(10):
                bar_labels[d].configure(text=f"{d}:   0.0 %", fg="#4a5568", font=("Consolas", 9))
            return image_28

        digit, confidence, probs = predict_digit(model, image_28)
        widgets["result"].configure(text=f"Erkannte Zahl: {digit}", fg="#2b6cb0")
        widgets["conf"].configure(text=f"Sicherheit: {confidence:.2f} % ({source_name})")

        for d, value in enumerate(probs):
            pct = float(value) * 100.0
            blocks = "█" * int(round(pct / 8.0))
            if d == digit:
                bar_labels[d].configure(
                    text=f"{d}: {pct:5.1f} % {blocks}", fg="#2b6cb0", font=("Consolas", 9, "bold")
                )
            else:
                bar_labels[d].configure(
                    text=f"{d}: {pct:5.1f} % {blocks}", fg="#4a5568", font=("Consolas", 9)
                )
        return image_28

    initial_dir = paths.samples_dir if paths.samples_dir.exists() else paths.app_dir
    window = InteractiveDrawingCanvas(
        title="KI Zahlen-Erkenner (TensorFlow)",
        headline="Gib dem Neuronalen Netz ein Bild!",
        subtitle="Wähle eine Bilddatei (.png / .jpg / .bmp) aus ODER zeichne direkt mit der Maus ins weiße Feld:",
        brush_radius=10,
        initial_dir=initial_dir,
        default_save_name="meine_zahl.png",
        on_evaluate=evaluate,
        build_results_panel=build_panel,
    )
    window.run()


def main(argv: list[str] | None = None) -> None:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser(
        prog="ki-studio-digits",
        description="Recognize handwritten digits (0-9) via CLI file or interactive drawing window.",
    )
    parser.add_argument("image", nargs="*", help="Optional path to an image file.")
    args = parser.parse_args(argv)

    model = load_digit_model(DEFAULT_PATHS)
    if args.image:
        image_path = " ".join(args.image).strip().strip('"').strip("'")
        try:
            recognize_file_in_terminal(model, image_path)
        except Exception as exc:
            print(f"\nFEHLER: {exc}")
            sys.exit(1)
        return

    print("Öffne interaktives KI-Fenster (Bilddatei auswählen oder mit der Maus zeichnen)...")
    try:
        launch_digit_window(model, DEFAULT_PATHS)
    except Exception as gui_err:
        print(f"(Fenster konnte nicht geöffnet werden: {gui_err})")
        entered = input("Bitte gib den Pfad zu deiner Bilddatei ein (.png/.jpg): ").strip().strip('"').strip("'")
        if entered:
            recognize_file_in_terminal(model, entered)


if __name__ == "__main__":
    main()
