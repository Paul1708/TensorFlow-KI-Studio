"""Shared Tkinter drawing canvas and image-dialog helpers for standalone predictors."""

from __future__ import annotations

import base64
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from ki_studio.core.preprocessing import load_image_file


def array_to_tk_photo(gray_2d: np.ndarray, target_size: int) -> Any:
    """Convert a normalized 2D grayscale array (`0..1`) into a `tk.PhotoImage`."""
    import tkinter as tk

    resized = tf.image.resize(
        gray_2d[:, :, np.newaxis],
        size=[target_size, target_size],
        method="nearest",
    ).numpy()
    uint8_rgb = np.repeat(np.clip(resized * 255.0, 0, 255).astype(np.uint8), 3, axis=-1)
    png_bytes = tf.io.encode_png(uint8_rgb).numpy()
    b64_data = base64.b64encode(png_bytes)
    return tk.PhotoImage(data=b64_data, format="png")


class InteractiveDrawingCanvas:
    """Reusable Tkinter interactive window pairing a mouse canvas with a custom result panel."""

    def __init__(
        self,
        *,
        title: str,
        headline: str,
        subtitle: str,
        brush_radius: int,
        initial_dir: Path,
        default_save_name: str,
        on_evaluate: Callable[[np.ndarray, str], np.ndarray],
        build_results_panel: Callable[[Any], None],
    ) -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox

        self._tk = tk
        self._filedialog = filedialog
        self._messagebox = messagebox
        self._initial_dir = initial_dir
        self._default_save_name = default_save_name
        self._on_evaluate = on_evaluate

        self.canvas_size = 280
        self.brush_radius = brush_radius
        self.buffer = np.full((self.canvas_size, self.canvas_size), 255.0, dtype=np.float32)
        self._last_pos: list[int | None] = [None, None]

        self.root = tk.Tk()
        self.root.title(title)
        self.root.resizable(False, False)
        self.root.configure(bg="#f4f6f9", padx=18, pady=16)

        tk.Label(
            self.root,
            text=headline,
            font=("Segoe UI", 15, "bold"),
            bg="#f4f6f9",
            fg="#1a202c",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))

        tk.Label(
            self.root,
            text=subtitle,
            font=("Segoe UI", 10),
            bg="#f4f6f9",
            fg="#4a5568",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 12))

        left = tk.Frame(self.root, bg="#f4f6f9")
        left.grid(row=2, column=0, sticky="n", padx=(0, 18))

        self.canvas = tk.Canvas(
            left,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="white",
            highlightthickness=2,
            highlightbackground="#cbd5e0",
            cursor="pencil",
        )
        self.canvas.pack()

        right = tk.Frame(
            self.root,
            bg="white",
            padx=16,
            pady=14,
            highlightthickness=1,
            highlightbackground="#e2e8f0",
        )
        right.grid(row=2, column=1, sticky="ns")
        build_results_panel(right)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        row1 = tk.Frame(left, bg="#f4f6f9")
        row1.pack(fill="x", pady=(10, 4))
        tk.Button(
            row1,
            text="Bilddatei öffnen (.png / .jpg)",
            command=self._open_image_dialog,
            bg="#2b6cb0",
            fg="white",
            activebackground="#2c5282",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            pady=6,
            relief="flat",
            cursor="hand2",
        ).pack(fill="x")

        row2 = tk.Frame(left, bg="#f4f6f9")
        row2.pack(fill="x")
        tk.Button(
            row2,
            text="Erkennen",
            command=lambda: self._on_evaluate(self.buffer, "Aktuelles Feld"),
            bg="#38a169",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            pady=5,
            relief="flat",
            cursor="hand2",
        ).pack(side="left", expand=True, fill="x", padx=(0, 4))

        tk.Button(
            row2,
            text="Feld leeren",
            command=self._clear_canvas,
            bg="#e53e3e",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            pady=5,
            relief="flat",
            cursor="hand2",
        ).pack(side="left", expand=True, fill="x", padx=(0, 4))

        tk.Button(
            row2,
            text="Als PNG speichern",
            command=self._save_png_dialog,
            bg="#718096",
            fg="white",
            font=("Segoe UI", 9),
            pady=5,
            relief="flat",
            cursor="hand2",
        ).pack(side="left", expand=True, fill="x")

    def run(self) -> None:
        """Initialize the preview state and start the Tkinter main event loop."""
        self._on_evaluate(self.buffer, "Start")
        self.root.mainloop()

    def _paint_circle(self, cx: float, cy: float) -> None:
        r = self.brush_radius
        y_min = max(0, int(cy - r))
        y_max = min(self.canvas_size, int(cy + r + 1))
        x_min = max(0, int(cx - r))
        x_max = min(self.canvas_size, int(cx + r + 1))
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
        self.buffer[y_min:y_max, x_min:x_max][mask] = 0.0

    def _on_press(self, event: Any) -> None:
        self._last_pos[0], self._last_pos[1] = event.x, event.y
        r = self.brush_radius
        self.canvas.create_oval(
            event.x - r, event.y - r, event.x + r, event.y + r, fill="black", outline="black"
        )
        self._paint_circle(event.x, event.y)

    def _on_drag(self, event: Any) -> None:
        x0, y0 = self._last_pos[0], self._last_pos[1]
        x1, y1 = event.x, event.y
        if x0 is None or y0 is None:
            x0, y0 = x1, y1

        self.canvas.create_line(
            x0,
            y0,
            x1,
            y1,
            width=self.brush_radius * 2,
            fill="black",
            capstyle=self._tk.ROUND,
            smooth=True,
        )
        steps = max(1, int(math.hypot(x1 - x0, y1 - y0) / 2))
        for i in range(steps + 1):
            t = i / steps
            self._paint_circle(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
        self._last_pos[0], self._last_pos[1] = x1, y1

    def _on_release(self, _event: Any) -> None:
        self._last_pos[0], self._last_pos[1] = None, None
        self._on_evaluate(self.buffer, "Zeichnung")

    def _clear_canvas(self) -> None:
        self.canvas.delete("all")
        self.buffer.fill(255.0)
        self._on_evaluate(self.buffer, "Zeichnung")

    def _open_image_dialog(self) -> None:
        file_path = self._filedialog.askopenfilename(
            title="Bilddatei auswählen",
            initialdir=str(self._initial_dir),
            filetypes=[
                ("Bilddateien", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("Alle Dateien", "*.*"),
            ],
        )
        if not file_path:
            return
        try:
            raw_pixels = load_image_file(file_path)
            prepared = self._on_evaluate(raw_pixels, Path(file_path).name)
            self.canvas.delete("all")
            self.buffer.fill(255.0)

            large_photo = array_to_tk_photo(1.0 - prepared, self.canvas_size)
            self.canvas.image_ref = large_photo  # type: ignore[attr-defined]
            self.canvas.create_image(0, 0, anchor="nw", image=large_photo)

            large_gray = tf.image.resize(
                ((1.0 - prepared) * 255.0)[:, :, np.newaxis],
                size=[self.canvas_size, self.canvas_size],
                method="nearest",
            ).numpy()[:, :, 0]
            self.buffer[:, :] = large_gray
        except Exception as exc:
            self._messagebox.showerror("Fehler beim Laden", str(exc))

    def _save_png_dialog(self) -> None:
        target = self._filedialog.asksaveasfilename(
            title="Zeichnung als PNG speichern",
            initialdir=str(self._initial_dir),
            defaultextension=".png",
            initialfile=self._default_save_name,
            filetypes=[("PNG-Bild", "*.png")],
        )
        if not target:
            return
        uint8_gray = np.clip(self.buffer, 0, 255).astype(np.uint8)[:, :, np.newaxis]
        png_bytes = tf.io.encode_png(uint8_gray).numpy()
        Path(target).write_bytes(png_bytes)
        self._messagebox.showinfo("Gespeichert", f"Dein Bild wurde gespeichert unter:\n{target}")
