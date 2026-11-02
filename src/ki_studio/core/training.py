"""Dynamic CNN/MLP model builder, training pipeline, and quick console evaluation."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
import json
import math
import random
import sys
import threading
import time
from typing import Any

import numpy as np
import tensorflow as tf

from ki_studio.config import DEFAULT_SEED, ModelMode, TrainingConfig
from ki_studio.core.ascii_art import render_ascii
from ki_studio.core.datasets import (
    DIGIT_IMAGE_SIZE,
    MOOD_NAMES,
    MOOD_SYMBOLS,
    SMILEY_IMAGE_SIZE,
    compute_happiness_percent,
    generate_smiley_dataset,
    load_digits_dataset,
    load_mnist_offline,
)
from ki_studio.paths import DEFAULT_PATHS, StudioPaths

LogFn = Callable[[str, bool], None]
MetricFn = Callable[[dict[str, Any]], None]


def default_log(text: str, overwrite_last: bool = False) -> None:
    """Write a log line to stdout when stdout is available."""
    if sys.stdout is None:
        return
    end = "\r" if overwrite_last else "\n"
    print(text, end=end, flush=True)


class LiveTrainingCallback(tf.keras.callbacks.Callback):
    """Stream batch progress and epoch metrics while honoring cooperative stop requests."""

    def __init__(
        self,
        total_epochs: int,
        steps_per_epoch: int,
        log_fn: LogFn = default_log,
        metric_fn: MetricFn | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        super().__init__()
        self.total_epochs = max(1, int(total_epochs))
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.log_fn = log_fn
        self.metric_fn = metric_fn
        self.stop_event = stop_event
        self._last_update = 0.0

    def _should_stop(self) -> bool:
        if self.stop_event is not None and self.stop_event.is_set():
            self.model.stop_training = True
            return True
        return False

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if self._should_stop():
            return
        self._last_update = 0.0
        self.log_fn(f"Epoche {epoch + 1}/{self.total_epochs}", False)

    def on_train_batch_end(self, batch: int, logs: dict[str, Any] | None = None) -> None:
        if self._should_stop():
            return
        now = time.monotonic()
        current = min(self.steps_per_epoch, int(batch) + 1)
        if current < self.steps_per_epoch and (now - self._last_update) < 0.14:
            return
        self._last_update = now
        logs = logs or {}
        filled = int(round((current / self.steps_per_epoch) * 20))
        bar = "█" * filled + "░" * (20 - filled)
        loss = float(logs.get("loss", 0.0))
        acc = float(logs.get("accuracy", 0.0)) * 100.0
        self.log_fn(
            f"   [{bar}] {current}/{self.steps_per_epoch} Batches • Loss: {loss:.4f} • Training: {acc:.2f} %",
            True,
        )

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if self._should_stop():
            return
        logs = logs or {}
        metric = {
            "epoch": int(epoch + 1),
            "total_epochs": int(self.total_epochs),
            "loss": round(float(logs.get("loss", 0.0)), 4),
            "acc": round(float(logs.get("accuracy", 0.0)) * 100.0, 2),
            "val_loss": round(float(logs.get("val_loss", 0.0)), 4),
            "val_acc": round(float(logs.get("val_accuracy", 0.0)) * 100.0, 2),
        }
        bar = "█" * 20
        self.log_fn(
            f"   [{bar}] {self.steps_per_epoch}/{self.steps_per_epoch} • "
            f"Loss: {metric['loss']:.4f} • Training: {metric['acc']:.2f} % • "
            f"Prüf-Genauigkeit: {metric['val_acc']:.2f} %",
            True,
        )
        self.log_fn("", False)
        if self.metric_fn is not None:
            self.metric_fn(metric)
        else:
            self.log_fn(f"[LABOR_METRIK] {json.dumps(metric)}", False)


def build_optimizer(name: str, learning_rate: float) -> tf.keras.optimizers.Optimizer:
    """Create the configured Keras optimizer."""
    key = name.lower().strip()
    if key == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    if key == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def compute_conv_filters(conv_layers: int, base_filters: int) -> list[int]:
    """Compute filter counts per Conv2D block (e.g. `[32, 64, 64]`)."""
    if conv_layers <= 0:
        return []
    filters = [int(base_filters)]
    for _ in range(1, conv_layers):
        filters.append(min(128, int(base_filters) * 2))
    return filters


def build_classifier_model(
    mode: ModelMode | str,
    config: TrainingConfig,
    log_fn: LogFn = default_log,
) -> tf.keras.Model:
    """Build and compile a CNN or MLP classifier for digits (`28x28`) or smileys (`48x48`)."""
    parsed_mode = ModelMode.parse(mode)
    is_digits = parsed_mode is ModelMode.DIGITS

    image_size = DIGIT_IMAGE_SIZE if is_digits else SMILEY_IMAGE_SIZE
    output_units = 10 if is_digits else 5
    filter_chain = compute_conv_filters(config.conv_layers, config.base_filters)

    arch_label = "CNN" if filter_chain else "MLP ohne Bildfilter"
    log_fn(f"2. Baue das Neuronale Netz ({arch_label}) auf...", False)
    log_fn(
        f"   -> Architektur: Filter={filter_chain or 'Keine'}, "
        f"Denkschichten={config.dense_layers}x{config.hidden_units} ({config.activation.upper()}), "
        f"Dropout={config.dropout * 100:.0f} %",
        False,
    )
    log_fn(f"   -> Lehrer:      {config.optimizer.upper()} mit Lernrate {config.learning_rate:g}\n", False)

    layers: list[tf.keras.layers.Layer] = [
        tf.keras.Input(
            shape=(image_size, image_size),
            name="Eingabe_Bild" if is_digits else "Eingabe_Smiley",
        )
    ]

    if filter_chain:
        layers.append(tf.keras.layers.Reshape((image_size, image_size, 1), name="Bild_Kanal"))
        padding = "same" if (not is_digits or len(filter_chain) >= 3) else "valid"
        smiley_names = ("Gesichts_Filter_1", "Mundwinkel_Filter_2", "Ausdrucks_Filter_3")
        for idx, count in enumerate(filter_chain):
            layer_name = (
                f"Formen_Filter_{idx + 1}"
                if is_digits
                else (smiley_names[idx] if idx < len(smiley_names) else f"Smiley_Filter_{idx + 1}")
            )
            layers.append(
                tf.keras.layers.Conv2D(
                    count,
                    (3, 3),
                    padding=padding,
                    activation=config.activation,
                    name=layer_name,
                )
            )
            layers.append(tf.keras.layers.MaxPooling2D((2, 2), name=f"Verkleinern_{idx + 1}"))

    layers.append(tf.keras.layers.Flatten(name="Flachklopfen"))
    if config.dropout > 0.001:
        layers.append(tf.keras.layers.Dropout(float(config.dropout), name="Auswendiglern_Schutz"))

    for d_idx in range(1, config.dense_layers + 1):
        name = "Denkschicht" if config.dense_layers == 1 else f"Denkschicht_{d_idx}"
        layers.append(tf.keras.layers.Dense(config.hidden_units, activation=config.activation, name=name))

    output_name = "Ausgabe_0_bis_9" if is_digits else "Gluecks_Stufen_0_bis_4"
    layers.append(tf.keras.layers.Dense(output_units, activation="softmax", name=output_name))

    model = tf.keras.Sequential(
        layers,
        name="Zahlen_Erkenner" if is_digits else "Smiley_Gluecks_Erkenner",
    )
    model.compile(
        optimizer=build_optimizer(config.optimizer, config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=lambda line: log_fn(line, False))
    log_fn("", False)
    return model


def _log_digit_examples(model: tf.keras.Model, x_test: np.ndarray, y_test: np.ndarray, log_fn: LogFn) -> None:
    log_fn("5. Live-Test: Wir zeigen dem Netz 3 Beispielbilder aus dem Testset:", False)
    sample_indices = [0, 1, 4]
    predictions = model.predict(x_test[sample_indices], verbose=0)
    for num, idx in enumerate(sample_indices, start=1):
        probs = predictions[num - 1]
        predicted = int(np.argmax(probs))
        confidence = float(np.max(probs)) * 100.0
        actual = int(y_test[idx])
        status = "RICHTIG" if predicted == actual else "FALSCH"
        log_fn(f"--- Beispiel {num} ---", False)
        log_fn(render_ascii(x_test[idx]), False)
        log_fn(
            f"   KI-Vorhersage: {predicted} (Sicherheit: {confidence:.2f} %) "
            f"| Echte Zahl: {actual} -> [{status}]\n",
            False,
        )


def _log_smiley_examples(model: tf.keras.Model, x_test: np.ndarray, y_test: np.ndarray, log_fn: LogFn) -> None:
    log_fn("5. Beispiel-Tests (von Sehr traurig bis Überglücklich):", False)
    for mood_class in (0, 2, 4):
        matches = np.where(y_test == mood_class)[0]
        if len(matches) == 0:
            continue
        idx = int(matches[0])
        probs = model.predict(x_test[idx : idx + 1], verbose=0)[0]
        predicted = int(np.argmax(probs))
        happiness = compute_happiness_percent(probs)
        log_fn(f"--- Beispiel: {MOOD_NAMES[mood_class]} ---", False)
        log_fn(render_ascii(x_test[idx], max_size=24), False)
        log_fn(
            f"   KI-Urteil: {MOOD_NAMES[predicted]} {MOOD_SYMBOLS[predicted]} "
            f"| Glücks-Level: {happiness:.1f} %!\n",
            False,
        )


def _build_metadata(
    mode: ModelMode,
    config: TrainingConfig,
    model: tf.keras.Model,
    train_samples: int,
    test_accuracy: float,
    test_loss: float,
    history_dict: dict[str, list[float]],
) -> dict[str, Any]:
    filter_chain = compute_conv_filters(config.conv_layers, config.base_filters)
    hidden_chain = [config.hidden_units] * config.dense_layers
    meta: dict[str, Any] = {
        "mode": mode.value,
        "is_factory": False,
        "arch_type": "CNN" if filter_chain else "MLP",
        "conv_layers": filter_chain,
        "base_filters": config.base_filters,
        "hidden_layers": hidden_chain,
        "dense_layers": config.dense_layers,
        "hidden_units": config.hidden_units,
        "output_units": 10 if mode is ModelMode.DIGITS else 5,
        "activation": config.activation,
        "optimizer": config.optimizer,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "dropout": round(config.dropout, 3),
        "epochs": config.epochs,
        "train_samples": int(train_samples),
        "total_params": int(model.count_params()),
        "test_accuracy": round(float(test_accuracy) * 100.0, 2),
        "test_loss": round(float(test_loss), 4),
        "trained_at": datetime.now().strftime("%d.%m.%Y, %H:%M:%S"),
        "history": {
            "loss": [round(float(v), 4) for v in history_dict.get("loss", [])],
            "acc": [round(float(v) * 100.0, 2) for v in history_dict.get("accuracy", [])],
            "val_loss": [round(float(v), 4) for v in history_dict.get("val_loss", [])],
            "val_acc": [round(float(v) * 100.0, 2) for v in history_dict.get("val_accuracy", [])],
        },
    }
    if mode is ModelMode.DIGITS:
        meta["data_fraction"] = round(config.data_fraction, 3)
    else:
        meta["samples_per_class"] = config.samples_per_class
    return meta


def _run_training_pipeline(
    mode: ModelMode,
    config: TrainingConfig,
    paths: StudioPaths,
    log_fn: LogFn,
    metric_fn: MetricFn | None,
    stop_event: threading.Event | None,
) -> bool:
    """Shared end-to-end training, evaluation, and model saving workflow."""
    tf.keras.utils.set_random_seed(DEFAULT_SEED)
    is_digits = mode is ModelMode.DIGITS

    header = (
        f" Starte TensorFlow Zahlen-Netz Training (TensorFlow v{tf.__version__})"
        if is_digits
        else f" Starte Training: Smiley-Glücks-Netz (TensorFlow v{tf.__version__})"
    )
    log_fn("=" * 68, False)
    log_fn(header, False)
    log_fn("=" * 68 + "\n", False)

    if is_digits:
        (x_train, y_train), (x_test, y_test) = load_digits_dataset(
            data_fraction=config.data_fraction,
            paths=paths,
            log_fn=log_fn,
        )
    else:
        (x_train, y_train), (x_test, y_test) = generate_smiley_dataset(
            samples_per_class_train=config.samples_per_class,
            samples_per_class_test=400,
            log_fn=log_fn,
            stop_event=stop_event,
        )

    if stop_event is not None and stop_event.is_set():
        return False

    model = build_classifier_model(mode, config, log_fn=log_fn)

    train_count = int(round(x_train.shape[0] * 0.9))
    steps_per_epoch = max(1, int(math.ceil(train_count / config.batch_size)))
    log_fn(f"3. Starte das Training ({config.epochs} Epochen, Batch-Größe {config.batch_size})...", False)

    history = model.fit(
        x_train,
        y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_split=0.1,
        callbacks=[
            LiveTrainingCallback(
                total_epochs=config.epochs,
                steps_per_epoch=steps_per_epoch,
                log_fn=log_fn,
                metric_fn=metric_fn,
                stop_event=stop_event,
            )
        ],
        verbose=0,
    )
    log_fn("", False)

    if stop_event is not None and stop_event.is_set():
        return False

    log_fn(f"4. Abschlussprüfung mit {len(x_test)} unbekannten Testbildern...", False)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    log_fn(f"   -> Fehlerwert (Loss):          {test_loss:.4f}", False)
    log_fn(f"   -> Trefferquote (Genauigkeit): {test_accuracy * 100:.2f} %\n", False)

    if is_digits:
        _log_digit_examples(model, x_test, y_test, log_fn)
    else:
        _log_smiley_examples(model, x_test, y_test, log_fn)

    if stop_event is not None and stop_event.is_set():
        return False

    model_path, meta_path = paths.prepare_training_paths(mode)
    model.save(model_path)
    meta = _build_metadata(
        mode=mode,
        config=config,
        model=model,
        train_samples=int(x_train.shape[0]),
        test_accuracy=float(test_accuracy),
        test_loss=float(test_loss),
        history_dict=history.history,
    )
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    log_fn(f"6. Fertig! Dein eigenes Netz ({meta['arch_type']}) wurde sichtbar gespeichert unter:", False)
    log_fn(f"   {model_path}", False)
    return True


def train_digits_model(
    config: TrainingConfig | None = None,
    paths: StudioPaths = DEFAULT_PATHS,
    log_fn: LogFn = default_log,
    metric_fn: MetricFn | None = None,
    stop_event: threading.Event | None = None,
) -> bool:
    """Train a digit (0-9) classifier on offline MNIST and save it to `KI_Studio_Daten`."""
    resolved_cfg = config or TrainingConfig.from_dict(ModelMode.DIGITS)
    return _run_training_pipeline(
        mode=ModelMode.DIGITS,
        config=resolved_cfg,
        paths=paths,
        log_fn=log_fn,
        metric_fn=metric_fn,
        stop_event=stop_event,
    )


def train_smiley_model(
    config: TrainingConfig | None = None,
    paths: StudioPaths = DEFAULT_PATHS,
    log_fn: LogFn = default_log,
    metric_fn: MetricFn | None = None,
    stop_event: threading.Event | None = None,
) -> bool:
    """Train a 5-level smiley mood classifier on synthetic drawings and save it to `KI_Studio_Daten`."""
    resolved_cfg = config or TrainingConfig.from_dict(ModelMode.SMILEY)
    return _run_training_pipeline(
        mode=ModelMode.SMILEY,
        config=resolved_cfg,
        paths=paths,
        log_fn=log_fn,
        metric_fn=metric_fn,
        stop_event=stop_event,
    )


def run_quick_digit_test(
    paths: StudioPaths = DEFAULT_PATHS,
    log_fn: LogFn = default_log,
) -> bool:
    """Load the active digit model and evaluate one random offline MNIST test sample."""
    model_path, _, is_custom = paths.resolve_model_paths(ModelMode.DIGITS)
    if not model_path.exists():
        log_fn("Fehler: 'zahlen_netz.keras' wurde nicht gefunden!", False)
        return False

    source_desc = "Eigenes Modell (KI_Studio_Daten)" if is_custom else "Integriertes Werks-Modell"
    log_fn(f"Lade KI-Gehirn ({source_desc}):\n  {model_path}\n", False)
    model = tf.keras.models.load_model(model_path)

    _, (x_test, y_test) = load_mnist_offline(paths)
    x_test = x_test.astype("float32") / 255.0

    index = random.randint(0, len(x_test) - 1)
    sample = x_test[index]
    actual = int(y_test[index])

    probabilities = model.predict(np.expand_dims(sample, axis=0), verbose=0)[0]
    predicted = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities)) * 100.0

    log_fn(f"Zufälliges Testbild #{index}:", False)
    log_fn(render_ascii(sample), False)
    log_fn(f"\n-> KI sagt:     {predicted} (mit {confidence:.2f} % Sicherheit)", False)
    log_fn(f"-> Echte Zahl:  {actual}", False)
    return True
