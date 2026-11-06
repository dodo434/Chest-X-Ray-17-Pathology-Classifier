"""TensorFlow model construction and prediction formatting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from labels import PATHOLOGY_LABELS


def build_model(
    image_size: int = 224,
    learning_rate: float = 1e-4,
    backbone: str = "DenseNet121",
    labels: tuple[str, ...] = PATHOLOGY_LABELS,
):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to build the model.") from exc
    backbone_class = {"DenseNet121": tf.keras.applications.DenseNet121, "EfficientNetB0": tf.keras.applications.EfficientNetB0}.get(backbone)
    if backbone_class is None:
        raise ValueError("backbone must be DenseNet121 or EfficientNetB0.")
    base = backbone_class(include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    features = base(inputs, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    outputs = tf.keras.layers.Dense(len(labels), activation="sigmoid")(features)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name="auc", multi_label=True, num_labels=len(labels))],
    )
    return model


def weighted_binary_cross_entropy(weights: np.ndarray):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for the loss function.") from exc
    weights_tensor = tf.constant(weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        per_label = -(weights_tensor * y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        return tf.reduce_mean(per_label)

    return loss


def save_metadata(
    directory: str | Path,
    thresholds: dict[str, float] | None = None,
    labels: tuple[str, ...] = PATHOLOGY_LABELS,
) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").write_text(json.dumps({
        "labels": labels,
        "thresholds": thresholds or {label: 0.5 for label in labels},
    }, indent=2), encoding="utf-8")


def load_metadata(directory: str | Path) -> dict:
    path = Path(directory) / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Model metadata not found: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not metadata.get("labels"):
        raise ValueError("Model metadata does not contain labels.")
    return metadata


def format_predictions(
    probabilities: np.ndarray,
    thresholds: dict[str, float] | None = None,
    labels: tuple[str, ...] = PATHOLOGY_LABELS,
) -> list[dict[str, object]]:
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    if values.size != len(labels):
        raise ValueError(f"Expected {len(labels)} probabilities, received {values.size}.")
    thresholds = thresholds or {}
    return [
        {"label": label, "probability": float(probability), "confidence": f"{probability:.1%}",
         "threshold": float(thresholds.get(label, 0.5)), "positive": bool(probability >= thresholds.get(label, 0.5))}
        for label, probability in zip(labels, values)
    ]
