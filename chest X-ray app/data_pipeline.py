"""Patient-aware splitting and TensorFlow input pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from labels import PATHOLOGY_LABELS


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def patient_split(
    frame: pd.DataFrame,
    patient_column: str,
    seed: int = 42,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> DatasetSplits:
    if not 0 < validation_fraction + test_fraction < 1:
        raise ValueError("Validation and test fractions must sum to a value between 0 and 1.")
    patients = frame[patient_column].drop_duplicates().to_numpy()
    if len(patients) < 3:
        raise ValueError("At least 3 unique patients are required for train, validation, and test splits.")
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    test_count = max(1, round(len(patients) * test_fraction))
    validation_count = max(1, round(len(patients) * validation_fraction))
    test_patients = set(patients[:test_count])
    validation_patients = set(patients[test_count:test_count + validation_count])
    train_patients = set(patients[test_count + validation_count:])
    if not train_patients:
        raise ValueError("The dataset is too small to create a non-empty training split.")

    train = frame[frame[patient_column].isin(train_patients)].copy()
    validation = frame[frame[patient_column].isin(validation_patients)].copy()
    test = frame[frame[patient_column].isin(test_patients)].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Unable to create non-empty train, validation, and test patient splits.")
    if set(train[patient_column]) & (set(validation[patient_column]) | set(test[patient_column])):
        raise RuntimeError("Patient leakage detected while creating dataset splits.")
    return DatasetSplits(train, validation, test)


def class_weights(frame: pd.DataFrame, labels: tuple[str, ...] = PATHOLOGY_LABELS) -> np.ndarray:
    values = frame.loc[:, labels].astype(float).to_numpy()
    positives = values.sum(axis=0)
    negatives = len(values) - positives
    return np.divide(negatives, np.maximum(positives, 1.0))


def make_dataset(
    frame: pd.DataFrame,
    image_column: str,
    image_size: int,
    batch_size: int,
    training: bool,
    labels: tuple[str, ...] = PATHOLOGY_LABELS,
):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to build the training pipeline.") from exc

    paths = frame[image_column].astype(str).to_numpy()
    targets = frame.loc[:, labels].astype("float32").to_numpy()

    def load(path, target):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [image_size, image_size])
        image = tf.cast(image, tf.float32) / 255.0
        if training:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.08)
        return image, target

    dataset = tf.data.Dataset.from_tensor_slices((paths, targets))
    dataset = dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
