"""Background Qt workers for inference and training."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from model import format_predictions, load_metadata
from image_validation import validate_prediction, validate_upload


try:
    from PySide6.QtCore import QObject, Signal, Slot
except ImportError:
    QObject = object
    Signal = lambda *args: None
    Slot = lambda function: function


class PredictionWorker(QObject):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, model_path: str, image_path: str):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path

    @Slot()
    def run(self):
        try:
            import tensorflow as tf
            validate_upload(self.image_path)
            metadata = load_metadata(Path(self.model_path).parent)
            model = tf.keras.models.load_model(self.model_path, compile=False)
            image = tf.keras.utils.load_img(self.image_path, target_size=(224, 224), color_mode="rgb")
            array = tf.keras.utils.img_to_array(image) / 255.0
            probabilities = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
            validate_prediction(probabilities)
            self.finished.emit(format_predictions(
                probabilities,
                metadata["thresholds"],
                tuple(metadata["labels"]),
            ))
        except Exception as exc:
            self.error.emit(str(exc))


class TrainingWorker(QObject):
    finished = Signal(object)
    progress = Signal(int, int, object)
    error = Signal(str)

    def __init__(self, output_directory: str, epochs: int, archive_path: str | None = None, fine_tune: bool = False):
        super().__init__()
        self.output_directory = output_directory
        self.epochs = epochs
        self.archive_path = archive_path
        self.fine_tune = fine_tune

    @Slot()
    def run(self):
        try:
            from training import fine_tune_feedback, train
            if self.fine_tune:
                history = fine_tune_feedback(
                    self.output_directory,
                    epochs=self.epochs,
                    progress_callback=lambda epoch, total, logs: self.progress.emit(epoch, total, logs),
                )
            else:
                from dataset_manager import load_folder_dataset
                from paths import user_data_dir
                if not self.archive_path:
                    raise ValueError("Prepare the dataset first with scripts/prepare_data.py.")
                prepared = Path(self.archive_path)
                frame, labels = load_folder_dataset(prepared)
                history = train(
                    frame,
                    "image_path",
                    "patient_id",
                    self.output_directory,
                    epochs=self.epochs,
                    labels=labels,
                    progress_callback=lambda epoch, total, logs: self.progress.emit(epoch, total, logs),
                )
            self.finished.emit(history)
        except Exception as exc:
            self.error.emit(str(exc))
