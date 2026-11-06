"""Training and evaluation orchestration."""

from __future__ import annotations

from pathlib import Path
import shutil

from data_pipeline import class_weights, make_dataset, patient_split
from model import build_model, save_metadata, weighted_binary_cross_entropy
from paths import model_dir


def train(
    frame,
    image_column: str,
    patient_column: str,
    output_directory: str = "models/chest_xray",
    image_size: int = 224,
    batch_size: int = 16,
    epochs: int = 10,
    seed: int = 42,
    progress_callback=None,
    labels=None,
):
    labels = tuple(labels or [column for column in frame.columns if column not in {image_column, patient_column, "label"}])
    splits = patient_split(frame, patient_column, seed=seed)
    model = build_model(image_size=image_size, labels=labels)
    import tensorflow as tf
    model.compile(
        optimizer=model.optimizer,
        loss=weighted_binary_cross_entropy(class_weights(splits.train, labels)),
        metrics=[tf.keras.metrics.AUC(name="auc", multi_label=True, num_labels=len(labels))],
    )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback:
                progress_callback(epoch + 1, epochs, logs or {})

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(output / "best.keras", monitor="val_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True),
        ProgressCallback(),
    ]
    history = model.fit(
        make_dataset(splits.train, image_column, image_size, batch_size, True, labels),
        validation_data=make_dataset(splits.validation, image_column, image_size, batch_size, False, labels),
        epochs=epochs,
        callbacks=callbacks,
    )
    model.save(output / "final.keras")
    save_metadata(output, labels=labels)
    return history.history


def fine_tune_feedback(
    model_directory: str | Path | None = None,
    image_size: int = 224,
    batch_size: int = 8,
    epochs: int = 1,
    progress_callback=None,
) -> dict:
    """Fine-tune the saved model on clinician-confirmed local feedback."""
    import tensorflow as tf
    from feedback import FeedbackStore
    from model import load_metadata

    directory = Path(model_directory) if model_directory else model_dir()
    feedback = FeedbackStore()
    pending = feedback.pending()
    if not pending:
        raise ValueError("No clinician feedback is pending for fine-tuning.")
    metadata = load_metadata(directory)
    labels = tuple(metadata["labels"])
    rows = []
    ids = []
    for item_id, image_path, corrected_labels in pending:
        row = {"image_path": image_path, "patient_id": f"feedback:{item_id}"}
        row.update({label: float(label in corrected_labels) for label in labels})
        rows.append(row)
        ids.append(item_id)
    import pandas as pd
    from dataset_manager import load_folder_dataset
    from paths import prepared_data_dir
    feedback_frame = pd.DataFrame(rows)
    if not prepared_data_dir().is_dir():
        raise ValueError("Prepared original data is required before feedback fine-tuning.")
    original_frame, original_labels = load_folder_dataset(prepared_data_dir())
    if tuple(original_labels) != labels:
        raise ValueError("Feedback labels do not match the prepared training labels.")
    frame = pd.concat([original_frame, feedback_frame], ignore_index=True)
    splits = patient_split(frame, "patient_id")
    old_model = tf.keras.models.load_model(directory / "best.keras", compile=False)
    old_score = old_model.evaluate(
        make_dataset(splits.validation, "image_path", image_size, batch_size, False, labels),
        verbose=0,
        return_dict=True,
    ).get("auc", 0.0)
    model = tf.keras.models.clone_model(old_model)
    model.set_weights(old_model.get_weights())
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=weighted_binary_cross_entropy(class_weights(frame, labels)),
        metrics=[tf.keras.metrics.AUC(name="auc", multi_label=True, num_labels=len(labels))],
    )
    history = model.fit(
        make_dataset(splits.train, "image_path", image_size, batch_size, True, labels),
        validation_data=make_dataset(splits.validation, "image_path", image_size, batch_size, False, labels),
        epochs=epochs,
        callbacks=[],
        verbose=0,
    )
    new_score = model.evaluate(
        make_dataset(splits.validation, "image_path", image_size, batch_size, False, labels),
        verbose=0,
        return_dict=True,
    ).get("auc", 0.0)
    if new_score + 1e-6 < old_score:
        raise ValueError(f"Feedback model was not better on validation AUC ({new_score:.4f} < {old_score:.4f}).")
    shutil.copy2(directory / "best.keras", directory / "best.backup.keras")
    model.save(directory / "best.keras")
    feedback.mark_used(ids)
    if progress_callback:
        progress_callback(epochs, epochs, history.history)
    return history.history


def evaluate(frame, image_column: str, patient_column: str, model_directory: str,
             image_size: int = 224, batch_size: int = 16) -> dict[str, float]:
    """Evaluate the saved checkpoint on the patient-disjoint test split."""
    import tensorflow as tf
    from sklearn.metrics import average_precision_score, roc_auc_score
    from labels import PATHOLOGY_LABELS

    directory = Path(model_directory)
    model = tf.keras.models.load_model(directory / "best.keras", compile=False)
    splits = patient_split(frame, patient_column)
    targets = splits.test.loc[:, PATHOLOGY_LABELS].astype(float).to_numpy()
    predictions = model.predict(
        make_dataset(splits.test, image_column, image_size, batch_size, False),
        verbose=0,
    )
    return {
        "macro_auroc": float(roc_auc_score(targets, predictions, average="macro")),
        "macro_auprc": float(average_precision_score(targets, predictions, average="macro")),
    }
