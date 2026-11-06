import numpy as np
import pandas as pd
import pytest
from PIL import Image

from dataset_manager import load_folder_dataset, sha256_file
from feedback import FeedbackStore
from data_pipeline import patient_split
from labels import PATHOLOGY_LABELS, validate_labels
from model import format_predictions


def make_frame():
    rows = []
    for patient in range(10):
        row = {"patient_id": f"p{patient}", "image_path": f"{patient}.png"}
        row.update({label: float(patient == 0) for label in PATHOLOGY_LABELS})
        rows.append(row)
    return pd.DataFrame(rows)


def test_labels_are_canonical():
    validate_labels(PATHOLOGY_LABELS)
    with pytest.raises(ValueError):
        validate_labels(tuple(reversed(PATHOLOGY_LABELS)))


def test_patient_split_has_no_overlap():
    splits = patient_split(make_frame(), "patient_id", seed=3)
    groups = [set(split.patient_id) for split in (splits.train, splits.validation, splits.test)]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]


def test_patient_split_rejects_too_few_patients():
    with pytest.raises(ValueError, match="At least 3"):
        patient_split(make_frame().iloc[:2], "patient_id")


def test_prediction_format():
    rows = format_predictions(np.zeros(len(PATHOLOGY_LABELS)))
    assert len(rows) == 17
    assert rows[0]["confidence"] == "0.0%"


def test_prediction_rejects_wrong_output_size():
    with pytest.raises(ValueError):
        format_predictions(np.zeros(16))


def test_folder_dataset_and_feedback(tmp_path):
    class_dir = tmp_path / "files" / "pneumonia"
    class_dir.mkdir(parents=True)
    Image.new("L", (8, 8)).save(class_dir / "one.jpg")
    frame, labels = load_folder_dataset(tmp_path)
    assert labels == ("pneumonia",)
    assert frame.iloc[0]["pneumonia"] == 1
    assert len(sha256_file(class_dir / "one.jpg")) == 64

    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    store.add(str(class_dir / "one.jpg"), [0.9], ["pneumonia"])
    assert len(store.pending()) == 1
