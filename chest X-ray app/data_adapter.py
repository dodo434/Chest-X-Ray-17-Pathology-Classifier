"""Typed adapter around the repository's existing api.py module."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from labels import PATHOLOGY_LABELS, validate_labels


@dataclass(frozen=True)
class DatasetColumns:
    image: str
    patient: str
    labels: tuple[str, ...] = PATHOLOGY_LABELS


def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    lowered = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise ValueError(f"Missing required dataset column; expected one of {candidates}.")


def load_records(file_path: str = "", api_module: str = "api") -> tuple[pd.DataFrame, DatasetColumns]:
    module = importlib.import_module(api_module)
    loader = getattr(module, "load_dataset", None)
    if not callable(loader):
        raise AttributeError(f"{api_module}.py must expose callable load_dataset().")
    frame = loader(file_path)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("api.load_dataset() must return a pandas.DataFrame.")

    image_column = _find_column(frame, ("image_path", "image", "path", "filename"))
    patient_column = _find_column(frame, ("patient_id", "patient", "subject_id"))
    missing = [label for label in PATHOLOGY_LABELS if label not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing pathology columns: {', '.join(missing)}")
    validate_labels(PATHOLOGY_LABELS)
    return frame.copy(), DatasetColumns(image=image_column, patient=patient_column)


def validate_image_paths(frame: pd.DataFrame, column: str) -> None:
    missing = [str(path) for path in frame[column].dropna() if not Path(path).is_file()]
    if missing:
        sample = ", ".join(missing[:3])
        raise FileNotFoundError(f"{len(missing)} image files are missing; examples: {sample}")
