"""Portable application data and model paths."""

from __future__ import annotations

import os
import sys
from pathlib import Path


APP_NAME = "ChestXRayClassifier"


def user_data_dir() -> Path:
    if sys.platform == "win32":
        root = os.environ.get("APPDATA")
        return Path(root) / APP_NAME if root else Path.home() / "AppData" / "Roaming" / APP_NAME
    root = os.environ.get("XDG_DATA_HOME")
    return Path(root) / APP_NAME if root else Path.home() / ".local" / "share" / APP_NAME


def model_dir() -> Path:
    path = user_data_dir() / "models" / "chest_xray"
    path.mkdir(parents=True, exist_ok=True)
    return path


def feedback_db_path() -> Path:
    path = Path(__file__).resolve().parent / "feedback"
    path.mkdir(parents=True, exist_ok=True)
    return path / "feedback.sqlite3"


def prepared_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"
