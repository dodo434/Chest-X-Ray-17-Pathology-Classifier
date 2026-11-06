"""Local clinician feedback storage for active learning."""

from __future__ import annotations

import json
import csv
import sqlite3
import shutil
from pathlib import Path

from paths import feedback_db_path


class FeedbackStore:
    def __init__(self, database_path: str | Path | None = None):
        self.path = Path(database_path) if database_path else feedback_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.path.parent / "feedback"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.images_dir / "labels.csv"
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT NOT NULL,
                    probabilities TEXT NOT NULL,
                    corrected_labels TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    used_for_training INTEGER NOT NULL DEFAULT 0
                )"""
            )

    def add(self, image_path: str, probabilities: list[float], corrected_labels: list[str]) -> int:
        source = Path(image_path)
        with sqlite3.connect(self.path) as connection:
            cursor = connection.execute(
                "INSERT INTO feedback(image_path, probabilities, corrected_labels) VALUES (?, ?, ?)",
                (image_path, json.dumps(probabilities), json.dumps(corrected_labels)),
            )
            record_id = int(cursor.lastrowid)
        destination = self.images_dir / f"{record_id}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if write_header:
                writer.writerow(["id", "image_path", "corrected_labels"])
            writer.writerow([record_id, str(destination), json.dumps(corrected_labels)])
        return record_id

    def pending(self) -> list[tuple[int, str, list[str]]]:
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT id, image_path, corrected_labels FROM feedback WHERE used_for_training = 0"
            ).fetchall()
        return [(int(row[0]), row[1], json.loads(row[2])) for row in rows]

    def mark_used(self, ids: list[int]) -> None:
        if not ids:
            return
        with sqlite3.connect(self.path) as connection:
            connection.executemany("UPDATE feedback SET used_for_training = 1 WHERE id = ?", ((item,) for item in ids))
