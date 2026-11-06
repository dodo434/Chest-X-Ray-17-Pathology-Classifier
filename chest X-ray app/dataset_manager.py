"""Verified local dataset archive management and folder-dataset loading."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_verified_archive(
    archive_path: str | Path,
    destination: str | Path,
    expected_sha256: str | None = None,
    overwrite: bool = False,
) -> Path:
    archive = Path(archive_path).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {archive}")
    if expected_sha256 and sha256_file(archive).lower() != expected_sha256.lower():
        raise ValueError("Dataset archive checksum verification failed.")
    target = Path(destination).expanduser().resolve()
    if target.exists() and overwrite:
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        root = target.resolve()
        for member in zipped.infolist():
            resolved = (target / member.filename).resolve()
            if root != resolved and root not in resolved.parents:
                raise ValueError(f"Unsafe archive member path: {member.filename}")
        zipped.extractall(target)
    return target


def load_folder_dataset(root: str | Path) -> tuple[pd.DataFrame, tuple[str, ...]]:
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {base}")
    candidates = [base / "files", base]
    data_root = next((path for path in candidates if path.is_dir()), None)
    if data_root is None:
        raise ValueError(f"No class-folder dataset found under {base}")
    rows: list[dict[str, str]] = []
    classes: set[str] = set()
    for class_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        images = sorted(path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            continue
        classes.add(class_dir.name)
        for image in images:
            rows.append({
                "image_path": str(image),
                "patient_id": f"{class_dir.name}:{image.stem}",
                "label": class_dir.name,
            })
    if not rows:
        raise ValueError(f"No supported images found under {data_root}")
    labels = tuple(sorted(classes))
    frame = pd.DataFrame(rows)
    for label in labels:
        frame[label] = (frame["label"] == label).astype("float32")
    return frame, labels


def save_manifest(directory: str | Path, archive: str | Path, labels: tuple[str, ...]) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    manifest = path / "dataset_manifest.json"
    manifest.write_text(json.dumps({
        "archive": str(Path(archive).resolve()),
        "sha256": sha256_file(archive),
        "labels": labels,
    }, indent=2), encoding="utf-8")
    return manifest
