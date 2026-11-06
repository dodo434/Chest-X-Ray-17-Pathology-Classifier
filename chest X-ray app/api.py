"""Dataset access for the Kaggle chest X-ray 17-disease dataset."""

from __future__ import annotations

import shutil
import urllib.request
from typing import Any
from pathlib import Path

from dataset_manager import extract_verified_archive, sha256_file


DATASET_HANDLE = "trainingdatapro/chest-xray-17-diseases"


def download_dataset(
    url: str,
    destination: str | Path,
    expected_sha256: str,
    extract_to: str | Path | None = None,
) -> Path:
    """Download, verify, and optionally extract a dataset archive."""
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    if sha256_file(target).lower() != expected_sha256.lower():
        target.unlink(missing_ok=True)
        raise ValueError("Downloaded dataset checksum verification failed.")
    if extract_to is not None:
        return extract_verified_archive(target, extract_to, expected_sha256=expected_sha256)
    return target


def load_dataset(file_path: str = "", **kwargs: Any):
    """Load a dataset table through kagglehub.

    The import and network request are intentionally deferred until this
    function is called so importing the application remains side-effect free.
    """
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub[pandas-datasets] is required to load the dataset."
        ) from exc

    return kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        DATASET_HANDLE,
        file_path,
        **kwargs,
    )


if __name__ == "__main__":
    dataset = load_dataset()
    print("First 5 records:", dataset.head())