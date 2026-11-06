"""Defensive validation for uploaded chest X-ray images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

MIN_IMAGE_SIZE = 128
MIN_ASPECT_RATIO = 0.35
MAX_ASPECT_RATIO = 2.85
# Keep this lower than DETECTION_THRESHOLD so valid scans with no detected
# pathology can still reach the no-findings notification.
INVALID_MAX_PROBABILITY = 0.10  # Tune with representative non-X-ray images.
FLAT_DISTRIBUTION_STD = 0.025  # Tune with validation data; lower means less decisive.
DETECTION_THRESHOLD = 0.30  # Tune per-label after calibration.


class ImageValidationError(ValueError):
    """Raised when an upload is unsuitable for model inference."""


def validate_upload(path: str | Path) -> Image.Image:
    image_path = Path(path)
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
        raise ImageValidationError("Please upload a JPG, JPEG, PNG, or BMP image.")
    try:
        image = Image.open(image_path)
        image.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise ImageValidationError("The selected file is not a readable image.") from exc
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIZE:
        raise ImageValidationError("The image is too small to be a reliable chest X-ray.")
    ratio = width / height
    if not MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO:
        raise ImageValidationError("The image dimensions are not suitable for a chest X-ray.")
    channels = np.asarray(image.convert("RGB"))
    channel_spread = channels.astype(np.float32).std(axis=2).mean()
    if channel_spread > 18:
        raise ImageValidationError("Colour photographs are not supported; upload a grayscale chest X-ray.")
    return image


def validate_prediction(probabilities: np.ndarray) -> None:
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    if values.size != 17:
        raise ImageValidationError("The model returned an invalid prediction shape.")
    if float(values.max()) < INVALID_MAX_PROBABILITY or float(values.std()) < FLAT_DISTRIBUTION_STD:
        raise ImageValidationError(
            "This image doesn't look like a valid chest X-ray for the 17 supported pathologies. "
            "Please upload a frontal chest X-ray."
        )
