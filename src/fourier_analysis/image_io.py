"""Image loading utilities for Fourier analysis pipelines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def load_grayscale_image(path: str | Path) -> NDArray[np.floating]:
    """Load an image as a float64 grayscale array in $[0, 255]$."""
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as img:
        gray = img.convert("L")
        return np.asarray(gray, dtype=np.float64)
