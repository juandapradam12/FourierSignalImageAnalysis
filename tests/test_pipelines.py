"""Integration tests for the end-to-end analysis pipelines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fourier_analysis.pipelines import analyze_images, analyze_signal


ROOT = Path(__file__).resolve().parents[1]


def test_analyze_signal_on_bundled_data(tmp_path: Path) -> None:
    result = analyze_signal(ROOT / "data" / "signals" / "sample_signal.dat", tmp_path)

    assert result.n_samples == 256
    assert result.dominant_frequency == pytest.approx(137.0, rel=1e-6)
    assert result.spectrum_path.exists()
    assert result.story_path.exists()


def test_analyze_images_writes_before_after_and_centered_spectra(tmp_path: Path) -> None:
    # Use tiny synthetic images so the integration test stays fast in CI.
    catalog: dict[str, Path] = {}
    for name in ("barcelona", "paris", "fractal", "triangles"):
        path = tmp_path / f"{name}.png"
        Image.fromarray(np.random.default_rng(0).integers(0, 255, (32, 40), dtype=np.uint8)).save(
            path
        )
        catalog[name] = path

    result = analyze_images(catalog, tmp_path / "out", filter_target="barcelona")

    assert result.images_path.exists()
    assert result.spectra_path.exists()
    assert result.cuts_path.exists()
    assert result.before_after_path.exists()
    assert result.before_after_path.name == "barcelona_before_after.png"
