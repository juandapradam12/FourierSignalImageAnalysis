"""Unit tests for spectrum display helpers."""

from __future__ import annotations

import numpy as np

from fourier_analysis.spectrum import centered_log_magnitude, log_magnitude


def test_centered_log_magnitude_places_dc_at_center() -> None:
    spectrum = np.zeros((8, 8), dtype=np.complex128)
    spectrum[0, 0] = 100.0 + 0j

    centered = centered_log_magnitude(spectrum, eps=1.0)
    peak = np.unravel_index(int(np.argmax(centered)), centered.shape)

    assert peak == (4, 4)
    assert centered[4, 4] == log_magnitude(np.array([100.0 + 0j]), eps=1.0)[0]
