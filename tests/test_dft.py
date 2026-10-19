"""Unit tests for the 1D DFT and frequency estimation."""

from __future__ import annotations

import numpy as np
import pytest

from fourier_analysis.dft import discrete_fourier_transform, dominant_frequency


def test_dft_matches_numpy_fft_on_sine() -> None:
    n = 128
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    signal = np.sin(2 * np.pi * 7 * t)

    custom = discrete_fourier_transform(signal)
    reference = np.fft.fft(signal)

    assert np.allclose(custom, reference, atol=1e-8)


def test_dominant_frequency_recovers_known_tone() -> None:
    n = 256
    dt = 1.0 / n
    t = np.arange(n) * dt
    true_freq = 12.0
    signal = np.cos(2 * np.pi * true_freq * t)

    spectrum = discrete_fourier_transform(signal)
    estimated = dominant_frequency(spectrum, sample_spacing=dt)

    assert estimated == pytest.approx(true_freq, abs=0.5)


def test_dft_rejects_2d_input() -> None:
    with pytest.raises(ValueError):
        discrete_fourier_transform(np.ones((4, 4)))
