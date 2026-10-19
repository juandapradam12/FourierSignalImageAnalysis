"""Helpers for frequency axes and spectral magnitude displays."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def frequency_axis(n_samples: int, sample_spacing: float) -> NDArray[np.floating]:
    """Return the DFT frequency grid for ``n_samples`` and spacing $\\Delta t$."""
    return np.fft.fftfreq(n_samples, d=sample_spacing)


def log_magnitude(
    spectrum: NDArray[np.complexfloating],
    eps: float = 1.0,
) -> NDArray[np.floating]:
    """Return $\\log(|F| + \\varepsilon)$ for stable visualization of spectra."""
    return np.log(np.abs(spectrum) + eps)


def centered_log_magnitude(
    spectrum: NDArray[np.complexfloating],
    eps: float = 1.0,
) -> NDArray[np.floating]:
    """Zero-frequency-centered log-magnitude spectrum for image display."""
    return np.fft.fftshift(log_magnitude(spectrum, eps=eps))
