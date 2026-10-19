"""Discrete Fourier Transform (DFT) for one-dimensional signals."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def discrete_fourier_transform(signal: NDArray[np.floating]) -> NDArray[np.complexfloating]:
    """Compute the DFT from its definition.

    For a length-$N$ signal $z[n]$, the transform is

    $$
    Z[k] = \\sum_{n=0}^{N-1} z[n]\\, e^{-2\\pi i\\, nk / N},
    \\qquad k = 0, \\ldots, N-1.
    $$

    Parameters
    ----------
    signal:
        Real-valued samples of shape ``(N,)``.

    Returns
    -------
    Complex spectrum of shape ``(N,)``.
    """
    z = np.asarray(signal, dtype=np.float64)
    if z.ndim != 1:
        raise ValueError(f"Expected a 1D signal, got shape {z.shape}")

    n_samples = z.size
    n = np.arange(n_samples)
    k = n.reshape(n_samples, 1)
    kernel = np.exp(-2j * np.pi * n * k / n_samples)
    return kernel @ z


def dominant_frequency(
    spectrum: NDArray[np.complexfloating],
    sample_spacing: float,
) -> float:
    """Estimate the strongest positive frequency in a DFT spectrum.

    Frequencies are evaluated with the standard DFT frequency grid
    $f_k = k / (N\\Delta t)$. The peak is taken over the magnitude
    spectrum, excluding the DC bin.

    Parameters
    ----------
    spectrum:
        Complex DFT coefficients.
    sample_spacing:
        Sampling interval $\\Delta t$ in the original domain.

    Returns
    -------
    Dominant frequency in cycles per unit of $\\Delta t$.
    """
    if sample_spacing <= 0:
        raise ValueError("sample_spacing must be positive")

    frequencies = np.fft.fftfreq(len(spectrum), d=sample_spacing)
    magnitudes = np.abs(spectrum)
    magnitudes[0] = 0.0  # ignore DC component
    peak_index = int(np.argmax(magnitudes))
    return float(abs(frequencies[peak_index]))
