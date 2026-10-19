"""Frequency-domain filters for two-dimensional image spectra."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def high_pass_vertical_kernel(
    height: int,
    strength: float = np.pi,
    subsample: int = 8,
) -> NDArray[np.floating]:
    """Build a 1D high-pass profile along the vertical frequency axis.

    The spatial kernel

    $$
    K(x) = 1 - e^{-C x^{2}}
    $$

    attenuates low vertical frequencies. When applied in the Fourier
    domain of an image with strong horizontal stripes (energy concentrated
    near $f_y = 0$), this suppresses those stripes while preserving most
    of the remaining structure.

    Parameters
    ----------
    height:
        Number of rows in the target image spectrum.
    strength:
        Positive constant $C$ controlling the high-pass sharpness.
    subsample:
        Downsampling factor used when constructing the compact kernel
        before zero-padded FFT matching of the image shape.
    """
    if height < 1:
        raise ValueError("height must be at least 1")
    if strength <= 0:
        raise ValueError("strength must be positive")
    if subsample < 1:
        raise ValueError("subsample must be at least 1")

    n_points = max(height // subsample, 2)
    x = np.linspace(-height, height, n_points)
    profile = 1.0 - np.exp(-strength * x**2)
    return profile.reshape(-1, 1)


def apply_frequency_filter(
    image_spectrum: NDArray[np.complexfloating],
    spatial_kernel: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Filter an image by multiplying spectra (convolution theorem).

    Given image spectrum $F$ and kernel spectrum $H$,

    $$
    G = \\mathcal{F}^{-1}\\{ H \\cdot F \\}.
    $$

    Returns the real part of the inverse transform.
    """
    kernel_spectrum = np.fft.fft2(spatial_kernel, s=image_spectrum.shape)
    filtered_spectrum = kernel_spectrum * image_spectrum
    restored = np.fft.ifft2(filtered_spectrum)
    return np.real(restored)
