"""Fourier analysis utilities for 1D signals and 2D images."""

from fourier_analysis.dft import discrete_fourier_transform, dominant_frequency
from fourier_analysis.filters import high_pass_vertical_kernel, apply_frequency_filter
from fourier_analysis.image_io import load_grayscale_image
from fourier_analysis.spectrum import frequency_axis, log_magnitude

__all__ = [
    "discrete_fourier_transform",
    "dominant_frequency",
    "high_pass_vertical_kernel",
    "apply_frequency_filter",
    "load_grayscale_image",
    "frequency_axis",
    "log_magnitude",
]

__version__ = "1.0.0"
