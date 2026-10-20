"""Unit tests for frequency-domain filtering helpers."""

from __future__ import annotations

import numpy as np
import pytest

from fourier_analysis.filters import apply_frequency_filter, high_pass_vertical_kernel


def test_high_pass_kernel_shape_and_range() -> None:
    kernel = high_pass_vertical_kernel(height=64, subsample=8)
    assert kernel.ndim == 2
    assert kernel.shape[1] == 1
    assert np.all(kernel >= 0.0)
    assert np.all(kernel <= 1.0)


def test_apply_frequency_filter_preserves_shape() -> None:
    image = np.random.default_rng(0).normal(size=(32, 40))
    spectrum = np.fft.fft2(image)
    kernel = high_pass_vertical_kernel(height=32, subsample=4)
    filtered = apply_frequency_filter(spectrum, kernel)

    assert filtered.shape == image.shape
    assert np.isrealobj(filtered)


def test_kernel_validation() -> None:
    with pytest.raises(ValueError):
        high_pass_vertical_kernel(height=0)
