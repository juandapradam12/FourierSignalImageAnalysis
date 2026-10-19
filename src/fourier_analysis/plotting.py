"""Plotting helpers for spectra and filtered images."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from fourier_analysis.spectrum import log_magnitude


def save_image_grid(
    images: Sequence[NDArray[np.floating]],
    titles: Sequence[str],
    output_path: str | Path,
    cmap: str = "gray",
) -> Path:
    """Save a $2 \\times 2$ grid of images."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, image, title in zip(axes.ravel(), images, titles):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def save_spectrum_grid(
    spectra: Sequence[NDArray[np.complexfloating]],
    titles: Sequence[str],
    output_path: str | Path,
    eps: float = 1.0,
) -> Path:
    """Save a $2 \\times 2$ grid of log-magnitude Fourier spectra."""
    magnitudes = [log_magnitude(spectrum, eps=eps) for spectrum in spectra]
    return save_image_grid(magnitudes, titles, output_path, cmap="inferno")


def save_horizontal_cuts(
    spectra: Sequence[NDArray[np.complexfloating]],
    titles: Sequence[str],
    output_path: str | Path,
    eps: float = 1.0,
) -> Path:
    """Plot mid-row horizontal cuts through each spectrum magnitude."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, spectrum, title in zip(axes.ravel(), spectra, titles):
        mid_row = spectrum.shape[0] // 2
        ax.plot(log_magnitude(spectrum[mid_row, :], eps=eps))
        ax.set_title(title)
        ax.set_xlabel("Frequency bin")
        ax.set_ylabel(r"$\log(|F| + \varepsilon)$")

    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def save_signal_spectrum(
    frequencies: NDArray[np.floating],
    spectrum: NDArray[np.complexfloating],
    output_path: str | Path,
) -> Path:
    """Plot the magnitude spectrum of a 1D signal."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    order = np.argsort(frequencies)
    figure, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frequencies[order], np.abs(spectrum[order]))
    ax.set_xlabel("Frequency")
    ax.set_ylabel(r"$|Z(f)|$")
    ax.set_title("1D DFT magnitude spectrum")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path
