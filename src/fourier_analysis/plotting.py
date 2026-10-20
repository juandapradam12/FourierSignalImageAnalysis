"""Plotting helpers for spectra and filtered images."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from fourier_analysis.spectrum import centered_log_magnitude, log_magnitude


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
    """Save a $2 \\times 2$ grid of DC-centered log-magnitude spectra."""
    magnitudes = [centered_log_magnitude(spectrum, eps=eps) for spectrum in spectra]
    return save_image_grid(magnitudes, titles, output_path, cmap="inferno")


def save_horizontal_cuts(
    spectra: Sequence[NDArray[np.complexfloating]],
    titles: Sequence[str],
    output_path: str | Path,
    eps: float = 1.0,
) -> Path:
    """Plot horizontal cuts through DC of each centered spectrum magnitude."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, spectrum, title in zip(axes.ravel(), spectra, titles):
        centered = centered_log_magnitude(spectrum, eps=eps)
        mid_row = centered.shape[0] // 2
        ax.plot(centered[mid_row, :])
        ax.set_title(title)
        ax.set_xlabel("Frequency bin (DC at center)")
        ax.set_ylabel(r"$\log(|F| + \varepsilon)$")

    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def save_before_after(
    original: NDArray[np.floating],
    filtered: NDArray[np.floating],
    output_path: str | Path,
    titles: tuple[str, str] = ("Original", "Filtered"),
) -> Path:
    """Save a side-by-side comparison of an image before and after filtering."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, image, title in zip(axes, (original, filtered), titles):
        ax.imshow(image, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

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


def save_signal_story(
    time: NDArray[np.floating],
    values: NDArray[np.floating],
    frequencies: NDArray[np.floating],
    spectrum: NDArray[np.complexfloating],
    dominant_frequency: float,
    output_path: str | Path,
) -> Path:
    """Save a two-panel figure: time series and magnitude spectrum with peak mark."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    order = np.argsort(frequencies)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(time, values, color="#1f4e79")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel(r"$f(t)$")
    axes[0].set_title("Sampled signal")

    axes[1].plot(frequencies[order], np.abs(spectrum[order]), color="#1f4e79")
    axes[1].axvline(
        dominant_frequency,
        color="#c45c26",
        linestyle="--",
        linewidth=1.5,
        label=rf"$f^\star = {dominant_frequency:.1f}$",
    )
    axes[1].set_xlabel("Frequency")
    axes[1].set_ylabel(r"$|Z(f)|$")
    axes[1].set_title("DFT magnitude spectrum")
    axes[1].legend(frameon=False)

    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path
