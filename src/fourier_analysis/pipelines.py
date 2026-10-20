"""End-to-end analysis pipelines for signals and images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fourier_analysis.dft import discrete_fourier_transform, dominant_frequency
from fourier_analysis.filters import apply_frequency_filter, high_pass_vertical_kernel
from fourier_analysis.image_io import load_grayscale_image
from fourier_analysis.plotting import (
    save_before_after,
    save_horizontal_cuts,
    save_image_grid,
    save_signal_spectrum,
    save_spectrum_grid,
)
from fourier_analysis.spectrum import frequency_axis


@dataclass(frozen=True)
class SignalAnalysisResult:
    """Summary of a one-dimensional Fourier analysis run."""

    dominant_frequency: float
    sample_spacing: float
    n_samples: int
    spectrum_path: Path


@dataclass(frozen=True)
class ImageAnalysisResult:
    """Paths produced by a two-dimensional Fourier analysis run."""

    images_path: Path
    spectra_path: Path
    cuts_path: Path
    before_after_path: Path


def analyze_signal(
    signal_path: str | Path,
    output_dir: str | Path,
) -> SignalAnalysisResult:
    """Load a two-column signal file and estimate its dominant frequency."""
    data = np.loadtxt(signal_path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Signal file must contain at least two columns: t, f(t)")

    time = data[:, 0]
    values = data[:, 1]
    sample_spacing = float(time[1] - time[0])

    spectrum = discrete_fourier_transform(values)
    frequencies = frequency_axis(len(values), sample_spacing)
    peak = dominant_frequency(spectrum, sample_spacing)

    out = Path(output_dir)
    spectrum_path = save_signal_spectrum(frequencies, spectrum, out / "signal_spectrum.png")

    return SignalAnalysisResult(
        dominant_frequency=peak,
        sample_spacing=sample_spacing,
        n_samples=len(values),
        spectrum_path=spectrum_path,
    )


def analyze_images(
    image_paths: dict[str, str | Path],
    output_dir: str | Path,
    filter_target: str = "barcelona",
) -> ImageAnalysisResult:
    """Compute 2D FFTs, DC-centered spectra, cuts, and a before/after filter demo."""
    if filter_target not in image_paths:
        raise KeyError(f"filter_target '{filter_target}' not found in image_paths")

    names = list(image_paths.keys())
    images = [load_grayscale_image(image_paths[name]) for name in names]
    spectra = [np.fft.fft2(image) for image in images]

    out = Path(output_dir)
    images_path = save_image_grid(images, names, out / "source_images.png")
    spectra_path = save_spectrum_grid(spectra, names, out / "fourier_spectra.png")
    cuts_path = save_horizontal_cuts(spectra, names, out / "spectrum_horizontal_cuts.png")

    target_index = names.index(filter_target)
    original = images[target_index]
    target_spectrum = spectra[target_index]
    kernel = high_pass_vertical_kernel(target_spectrum.shape[0])
    filtered = apply_frequency_filter(target_spectrum, kernel)

    before_after_path = save_before_after(
        original,
        filtered,
        out / f"{filter_target}_before_after.png",
        titles=(
            f"{filter_target} (original)",
            f"{filter_target} (horizontal structure attenuated)",
        ),
    )

    return ImageAnalysisResult(
        images_path=images_path,
        spectra_path=spectra_path,
        cuts_path=cuts_path,
        before_after_path=before_after_path,
    )


def default_image_catalog(data_root: str | Path) -> dict[str, Path]:
    """Return the default image catalog relative to ``data_root``."""
    root = Path(data_root)
    return {
        "barcelona": root / "images" / "barcelona.jpg",
        "paris": root / "images" / "paris.jpg",
        "fractal": root / "images" / "fractal.jpeg",
        "triangles": root / "images" / "triangles.png",
    }
