#!/usr/bin/env python3
"""Run the 1D Fourier analysis use case on the bundled sample signal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fourier_analysis.pipelines import analyze_signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate the dominant frequency of a 1D signal via DFT.",
    )
    parser.add_argument(
        "--signal",
        type=Path,
        default=ROOT / "data" / "signals" / "sample_signal.dat",
        help="Two-column file with time and amplitude samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs",
        help="Directory for generated figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_signal(args.signal, args.output_dir)
    print(
        f"Dominant frequency: {result.dominant_frequency:.6f} "
        f"(N={result.n_samples}, dt={result.sample_spacing:.6g})"
    )
    print(f"Spectrum figure: {result.spectrum_path}")
    print(f"Story figure:    {result.story_path}")


if __name__ == "__main__":
    main()
