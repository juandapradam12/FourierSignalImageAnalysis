#!/usr/bin/env python3
"""Compute 2D spectra and a directional filter on the bundled images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fourier_analysis.pipelines import analyze_images, default_image_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute 2D FFTs, spectra, and a frequency-domain filter demo.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data",
        help="Root directory containing images/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs",
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--filter-target",
        default="barcelona",
        help="Catalog key of the image used for horizontal-line attenuation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = default_image_catalog(args.data_root)
    result = analyze_images(catalog, args.output_dir, filter_target=args.filter_target)

    print(f"Source images:      {result.images_path}")
    print(f"Fourier spectra:    {result.spectra_path}")
    print(f"Horizontal cuts:    {result.cuts_path}")
    print(f"Before / after:     {result.before_after_path}")


if __name__ == "__main__":
    main()
