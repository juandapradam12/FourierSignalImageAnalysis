# Architecture

The repository is organized as a small installable library plus thin example
entry points. Responsibilities are separated so transforms, I/O, filters,
and plotting can be reused independently.

```text
src/fourier_analysis/
  dft.py          # 1D DFT from definition + peak frequency
  spectrum.py     # frequency grid and log-magnitude helpers
  filters.py      # frequency-domain high-pass / convolution filter
  image_io.py     # grayscale image loading
  plotting.py     # figure writers (PNG)
  pipelines.py    # orchestration for the two use cases
examples/
  analyze_1d_signal.py
  analyze_2d_images.py
data/
  signals/        # tabular 1D samples
  images/         # grayscale-ready input imagery
docs/figures/     # curated PNGs embedded in the README story
tests/            # unit checks against NumPy FFT reference
```

## Design choices

- **Explicit DFT for 1D.** The signal use case implements the transform from
  the definition to expose the linear-algebra view $Z = Mz$, while tests
  assert numerical agreement with `numpy.fft.fft`.
- **Library FFT for 2D.** Image sizes make an $O(N^2)$ dense kernel
  impractical; `numpy.fft.fft2` / `ifft2` are used for spectra and filtering.
- **Pipelines as facades.** `analyze_signal` and `analyze_images` compose
  lower-level modules and return typed result objects with output paths.
- **DC-centered spectra.** 2D magnitude plots and cuts use `fftshift` so
  zero frequency sits at the image center.
- **Artifacts stay out of git.** Figures are written under `outputs/`
  (gitignored) so the repository tracks sources and data, not regenerated
  binaries.

## Data contracts

| Asset | Format | Role |
| --- | --- | --- |
| `data/signals/sample_signal.dat` | two columns: $t$, $f(t)$ | 1D frequency estimation |
| `data/images/*.jpg\|png\|jpeg` | RGB or gray images | 2D spectrum + filter demo |
