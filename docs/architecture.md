# Numerical Layout

The code separates the transform definitions, spectral helpers, filters, and
I/O so each piece can be checked independently.

```text
src/fourier_analysis/
  dft.py          # 1D DFT from definition + peak frequency
  spectrum.py     # frequency grid and log-magnitude helpers
  filters.py      # frequency-domain high-pass / convolution filter
  image_io.py     # grayscale image loading
  plotting.py     # figure writers
  pipelines.py    # end-to-end 1D and 2D analyses
examples/
  analyze_1d_signal.py
  analyze_2d_images.py
data/
  signals/        # tabular 1D samples
  images/         # input imagery
docs/figures/     # result figures shown in the README
tests/            # checks against NumPy FFT and pipeline outputs
```

## Method choices

- **Explicit DFT for 1D.** The length-$N$ signal is small enough that the
  dense kernel $M_{kn} = e^{-2\pi i\, nk / N}$ can be formed and applied as
  $Z = Mz$, matching the definition used in the theory note.
- **Library FFT for 2D.** Image sizes make an $O(N^2)$ dense kernel
  impractical; `numpy.fft.fft2` / `ifft2` evaluate the same transform at
  FFT complexity for spectra and filtering.
- **DC-centered display.** Magnitude maps and cuts use `fftshift` so zero
  frequency sits at the image center, which matches the usual physical
  reading of $(f_x, f_y)$.
- **Generated vs curated figures.** Runtime figures write to `outputs/`
  (gitignored). Panels embedded in the README are stored under
  `docs/figures/`.

## Data

| Asset | Format | Role |
| --- | --- | --- |
| `data/signals/sample_signal.dat` | two columns: $t$, $f(t)$ | 1D frequency estimation |
| `data/images/*` | RGB or gray images | 2D spectra and directional filtering |
