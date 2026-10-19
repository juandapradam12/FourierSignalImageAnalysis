# Fourier Signal & Image Analysis

Estimate a signal’s dominant frequency with a from-definition 1D DFT, then
inspect image spectra with the 2D FFT and attenuate horizontal structure in the
frequency domain — a compact Python portfolio use case with tests and CI.

**Topics:** `fourier-transform` · `dft` · `fft` · `signal-processing` ·
`image-processing` · `numpy` · `python` · `scientific-computing` · `portfolio`

| Layer | Role |
| --- | --- |
| `src/fourier_analysis` | Reusable transforms, filters, and pipelines |
| `examples/` | Runnable 1D and 2D demos |
| `data/` | Bundled signal and images |
| `docs/` | Theory and architecture notes |

## Use cases

### 1. Dominant frequency of a 1D signal

A two-column time series $(t, f(t))$ is transformed with the discrete Fourier
transform

$$
Z[k] = \sum_{n=0}^{N-1} z[n]\, e^{-2\pi i\, nk / N}.
$$

The reported frequency is the peak of $\lvert Z[k]\rvert$ on the DFT grid
$f_k = k / (N\Delta t)$, excluding the DC bin.

```bash
python examples/analyze_1d_signal.py
```

### 2. Image spectra and horizontal-line attenuation

Four grayscale images are transformed with the 2D FFT. Log-magnitude spectra

$$
\log\bigl(\lvert F(u, v)\rvert + \varepsilon\bigr)
$$

and horizontal cuts through DC are exported with spectra centered via
`fftshift`. For the Barcelona city map, a vertical high-pass kernel

$$
K(x) = 1 - e^{-C x^{2}}
$$

is applied in the Fourier domain,

$$
G = \mathcal{F}^{-1}\{ \mathcal{F}\{K\} \cdot F \},
$$

which reduces the dense horizontal street pattern while preserving most of the
remaining geometry. The pipeline writes a side-by-side before/after figure.

```bash
python examples/analyze_2d_images.py
```

Figures are written to `outputs/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Requires Python 3.10+.

## Project layout

```text
├── src/fourier_analysis/   # library
├── examples/               # CLI-style demos
├── data/signals/           # 1D samples
├── data/images/            # input imagery
├── docs/theory.md          # equations behind the pipelines
├── docs/architecture.md    # module boundaries
└── tests/                  # DFT and filter checks
```

## Tests

```bash
pytest
```

GitHub Actions runs the same suite on Python 3.10 and 3.12 for every push and
pull request. The 1D DFT is compared against `numpy.fft.fft` on a known
sinusoid; frequency estimation recovers a planted tone within half a bin;
pipelines are checked end-to-end.

## Documentation

- [Mathematical background](docs/theory.md) — DFT definition, 2D spectrum
  display, and the high-pass filter used on Barcelona.
- [Architecture](docs/architecture.md) — package boundaries and data contracts.

## Stack

NumPy, Matplotlib, Pillow. No external FFT service: 1D uses an explicit kernel
matrix; 2D uses NumPy’s FFT for scale.
