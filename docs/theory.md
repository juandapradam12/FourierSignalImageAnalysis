# Mathematical Background

This note records the transforms and filter used by the portfolio use case.
It is not a tutorial; it states the definitions that the code implements.

## Discrete Fourier Transform (1D)

Given samples $z[n]$ of length $N$, the DFT is

$$
Z[k] = \sum_{n=0}^{N-1} z[n]\, e^{-2\pi i\, nk / N},
\qquad k = 0, \ldots, N-1.
$$

With sampling interval $\Delta t$, the corresponding frequency grid is

$$
f_k = \frac{k}{N\Delta t},
\qquad k = -\tfrac{N}{2}, \ldots, \tfrac{N}{2}-1
$$

(in the usual FFT frequency ordering). The dominant frequency reported by
the 1D pipeline is

$$
f^\star = \arg\max_{k \neq 0} \lvert Z[k]\rvert
\quad\text{mapped to}\quad
\lvert f_k\rvert.
$$

The implementation builds the DFT kernel matrix explicitly and multiplies
it by the signal vector, then cross-checks against `numpy.fft.fft` in tests.

## Two-Dimensional FFT

For an image $I(x, y)$, the discrete 2D transform is

$$
F(u, v) = \sum_{x}\sum_{y}
I(x, y)\, e^{-2\pi i\,(ux/M + vy/N)}.
$$

Spectral figures show a DC-centered log-magnitude,

$$
\log\bigl(\lvert F(u, v)\rvert + \varepsilon\bigr)
\quad\text{after}\quad
\texttt{fftshift},
$$

with $\varepsilon > 0$ to avoid $\log 0$. Horizontal cuts are taken through
the center row of that shifted array (the $f_y \approx 0$ line).

## Horizontal-structure attenuation

City maps with dense east–west streets concentrate energy near low vertical
frequencies. A compact high-pass profile along the vertical axis,

$$
K(x) = 1 - e^{-C x^{2}},
$$

is transformed and applied by pointwise multiplication in the Fourier
domain (convolution theorem):

$$
G = \mathcal{F}^{-1}\{ H \cdot F \},
\qquad H = \mathcal{F}\{K\}.
$$

The restored image $G$ is shown next to the original in a before/after
figure so the effect of attenuating horizontal structure is immediate.
