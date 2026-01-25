"""FIR least-squares (cosine-series) helpers.

We assume a *zero-phase* (even-symmetric) FIR magnitude model on a discrete
frequency grid:

    h[f] = sum_{k=0..K-1} a[k] * cos(2*pi*k*f/Nfft)

for f in (0..Nfft/2], where Nfft is the DFT length that defines the grid.

This is convenient for educational / exploratory work and for building the
normal equations explicitly.

Notes
-----
- This is *not* a full replacement for scipy.signal.firls.
- For numerical stability we solve with np.linalg.lstsq by default.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class FirLeastSquaresCosine:
    """Least-squares fitter for a cosine-series magnitude model.

    Model
    -----
    On a grid of bins f = 0..N (inclusive), we model magnitude as:

        h[f] = sum_{k=0..K-1} a[k] * cos(2*pi*k*f/Nfft)

    where Nfft = 2*N (the implied full FFT length). This class focuses on the
    *half-spectrum* 0..Nyquist grid.

    Design constraints (by convention in this project)
    --------------------------------------------------
    We require:
    - num_coeffs is even

    Then the number of explicit bins (0..Nyquist, inclusive) is derived as:

        nfft = num_coeffs/2 + 1

    and the implied full FFT length is:

        Nfft = 2*(nfft-1) = num_coeffs

    This matches the idea that we only represent bins 0..Nyquist explicitly.
    """

    def __init__(
        self,
        *,
        num_coeffs: int,
        include_dc: bool = False,
        dtype=np.float64,
    ):
        self.num_coeffs = int(num_coeffs)
        self.include_dc = bool(include_dc)
        self.dtype = dtype

        if self.num_coeffs <= 0:
            raise ValueError(f"num_coeffs must be > 0, got {num_coeffs!r}")
        if (self.num_coeffs % 2) != 0:
            raise ValueError(f"num_coeffs must be even, got {num_coeffs!r}")

        # Derived sizes
        self.nfft = self.num_coeffs // 2 + 1

        # Validate dtype early so we fail fast.
        try:
            np.dtype(self.dtype)
        except Exception as e:  # pragma: no cover
            raise TypeError(f"Invalid dtype {dtype!r}") from e

    @property
    def n_bins(self) -> int:
        """Number of explicit frequency bins from 0..Nyquist (inclusive)."""
        return self.nfft

    @property
    def implied_fft_len(self) -> int:
        """Implied full FFT length for the cosine grid."""
        # With our convention Nfft == num_coeffs
        return self.num_coeffs

    def frequency_bins(self) -> np.ndarray:
        """Return the integer bin indices used for fitting."""
        f0 = 0 if self.include_dc else 1
        f_idx = np.arange(f0, self.nfft, dtype=np.int64)
        if f_idx.size == 0:
            raise ValueError("Empty frequency grid: check include_dc")
        return f_idx

    def design_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create the cosine-series design matrix for this instance."""
        f_idx = self.frequency_bins()
        k = np.arange(self.num_coeffs, dtype=np.int64)
        n = float(self.implied_fft_len)
        A = np.cos((2.0 * np.pi / n) * f_idx[:, None] * k[None, :]).astype(self.dtype, copy=False)
        return A, f_idx

    def fit(
        self,
        g: np.ndarray,
        *,
        lam: float = 0.0,
        rcond: float | None = None,
    ) -> np.ndarray:
        """Fit cosine-series coefficients a to desired samples g on this grid.

        Parameters
        ----------
        g:
            Desired magnitude samples on this instance's frequency grid.
        lam:
            Optional L2/Tikhonov regularization strength (λ ≥ 0). This solves

                min_a ||A a - g||^2 + λ ||a||^2

            which is equivalent to solving the regularized normal equations

                (AᵀA + λI)a = Aᵀg.

            (Same idea as adding `sqrt(λ) I` rows to A and zeros to g.)
        rcond:
            Passed through to `np.linalg.lstsq`.

        Returns
        -------
        a:
            1D array of length `num_coeffs`.
        """
        if lam < 0:
            raise ValueError(f"lam must be >= 0, got {lam!r}")

        A, f_idx = self.design_matrix()
        g = np.asarray(g, dtype=np.float64)
        if g.ndim != 1 or g.size != f_idx.size:
            raise ValueError(
                f"g must be a 1D array of length {f_idx.size} for include_dc={self.include_dc}, "
                f"got shape={g.shape!r}"
            )

        if lam == 0.0:
            a, *_ = np.linalg.lstsq(A, g, rcond=rcond)
            return a

        # Regularized least squares by augmentation:
        # [A         ] a ≈ [g]
        # [sqrt(lam)I]     [0]
        sqrt_lam = float(np.sqrt(lam))
        A_aug = np.vstack([A, sqrt_lam * np.eye(self.num_coeffs, dtype=A.dtype)])
        g_aug = np.concatenate([g, np.zeros(self.num_coeffs, dtype=g.dtype)])
        a, *_ = np.linalg.lstsq(A_aug, g_aug, rcond=rcond)
        return a

    def predict(self, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute fitted samples h = A a on this grid."""
        a = np.asarray(a)
        if a.ndim != 1 or a.size != self.num_coeffs:
            raise ValueError(
                f"a must be a 1D array of length {self.num_coeffs}, got shape={a.shape!r}"
            )
        A, f_idx = self.design_matrix()
        return A @ a, f_idx

    def design_single_bin_bandpass(
        self,
        bin_idx: int,
        *,
        lam: float = 0.0,
        rcond: float | None = None,
        amplitude: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Design an optimal "single-bin bandpass" in the LS sense.

        The target magnitude is zero everywhere on the fitting grid except for
        one frequency bin where it equals `amplitude`.

        Parameters
        ----------
        bin_idx:
            Integer bin index on the 0..Nyquist grid.
            Must be in the instance's `frequency_bins()`.
        lam:
            Optional L2 regularization passed to :meth:`fit`.
        rcond:
            Passed through to :meth:`fit`.
        amplitude:
            Target magnitude at `bin_idx`.

        Returns
        -------
        a:
            Cosine-series coefficients (length `num_coeffs`).
        g:
            Target magnitude samples on the fitting grid.
        f_idx:
            Frequency-bin vector corresponding to `g` (same as from
            :meth:`frequency_bins`).

        Raises
        ------
        ValueError
            If `bin_idx` is outside the fitting grid.
        """
        f_idx = self.frequency_bins()
        # Validate bin_idx is part of the grid (fast path with bounds check + membership)
        if bin_idx < int(f_idx[0]) or bin_idx > int(f_idx[-1]) or int(bin_idx) not in set(map(int, f_idx)):
            raise ValueError(
                f"bin_idx={bin_idx} is not in the fitting grid [{int(f_idx[0])}..{int(f_idx[-1])}] "
                f"with include_dc={self.include_dc}"
            )

        g = np.zeros(f_idx.size, dtype=np.float64)
        pos = int(np.where(f_idx == int(bin_idx))[0][0])
        g[pos] = float(amplitude)

        a = self.fit(g, lam=lam, rcond=rcond)
        return a, g, f_idx

    def design_single_bin_bandpass_constrained(
        self,
        bin_idx: int,
        *,
        amplitude: float = 1.0,
        f_weight: np.ndarray | None = None,
        lam: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Design a single-bin bandpass with *exact* peak and *flat-top* constraints.

        This uses a 2-constraint Lagrange multiplier (KKT) formulation.

        Objective
        ---------
        Minimize weighted energy (optionally regularized):

            min_a  || W (A a) ||^2 + lam ||a||^2

        subject to two equality constraints at one frequency bin:

            1) (A_c a) = amplitude
            2) (dA_c a) = 0

        where A_c is the single row of A corresponding to `bin_idx`.
        dA_c is the derivative of that row with respect to the (continuous)
        bin index f, i.e. a "flat top" constraint.

        Notes
        -----
        For the cosine series

            h(f) = sum_k a[k] cos(2π k f / Nfft)

        the derivative is

            dh/df = sum_k a[k] * (-(2π k / Nfft)) * sin(2π k f / Nfft)

        We evaluate this at f = bin_idx.

        Parameters
        ----------
        bin_idx:
            Integer bin index in the instance's fitting grid.
        amplitude:
            Desired magnitude at `bin_idx`.
        f_weight:
            Optional nonnegative weights per frequency sample on the fitting grid.
            If provided, must be shape (F,). A common choice is to set the
            constrained bin's weight to 0.
        lam:
            Optional ridge regularization (>= 0) on the coefficients.

        Returns
        -------
        a:
            Cosine-series coefficients (length `num_coeffs`).
        g:
            Target vector (all zeros except at constrained bin = amplitude).
            Returned for plotting/inspection.
        f_idx:
            Frequency-bin vector for the fitting grid.

        Raises
        ------
        ValueError
            On invalid inputs.
        np.linalg.LinAlgError
            If the KKT system is singular.
        """
        if lam < 0:
            raise ValueError(f"lam must be >= 0, got {lam!r}")

        A, f_idx = self.design_matrix()

        # Locate constrained row
        if bin_idx < int(f_idx[0]) or bin_idx > int(f_idx[-1]):
            raise ValueError(
                f"bin_idx={bin_idx} is not in the fitting grid [{int(f_idx[0])}..{int(f_idx[-1])}] "
                f"with include_dc={self.include_dc}"
            )
        hits = np.where(f_idx == int(bin_idx))[0]
        if hits.size != 1:
            raise ValueError(f"bin_idx={bin_idx} not in fitting grid (include_dc={self.include_dc})")
        c = int(hits[0])

        # Weights
        F = f_idx.size
        if f_weight is None:
            w = np.ones(F, dtype=np.float64)
        else:
            w = np.asarray(f_weight, dtype=np.float64)
            if w.ndim != 1 or w.size != F:
                raise ValueError(f"f_weight must be 1D of length {F}, got shape={w.shape!r}")
            if np.any(w < 0) or not np.all(np.isfinite(w)):
                raise ValueError("f_weight must be finite and nonnegative")

        # Build quadratic form H = 2*(Aᵀ W² A + lam I)
        W2 = w ** 2
        AW = A * W2[:, None]
        H = 2.0 * (A.T @ AW)
        if lam != 0.0:
            H = H + 2.0 * float(lam) * np.eye(self.num_coeffs, dtype=H.dtype)

        # Constraint rows
        C0 = A[c : c + 1, :].astype(np.float64, copy=False)  # (1,K)

        # Derivative row at f=bin_idx
        k = np.arange(self.num_coeffs, dtype=np.float64)
        n = float(self.implied_fft_len)
        omega = (2.0 * np.pi / n) * k * float(bin_idx)
        dC1 = (-(2.0 * np.pi / n) * k) * np.sin(omega)
        C1 = dC1.reshape(1, -1)

        C = np.vstack([C0, C1])  # (2,K)
        d = np.array([float(amplitude), 0.0], dtype=np.float64)

        # KKT system:
        # [H  Cᵀ][a] = [0]
        # [C   0][μ]   [d]
        K = self.num_coeffs
        m = 2
        KKT = np.zeros((K + m, K + m), dtype=np.float64)
        KKT[:K, :K] = H
        KKT[:K, K:] = C.T
        KKT[K:, :K] = C

        rhs = np.zeros(K + m, dtype=np.float64)
        rhs[K:] = d

        sol = np.linalg.solve(KKT, rhs)
        a = sol[:K]

        # Return a synthetic "desired" vector for plotting/inspection
        g = np.zeros(F, dtype=np.float64)
        g[c] = float(amplitude)
        return a, g, f_idx

