import numpy as np
from numpy.linalg import eigh


class MinDelayFilterDesigner:
    """
    Minimum-delay FIR designer from piecewise-linear log-magnitude,
    with minimum-phase reconstruction and optimal smoothing window.
    """

    def __init__(self,
                 fs=1.0,
                 n_fft=4096,
                 lam=1e-3,
                 mag_eps=1e-6,
                 trunc_thresh=1e-6):
        """
        Parameters
        ----------
        fs : float
            Sampling frequency (for reference; normalized design if fs=1).
        n_fft : int
            FFT size / dense frequency grid size (even).
        lam : float
            Regularization parameter for window optimization.
        mag_eps : float
            Floor for magnitude to avoid log(0).
        trunc_thresh : float
            Threshold for truncating the final impulse response.
        """
        self.fs = fs
        self.n_fft = int(n_fft)
        if self.n_fft % 2 != 0:
            raise ValueError("n_fft must be even.")
        self.lam = lam
        self.mag_eps = mag_eps
        self.trunc_thresh = trunc_thresh

        # Frequency grid [0, π] (normalized rad/s)
        self.w = np.linspace(0, np.pi, self.n_fft // 2 + 1)

    # ------------------------------------------------------------------
    # 1. Piecewise-linear log-magnitude
    # ------------------------------------------------------------------
    def build_logmag_lowpass(self,
                             wp,
                             ws,
                             Lp=0.0,
                             Ls=-60.0,
                             allow_passband_tilt=False):
        """
        Build a piecewise-linear log-magnitude for a lowpass.

        Parameters
        ----------
        wp : float
            Passband edge (rad, 0..π).
        ws : float
            Stopband edge (rad, 0..π, ws > wp).
        Lp : float
            Passband level in dB (typically 0).
        Ls : float
            Stopband level in dB (negative).
        allow_passband_tilt : bool
            If True, allow a small linear tilt in the passband.

        Returns
        -------
        L : ndarray
            Log-magnitude (natural log) on [0, π], length n_fft//2+1.
        """
        w = self.w
        L_db = np.zeros_like(w)

        # Passband region
        pass_mask = w <= wp
        stop_mask = w >= ws
        trans_mask = (~pass_mask) & (~stop_mask)

        # Passband: flat or slightly tilted
        if allow_passband_tilt:
            # Simple example: small tilt from 0 dB at DC to Lp at wp
            L_db[pass_mask] = np.linspace(0.0, Lp, pass_mask.sum())
        else:
            L_db[pass_mask] = Lp

        # Stopband: flat at Ls
        L_db[stop_mask] = Ls

        # Transition: linear in dB between Lp at wp and Ls at ws
        if trans_mask.any():
            w_t = w[trans_mask]
            L_db[trans_mask] = Lp + (Ls - Lp) * (w_t - wp) / (ws - wp)

        # Convert dB to natural log magnitude
        mag = 10.0 ** (L_db / 20.0)
        mag = np.maximum(mag, self.mag_eps)
        L = np.log(mag)
        return L

    # ------------------------------------------------------------------
    # 2. Minimum-phase reconstruction from log-magnitude
    # ------------------------------------------------------------------
    def minphase_from_logmag(self, L):
        """
        Compute minimum-phase spectrum from one-sided log-magnitude
        using real-cepstrum folding.

        Parameters
        ----------
        L : ndarray
            Log-magnitude spectrum (natural log, one-sided, length n_fft//2+1).

        Returns
        -------
        H_min : ndarray
            Minimum-phase spectrum (complex, one-sided, length n_fft//2+1).
        """
        # Real cepstrum via inverse real FFT of log-magnitude
        cep = np.fft.irfft(L, n=self.n_fft)  # length n_fft, real
        n_fft = len(cep)
        n2 = n_fft // 2

        # Fold cepstrum: double causal part, zero anti-causal
        # cep[0] stays, cep[1:n2] += cep[-1:-n2:-1], cep[-1:-n2:-1] = 0
        cep[1:n2] += cep[-1:-n2:-1]
        cep[-1:-n2:-1] = 0.0

        # Back to log-spectrum (complex)
        logH_min = np.fft.rfft(cep, n=self.n_fft)
        H_min = np.exp(logH_min)
        return H_min

    # ------------------------------------------------------------------
    # 3. Impulse response from minimum-phase spectrum
    # ------------------------------------------------------------------
    def impulse_from_spectrum(self, H):
        """
        Compute real impulse response from one-sided spectrum.

        Parameters
        ----------
        H : ndarray
            One-sided complex spectrum (length n_fft//2+1).

        Returns
        -------
        h : ndarray
            Real impulse response (length n_fft).
        """
        h = np.fft.irfft(H, n=self.n_fft)
        return h

    # ------------------------------------------------------------------
    # 4. Optimal smoothing window (quadratic curvature minimization)
    # ------------------------------------------------------------------
    def optimal_smoothing_window(self, h, K=None):
        """
        Compute a regularized optimal window for smoothing the complex
        frequency response of a real prototype h[n].

        Parameters
        ----------
        h : ndarray
            Real impulse response (e.g. minimum-phase).
        K : int or None
            Number of frequency samples for optimization. If None, use n_fft.

        Returns
        -------
        w_opt : ndarray
            Optimal window (unit-norm, same length as h).
        """
        h = np.asarray(h).ravel()
        N = len(h)
        if K is None:
            K = self.n_fft
        K = int(K)
        if K % 2 != 0:
            raise ValueError("K must be even.")

        # Frequency grid
        k = np.arange(K)
        omega = 2 * np.pi * k / K
        n = np.arange(N)

        # Complex frequency response matrix B_{k,n} = h[n] * exp(-j ω_k n)
        B = h[None, :] * np.exp(-1j * omega[:, None] * n[None, :])  # (K, N)

        # Stack real and imaginary parts into A (2K x N)
        A = np.vstack([B.real, B.imag])

        # Second-difference operator D over K samples
        D1 = np.zeros((K - 2, K))
        for i in range(K - 2):
            D1[i, i]     = 1.0
            D1[i, i + 1] = -2.0
            D1[i, i + 2] = 1.0

        # Block-diagonal D for real/imag
        D = np.block([
            [D1,                  np.zeros_like(D1)],
            [np.zeros_like(D1),   D1              ]
        ])  # shape (2(K-2), 2K)

        # Quadratic form Q = A^T D^T D A
        DA = D @ A          # (2(K-2), N)
        Q = DA.T @ DA       # (N, N), symmetric PSD

        # Regularization
        Q_reg = Q + self.lam * np.eye(N)

        # Eigenproblem Q_reg w = λ w
        evals, evecs = eigh(Q_reg)
        w_opt = evecs[:, 0].real

        # Normalize
        w_opt /= np.linalg.norm(w_opt)

        return w_opt

    # ------------------------------------------------------------------
    # 5. Truncate impulse response
    # ------------------------------------------------------------------
    def truncate_impulse(self, h):
        """
        Truncate impulse response when magnitude falls below threshold.

        Parameters
        ----------
        h : ndarray
            Impulse response.

        Returns
        -------
        h_trunc : ndarray
            Truncated impulse response.
        """
        h = np.asarray(h).ravel()
        idx = np.where(np.abs(h) > self.trunc_thresh)[0]
        if len(idx) == 0:
            return np.array([0.0])
        last = idx[-1]
        return h[:last + 1]

    # ------------------------------------------------------------------
    # 6. Full pipeline: design lowpass minimum-delay FIR
    # ------------------------------------------------------------------
    def design_lowpass(self,
                       wp,
                       ws,
                       Lp=0.0,
                       Ls=-60.0,
                       allow_passband_tilt=False,
                       K_window=None):
        """
        Full pipeline: piecewise-linear log-magnitude -> minimum-phase
        -> optimal smoothing window -> truncated FIR.

        Parameters
        ----------
        wp : float
            Passband edge (rad, 0..π).
        ws : float
            Stopband edge (rad, 0..π, ws > wp).
        Lp : float
            Passband level in dB.
        Ls : float
            Stopband level in dB.
        allow_passband_tilt : bool
            Allow small tilt in passband log-magnitude.
        K_window : int or None
            Frequency grid size for window optimization.

        Returns
        -------
        h_final : ndarray
            Final truncated FIR impulse response.
        h_min : ndarray
            Minimum-phase prototype impulse response (before windowing).
        w_opt : ndarray
            Optimal window (same length as h_min).
        """
        # 1) Log-magnitude
        L = self.build_logmag_lowpass(wp, ws, Lp=Lp, Ls=Ls,
                                      allow_passband_tilt=allow_passband_tilt)

        # 2) Minimum-phase spectrum
        H_min = self.minphase_from_logmag(L)

        # 3) Impulse response
        h_min = self.impulse_from_spectrum(H_min)

        # 4) Optimal smoothing window
        w_opt = self.optimal_smoothing_window(h_min, K=K_window)

        # 5) Apply window
        h_win = h_min * w_opt

        # 6) Truncate
        h_final = self.truncate_impulse(h_win)

        return h_final, h_min, w_opt