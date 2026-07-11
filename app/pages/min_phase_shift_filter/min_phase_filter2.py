import numpy as np


class MinDelayLogMagDesigner:
    """
    Minimum-delay FIR designer using log-magnitude smoothing (Option B),
    extended with a scipy.signal.firls-like multi-band interface.
    """

    def __init__(self,
                 n_fft=4096,
                 mu=1e-2,
                 mag_eps=1e-12,
                 trunc_thresh=1e-6):
        """
        Parameters
        ----------
        n_fft : int
            FFT size / dense frequency grid (even).
        mu : float
            Curvature regularization parameter for log-magnitude smoothing.
        mag_eps : float
            Magnitude floor to avoid log(0).
        trunc_thresh : float
            Threshold for truncating the final impulse response.
        """
        self.n_fft = int(n_fft)
        if self.n_fft % 2 != 0:
            raise ValueError("n_fft must be even.")
        self.mu = mu
        self.mag_eps = mag_eps
        self.trunc_thresh = trunc_thresh

        # Frequency grid [0, π]
        self.w = np.linspace(0, np.pi, self.n_fft // 2 + 1)

    # --------------------------------------------------------------
    # 1. Build multi-band piecewise-linear log-magnitude
    # --------------------------------------------------------------
    def build_logmag_multiband(self, bands, levels):
        """
        Multi-band piecewise-linear log-magnitude (firls-style).

        Parameters
        ----------
        bands : list or ndarray
            Frequency edges in radians, e.g. [0, 0.2π, 0.3π, 0.5π].
            Must be even-length: [b0, b1, b2, b3, ...].
        levels : list or ndarray
            Desired magnitude level per band (linear scale).
            Must have len(levels) == len(bands)/2.

        Returns
        -------
        L0 : ndarray
            Raw log-magnitude (natural log), length n_fft//2+1.
        """
        bands = np.asarray(bands, float)
        levels = np.asarray(levels, float)

        if len(bands) % 2 != 0:
            raise ValueError("bands must have even length.")
        if len(levels) != len(bands) // 2:
            raise ValueError("levels must have len(bands)/2 entries.")

        w = self.w
        L = np.zeros_like(w)

        # Build piecewise-linear magnitude in linear scale
        for i in range(len(levels)):
            w0 = bands[2*i]
            w1 = bands[2*i + 1]
            m0 = levels[i]
            m1 = levels[i]

            # Flat band
            mask = (w >= w0) & (w <= w1)
            L[mask] = m0

            # If not last band, connect to next band linearly
            if i < len(levels) - 1:
                w2 = bands[2*i + 2]
                m2 = levels[i + 1]
                trans_mask = (w > w1) & (w < w2)
                if trans_mask.any():
                    wt = w[trans_mask]
                    L[trans_mask] = m1 + (m2 - m1) * (wt - w1) / (w2 - w1)

        # Convert to log-magnitude
        L = np.maximum(L, self.mag_eps)
        return np.log(L)

    # --------------------------------------------------------------
    # 2. Smooth log-magnitude via curvature penalty
    # --------------------------------------------------------------
    def smooth_logmag(self, L0):
        L0 = np.asarray(L0).ravel()
        K = len(L0)

        D = np.zeros((K - 2, K))
        for i in range(K - 2):
            D[i, i]     = 1.0
            D[i, i + 1] = -2.0
            D[i, i + 2] = 1.0

        A = np.eye(K) + self.mu * (D.T @ D)
        return np.linalg.solve(A, L0)

    # --------------------------------------------------------------
    # 3. Minimum-phase reconstruction
    # --------------------------------------------------------------
    def minphase_from_logmag(self, L):
        cep = np.fft.irfft(L, n=self.n_fft)
        n_fft = len(cep)
        n2 = n_fft // 2

        cep[1:n2] += cep[-1:-n2:-1]
        cep[-1:-n2:-1] = 0.0

        logH_min = np.fft.rfft(cep, n=self.n_fft)
        return np.exp(logH_min)

    # --------------------------------------------------------------
    # 4. Impulse response
    # --------------------------------------------------------------
    def impulse_from_spectrum(self, H):
        return np.fft.irfft(H, n=self.n_fft)

    # --------------------------------------------------------------
    # 5. Truncate impulse response
    # --------------------------------------------------------------
    def truncate_impulse(self, h):
        h = np.asarray(h).ravel()
        idx = np.where(np.abs(h) > self.trunc_thresh)[0]
        if len(idx) == 0:
            return np.array([0.0])
        return h[:idx[-1] + 1]

    # --------------------------------------------------------------
    # 6. Full multi-band pipeline
    # --------------------------------------------------------------
    def design_multiband(self, bands, levels):
        """
        Full pipeline:
          L0 → smoothed L → minimum-phase → impulse → truncate

        Parameters
        ----------
        bands : list
            Frequency edges (rad).
        levels : list
            Desired magnitude per band (linear).

        Returns
        -------
        h_final : ndarray
            Final FIR.
        L_smooth : ndarray
            Smoothed log-magnitude.
        h_min : ndarray
            Minimum-phase impulse before truncation.
        """
        L0 = self.build_logmag_multiband(bands, levels)
        L_smooth = self.smooth_logmag(L0)
        H_min = self.minphase_from_logmag(L_smooth)
        h_min = self.impulse_from_spectrum(H_min)
        h_final = self.truncate_impulse(h_min)
        return h_final, L_smooth, h_min