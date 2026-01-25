"""
Minimum-phase FIR design with cubic log-magnitude transitions.

- Similar interface to scipy.signal.firls
- Uses cubic minimal-curvature transitions between band points
- Smooth log-magnitude interpolation minimizes phase distortion
- Minimum-phase FIR via real-cepstrum spectral factorization
- Half-windowing for minimum-phase-friendly truncation
"""

import numpy as np
import scipy.linalg as linalg
from matplotlib import pyplot as plt

from numlib import numlegend, numplot, numutil
import scipy.signal as ssig

SIGMA = 0.5
# --------------------------------------------------------------------
# 1. Transition functions for log-magnitude
# --------------------------------------------------------------------
from scipy.special import erf


def L_transition(x, x0, x1, L0, L1, transition='cubic', sigma=SIGMA):
    """
    Generalized transition function for log-magnitude.

    Supports multiple transition types with different smoothness characteristics:
    - cubic: Minimal curvature (L² optimal), C¹ continuous
    - quintic: C² continuous (zero acceleration at boundaries)
    - raised_cosine: Half-period cosine, smooth transition
    - erf: C∞ continuous (Gaussian error function), sigma controls sharpness

    Parameters
    ----------
    x : array-like
        Input coordinates (e.g., log-frequency values).
    x0 : float
        Start coordinate.
    x1 : float
        End coordinate.
    L0 : float
        Log-magnitude at start (x=x0).
    L1 : float
        Log-magnitude at end (x=x1).
    transition : str, optional
        Type of transition: 'cubic', 'quintic', 'raised_cosine', or 'erf'.
        Default is 'cubic'.
    sigma : float, optional
        Sharpness parameter for 'erf' transition. Default is SIGMA.
        Smaller values give sharper transitions.

    Returns
    -------
    L : ndarray
        Log-magnitude values at each x.
    """
    x = np.asarray(x)
    # Normalize to t in [0, 1]
    t = np.clip((x - x0) / (x1 - x0), 0, 1)

    if transition == 'cubic':
        # Cubic interpolation: L(t) = L0 + (L1-L0) * (3t² - 2t³)
        blend = 3.0 * t**2 - 2.0 * t**3
    elif transition == 'quintic':
        # Quintic interpolation: 10t³ - 15t⁴ + 6t⁵ (C² continuous)
        blend = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
    elif transition == 'raised_cosine':
        # Raised cosine: 0.5 * (1 - cos(π*t))
        blend = 0.5 * (1 - np.cos(np.pi * t))
    elif transition == 'erf':
        # ERF transition: C∞ continuous
        # Center at t=0.5, scale so ERF covers transition range
        t_mid = 0.5
        scale = SIGMA  # ERF covers roughly ±2*sigma
        z = (t - t_mid) / (sigma * scale)
        blend = 0.5 * (1 + erf(z))
    else:
        raise ValueError(f"Unknown transition type: {transition}. "
                         f"Choose from 'cubic', 'quintic', 'raised_cosine', 'erf'.")

    return L0 + (L1 - L0) * blend


# --------------------------------------------------------------------
# 2. Build log-magnitude from bands/desired using transitions
# --------------------------------------------------------------------
def logmag_from_bands(ww, bands, desired, transition='cubic', sigma=SIGMA):
    """
    Construct log-magnitude using smooth transitions between band points.

    Uses log-frequency domain for transitions to minimize curvature
    and thus minimize phase distortion (via Hilbert/Parseval relation).

    Parameters
    ----------
    ww : array-like
        Frequency grid (normalized, 0 to 1 where 1 = Nyquist), including DC.
    bands : array-like
        Frequency band edges, normalized [0, 1]. Must be monotonically increasing.
        Must start at 0 (DC) and end at 1 (Nyquist).
        Example: [0, 0.3, 0.5, 1.0] for lowpass.
    desired : array-like
        Desired linear magnitude at each band edge. Cannot be zero.
        Example: [1, 1, 0.001, 0.001] for lowpass with -60 dB stopband.
    transition : str, optional
        Type of transition: 'cubic', 'quintic', 'raised_cosine', or 'erf'.
        Default is 'cubic'.
    sigma : float, optional
        Sharpness parameter for 'erf' transition. Default is SIGMA.

    Returns
    -------
    L : ndarray
        Log-magnitude at each frequency in ww.
    log_bands : ndarray
        Log of band frequencies (with DC adjusted).
    log_desired : ndarray
        Log of desired magnitudes.

    Raises
    ------
    ValueError
        If any desired magnitude is zero or negative, or bands are invalid.
    """
    bands = np.asarray(bands, dtype=float)
    desired = np.asarray(desired, dtype=float)

    # Validate inputs
    if np.any(desired <= 0):
        raise ValueError("Desired magnitudes must be positive (cannot be zero or negative).")
    if bands[0] != 0:
        raise ValueError("Bands need to start at zero (DC).")
    if bands[-1] != 1:
        raise ValueError("Bands must end at one (Nyquist).")
    if any(np.diff(bands) <= 0):
        raise ValueError("Bands must be monotonically increasing.")

    # Convert to log domain
    log_desired = np.log(desired)

    # Initialize output
    L = np.zeros_like(ww, dtype=float)

    # Skip DC (index 0), work in log-frequency for non-zero frequencies
    mask = ww > 0
    ww_pos = ww[mask]
    log_ww = np.log(ww_pos)
    L_pos = np.zeros_like(ww_pos)

    # Handle bands[0] == 0 by using first non-zero frequency
    log_bands = np.log(np.maximum(bands, ww_pos[0]))

    # Apply cubic transitions between each pair of band points
    for i in range(len(bands) - 1):
        # Find frequencies in this segment
        if i == 0:
            seg_mask = ww_pos <= bands[i + 1]
        else:
            seg_mask = (ww_pos > bands[i]) & (ww_pos <= bands[i + 1])

        if not np.any(seg_mask):
            continue

        # Apply transition (normalization done inside L_transition)
        L_pos[seg_mask] = L_transition(
            log_ww[seg_mask],
            log_bands[i], log_bands[i + 1],
            log_desired[i], log_desired[i + 1],
            transition=transition,
            sigma=sigma
        )

    L[mask] = L_pos

    # DC: extend from first non-DC point
    L[0] = L[1] if np.any(mask) else log_desired[0]

    return L, log_bands, log_desired


def plot_logmag_from_bands(bands, desired, n_points=1000):
    """
    Plot the log-magnitude response from bands/desired specification.

    Parameters
    ----------
    bands : array-like
        Frequency band edges, normalized [0, 1].
    desired : array-like
        Desired linear magnitude at each band edge.
    n_points : int, optional
        Number of frequency points for plotting. Default 1000.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes.
    """
    bands = np.asarray(bands, dtype=float)
    desired = np.asarray(desired, dtype=float)

    # Frequency grid
    ww = np.linspace(0, 1, n_points)

    # Compute log-magnitude
    L, log_bands, log_desired = logmag_from_bands(ww, bands, desired)

    # Convert to dB for plotting
    L_db = 20 * L / np.log(10)  # L is ln(mag), convert to 20*log10(mag)
    bands = np.exp(log_bands)
    desired_db = 20 * log_desired / np.log(10)
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot magnitude in dB
    ax[0].semilogx(ww[1:], L_db[1:], 'b-', linewidth=1.5, label='Cubic spline interpolation')
    ax[0].scatter(bands, desired_db, color='red', s=80, zorder=5, label='Band/desired points')
    ax[0].set_ylabel('Magnitude [dB]')
    ax[0].set_title('Log-Magnitude from Bands (CubicSpline with natural BC)')
    ax[0].grid(True, which='both', alpha=0.3)
    ax[0].legend()

    # Plot linear magnitude
    ax[1].semilogx(ww[1:], np.exp(L[1:]), 'b-', linewidth=1.5, label='Cubic spline interpolation')
    ax[1].scatter(bands, desired, color='red', s=80, zorder=5, label='Band/desired points')
    ax[1].set_ylabel('Magnitude [linear]')
    ax[1].set_xlabel('Normalized Frequency (1 = Nyquist)')
    ax[1].grid(True, which='both', alpha=0.3)
    ax[1].legend()

    fig.tight_layout()

    return fig, ax
def minphase_from_logmag(L):
    """
    Compute minimum-phase spectrum from log-magnitude using cepstral folding.

    Parameters
    ----------
    L : ndarray
        Log-magnitude spectrum (one-sided, DC to Nyquist).

    Returns
    -------
    H : ndarray
        Minimum-phase spectrum (complex, same length as L).
    """
    # Real cepstrum via inverse FFT of log-magnitude
    cep = np.fft.irfft(L)
    n_fft = len(cep)
    n2 = n_fft // 2

    # Fold cepstrum to create minimum-phase: double causal part, zero anti-causal
    cep[1:n2] += cep[-1:-n2:-1]  # fold (cepstrum is real, no conj needed)
    cep[-1:-n2:-1] = 0.0

    # Back to spectrum
    H = np.exp(np.fft.rfft(cep))

    return H


def firmp_full(numtaps, bands, desired, n_fft=None, fs=None, transition='cubic', sigma=SIGMA):
    """
    Design a minimum-phase FIR filter, returning full diagnostic info.

    Parameters
    ----------
    numtaps : int
        Desired FIR length (number of taps).
    bands : array-like
        Frequency band edges, normalized [0, 1] where 1 = Nyquist.
    desired : array-like
        Desired linear magnitude at each band edge.
    n_fft : int, optional
        FFT length for spectrum construction.
    fs : float, optional
        Sample rate in Hz.
    transition : str, optional
        Type of transition: 'cubic', 'quintic', 'raised_cosine', or 'erf'.
        Default is 'cubic'.
    sigma : float, optional
        Sharpness parameter for 'erf' transition. Default is SIGMA.

    Returns
    -------
    fir : ndarray
        Minimum-phase FIR impulse response.
    H : ndarray
        Minimum-phase spectrum (complex).
    L : ndarray
        Log-magnitude spectrum.
    """
    bands = np.asarray(bands, dtype=float)
    desired = np.asarray(desired, dtype=float)

    if fs is not None:
        bands = bands / (fs / 2)

    if n_fft is None:
        n_fft = numutil.ceil_power2(4 * numtaps)

    nf = n_fft // 2 + 1
    ww = np.linspace(0, 1, nf)

    L, log_bands, log_desired = logmag_from_bands(ww, bands, desired, transition=transition, sigma=sigma)
    H = minphase_from_logmag(L)
    ir = np.fft.irfft(H)

    window = np.hanning(2 * numtaps)[-numtaps:]
    fir = ir[:numtaps] * window
    fir *= linalg.norm(ir[:numtaps]) / linalg.norm(fir)

    return fir, H, L, log_bands, log_desired

# --------------------------------------------------------------------
# 3. Main design function (firls-like interface)
# --------------------------------------------------------------------
def firmp(numtaps, bands, desired, n_fft=None, fs=None, transition='cubic', sigma=SIGMA):
    fir, H, L, log_bands, log_desired = firmp_full(numtaps, bands, desired,
                                                   n_fft=n_fft, fs=fs, transition=transition, sigma=sigma)
    return fir

# --------------------------------------------------------------------
# 4. Group delay calculation for minimum-phase FIR
# --------------------------------------------------------------------
def group_delay_minphase_fir(fir, n_fft=4096):
    """
    Vectorized analytic group delay of a minimum-phase FIR using the real cepstrum.

    Parameters
    ----------
    fir : array_like
        Minimum-phase FIR coefficients.
    n_fft : int
        FFT size for spectral/cepstral resolution.

    Returns
    -------
    w : ndarray
        Frequency grid (0..1 normalized), n_fft//2 + 1 samples including DC and Nyquist.
    gd : ndarray
        Group delay at each frequency (in samples).
    """
    nf = n_fft // 2 + 1

    # Frequency response
    H = np.fft.fft(fir, n_fft)
    mag = np.abs(H)

    # Real cepstrum
    log_mag = np.log(np.maximum(mag, 1e-12))
    c = np.real(np.fft.ifft(log_mag))

    # Positive quefrencies (1..N/2-1)
    c_pos = c[1:n_fft//2]
    n = np.arange(1, n_fft//2)

    # Frequency grid: 0 to 1 normalized (nf samples)
    w = np.linspace(0, 1, nf)
    w_rad = w * np.pi

    # Vectorized cosine matrix: shape (freqs, quefrencies)
    cos_matrix = np.cos(w_rad[:, None] * n[None, :])

    # Analytic group delay: sum over quefrencies
    # tau(w) = Σ 2 n c[n] cos(n w)
    gd = cos_matrix @ (2 * n * c_pos)

    # Extend DC (frequency 0) from frequency 1
    gd[0] = gd[1]

    return w, gd


def step_response(fir, n_samples=None):
    """
    Compute the step response of a FIR filter using Simpson's rule integration.

    The step response is the integral of the impulse response (FIR coefficients).
    Using scipy.integrate.simpson for accurate numerical integration.

    Parameters
    ----------
    fir : array-like
        FIR filter coefficients (impulse response).
    n_samples : int, optional
        Number of samples for the step response. Default is 2 * len(fir).

    Returns
    -------
    t : ndarray
        Time index (sample numbers).
    step : ndarray
        Step response at each sample.
    """
    from scipy.integrate import simpson

    fir = np.asarray(fir)

    if n_samples is None:
        n_samples = 2 * len(fir)

    # Pad FIR with zeros to desired length
    ir = np.zeros(n_samples)
    ir[:len(fir)] = fir

    # Compute step response using cumulative Simpson integration
    # Step response at sample k is integral of impulse response from 0 to k
    step = np.zeros(n_samples)

    for k in range(1, n_samples):
        # Integrate from 0 to k using Simpson's rule
        # Simpson requires at least 2 points, use cumulative approach
        step[k] = simpson(ir[:k+1], dx=1.0)

    # Time index
    t = np.arange(n_samples)

    return t, step


def step_response_cumsum(fir, n_samples=None):
    """
    Compute the step response using simple cumulative sum.

    This is mathematically equivalent to discrete integration and serves
    as a fast alternative to Simpson's rule for comparison.

    Parameters
    ----------
    fir : array-like
        FIR filter coefficients (impulse response).
    n_samples : int, optional
        Number of samples for the step response. Default is 2 * len(fir).

    Returns
    -------
    t : ndarray
        Time index (sample numbers).
    step : ndarray
        Step response at each sample.
    """
    fir = np.asarray(fir)

    if n_samples is None:
        n_samples = 2 * len(fir)

    # Pad FIR with zeros to desired length
    ir = np.zeros(n_samples)
    ir[:len(fir)] = fir

    # Step response is cumulative sum of impulse response
    step = np.cumsum(ir)

    # Time index
    t = np.arange(n_samples)

    return t, step


def plot_firmp(numtaps, bands, desired, n_fft=None, transition='cubic', sigma=SIGMA,
               compare_transitions=False):
    """
    Plot minimum-phase FIR filter designed with firmp.

    Parameters
    ----------
    numtaps : int
        Number of filter taps.
    bands : array-like
        Frequency band edges, normalized [0, 1].
    desired : array-like
        Desired magnitude at each band edge.
    n_fft : int, optional
        FFT length for plotting.
    transition : str, optional
        Type of transition: 'cubic', 'quintic', 'raised_cosine', or 'erf'.
        Default is 'cubic'.
    sigma : float, optional
        Sharpness parameter for 'erf' transition. Default is SIGMA.
    compare_transitions : bool, optional
        If True, plot all transition types for comparison. Default is False.
    """
    if n_fft is None:
        n_fft = numutil.ceil_power2(4 * numtaps)

    nf = n_fft // 2 + 1
    ff = np.linspace(0, 1, nf)

    if compare_transitions:
        # Compare all transition types
        transitions = ['cubic', 'quintic', 'raised_cosine', 'erf']
        fax = None

        # Impulse response plot
        fig_ir, ax_ir = plt.subplots()

        for trans in transitions:
            fir, H, L, _, _ = firmp_full(numtaps, bands, desired, n_fft=n_fft,
                                         transition=trans, sigma=sigma)
            hh_fir = np.fft.rfft(fir, n=n_fft)

            # Plot impulse responses
            ax_ir.plot(fir, label=trans, alpha=0.7)

            # Plot spectra
            if fax is None:
                fax = numplot.plotspek(hh_fir, ff, label=trans)
            else:
                numplot.plotspek(hh_fir, ff, fax=fax, label=trans)

        ax_ir.set_xlabel('Sample')
        ax_ir.set_ylabel('Amplitude')
        ax_ir.set_title('Minimum-Phase FIR: Impulse Response Comparison')
        numlegend.gridsetup(ax_ir)
        fig_ir.legend_handler = numlegend.LegendHandler(fig_ir.canvas)
        fig_ir.legend_handler.setup(ax_ir)

        fax[0].legend_handler = numlegend.LegendHandler(fax[0].canvas)
        fax[0].legend_handler.setup(fax[1])
        fax[0].suptitle('Minimum-Phase FIR: Transition Type Comparison')

        # Group delay comparison
        fig_gd, ax_gd = plt.subplots()
        for trans in transitions:
            fir, _, _, _, _ = firmp_full(numtaps, bands, desired, n_fft=n_fft,
                                         transition=trans, sigma=sigma)
            _, gd = group_delay_minphase_fir(fir, n_fft)
            ax_gd.semilogx(ff[1:], gd[1:], label=trans, alpha=0.7)

        ax_gd.set_xlabel('Normalized Frequency')
        ax_gd.set_ylabel('Group Delay [samples]')
        ax_gd.set_title('Group Delay: Transition Type Comparison')
        numlegend.gridsetup(ax_gd)
        fig_gd.legend_handler = numlegend.LegendHandler(fig_gd.canvas)
        fig_gd.legend_handler.setup(ax_gd)

        # Step response comparison
        fig_step, ax_step = plt.subplots()
        for trans in transitions:
            fir, _, _, _, _ = firmp_full(numtaps, bands, desired, n_fft=n_fft,
                                         transition=trans, sigma=sigma)
            t, step = step_response(fir)
            ax_step.plot(t, step, label=trans, alpha=0.7)

        ax_step.set_xlabel('Sample')
        ax_step.set_ylabel('Amplitude')
        ax_step.set_title('Step Response: Transition Type Comparison')
        ax_step.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Target')
        numlegend.gridsetup(ax_step)
        fig_step.legend_handler = numlegend.LegendHandler(fig_step.canvas)
        fig_step.legend_handler.setup(ax_step)

    else:
        # Single transition type
        fir, H, L, log_bands, log_desired = firmp_full(numtaps, bands, desired, n_fft=n_fft,
                                                        transition=transition, sigma=sigma)

        # Impulse response plot
        fig, ax = plt.subplots()
        ax.plot(fir)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Minimum-Phase FIR Impulse Response ({transition})')
        numlegend.gridsetup(ax)

        # Spectrum comparison
        hh_fir = np.fft.rfft(fir, n=n_fft)
        fax = numplot.plotspek(H, ff, label="Min-phase spectrum")
        numplot.plotspek(hh_fir, ff, fax=fax, label="FIR response")
        fax[1][0].semilogx(ff[1:], np.exp(L[1:]), label="Target magnitude")
        fax[0].legend_handler = numlegend.LegendHandler(fax[0].canvas)
        fax[0].legend_handler.setup(fax[1])
        fax[0].suptitle(f'Minimum-Phase FIR: Spectrum Comparison ({transition})')

        # Step response plot
        fig_step, ax_step = plt.subplots()
        t, step = step_response(fir)
        ax_step.plot(t, step, label=f'Step response ({transition})')
        ax_step.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Target')
        ax_step.set_xlabel('Sample')
        ax_step.set_ylabel('Amplitude')
        ax_step.set_title(f'Minimum-Phase FIR Step Response ({transition})')
        numlegend.gridsetup(ax_step)
        fig_step.legend_handler = numlegend.LegendHandler(fig_step.canvas)
        fig_step.legend_handler.setup(ax_step)

    return fax


def minphase_butterworth_pair(fp, fs_edge, A_s_db, fs=2.0):
    """
    Design a pair of cascaded Butterworth filters for minimum phase shift.

    The strategy:
    1. First Butterworth lowpass with -3dB at fp (passband edge), order chosen
       so that attenuation at fs_edge is A_s_db
    2. Second: inverted analog lowpass to flatten stopband AFTER fs_edge

    Combined response: H_total = H_lowpass1 / H_lowpass2

    The inverted lowpass rises at +20N dB/decade, countering the first lowpass's
    -20N dB/decade rolloff, resulting in a flat stopband at exactly A_s_db.

    The design is done in analog domain (where inversion is stable) then converted
    to digital using bilinear transform.

    Parameters
    ----------
    fp : float
        Passband edge (normalized frequency, 0 to 1). First filter has -3dB here.
    fs_edge : float
        Stopband edge (normalized frequency, 0 to 1). First filter has A_s_db here.
    A_s_db : float
        Required stopband attenuation in dB (positive value) at fs_edge and beyond.
    fs : float
        Sample rate (default 2.0 for normalized frequencies).

    Returns
    -------
    b : ndarray
        Numerator coefficients.
    a : ndarray
        Denominator coefficients.
    info : dict
        Dictionary with design details:
        - 'order': order of both Butterworths
        - 'wn1': cutoff of first lowpass Butterworth (= fp, the -3dB point)
        - 'wn2': cutoff of second (inverted) lowpass Butterworth
        - 'atten_at_nyquist': actual attenuation at Nyquist
    """
    # First Butterworth: -3dB at fp, and A_s_db at fs_edge
    # For Butterworth: |H(w)|^2 = 1 / (1 + (w/wc)^(2n))
    # At w=fp (cutoff): |H|^2 = 0.5 => -3dB
    # At w=fs_edge: |H|^2 = 10^(-A_s_db/10)
    #
    # From: 1/(1 + (fs_edge/fp)^(2n)) = 10^(-A_s_db/10)
    # => (fs_edge/fp)^(2n) = 10^(A_s_db/10) - 1
    # => n = log(10^(A_s_db/10) - 1) / (2 * log(fs_edge/fp))

    ratio = fs_edge / fp
    target_atten_linear = 10 ** (A_s_db / 10) - 1
    order = int(np.ceil(np.log(target_atten_linear) / (2 * np.log(ratio))))
    order = max(order, 1)  # at least first order

    # Pre-warp frequencies for bilinear transform
    fp_analog = 2 * fs * np.tan(np.pi * fp / fs)
    fs_edge_analog = 2 * fs * np.tan(np.pi * fs_edge / fs)

    # Target frequency for flat stopband (close to Nyquist)
    w_target = 0.95
    w_target_analog = 2 * fs * np.tan(np.pi * w_target / fs)

    # First filter: -3dB at fp
    wn1_analog = fp_analog

    # Calculate attenuation of first filter at fs_edge
    # |H1(fs_edge)|^2 = 1 / (1 + (fs_edge/wn1)^(2n))
    denom1_fs = 1 + (fs_edge_analog / wn1_analog) ** (2*order)
    h1_sq_fs = 1 / denom1_fs
    h1_db_fs = -10 * np.log10(denom1_fs)  # should be approximately -A_s_db

    # We want combined filter to have exactly A_s_db at fs_edge and stay flat after
    # The second inverted filter cutoff should be at fs_edge so it starts boosting there
    # |H_total(fs_edge)|^2 = |H1|^2 * |1/H2|^2 = h1_sq * (1 + (fs_edge/wn2)^(2n)) = target
    #
    # At fs_edge, we want: h1_sq_fs * (1 + (fs_edge/wn2)^(2n)) = 10^(-A_s_db/10)
    # => (fs_edge/wn2)^(2n) = target/h1_sq_fs - 1
    # => wn2 = fs_edge / (target/h1_sq_fs - 1)^(1/(2n))

    target_ratio = 10 ** (-A_s_db / 10)
    ratio_pow = target_ratio / h1_sq_fs - 1

    if ratio_pow <= 0:
        # First filter alone has less attenuation than target at fs_edge
        # No flattening needed - just use first filter
        b, a = ssig.butter(order, fp, btype='low', analog=False, output='ba')
        _, hh = ssig.freqz(b, a, worN=[np.pi])
        actual_atten = -20 * np.log10(np.abs(hh[0]))
        return b, a, {
            'order': order, 'wn1': fp, 'wn2': None,
            'atten_at_nyquist': actual_atten
        }

    wn2_analog = fs_edge_analog / (ratio_pow ** (1 / (2*order)))

    # First lowpass (analog)
    b1_a, a1_a = ssig.butter(order, wn1_analog, btype='low', analog=True, output='ba')

    # Second lowpass (analog) - we will invert it
    b2_a, a2_a = ssig.butter(order, wn2_analog, btype='low', analog=True, output='ba')

    # Invert second filter: H2_inv = a2/b2
    # Combined analog: H_total = H1 * H2_inv = (b1/a1) * (a2/b2) = (b1*a2) / (a1*b2)
    b_analog = np.convolve(b1_a, a2_a)
    a_analog = np.convolve(a1_a, b2_a)

    # Convert to digital using bilinear transform
    b, a = ssig.bilinear(b_analog, a_analog, fs)

    # Normalize gain at DC
    dc_gain = np.sum(b) / np.sum(a)
    b = b / dc_gain

    # Convert wn2_analog back to digital for info
    wn2_digital = 2 * np.arctan(wn2_analog / (2*fs)) * fs / np.pi

    # Verify actual attenuation at Nyquist
    _, hh = ssig.freqz(b, a, worN=[np.pi])
    actual_atten = -20 * np.log10(np.abs(hh[0]))

    return b, a, {
        'order': order, 'wn1': fp, 'wn2': wn2_digital,
        'atten_at_nyquist': actual_atten
    }


def compare_filters(numtaps, bands, desired, n_fft=None, include_transitions=True):
    """
    Compare custom minimum-phase FIR with scipy minimum_phase and Butterworth.

    Parameters
    ----------
    numtaps : int
        Number of filter taps.
    bands : array-like
        Frequency band edges, normalized [0, 1].
    desired : array-like
        Desired magnitude at each band edge.
    n_fft : int, optional
        FFT length for plotting.
    include_transitions : bool, optional
        If True, include all transition types (cubic, quintic, raised_cosine, erf).
        Default is True.
    """
    if n_fft is None:
        n_fft = numutil.ceil_power2(4 * numtaps)

    bands = np.asarray(bands)
    desired = np.asarray(desired)

    nf = n_fft // 2 + 1
    ff = np.linspace(0, 1, nf)
    ww = ff * np.pi

    fax = None
    fig_gd, ax_gd = plt.subplots()

    # 1. Custom minimum-phase FIR with different transitions
    if include_transitions:
        transitions = ['cubic', 'quintic', 'raised_cosine', 'erf']
        for trans in transitions:
            fir_mp, _, _, _, _ = firmp_full(numtaps, bands, desired, n_fft=n_fft, transition=trans)
            hh_mp = np.fft.rfft(fir_mp, n=n_fft)
            _, gd_mp = group_delay_minphase_fir(fir_mp, n_fft)

            label = f"firmp ({trans})"
            if fax is None:
                fax = numplot.plotspek(hh_mp, ff, label=label)
            else:
                numplot.plotspek(hh_mp, ff, fax=fax, label=label)
            ax_gd.semilogx(ff[1:], gd_mp[1:], label=label, alpha=0.7)
    else:
        # Just cubic
        fir_mp, _, _, _, _ = firmp_full(numtaps, bands, desired, n_fft=n_fft)
        hh_mp = np.fft.rfft(fir_mp, n=n_fft)
        _, gd_mp = group_delay_minphase_fir(fir_mp, n_fft)
        fax = numplot.plotspek(hh_mp, ff, label="firmp (cubic)")
        ax_gd.semilogx(ff[1:], gd_mp[1:], label="firmp (cubic)")

    # 2. Scipy firwin + minimum_phase
    # Find cutoff at midpoint of transition band (assumes lowpass: bands[1] to bands[2])
    fc = (bands[1] + bands[2]) / 2
    fir_firwin = ssig.firwin(numtaps, fc, window='hamming')
    fir_firwin_mp = ssig.minimum_phase(fir_firwin, method='homomorphic', half=False)
    hh_firwin = np.fft.rfft(fir_firwin_mp, n=n_fft)
    _, gd_firwin = group_delay_minphase_fir(fir_firwin_mp, n_fft)
    numplot.plotspek(hh_firwin, ff, fax=fax, label="firwin + minimum_phase")
    ax_gd.semilogx(ff[1:], gd_firwin[1:], label="firwin + min_phase", linestyle='--')

    # 3. Butterworth lowpass (standard)
    # Estimate attenuation from desired stopband level
    A_s_db = -20 * np.log10(np.maximum(desired[-1], 1e-12))
    fp, fs_edge = bands[1], bands[2]
    butter_order, butter_wn = ssig.buttord(fp, fs_edge, 3.0, A_s_db)
    b_butter, a_butter = ssig.butter(butter_order, butter_wn, btype='low', analog=False, output='ba')
    _, hh_butter = ssig.freqz(b_butter, a_butter, worN=ww)
    _, gd_butter = ssig.group_delay((b_butter, a_butter), ww)
    numplot.plotspek(hh_butter, ff, fax=fax, label=f"Butterworth (order={butter_order})")
    ax_gd.semilogx(ff[1:], gd_butter[1:], label="Butterworth", linestyle='--')

    # 4. Minimum-phase Butterworth pair (lowpass / inverted lowpass for flattening)
    b_mpb, a_mpb, info_mpb = minphase_butterworth_pair(fp, fs_edge, A_s_db)
    _, hh_mpb = ssig.freqz(b_mpb, a_mpb, worN=ww)
    _, gd_mpb = ssig.group_delay((b_mpb, a_mpb), ww)
    wn2_str = f"{info_mpb['wn2']:.3f}" if info_mpb['wn2'] else "N/A"
    numplot.plotspek(hh_mpb, ff, fax=fax, label=f"MinPhase Butter pair (n={info_mpb['order']}, wn2={wn2_str})")
    ax_gd.semilogx(ff[1:], gd_mpb[1:], label="MinPhase Butter pair", linestyle='--')

    # Setup legends
    fax[0].legend_handler = numlegend.LegendHandler(fax[0].canvas)
    fax[0].legend_handler.setup(fax[1])
    fax[0].suptitle("Filter Comparison: Magnitude and Phase")

    ax_gd.set_xlabel('Normalized Frequency')
    ax_gd.set_ylabel('Group Delay [samples]')
    numlegend.gridsetup(ax_gd)
    fig_gd.legend_handler = numlegend.LegendHandler(fig_gd.canvas)
    fig_gd.legend_handler.setup(ax_gd)
    fig_gd.suptitle("Group Delay Comparison")

    return fax


def compare_band_filters(f0, Q_values, G_db, numtaps=256, n_fft=None, fs=2.0,
                         transitions=None):
    """
    Compare 2nd order S-domain bandpass filters with minimum phase FIR filters.

    The minimum phase FIR filters are designed to match the bandwidth of the
    S-domain filters at the specified attenuation level G_db.

    For a 2nd order bandpass filter:
        H(s) = (s * w0/Q) / (s^2 + s*w0/Q + w0^2)

    The -G dB bandwidth can be calculated from the Q factor.

    Parameters
    ----------
    f0 : float
        Center frequency (normalized, 0 to 1 where 1 = Nyquist).
    Q_values : array-like
        List of Q factors to compare.
    G_db : float
        Attenuation level in dB (positive) at which to match bandwidth.
        E.g., G_db=3 means match bandwidth at -3dB points.
    numtaps : int, optional
        Number of FIR taps. Default 256.
    n_fft : int, optional
        FFT length for plotting. Default is 4*numtaps.
    fs : float, optional
        Sample rate (default 2.0 for normalized frequencies).
    transitions : list of str, optional
        Transition types to include. Default is ['cubic', 'quintic'].

    Returns
    -------
    fax : tuple
        Figure and axes from plotspek.
    """
    if n_fft is None:
        n_fft = numutil.ceil_power2(4 * numtaps)

    if transitions is None:
        transitions = ['cubic', 'quintic']

    Q_values = np.asarray(Q_values)

    nf = n_fft // 2 + 1
    ff = np.linspace(0, 1, nf)  # normalized frequency 0 to Nyquist
    ww = ff * np.pi  # digital frequency
    ss = ww*1j # s domain

    # Analog frequency for bilinear transform
    # Pre-warp center frequency
    w0_analog = 2 * fs * np.tan(np.pi * f0 / fs)

    fax = None
    fig_gd, ax_gd = plt.subplots()
    fig_step, ax_step = plt.subplots()

    # Target linear magnitude at -G dB
    G_linear = 10 ** (-G_db / 20)

    # For each Q value, design S-domain bandpass and matching FIR filters
    for Q in Q_values:
        # --- 2nd order analog bandpass ---
        # H(s) = (s * w0/Q) / (s^2 + s*w0/Q + w0^2)
        b_analog = [w0_analog / Q, 0]
        a_analog = [1, w0_analog / Q, w0_analog ** 2]

        # Convert to digital using bilinear transform
        b_digital, a_digital = ssig.bilinear(b_analog, a_analog, fs)

        # Frequency response
        _, hh_bp = ssig.freqz(b_digital, a_digital, worN=ww)
        mag_bp = np.abs(hh_bp)

        # --- Find -G dB points numerically from actual response ---
        # Normalize magnitude (peak should be ~1 at f0)
        mag_bp_norm = mag_bp / np.max(mag_bp)

        # Find index closest to f0
        f0_idx = np.argmin(np.abs(ff - f0))

        # Find lower -G dB point (search from f0 down to DC)
        f_low = None
        for i in range(f0_idx, 0, -1):
            if mag_bp_norm[i] >= G_linear and mag_bp_norm[i-1] < G_linear:
                # Linear interpolation for more accurate crossing
                t = (G_linear - mag_bp_norm[i-1]) / (mag_bp_norm[i] - mag_bp_norm[i-1])
                f_low = ff[i-1] + t * (ff[i] - ff[i-1])
                break

        # Find upper -G dB point (search from f0 up to Nyquist)
        f_high = None
        for i in range(f0_idx, nf - 1):
            if mag_bp_norm[i] >= G_linear and mag_bp_norm[i+1] < G_linear:
                # Linear interpolation
                t = (G_linear - mag_bp_norm[i]) / (mag_bp_norm[i+1] - mag_bp_norm[i])
                f_high = ff[i] + t * (ff[i+1] - ff[i])
                break

        # Fallback if crossings not found (shouldn't happen for reasonable Q)
        if f_low is None:
            f_low = max(0.01, f0 - f0 / Q)
        if f_high is None:
            f_high = min(0.99, f0 + f0 / Q)

        print(f"Q={Q:.1f}: f_low={f_low:.4f}, f0={f0:.4f}, f_high={f_high:.4f}, "
              f"BW={f_high - f_low:.4f}")

        # Group delay
        _, gd_bp = ssig.group_delay((b_digital, a_digital), ww)

        # Step response (impulse response convolved with step)
        # For IIR, use scipy.signal.dlti
        t_iir = np.arange(2 * numtaps)
        _, step_bp = ssig.dstep((b_digital, a_digital, 1), t=t_iir)
        step_bp = np.squeeze(step_bp)

        # Plot S-domain bandpass
        label_bp = f"S-domain BP (Q={Q:.1f})"
        if fax is None:
            fax = numplot.plotspek(hh_bp, ff, label=label_bp)
        else:
            numplot.plotspek(hh_bp, ff, fax=fax, label=label_bp)
        ax_gd.semilogx(ff[1:], gd_bp[1:], label=label_bp, linewidth=2)
        ax_step.plot(t_iir, step_bp, label=label_bp, linewidth=2)

        # --- Minimum phase FIR filters matching bandwidth ---
        # Design bandpass FIR with same bandwidth at -G dB
        # We want unity gain at f0, and G_linear at f_low and f_high
        bands_fir = [0, f_low, f0, f_high, 1.0]
        desired_fir = [G_linear, G_linear, 1.0, G_linear, G_linear]

        for trans in transitions:
            try:
                fir_mp, H_mp, L_mp, _, _ = firmp_full(
                    numtaps, bands_fir, desired_fir, n_fft=n_fft, transition=trans
                )
                hh_fir = np.fft.rfft(fir_mp, n=n_fft)
                _, gd_fir = group_delay_minphase_fir(fir_mp, n_fft)
                t_fir, step_fir = step_response(fir_mp, n_samples=2 * numtaps)

                label_fir = f"FIR {trans} (Q={Q:.1f})"
                numplot.plotspek(hh_fir, ff, fax=fax, label=label_fir)
                ax_gd.semilogx(ff[1:], gd_fir[1:], label=label_fir, alpha=0.7, linestyle='--')
                ax_step.plot(t_fir, step_fir, label=label_fir, alpha=0.7, linestyle='--')
            except ValueError as e:
                print(f"Warning: Could not design FIR for Q={Q}, {trans}: {e}")

    # Setup legends and titles
    fax[0].legend_handler = numlegend.LegendHandler(fax[0].canvas)
    fax[0].legend_handler.setup(fax[1])
    fax[0].suptitle(f"Bandpass Filter Comparison (f0={f0:.2f}, G={G_db}dB)")

    ax_gd.set_xlabel('Normalized Frequency')
    ax_gd.set_ylabel('Group Delay [samples]')
    ax_gd.set_title(f'Group Delay Comparison (f0={f0:.2f})')
    numlegend.gridsetup(ax_gd)
    fig_gd.legend_handler = numlegend.LegendHandler(fig_gd.canvas)
    fig_gd.legend_handler.setup(ax_gd)

    ax_step.set_xlabel('Sample')
    ax_step.set_ylabel('Amplitude')
    ax_step.set_title(f'Step Response Comparison (f0={f0:.2f})')
    numlegend.gridsetup(ax_step)
    fig_step.legend_handler = numlegend.LegendHandler(fig_step.canvas)
    fig_step.legend_handler.setup(ax_step)

    return fax


def main():
    import matplotlib.pyplot as plt

    # # Specs in firls-like format
    # numtaps = 256
    # bands = [0, 0.3, 0.5, 1.0]  # normalized frequency edges
    # desired = [1, 1, 0.001, 0.001]  # -60 dB stopband
    # n_fft = 4096
    # plot_logmag_from_bands(bands, desired, n_points=n_fft)
    #
    # # Plot the designed filter with transition comparison
    # plot_firmp(numtaps, bands, desired, n_fft=n_fft, compare_transitions=True)
    #
    # # Compare with other methods (includes all transition types)
    # compare_filters(numtaps, bands, desired, n_fft=n_fft, include_transitions=True)

    # Compare bandpass filters with different Q values
    compare_band_filters(f0=0.3, Q_values=[1, 2, 5], G_db=20, numtaps=256)
    plt.show()


if __name__ == "__main__":
    main()

