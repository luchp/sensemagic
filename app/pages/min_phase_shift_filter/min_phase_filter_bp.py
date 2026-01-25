import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

def erf_bandpass(u, uL, uH, Gleft, Gright, sigmaL=0.25, sigmaH=0.25):
    """
    ERF-based bandpass bump in log-frequency domain.

    Parameters
    ----------
    u : array
        Log-frequency axis (e.g., ln(omega)).
    uL : float
        Low-frequency transition center.
    uH : float
        High-frequency transition center.
    Gleft : float
        Stopband attenuation on the low side (positive number, in dB).
    Gright : float
        Stopband attenuation on the high side (positive number, in dB).
    sigmaL : float
        Width parameter for the low-side ERF.
    sigmaH : float
        Width parameter for the high-side ERF.

    Returns
    -------
    A : array
        Log-magnitude in dB (negative values).
    """

    # Low-side transition: -Gleft → 0
    A_low = -0.5 * Gleft * (1 + erf((u - uL) / sigmaL))

    # High-side transition: 0 → -Gright
    A_high = -0.5 * Gright * (1 - erf((u - uH) / sigmaH))

    # Combined bandpass bump
    return A_low + A_high


# --- Example usage ---
u = np.linspace(0, 10, 2000)

# Transition centers in log-frequency
uL = 3.0
uH = 7.0

# Independent left/right attenuation
Gleft = 50     # dB
Gright = 70    # dB

A = erf_bandpass(u, uL, uH, Gleft, Gright, sigmaL=0.3, sigmaH=0.3)

plt.figure(figsize=(10, 6))
plt.plot(u, A, label="ERF Bandpass Bump")
plt.axvline(uL, color='k', alpha=0.2)
plt.axvline(uH, color='k', alpha=0.2)
plt.title("ERF-Based Bandpass Log-Magnitude")
plt.xlabel("Log Frequency (u)")
plt.ylabel("Attenuation (dB)")
plt.grid(True)
plt.legend()
plt.show()