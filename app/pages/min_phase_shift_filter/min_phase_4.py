import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

def normalize(u, u1, u2):
    """Normalize log-frequency to the [0, 1] range for interpolation."""
    return np.clip((u - u1) / (u2 - u1), 0, 1)

# 1. Cubic Transition (Minimum Energy/Curvature)
# Solves A''' = constant, A(0)=1, A(1)=0, A'(0)=0, A'(1)=0
def cubic_transition(u, u1, u2, G=1):
    t = normalize(u, u1, u2)
    # Standard smoothstep: 3t^2 - 2t^3 (normalized 0 to 1)
    # To transition from 0 down to -G:
    return -G * (3*t**2 - 2*t**3)

# 2. Quintic Transition (L'' Continuous)
# Solves A''=0 at boundaries, A(0)=1, A(1)=0, A'(0)=0, A'(1)=0, A''(0)=0, A''(1)=0
def quintic_transition(u, u1, u2, G=1):
    t = normalize(u, u1, u2)
    # Higher-order smoothstep: 10t^3 - 15t^4 + 6t^5
    return -G * (10*t**3 - 15*t**4 + 6*t**5)

# 3. Raised Cosine Transition
# Uses a half-period of a cosine to transition the slope smoothly.
def raised_cosine_transition(u, u1, u2, G=1):
    t = normalize(u, u1, u2)
    # Transitions from 0 to -G using 0.5 * (1 - cos(pi * t))
    return -G * 0.5 * (1 - np.cos(np.pi * t))

# 4. ERF (Error Function) Transition
# Provides C-infinity continuity; sigma controls transition "sharpness".
def erf_transition(u, u1, u2, G=1, sigma=0.25):
    # Center the ERF in the middle of the transition band
    u_mid = (u1 + u2) / 2
    # Adjust scale so ERF covers the transition range roughly at +/- 2*sigma
    scale = (u2 - u1) / 4 
    z = (u - u_mid) / (sigma * scale)
    # erf(z) goes from -1 to 1; we shift and scale to 0 to -G
    return -G * 0.5 * (1 + erf(z))

# --- Visualization ---
u = np.linspace(0, 10, 1000)
u1, u2 = 3, 7  # Transition band edges
G = 60         # Stopband attenuation in dB

plt.figure(figsize=(10, 6))
plt.plot(u, cubic_transition(u, u1, u2, G), label='Cubic ($L^2$ Min)')
plt.plot(u, quintic_transition(u, u1, u2, G), label='Quintic ($L\'\'$ Cont.)', linestyle='--')
plt.plot(u, raised_cosine_transition(u, u1, u2, G), label='Raised Cosine', alpha=0.7)
plt.plot(u, erf_transition(u, u1, u2, G), label='ERF (Gaussian)', linestyle=':')

plt.axvline(u1, color='k', alpha=0.2, label='Passband Edge')
plt.axvline(u2, color='k', alpha=0.2, label='Stopband Edge')
plt.title("Log-Magnitude Transition Curves for Minimum Group Delay")
plt.ylabel("Attenuation (dB)")
plt.xlabel("Log Frequency ($u$)")
plt.legend()
plt.grid(True)
plt.show()
