"""
Minimum Phase Lowpass Filter Optimization using Real Cepstrum

Optimizes log-magnitude values directly at each frequency point with constraints:
  - log(H(0)) = 0  (DC gain = 1)
  - log(H) is monotonically decreasing
  - Concave shape (no bumps)

The minimum phase is computed via the Hilbert transform relationship:
  Phase = Hilbert(log|H|)

For a minimum-phase system:
  - Cepstrum c[n] = IDFT(log|H|)
  - Phase is derived from the causal cepstrum
"""

import numpy as np
from scipy.optimize import minimize
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt


def log_mag_to_min_phase_response(log_mag: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert log-magnitude spectrum to minimum-phase frequency response.

    Uses the real cepstrum approach:
    1. Create symmetric log spectrum: log|H(-f)| = log|H(f)|
    2. IFFT to get real cepstrum
    3. Make cepstrum causal (multiply by 2 for n>0)
    4. FFT back to get log|H| + j*phase

    Parameters:
        log_mag: Log-magnitude at frequencies 0 to Nyquist (n_freq points)
                 This is ln|H|, not dB!

    Returns:
        freqs: Normalized frequencies 0 to 1 (Nyquist)
        mag_db: Magnitude in dB
        phase_deg: Minimum phase in degrees
    """
    n_freq = len(log_mag)
    n_fft = 2 * (n_freq - 1)

    # Create full symmetric spectrum for FFT
    log_mag_full = np.zeros(n_fft)
    log_mag_full[:n_freq] = log_mag
    log_mag_full[n_freq:] = log_mag[-2:0:-1]  # Mirror (excluding DC and Nyquist)

    # Get real cepstrum via IFFT
    cepstrum = np.real(ifft(log_mag_full))

    # Create minimum-phase cepstrum (causal part doubled)
    cep_mp = np.zeros(n_fft)
    cep_mp[0] = cepstrum[0]
    cep_mp[1:n_fft//2] = 2 * cepstrum[1:n_fft//2]
    cep_mp[n_fft//2] = cepstrum[n_fft//2]  # Nyquist

    # FFT to get complex log spectrum
    log_spectrum = fft(cep_mp)

    # Imaginary part is phase
    phase_rad = np.imag(log_spectrum[:n_freq])

    # Convert to dB and degrees
    mag_db = log_mag * (20.0 / np.log(10.0))
    phase_deg = np.degrees(phase_rad)
    freqs = np.linspace(0, 1, n_freq)

    return freqs, mag_db, phase_deg


class MinPhaseFilter:
    """
    Minimum-phase lowpass filter designed by optimizing log-magnitude.

    Constraints:
    - log(H(0)) = 0 (unity DC gain)
    - log(H) is monotonically decreasing (with 0.1 dB tolerance)
    - Concave shape (no bumps)
    - Phase monotonically decreasing in passband

    Works well for attenuations up to ~15 dB at the target frequency.
    """

    def __init__(self, target_freq: float, attenuation_db: float, n_freq: int = 256):
        """
        Design a minimum-phase lowpass filter.

        Parameters:
            target_freq: Target frequency as fraction of Nyquist (0 to 1)
            attenuation_db: Minimum attenuation required at target_freq (positive dB)
            n_freq: Number of frequency points
        """
        self.target_freq = target_freq
        self.attenuation_db = attenuation_db
        self.n_freq = n_freq
        self.target_idx = int(target_freq * (n_freq - 1))

        # Run optimization
        self._optimize()

    def _optimize(self):
        """Run the optimization to find optimal log-magnitude."""
        n_freq = self.n_freq
        target_idx = self.target_idx
        target_atten_neper = self.attenuation_db * np.log(10) / 20

        def objective(log_mag):
            """Minimize phase at target frequency while meeting constraints."""
            log_mag_full = np.zeros(n_freq)
            log_mag_full[0] = 0.0
            log_mag_full[1:] = log_mag

            _, mag_db, phase_deg = log_mag_to_min_phase_response(log_mag_full)

            phase_cost = abs(phase_deg[target_idx])

            # Tolerances
            mag_tol_neper = 0.1 * np.log(10) / 20  # 0.1 dB
            phase_tol_deg = 0.05

            # 1. Monotonically decreasing magnitude in passband
            diff_passband = np.diff(log_mag_full[:target_idx+1])
            mono_violation = np.maximum(0, diff_passband - mag_tol_neper)
            mono_penalty = 1e7 * np.sum(mono_violation**2)

            # 2. Concave (no bumps) in passband
            if target_idx >= 2:
                d2 = log_mag_full[2:target_idx+1] - 2*log_mag_full[1:target_idx] + log_mag_full[:target_idx-1]
                bump_violation = np.maximum(0, d2)
                bump_penalty = 1e7 * np.sum(bump_violation**2)
            else:
                bump_penalty = 0

            # 3. Phase monotonically decreasing in passband
            diff_phase = np.diff(phase_deg[:target_idx+1])
            phase_increase = np.maximum(0, diff_phase - phase_tol_deg)
            phase_mono_penalty = 1e8 * np.sum(phase_increase**2)

            # 4. Must reach target attenuation
            target_violation = np.maximum(0, log_mag_full[target_idx] + target_atten_neper)
            target_penalty = 1e6 * target_violation**2

            # 5. Stopband must stay at or below target
            stopband_violation = np.maximum(0, log_mag_full[target_idx:] + target_atten_neper)
            stopband_penalty = 1e6 * np.sum(stopband_violation**2)

            # 6. Encourage flat stopband (minimizes phase at target)
            stopband_deviation = np.abs(log_mag_full[target_idx:] + target_atten_neper)
            flat_penalty = 10 * np.sum(stopband_deviation**2)

            return (phase_cost + mono_penalty + bump_penalty + phase_mono_penalty +
                    target_penalty + stopband_penalty + flat_penalty)

        # Initial guess: first-order response clipped to flat stopband
        fc = self.target_freq / np.sqrt(10**(self.attenuation_db/10) - 1)
        f = np.linspace(0, 1, n_freq)
        f_ratio = f[1:] / fc
        initial = -0.5 * np.log(1 + f_ratio**2)
        initial = np.maximum(initial, -target_atten_neper)

        # Bounds
        bounds = [(-target_atten_neper * 2, 0.0)] * (n_freq - 1)

        # Optimize
        result = minimize(
            objective, initial, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 10000, 'ftol': 1e-12}
        )

        # Store final response
        final_log_mag = np.zeros(n_freq)
        final_log_mag[0] = 0.0
        final_log_mag[1:] = result.x

        self.freqs, self.magnitude_db, self.phase_deg = log_mag_to_min_phase_response(final_log_mag)
        self.log_mag = final_log_mag
        self.converged = result.success

    @property
    def phase_at_target(self) -> float:
        """Phase in degrees at target frequency."""
        return self.phase_deg[self.target_idx]

    @property
    def attenuation_at_target(self) -> float:
        """Actual attenuation in dB at target frequency."""
        return -self.magnitude_db[self.target_idx]


class FirstOrderFilter:
    """
    Analytical first-order (RC) lowpass filter.

    Transfer function: H(s) = 1 / (1 + s/ωc)

    |H(f)| = 1 / √(1 + (f/fc)²)
    ∠H(f) = -arctan(f/fc)
    """

    def __init__(self, target_freq: float, attenuation_db: float, n_freq: int = 256):
        """
        Design a first-order filter for specific attenuation at target frequency.

        Parameters:
            target_freq: Target frequency as fraction of Nyquist
            attenuation_db: Desired attenuation at target frequency (positive dB)
            n_freq: Number of frequency points
        """
        self.target_freq = target_freq
        self.attenuation_db = attenuation_db
        self.n_freq = n_freq
        self.target_idx = int(target_freq * (n_freq - 1))

        # Calculate cutoff frequency for desired attenuation
        self.cutoff_freq = target_freq / np.sqrt(10**(attenuation_db / 10) - 1)

        # Compute response
        self.freqs = np.linspace(0, 1, n_freq)
        f_ratio = self.freqs / self.cutoff_freq

        self.magnitude_db = -10 * np.log10(1 + f_ratio**2)
        self.phase_deg = -np.degrees(np.arctan(f_ratio))

    @property
    def phase_at_target(self) -> float:
        """Phase in degrees at target frequency."""
        return self.phase_deg[self.target_idx]

    @property
    def attenuation_at_target(self) -> float:
        """Actual attenuation in dB at target frequency."""
        return -self.magnitude_db[self.target_idx]


class LeadLagFilter:
    """
    Lead-lag filter (first-order zero and pole).

    Transfer function: H(s) = (1 + s/ωz) / (1 + s/ωp)

    Or equivalently: H(s) = (1 + a·s) / (1 + b·s)  where a = 1/ωz, b = 1/ωp

    For attenuation (a < b, i.e., ωz > ωp):
    - DC gain = 1 (0 dB)
    - High-freq gain = a/b = ωp/ωz < 1 (negative dB)
    - Phase: arctan(ω/ωz) - arctan(ω/ωp), starts at 0, goes negative, returns to 0

    |H(f)| = √(1 + (f/fz)²) / √(1 + (f/fp)²)
    ∠H(f) = arctan(f/fz) - arctan(f/fp)

    We optimize fz and fp to achieve target attenuation at target_freq
    while minimizing phase lag at that frequency.
    """

    def __init__(self, target_freq: float, attenuation_db: float, n_freq: int = 256):
        """
        Design a lead-lag filter for specific attenuation at target frequency,
        optimized to minimize phase lag.

        Parameters:
            target_freq: Target frequency as fraction of Nyquist
            attenuation_db: Desired attenuation at target frequency (positive dB)
            n_freq: Number of frequency points
        """
        self.target_freq = target_freq
        self.attenuation_db = attenuation_db
        self.n_freq = n_freq
        self.target_idx = int(target_freq * (n_freq - 1))

        # Optimize fz and fp to minimize phase while achieving target attenuation
        self._optimize()

    def _optimize(self):
        """Find optimal zero and pole frequencies.

        For minimum magnitude exactly at target_freq:
        - Geometric mean of pole and zero = target_freq: sqrt(fp * fz) = target_freq
        - At f = sqrt(fp*fz): |H|² = (1 + fz/fp) / (1 + fp/fz)

        Let r = fz/fp (> 1 for attenuation). At geometric mean:
        |H|² = (1 + r) / (1 + 1/r) = (1 + r) * r / (r + 1) = r
        So |H| = sqrt(r) = sqrt(fz/fp)

        For attenuation A dB: |H| = 10^(-A/20)
        sqrt(fz/fp) = 10^(-A/20)
        fz/fp = 10^(-A/10)

        With sqrt(fp * fz) = target_freq:
        fp = target_freq * (fz/fp)^(-1/2) = target_freq * 10^(A/20)
        fz = target_freq * (fz/fp)^(1/2) = target_freq * 10^(-A/20)

        Wait, let's verify: fz/fp = 10^(-A/20) / 10^(A/20) = 10^(-A/10) ✓
        """
        target_freq = self.target_freq
        target_atten = self.attenuation_db
        n_freq = self.n_freq

        # At geometric mean f = sqrt(fp*fz), magnitude = sqrt(fz/fp)
        # For target attenuation: sqrt(fz/fp) = 10^(-atten/20)
        # So fz/fp = 10^(-atten/10)
        #
        # With fp * fz = target_freq², and fz = fp * 10^(-atten/10):
        # fp² * 10^(-atten/10) = target_freq²
        # fp = target_freq * 10^(atten/20)
        # fz = target_freq * 10^(-atten/20)

        self.pole_freq = target_freq * 10**(target_atten / 20)
        self.zero_freq = target_freq * 10**(-target_atten / 20)

        # Compute full response
        self.freqs = np.linspace(0, 1, n_freq)
        fp_ratio = self.freqs / self.pole_freq
        fz_ratio = self.freqs / self.zero_freq

        self.magnitude_db = 10 * np.log10(1 + fz_ratio**2) - 10 * np.log10(1 + fp_ratio**2)
        self.phase_deg = np.degrees(np.arctan(fz_ratio) - np.arctan(fp_ratio))

    @property
    def phase_at_target(self) -> float:
        """Phase in degrees at target frequency."""
        return self.phase_deg[self.target_idx]

    @property
    def attenuation_at_target(self) -> float:
        """Actual attenuation in dB at target frequency."""
        return -self.magnitude_db[self.target_idx]



def plot_comparison(target_freq: float = 0.2, attenuation_db: float = 10, n_freq: int = 256):
    """Plot comparison between first-order and optimized filter."""

    fo = FirstOrderFilter(target_freq, attenuation_db, n_freq)
    opt = MinPhaseFilter(target_freq, attenuation_db, n_freq)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    freq_min = 0.01
    freq_mask = fo.freqs >= freq_min

    # Magnitude
    ax = axes[0]
    ax.semilogx(fo.freqs[freq_mask], fo.magnitude_db[freq_mask], 'b--',
                linewidth=2, label='First-Order')
    ax.semilogx(opt.freqs[freq_mask], opt.magnitude_db[freq_mask], 'r-',
                linewidth=2, label='Optimized')
    ax.axvline(target_freq, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(-attenuation_db, color='green', linestyle=':', alpha=0.5,
               label=f'-{attenuation_db} dB')
    ax.set_xlabel('Frequency (× Nyquist)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(f'Magnitude Response: {attenuation_db} dB @ f={target_freq}')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([freq_min, 1])
    ax.set_ylim([-30, 5])

    # Phase
    ax = axes[1]
    ax.semilogx(fo.freqs[freq_mask], fo.phase_deg[freq_mask], 'b--',
                linewidth=2, label=f"First-Order: {fo.phase_at_target:.1f}°")
    ax.semilogx(opt.freqs[freq_mask], opt.phase_deg[freq_mask], 'r-',
                linewidth=2, label=f"Optimized: {opt.phase_at_target:.1f}°")
    ax.axvline(target_freq, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Frequency (× Nyquist)')
    ax.set_ylabel('Phase (°)')
    improvement = abs(fo.phase_at_target) - abs(opt.phase_at_target)
    ax.set_title(f'Phase Response (Optimized saves {improvement:.1f}° at target)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([freq_min, 1])

    plt.tight_layout()
    return fig


def plot_multiple_attenuations(target_freq: float = 0.2, attenuations: list = [10, 20, 30], n_freq: int = 256):
    """Plot comparison for multiple attenuation levels."""

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    freq_min = 0.01

    colors = ['blue', 'green', 'red']

    for atten, color in zip(attenuations, colors):
        fo = FirstOrderFilter(target_freq, atten, n_freq)
        ll = LeadLagFilter(target_freq, atten, n_freq)
        opt = MinPhaseFilter(target_freq, atten, n_freq)

        freq_mask = fo.freqs >= freq_min

        # Magnitude
        axes[0].semilogx(fo.freqs[freq_mask], fo.magnitude_db[freq_mask], '--',
                         color=color, linewidth=1.5, alpha=0.5)
        axes[0].semilogx(ll.freqs[freq_mask], ll.magnitude_db[freq_mask], ':',
                         color=color, linewidth=1.5, alpha=0.7)
        axes[0].semilogx(opt.freqs[freq_mask], opt.magnitude_db[freq_mask], '-',
                         color=color, linewidth=2, label=f'{atten} dB')

        # Phase
        axes[1].semilogx(fo.freqs[freq_mask], fo.phase_deg[freq_mask], '--',
                         color=color, linewidth=1.5, alpha=0.5)
        axes[1].semilogx(ll.freqs[freq_mask], ll.phase_deg[freq_mask], ':',
                         color=color, linewidth=1.5, alpha=0.7)
        axes[1].semilogx(opt.freqs[freq_mask], opt.phase_deg[freq_mask], '-',
                         color=color, linewidth=2,
                         label=f'{atten} dB: 1st={fo.phase_at_target:.0f}°, LL={ll.phase_at_target:.0f}°, opt={opt.phase_at_target:.0f}°')

    # Magnitude plot
    ax = axes[0]
    ax.axvline(target_freq, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Frequency (× Nyquist)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(f'Magnitude Response @ f={target_freq} (dashed=1st-order, dotted=lead-lag, solid=optimized)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([freq_min, 1])
    ax.set_ylim([-50, 5])

    # Phase plot
    ax = axes[1]
    ax.axvline(target_freq, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Frequency (× Nyquist)')
    ax.set_ylabel('Phase (°)')
    ax.set_title('Phase Response (dashed=1st-order, dotted=lead-lag, solid=optimized)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([freq_min, 1])

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("=" * 70)
    print("COMPARISON: First-Order vs Lead-Lag vs Optimized Filter")
    print("=" * 70)

    target_freq = 0.2
    attenuations = [10, 20, 30]
    n_freq = 256

    print(f"\nTarget frequency: {target_freq} × Nyquist")
    print("-" * 70)
    print(f"{'Atten (dB)':<12} {'1st-Order':<14} {'Lead-Lag':<14} {'Optimized':<14} {'Saved vs 1st':<12}")
    print("-" * 70)

    for atten in attenuations:
        fo = FirstOrderFilter(target_freq, atten, n_freq)
        ll = LeadLagFilter(target_freq, atten, n_freq)
        opt = MinPhaseFilter(target_freq, atten, n_freq)
        improvement = abs(fo.phase_at_target) - abs(opt.phase_at_target)
        print(f"{atten:<12.0f} {fo.phase_at_target:<14.1f}° {ll.phase_at_target:<14.1f}° {opt.phase_at_target:<14.1f}° {improvement:+.1f}°")

    print("-" * 70)
    print("\nFirst-order: H(s) = 1/(1+s/ωp)              -20 dB/decade, monotonic phase")
    print("Lead-lag:    H(s) = (1+s/ωz)/(1+s/ωp)       phase returns to 0 at high freq")
    print("Optimized:   flat stopband via cepstrum     minimizes phase at target")

    # Plot all 3 cases
    plot_multiple_attenuations(target_freq, attenuations, n_freq)
    plt.show()

