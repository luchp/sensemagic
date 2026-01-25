"""
Lead Compensator Design using COBYLA Optimization

This module designs optimal unity-gain lead compensators for ALL phase margins (0-90°)
using 2D constrained optimization with the COBYLA method.

The lead compensator has the transfer function:
    H(s) = (1 + s/ω_z) / (1 + s/ω_p)

Where ω_z (zero) < ω_p (pole), providing:
- Unity DC gain (does not affect low-frequency PID region)
- High-frequency gain = α = ω_p/ω_z (constrained to ≤ 10 for robustness)
- Positive phase shift between ω_z and ω_p

The α ≤ 10 constraint is critical for practical implementation:
- Prevents noise amplification at high frequencies
- Avoids exciting unmodeled parasitic dynamics
- Ensures implementability with real hardware
"""

import numpy as np
from scipy.optimize import minimize
from typing import Any, Dict, List


class LeadCompensatorDesign:
    """
    Design a lead compensator using 3D constrained optimization.

    Uses COBYLA to find optimal (A, wz, alpha) that maximizes the DC gain A
    while maintaining the desired phase margin.

    The key insight: lead compensation allows pushing the crossover higher,
    which enables increasing the DC gain in the PID region.

    Attributes
    ----------
    tau : float
        System delay in seconds
    phi_m_deg : float
        Desired phase margin in degrees
    wc1 : float
        Original (baseline) crossover frequency (rad/s)
    A : float
        DC gain boost factor (A >= 1)
    alpha : float
        Pole/zero ratio (wp/wz, also equals high-frequency gain of lead)
    wc2 : float
        New crossover frequency (rad/s) with lead compensation
    wz : float
        Lead zero frequency (rad/s)
    wp : float
        Lead pole frequency (rad/s)
    """

    # Practical numerical tolerances used throughout.
    _REL_TOL = 1e-2
    _ABS_TOL = 1e-12

    def __init__(self, tau, phi_m_deg, *, alpha_max: float = 10.0):
        """
        Initialize and design the lead compensator.

        Parameters
        ----------
        tau : float
            System delay in seconds (must be positive)
        phi_m_deg : float
            Desired phase margin in degrees (must be between 0 and 90)
        alpha_max : float, optional
            Maximum allowed high-frequency gain of the lead compensator (alpha).
            Defaults to 10.

        Raises
        ------
        ValueError
            If tau or phi_m_deg are invalid
        RuntimeError
            If optimization fails to converge
        """
        # Validate inputs
        if tau <= 0:
            raise ValueError(f"Delay tau must be positive, got {tau}")
        if not (0 < phi_m_deg < 90):
            # Exclude 0 and 90 to avoid degenerate crossover (wc1 -> pi/(2*tau) or 0)
            raise ValueError(f"Phase margin must be strictly between 0° and 90°, got {phi_m_deg}°")
        if alpha_max <= 1:
            raise ValueError(f"alpha_max must be > 1, got {alpha_max}")

        self.tau = float(tau)
        self.phi_m_deg = float(phi_m_deg)
        self.alpha_max = float(alpha_max)

        # Original (baseline) crossover from delay and phase margin
        self.wc1 = self._compute_wc1_rad_s()

        # Compute baseline gain A1:
        # Model: |H(jω)| = A / ω  (first-order slope)
        # At baseline crossover wc1: |H(jwc1)| = A1 / wc1 = 1
        # Therefore: A1 = wc1
        self.A1 = self.wc1

        # Design via optimization to maximize A
        self._design_with_cobyla()

    def _compute_wc1_rad_s(self) -> float:
        """Compute the baseline crossover ωc1 [rad/s] from delay and phase margin.

        Without lead filter:
        - At crossover: phase = -π/2 - ωc1*τ = -π + margin_rad
        - Solving: ωc1 = (π/2 - margin_rad) / τ

        Note: This is in rad/s, not Hz!

        Returns
        -------
        float
            Baseline crossover frequency in rad/s.
        """
        margin_rad = self.phi_m_deg * np.pi / 180.0
        return (np.pi / 2.0 - margin_rad) / self.tau

    def _eval_wc_pm(self, A: float, wz: float, alpha: float) -> tuple[float, float]:
        """Evaluate crossover and phase margin robustly.

        Parameters
        ----------
        A : float
            DC gain boost factor
        wz : float
            Lead zero frequency
        alpha : float
            Pole/zero ratio
        """
        # clamp for numerical safety only
        if not np.isfinite(A) or not np.isfinite(wz) or not np.isfinite(alpha):
            return float('nan'), float('nan')

        A_eff = max(float(A), 1.0)
        wz_eff = max(float(wz), self._ABS_TOL)
        alpha_eff = max(float(alpha), 1.0)

        try:
            wc = self._solve_crossover_rad_s(A=A_eff, wz=wz_eff, alpha=alpha_eff)
            pm = self._phase_margin_deg_at(wc=wc, wz=wz_eff, alpha=alpha_eff)
            return float(wc), float(pm)
        except Exception:
            # If the analytic solve fails, return NaNs
            return float('nan'), float('nan')


    def _solve_crossover_rad_s(self, A: float, wz: float, alpha: float) -> float:
        """Solve for crossover frequency ωc [rad/s] from the magnitude equation.

        Model: |L(jω)| = A/ω * |F(jω)| where A is the DC gain (slope gain).
        At crossover: A/ωc * |F(jωc)| = 1

        Parameters
        ----------
        A : float
            DC gain (slope gain, A >= A1 = wc1)
        wz : float
            Lead zero frequency (rad/s)
        alpha : float
            Pole/zero ratio (wp/wz)

        Returns
        -------
        float
            Crossover frequency in rad/s
        """
        if A <= 0:
            raise ValueError(f"A must be > 0, got {A}")
        if wz <= 0:
            raise ValueError(f"wz must be > 0, got {wz}")

        if alpha <= 1.0 + self._REL_TOL:
            # No lead, just A/ω: A/ωc = 1 => ωc = A
            return float(A)

        # Derivation: (A/wc) * sqrt((1+(wc/wz)²)/(1+(wc/wp)²)) = 1
        # Square: A²/x * (1 + x/wz²) = 1 + x/wp²  where x=wc²
        # A² * (1 + x/wz²) = x * (1 + x/wp²)
        # A² + A²*x/wz² = x + x²/wp²
        # Multiply by wz²*wp²:
        # A²*wz²*wp² + A²*wp²*x = x*wz²*wp² + x²*wz²
        # Rearrange:
        # x²*wz² + x*(wz²*wp² - A²*wp²) - A²*wz²*wp² = 0
        # Divide by wz²:
        # x² + x*wp²*(1 - A²/wz²) - A²*wp² = 0
        wp = alpha * wz
        A_sq = A * A

        a_coef = 1.0
        b_coef = (wp * wp) * (wz * wz - A_sq) / (wz * wz)
        c_coef = -A_sq * (wp * wp)

        disc = b_coef * b_coef - 4.0 * a_coef * c_coef
        if disc < 0:
            raise RuntimeError(f"No real crossover solution (discriminant < 0)")

        sqrt_disc = float(np.sqrt(disc))
        x1 = (-b_coef + sqrt_disc) / (2.0 * a_coef)
        x2 = (-b_coef - sqrt_disc) / (2.0 * a_coef)

        xs = [x for x in (x1, x2) if x > 0]
        if not xs:
            raise RuntimeError(f"No positive crossover solution")

        x = max(xs)
        wc = float(np.sqrt(x))
        if not np.isfinite(wc) or wc <= 0:
            raise RuntimeError(f"Invalid crossover result wc={wc}")
        return wc

    def _phase_margin_deg_at(self, wc: float, wz: float, alpha: float) -> float:
        """Compute phase margin at ω=wc for the modeled loop."""
        if wz <= 0:
            raise ValueError(f"wz must be > 0, got {wz}")

        wp = alpha * wz
        phase_plant = -90.0
        phase_delay = -(wc * self.tau) * (180.0 / np.pi)
        phase_lead = np.degrees(np.arctan(wc / wz) - np.arctan(wc / wp))
        total_phase = phase_plant + phase_delay + phase_lead
        return 180.0 + total_phase

    def _design_with_cobyla(self):
        """
        Design lead compensator using 3D constrained optimization (COBYLA).

        Optimization variables: (A, wz, alpha) where:
        - A: DC gain boost (>= 1)
        - wz: zero frequency (>= wc1, the original crossover)
        Optimization variables (normalized to be O(1)):
        - A_ratio = A/A1 (DC gain ratio, >= 1)
        - wz_ratio = wz/wc1 (zero frequency ratio, >= 1)
        - alpha: pole/zero ratio (1 <= alpha <= alpha_max)

        This normalization makes the problem tau-independent and numerically
        better conditioned.

        Objective: Maximize A_ratio (the DC gain improvement)
        """
        # Normalized bounds (all O(1))
        A_ratio_min = 1.0  # No worse than baseline
        A_ratio_max = 100.0  # Very large upper bound

        wz_ratio_min = 1.0  # Zero at or above baseline crossover
        wz_ratio_max = 100.0  # Very large upper bound

        alpha_min = 1.0
        alpha_max = self.alpha_max

        def objective(x):
            A_ratio, wz_ratio, alpha = x
            # Maximize A_ratio (minimize -A_ratio)
            return -A_ratio

        def constraint_phase_margin(x):
            A_ratio, wz_ratio, alpha = x
            # Convert back to absolute values
            A = A_ratio * self.A1
            wz = wz_ratio * self.wc1
            _wc, pm = self._eval_wc_pm(A=A, wz=wz, alpha=alpha)
            # If pm invalid, treat as violated
            if not np.isfinite(pm):
                return -1e6
            return pm - self.phi_m_deg

        # Initial guess (all O(1)) - start conservatively
        A_ratio_init = 1.0  # Start at baseline (always feasible)
        wz_ratio_init = 1.0  # Zero well above baseline
        alpha_init = min(2.0, self.alpha_max)  # Start with more lead
        x0 = [A_ratio_init, wz_ratio_init, alpha_init]

        # Box bounds (all O(1))
        bounds = [(A_ratio_min, A_ratio_max), (wz_ratio_min, wz_ratio_max), (alpha_min, alpha_max)]

        constraints: List[Dict[str, Any]] = [
            {'type': 'ineq', 'fun': constraint_phase_margin},
        ]

        result = minimize(
            objective,
            x0,
            method='COBYLA',
            bounds=bounds,
            constraints=constraints,  # type: ignore[arg-type]
            options={'maxiter': 10000, 'rhobeg': 1.0, 'catol': 1e-4, 'disp': False},
        )

        if not result.success:
            raise RuntimeError(
                f"COBYLA optimization failed for phi_m_deg={self.phi_m_deg}°. "
                f"Message: {result.message}"
            )

        # Extract normalized results
        A_ratio = float(result.x[0])
        wz_ratio = float(result.x[1])
        self.alpha = float(result.x[2])

        # Convert back to absolute values
        self.A = A_ratio * self.A1
        self.wz = wz_ratio * self.wc1
        self.wp = self.alpha * self.wz

        self.wc2, pm2 = self._eval_wc_pm(A=self.A, wz=self.wz, alpha=self.alpha)

        # Minimal post-checks: trust COBYLA/bounds if it says success.
        for name, val in (('A', self.A), ('wz', self.wz), ('alpha', self.alpha), ('wp', self.wp), ('wc2', self.wc2), ('pm2', pm2)):
            if not np.isfinite(val):
                raise RuntimeError(f"Internal error: non-finite {name}={val} after successful optimization")

        if self.A <= 0 or self.wz <= 0 or self.wp <= 0 or self.wc2 <= 0:
            raise RuntimeError(
                f"Internal error: designed non-positive value (A={self.A}, wz={self.wz}, wp={self.wp}, wc2={self.wc2})"
            )

        if self.wp < self.wz:
            raise RuntimeError(
                f"Internal error: designed wp < wz (wz={self.wz}, wp={self.wp})."
            )


    def L0_phase(self, w):
        """Original loop phase (asymptotic).

        Parameters
        ----------
        w : float or ndarray
            Frequency in rad/s

        Returns
        -------
        float or ndarray
            Phase in degrees
        """
        w = np.asarray(w)
        return -90.0 - (w * self.tau) * (180.0 / np.pi)

    def L0_mag(self, w):
        """Original loop magnitude (asymptotic).

        Model: |L0(jω)| ≈ ωc1 / ω

        Parameters
        ----------
        w : float or ndarray
            Frequency in rad/s. Must be strictly positive.

        Returns
        -------
        float or ndarray
            Magnitude (linear, not dB)

        Raises
        ------
        ValueError
            If any frequency is <= 0.
        """
        w_arr = np.asarray(w)
        if np.any(w_arr <= 0):
            raise ValueError(f"Frequency w must be > 0 rad/s, got min(w)={float(np.min(w_arr))}")
        return self.wc1 / w_arr

    def lead_mag(self, w):
        """
        Lead compensator magnitude.

        Parameters
        ----------
        w : float or ndarray
            Frequency in rad/s

        Returns
        -------
        float or ndarray
            Magnitude (linear, not dB)
        """
        return np.sqrt((1 + (w/self.wz)**2) / (1 + (w/self.wp)**2))

    def lead_phase(self, w):
        """
        Lead compensator phase.

        Parameters
        ----------
        w : float or ndarray
            Frequency in rad/s

        Returns
        -------
        float or ndarray
            Phase in degrees
        """
        return np.degrees(np.arctan(w/self.wz) - np.arctan(w/self.wp))

    def compensated_mag(self, w):
        """
        Compensated loop magnitude.

        Parameters
        ----------
        w : float or ndarray
            Frequency in rad/s

        Returns
        -------
        float or ndarray
            Magnitude (linear, not dB)
        """
        return self.L0_mag(w) * self.lead_mag(w)

    def compensated_phase(self, w):
        """
        Compensated loop phase.

        Parameters
        ----------
        w : float or ndarray
            Frequency in rad/s

        Returns
        -------
        float or ndarray
            Phase in degrees
        """
        return self.L0_phase(w) + self.lead_phase(w)

    def get_summary(self):
        """
        Get a summary dictionary of the design.

        Returns
        -------
        dict
            Design parameters
        """
        return {
            'tau': self.tau,
            'phi_m_deg': self.phi_m_deg,
            'wc1': self.wc1,
            'A': self.A,
            'alpha': self.alpha,
            'wc2': self.wc2,
            'bandwidth_gain': self.wc2 / self.wc1,
            'wz': self.wz,
            'wp': self.wp
        }

    def print_summary(self):
        """Print a formatted summary of the design."""
        summary = self.get_summary()
        print("=== Lead Compensator Design Summary ===")
        print(f"Delay (tau)       : {summary['tau']:.6f} s")
        print(f"Phase margin      : {summary['phi_m_deg']:.2f} deg")
        print(f"Original crossover: {summary['wc1']:.4f} rad/s")
        print(f"DC Gain (A)       : {summary['A']:.4f}x")
        print(f"Alpha (pole/zero) : {summary['alpha']:.4f}")
        print(f"New crossover     : {summary['wc2']:.4f} rad/s")
        print(f"Bandwidth gain    : {summary['bandwidth_gain']:.4f}x")
        print(f"Lead zero         : {summary['wz']:.4f} rad/s")
        print(f"Lead pole         : {summary['wp']:.4f} rad/s")

    @staticmethod
    def compute_alpha_vs_margin(phi_m_range):
        """
        Compute alpha for a range of phase margins.

        This is a static method because alpha depends only on phase margin,
        not on the delay (tau). The delay only scales the absolute frequencies
        but not the bandwidth gain ratio.

        Parameters
        ----------
        phi_m_range : array-like
            Phase margins in degrees

        Returns
        -------
        ndarray
            Alpha values corresponding to each phase margin

        Raises
        ------
        ValueError
            If any phase margin value is invalid or alpha cannot be computed
        """
        phi_m_range = np.asarray(phi_m_range)
        alphas = np.zeros_like(phi_m_range, dtype=float)

        for i, phi_m in enumerate(phi_m_range):
            try:
                # Create temporary design with dummy tau (doesn't affect alpha)
                temp_design = LeadCompensatorDesign(tau=1.0, phi_m_deg=phi_m)
                alphas[i] = temp_design.alpha
            except (ValueError, RuntimeError) as e:
                raise ValueError(
                    f"Failed to compute alpha for phase margin {phi_m}° "
                    f"(index {i} in input array). "
                    f"Original error: {e}"
                ) from e

        return alphas

    # Backward-compatible aliases used by older example scripts.
    # These DO NOT change behavior; they forward to the strict implementations.
    # def L0_mag(self, w):
    #     return self.L0_mag(w)

    # def L0_phase(self, w):
    #     return self.L0_phase(w)


# ------------------------------------------------------------
# Plotting functions (optional, requires matplotlib)
# ------------------------------------------------------------

def plot_bode(design, w_range=None, figsize=(10, 8)):
    """
    Plot Bode magnitude and phase for original and compensated loops.

    Parameters
    ----------
    design : LeadCompensatorDesign
        Design object containing all parameters
    w_range : tuple of (w_min, w_max) or None
        Frequency range in rad/s. If None, auto-range around crossovers.
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    fig, axes
        Matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt

    # Determine frequency range
    if w_range is None:
        w_min = design.wc1 / 10
        w_max = design.wc2 * 10
    else:
        w_min, w_max = w_range

    w = np.logspace(np.log10(w_min), np.log10(w_max), 2000)

    # Compute responses
    mag0 = design.L0_mag(w)
    ph0 = design.L0_phase(w)
    mag1 = design.compensated_mag(w)
    ph1 = design.compensated_phase(w)

    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Plot magnitude
    axes[0].semilogx(w, 20*np.log10(mag0), 'b-', lw=2, label="Original L0")
    axes[0].semilogx(w, 20*np.log10(mag1), 'r-', lw=2, label="With Lead")
    axes[0].axvline(design.wc1, color='b', linestyle='--', alpha=0.5,
                    label=f'wc1 = {design.wc1:.2f} rad/s')
    axes[0].axvline(design.wc2, color='r', linestyle='--', alpha=0.5,
                    label=f'wc2 = {design.wc2:.2f} rad/s')
    axes[0].axhline(0, color='k', linestyle='-', alpha=0.3, lw=0.5)
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].set_title("Open-Loop Bode Plot: Bandwidth Extension with Unity-Gain Lead (COBYLA)")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, which="both", alpha=0.3)

    # Plot phase
    axes[1].semilogx(w, ph0, 'b-', lw=2, label="Original L0")
    axes[1].semilogx(w, ph1, 'r-', lw=2, label="With Lead")
    axes[1].axvline(design.wc1, color='b', linestyle='--', alpha=0.5)
    axes[1].axvline(design.wc2, color='r', linestyle='--', alpha=0.5)
    axes[1].axhline(-180 + design.phi_m_deg, color='g', linestyle=':', alpha=0.7,
                    label=f'Phase margin = {design.phi_m_deg}°')
    axes[1].axhline(-180, color='k', linestyle='-', alpha=0.3, lw=0.5)
    axes[1].set_ylabel("Phase (deg)")
    axes[1].set_xlabel("Frequency (rad/s)")
    axes[1].legend(loc='lower left')
    axes[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    return fig, axes


def plot_lead_only(design, w_range=None, figsize=(10, 8)):
    """
    Plot the lead compensator frequency response alone.

    Parameters
    ----------
    design : LeadCompensatorDesign
        Design object containing all parameters
    w_range : tuple of (w_min, w_max) or None
        Frequency range in rad/s. If None, auto-range around zero/pole.
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    fig, axes
        Matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt

    # Determine frequency range
    if w_range is None:
        w_min = design.wz / 10
        w_max = design.wp * 10
    else:
        w_min, w_max = w_range

    w = np.logspace(np.log10(w_min), np.log10(w_max), 2000)

    # Compute lead response
    mag = design.lead_mag(w)
    phase = design.lead_phase(w)

    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Plot magnitude
    axes[0].semilogx(w, 20*np.log10(mag), 'g-', lw=2)
    axes[0].axvline(design.wz, color='b', linestyle='--', alpha=0.5,
                    label=f'Zero = {design.wz:.2f} rad/s')
    axes[0].axvline(design.wp, color='r', linestyle='--', alpha=0.5,
                    label=f'Pole = {design.wp:.2f} rad/s')
    axes[0].axhline(0, color='k', linestyle='-', alpha=0.3, lw=0.5,
                    label='DC gain = 0 dB (unity)')
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].set_title(f"Unity-Gain Lead Compensator (α = {design.alpha:.2f})")
    axes[0].legend(loc='upper left')
    axes[0].grid(True, which="both", alpha=0.3)

    # Plot phase
    axes[1].semilogx(w, phase, 'g-', lw=2)
    axes[1].axvline(design.wz, color='b', linestyle='--', alpha=0.5)
    axes[1].axvline(design.wp, color='r', linestyle='--', alpha=0.5)
    axes[1].axvline(np.sqrt(design.wz * design.wp), color='purple',
                    linestyle=':', alpha=0.7,
                    label=f'Max phase @ {np.sqrt(design.wz * design.wp):.2f} rad/s')
    axes[1].axhline(0, color='k', linestyle='-', alpha=0.3, lw=0.5)
    max_phase = np.degrees(np.arcsin((design.alpha - 1) / (design.alpha + 1)))
    axes[1].axhline(max_phase, color='purple', linestyle=':', alpha=0.5,
                    label=f'Max phase = {max_phase:.2f}°')
    axes[1].set_ylabel("Phase (deg)")
    axes[1].set_xlabel("Frequency (rad/s)")
    axes[1].legend(loc='upper left')
    axes[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    return fig, axes


def plot_bandwidth_gain_vs_margin(phi_m_range=None, alpha_max=10, figsize=(10, 6)):
    """
    Plot DC gain improvement (A/A1) vs phase margin.

    This plot shows how much we can increase the DC gain A (the PID gain)
    while maintaining phase margin using lead compensation.

    Parameters
    ----------
    phi_m_range : array-like or None
        Phase margin range in degrees. If None, uses [15, 85] degrees.
    alpha_max : float
        Maximum alpha for the designs. Default 10 (practical limit).
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt

    # Default range
    if phi_m_range is None:
        phi_m_range = np.linspace(15, 85, 50)

    phi_m_range = np.asarray(phi_m_range)

    # Compute A/A1 ratio for each phase margin
    gain_ratios = np.full_like(phi_m_range, np.nan, dtype=float)
    failed_indices = []

    for i, phi_m in enumerate(phi_m_range):
        try:
            temp_design = LeadCompensatorDesign(tau=0.02, phi_m_deg=phi_m, alpha_max=alpha_max)
            gain_ratios[i] = temp_design.A / temp_design.A1
        except (ValueError, RuntimeError):
            failed_indices.append(i)
            gain_ratios[i] = np.nan

    # Separate valid and invalid points
    valid_mask = ~np.isnan(gain_ratios)
    failed_mask = np.isnan(gain_ratios)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot valid points
    if np.any(valid_mask):
        ax.plot(phi_m_range[valid_mask], gain_ratios[valid_mask],
                'b-', lw=2.5, label='Lead compensation (COBYLA)')

    # Mark failed points with red X
    if np.any(failed_mask):
        ax.plot(phi_m_range[failed_mask],
                np.ones(np.sum(failed_mask)),
                'rx', markersize=10, markeredgewidth=2,
                label=f'Solver failed ({np.sum(failed_mask)} points)')

    ax.axhline(1, color='k', linestyle='--', alpha=0.5, lw=1.5,
               label='Baseline (no lead)')
    ax.axhline(alpha_max**0.5, color='r', linestyle=':', alpha=0.5, lw=1.5,
               label=f'Approx. limit (sqrt(alpha_max)={alpha_max**0.5:.2f})')

    # Highlight some common phase margins
    common_margins = [30, 45, 60]
    for pm in common_margins:
        if pm >= phi_m_range[0] and pm <= phi_m_range[-1]:
            idx = np.argmin(np.abs(phi_m_range - pm))
            if valid_mask[idx]:
                ratio = gain_ratios[idx]
                ax.plot(pm, ratio, 'go', markersize=8, zorder=10)
                ax.annotate(f'{pm}deg\nA/A1={ratio:.2f}x',
                           xy=(pm, ratio),
                           xytext=(10, 10),
                           textcoords='offset points',
                           fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7))

    ax.set_xlabel('Phase Margin (degrees)', fontsize=12, fontweight='bold')
    ax.set_ylabel('DC Gain Improvement (A/A1)', fontsize=12, fontweight='bold')
    ax.set_title(f'Lead Compensation: DC Gain vs Phase Margin\n(alpha_max={alpha_max})',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Set reasonable y-axis limits
    if np.any(valid_mask):
        y_min = max(0.9, float(np.nanmin(gain_ratios) * 0.95))
        y_max = min(5.0, float(np.nanmax(gain_ratios) * 1.1))
        ax.set_ylim((y_min, y_max))

    # Add explanatory text
    ax.text(0.98, 0.02,
            f'Lead compensator allows increasing DC gain (PID gain)\n' +
            f'while maintaining phase margin. Constraint: alpha <= {alpha_max}',
            transform=ax.transAxes,
            fontsize=9,
            ha='right',
            va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig, ax


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------

if __name__ == "__main__":
    # Create design
    tau = 0.02          # 20 ms delay
    phi_m_deg = 45      # 45° phase margin

    design = LeadCompensatorDesign(tau, phi_m_deg)
    design.print_summary()

    # Plot results
    print("\nGenerating plots...")
    fig1, ax1 = plot_bode(design)
    fig2, ax2 = plot_lead_only(design)
    fig3, ax3 = plot_bandwidth_gain_vs_margin()

    import matplotlib.pyplot as plt
    plt.show()
