"""
Example usage of the LeadCompensatorDesign class.

This demonstrates a COBYLA-based design for a unity-gain lead compensator.

Note: depending on constraints and numerical tolerances, some parameter
combinations can be infeasible and will raise exceptions (by design).
"""

# Example 1: Basic usage
print("=" * 60)
print("Example 1: Lead Compensator Design (COBYLA)")
print("=" * 60)

from lead_lag_design import LeadCompensatorDesign
import numpy as np

# Design a lead compensator for a system with 20ms delay and 45deg phase margin
tau = 0.02          # 20 ms delay
phi_m_deg = 45      # 45deg phase margin

design = LeadCompensatorDesign(tau, phi_m_deg)
design.print_summary()

# Access design parameters
print(f"\nYou can access parameters directly:")
print(f"  design.wc1 = {design.wc1:.4f} rad/s")
print(f"  design.alpha = {design.alpha:.4f}")
print(f"  design.wz = {design.wz:.4f} rad/s")
print(f"  design.wp = {design.wp:.4f} rad/s")

# Or get all parameters as a dictionary
summary = design.get_summary()
print(f"\nBandwidth improvement: {summary['bandwidth_gain']:.2f}x")

# Example 2: Designs across phase margins (some may be infeasible under constraints)
print("\n" + "=" * 60)
print("Example 2: Designs Across Phase Margins (Some May Be Infeasible)")
print("=" * 60)

print("\nComparing designs across different phase margins:")
print("-" * 60)
print(f"{'PM [deg]':<8} {'Alpha':<8} {'BW Gain':<10} {'wc1 [Hz]':<10} {'wc2 [Hz]':<10}")
print("-" * 60)

for phi_m in [20, 30, 35, 45, 60, 75, 85]:
    try:
        d = LeadCompensatorDesign(tau, phi_m)
        s = d.get_summary()
        print(f"{phi_m:<8} {s['alpha']:<8.2f} {s['bandwidth_gain']:<10.2f} "
              f"{s['wc1']/(2*np.pi):<10.2f} {s['wc2']/(2*np.pi):<10.2f}")
    except Exception as e:
        print(f"{phi_m:<8} FAILED: {str(e)[:60]}")

print("\nNote: failures mean the constraints couldn’t be satisfied (no silent fallbacks).")

# Example 3: Compute frequency response
print("\n" + "=" * 60)
print("Example 3: Compute Frequency Response")
print("=" * 60)

# Define some frequencies of interest
freqs = np.array([1, 5, 10, 20, 50])  # rad/s

print("\nOriginal loop response:")
for w in freqs:
    mag = design.L0_mag(w)
    phase = design.L0_phase(w)
    print(f"  w={w:5.1f} rad/s: mag={mag:7.4f}, phase={phase:7.2f}deg")

print("\nLead compensator response:")
for w in freqs:
    mag = design.lead_mag(w)
    phase = design.lead_phase(w)
    print(f"  w={w:5.1f} rad/s: mag={mag:7.4f}, phase={phase:7.2f}deg")

print("\nCompensated loop response:")
for w in freqs:
    mag = design.compensated_mag(w)
    phase = design.compensated_phase(w)
    print(f"  w={w:5.1f} rad/s: mag={mag:7.4f}, phase={phase:7.2f}deg")

# Example 4: Optional plotting (requires matplotlib)
print("\n" + "=" * 60)
print("Example 4: Plotting (Requires matplotlib)")
print("=" * 60)

try:
    from lead_lag_design import plot_bode, plot_lead_only, plot_bandwidth_gain_vs_margin
    import matplotlib.pyplot as plt

    print("Generating plots...")

    # Plot the full Bode comparison
    fig1, ax1 = plot_bode(design)

    # Plot just the lead compensator
    fig2, ax2 = plot_lead_only(design)

    # Plot bandwidth gain vs phase margin
    fig3, ax3 = plot_bandwidth_gain_vs_margin()

    plt.show()

except ImportError as e:
    print(f"Matplotlib not available: {e}")
    print("Skipping plots.")

# Example 5: Effect of Delay (Same Phase Margin)
print("\n" + "=" * 60)
print("Example 5: Effect of Delay on Design")
print("=" * 60)

phi_m_fixed = 45  # Fixed phase margin

print(f"\nPhase margin = {phi_m_fixed}deg, varying delay:")
print("-" * 60)
print(f"{'Delay [ms]':<12} {'wc1 [Hz]':<12} {'Alpha':<8} {'wc2 [Hz]':<12} {'BW Gain':<10}")
print("-" * 60)

for tau_ms in [5, 10, 20, 50]:
    tau_val = tau_ms / 1000.0
    try:
        d = LeadCompensatorDesign(tau=tau_val, phi_m_deg=phi_m_fixed)
        s = d.get_summary()
        print(f"{tau_ms:<12.1f} {s['wc1']/(2*np.pi):<12.2f} {s['alpha']:<8.2f} "
              f"{s['wc2']/(2*np.pi):<12.2f} {s['bandwidth_gain']:<10.2f}")
    except Exception as e:
        print(f"{tau_ms:<12.1f} FAILED: {str(e)[:60]}")

print("\nKey insight: for a fixed phase margin, frequencies scale ~ 1/tau.")

# Example 6: Edge cases and constraints
print("\n" + "=" * 60)
print("Example 6: Constraints and Edge Cases")
print("=" * 60)

print("\nTesting very low phase margins (with 10x max gain constraint):")
print("-" * 60)

for phi_m in [10, 15, 20, 25, 30]:
    try:
        d = LeadCompensatorDesign(0.02, phi_m)
        s = d.get_summary()
        bw_gain = s['bandwidth_gain']
        status = "[OK]" if bw_gain <= 10 else "[WARNING]"
        print(f"  PM={phi_m:2d}deg: BW gain = {bw_gain:5.2f}x, Alpha = {s['alpha']:5.2f}  {status}")
    except Exception as e:
        print(f"  PM={phi_m:2d}deg: Failed - {str(e)[:60]}")

print("\nTesting invalid inputs:")
print("-" * 40)

# Test negative delay
try:
    d = LeadCompensatorDesign(-0.02, 45)
    print("  [X] Negative delay: Should have raised ValueError!")
except ValueError as e:
    print(f"  [OK] Negative delay: Correctly raised ValueError")

# Test invalid phase margin
try:
    d = LeadCompensatorDesign(0.02, 120)
    print("  [X] Phase margin > 90deg: Should have raised ValueError!")
except ValueError as e:
    print(f"  [OK] Phase margin > 90deg: Correctly raised ValueError")

print("\n" + "=" * 60)
print("All examples completed successfully!")
print("=" * 60)
print("\nSummary:")
print("  - Unity-gain lead design via COBYLA (wz, alpha)")
print("  - Strict exceptions on infeasible designs (no silent failures)")
print("  - alpha is capped (default alpha_max=10)")
