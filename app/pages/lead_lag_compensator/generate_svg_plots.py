"""Generate SVG plots for the LinkedIn post"""
import sys
sys.path.insert(0, '.')

from lead_lag_design import LeadCompensatorDesign, plot_bode
import matplotlib.pyplot as plt
import numpy as np

# Create a design example
tau = 0.02  # 20 ms delay
phi_m = 45  # 45° phase margin

design = LeadCompensatorDesign(tau, phi_m)

# Generate Bode plot
fig, axes = plot_bode(design, figsize=(10, 8))

# Save as SVG
plt.savefig('lead_compensator_bode.svg', format='svg', bbox_inches='tight')
print("✓ Saved: lead_compensator_bode.svg")

# Also create a simple bandwidth gain vs phase margin plot
from lead_lag_design import plot_bandwidth_gain_vs_margin

fig2, ax2 = plot_bandwidth_gain_vs_margin(figsize=(10, 6))
plt.savefig('bandwidth_gain_vs_margin.svg', format='svg', bbox_inches='tight')
print("✓ Saved: bandwidth_gain_vs_margin.svg")

plt.close('all')

print("\nPlot details:")
print(f"  Delay: {tau*1000:.0f} ms")
print(f"  Phase margin: {phi_m}°")
print(f"  Alpha: {design.alpha:.2f}")
print(f"  Bandwidth gain: {design.wc2/design.wc1:.2f}x")
print(f"  Zero: {design.wz:.2f} rad/s = {design.wz/(2*np.pi):.2f} Hz")
print(f"  Pole: {design.wp:.2f} rad/s = {design.wp/(2*np.pi):.2f} Hz")
