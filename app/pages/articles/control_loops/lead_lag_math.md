# Lead Compensator Design: Mathematical Derivation

This document derives the complete mathematical framework for lead compensator design to maximize DC gain while maintaining phase margin.

---

## 1. System Model

### Why the First-Order Slope Model is Valid

In a typical PID control loop, the open-loop gain has several regions:

1. **Integration region** (low frequency): Steep slope for zero steady-state error
2. **Proportional region** (mid frequency): Flat plateau
3. **Differentiation region** (higher frequency): Small gain boost
4. **Rolloff region** (around crossover): Descent to unity gain

The **crossover frequency must be far beyond the PID region** to maintain phase margin. At these frequencies:

- The integrator contribution has rolled off to a simple $1/\omega$ slope
- The proportional gain is just a constant (which we absorb into $A$)
- The derivative term (if any) has minimal effect
- The plant often provides natural first-order rolloff (motor inertia, thermal time constants, etc.)

Therefore, around crossover, we can approximate the loop as:

$$
|L(j\omega)| \approx \frac{A}{\omega}
$$

This is the **same reasoning** used in the [Control Loop Bandwidth](/app_articles/control-loops/control-loop-bandwidth) analysis, where we showed that delay-limited loops naturally have a first-order slope at crossover.

### Plant + Controller Around Crossover

The open-loop transfer function around crossover is modeled as a **first-order slope**:

$$
|L(j\omega)| = \frac{A}{\omega}
$$

where:
- $A$ is the **DC gain** (the slope gain coefficient)
- The phase is approximately $-90°$ (first-order rolloff)

This model is accurate because:
- Crossover is far from the PID corner frequencies
- The $1/\omega$ behavior dominates
- Higher-order dynamics are negligible at these frequencies

### Delay

A pure delay $\tau$ adds phase:

$$
\angle\text{delay}(j\omega) = -\omega\tau \quad \text{[radians]}
$$

---

## 2. Baseline Design (No Lead Compensation)

### Phase Margin Condition

At the baseline crossover $\omega_{c1}$, we require phase margin $\phi_m$:

$$
-\frac{\pi}{2} - \omega_{c1}\tau = -\pi + \phi_m \quad \text{[radians]}
$$

Solving for $\omega_{c1}$:

$$
\boxed{\omega_{c1} = \frac{\pi/2 - \phi_m}{\tau}} \quad \text{[rad/s]}
$$

where $\phi_m$ is in **radians**. If given in degrees:

$$
\omega_{c1} = \frac{\pi}{180} \cdot \frac{90° - \phi_m°}{\tau}
$$

### Baseline DC Gain

At crossover, the magnitude condition is:

$$
|L(j\omega_{c1})| = \frac{A_1}{\omega_{c1}} = 1
$$

Therefore, the **baseline DC gain** is:

$$
\boxed{A_1 = \omega_{c1}}
$$

This is the maximum gain achievable with the specified phase margin and delay, **without lead compensation**.

---

## 3. Lead Compensator

### Transfer Function

The lead compensator has the form:

$$
F(s) = \frac{1 + s/\omega_z}{1 + s/\omega_p}
$$

where $\omega_z < \omega_p$ (zero before pole).

Key properties:
- **Unity DC gain:** $F(0) = 1$ (does not affect low-frequency PID gain)
- **High-frequency gain:** $|F(j\omega)| \to \alpha$ as $\omega \to \infty$

### Pole-Zero Ratio

Define:

$$
\alpha = \frac{\omega_p}{\omega_z} > 1
$$

This is the **high-frequency gain** and the primary design constraint (typically $\alpha \le 10$).

### Magnitude Response

$$
|F(j\omega)| = \sqrt{\frac{1 + (\omega/\omega_z)^2}{1 + (\omega/\omega_p)^2}}
$$

### Phase Response

$$
\angle F(j\omega) = \arctan\left(\frac{\omega}{\omega_z}\right) - \arctan\left(\frac{\omega}{\omega_p}\right)
$$

The lead compensator provides **positive phase** (phase lead) in the frequency range between $\omega_z$ and $\omega_p$.

---

## 4. Compensated System

### Magnitude Condition

With lead compensation, the loop magnitude is:

$$
|L_{\text{comp}}(j\omega)| = \frac{A}{\omega} \cdot |F(j\omega)|
$$

At the new crossover $\omega_{c2}$:

$$
\frac{A}{\omega_{c2}} \cdot |F(j\omega_{c2})| = 1
$$

### Analytic Crossover Solution

Substituting the lead magnitude and squaring:

$$
\frac{A^2}{\omega_c^2} \cdot \frac{1 + (\omega_c/\omega_z)^2}{1 + (\omega_c/\omega_p)^2} = 1
$$

Let $x = \omega_c^2$ and $\omega_p = \alpha\omega_z$:

$$
\frac{A^2}{x} \cdot \frac{1 + x/\omega_z^2}{1 + x/(\alpha^2\omega_z^2)} = 1
$$

Expanding:

$$
A^2 \left(1 + \frac{x}{\omega_z^2}\right) = x \left(1 + \frac{x}{\omega_p^2}\right)
$$

$$
A^2 + \frac{A^2 x}{\omega_z^2} = x + \frac{x^2}{\omega_p^2}
$$

Multiply by $\omega_z^2\omega_p^2$ and rearrange:

$$
\boxed{x^2 + x\omega_p^2\left(\frac{\omega_z^2 - A^2}{\omega_z^2}\right) - A^2\omega_p^2 = 0}
$$

Solve using the quadratic formula, taking the positive root:

$$
\omega_c = \sqrt{x}
$$

**Special case:** When $\alpha = 1$ (no lead), this reduces to $\omega_c = A$.

### Phase Condition

The total phase at crossover is:

$$
\angle L_{\text{comp}}(j\omega_c) = -\frac{\pi}{2} - \omega_c\tau + \arctan\left(\frac{\omega_c}{\omega_z}\right) - \arctan\left(\frac{\omega_c}{\omega_p}\right)
$$

The phase margin is:

$$
\text{PM} = \pi + \angle L_{\text{comp}}(j\omega_c)
$$

---

## 5. Optimization Problem

### Variables (Normalized)

To make the problem **tau-independent** and numerically well-conditioned, we optimize normalized variables:

$$
\begin{aligned}
A_{\text{ratio}} &= \frac{A}{A_1} \ge 1 \\
\omega_{z,\text{ratio}} &= \frac{\omega_z}{\omega_{c1}} \ge 1 \\
\alpha &= \frac{\omega_p}{\omega_z}, \quad 1 \le \alpha \le \alpha_{\max}
\end{aligned}
$$

All variables are $O(1)$, making COBYLA's numerical scaling work well.

### Objective

**Maximize** $A_{\text{ratio}}$ (the DC gain improvement):

$$
\max_{A_{\text{ratio}}, \omega_{z,\text{ratio}}, \alpha} A_{\text{ratio}}
$$

### Constraint

Phase margin at the new crossover must meet or exceed the target:

$$
\text{PM}(A_{\text{ratio}} \cdot A_1, 
\omega_{z,\text{ratio}} \cdot \omega_{c1}, 
\alpha) \ge \phi_m
$$

### Bounds

$$
\begin{aligned}
A_{\text{ratio}} &\in [1, 100] \quad \text{(effectively unbounded above)} \\
\omega_{z,\text{ratio}} &\in [1, 100] \quad \text{(zero at or above baseline)} \\
\alpha &\in [1, \alpha_{\max}] \quad \text{(typically } \alpha_{\max} = 10\text{)}
\end{aligned}
$$

---

## 6. Physical Interpretation

### The DC Gain A

The DC gain $A$ determines:
- **Disturbance rejection:** Higher $A$ → better rejection
- **Tracking accuracy:** Higher $A$ → tighter tracking  
- **Steady-state error:** Higher $A$ → lower error

In a PID controller, $A$ is proportional to the **integral gain**.

### The Improvement Ratio

The ratio $A/A_1$ represents how much we can **increase the PID gain** beyond what's possible without lead compensation, while maintaining the same phase margin.

**Example Results** (for $\tau = 0.02$ s):

| Phase Margin | $A_1$ | $A$ | $A/A_1$ | Improvement |
|--------------|-------|-----|---------|-------------|
| 45°          | 39.3  | 54.5| 1.39    | 39%         |
| 60°          | 26.2  | 43.8| 1.67    | 67%         |
| 75°          | 13.1  | 33.8| 2.58    | 158%        |

### Why Alpha Matters

The parameter $\alpha$ (high-frequency gain) is constrained because:
- **Noise amplification:** High $\alpha$ amplifies sensor noise
- **Unmodeled dynamics:** Excites resonances and parasitics
- **Actuator stress:** Higher bandwidth demands more actuator effort

Typical practical limit: $\alpha_{\max} = 10$.

---

## 7. Design Procedure

1. **Specify requirements:**
   - Delay: $\tau$
   - Phase margin: $\phi_m$
   - Maximum alpha: $\alpha_{\max}$ (default: 10)

2. **Compute baseline:**
   - $\omega_{c1} = (\pi/2 - \phi_m)/\tau$
   - $A_1 = \omega_{c1}$

3. **Optimize:**
   - Variables: $(A_{\text{ratio}}, \omega_{z,\text{ratio}}, \alpha)$
   - Objective: Maximize $A_{\text{ratio}}$
   - Constraint: Phase margin $\ge \phi_m$
   - Solver: COBYLA

4. **Extract results:**
   - Optimal DC gain: $A = A_{\text{ratio}} \cdot A_1$
   - Lead zero: $\omega_z = \omega_{z,\text{ratio}} \cdot \omega_{c1}$
   - Lead pole: $\omega_p = \alpha \cdot \omega_z$
   - New crossover: $\omega_{c2}$ (from analytic solution)

---

## 8. Key Insights

### Tau Independence (Theoretical)

The normalized optimization problem is **mathematically independent of $\tau$**:
- All variables scale with baseline values
- The constraint involves only ratios
- The product $\omega_{c1} \cdot \tau = \pi/2 - \phi_m$ is constant

### Numerical Considerations

For very large $\tau$ (> 1 s), numerical convergence can be slow because absolute frequencies become very small. However, for all practical control systems ($\tau = 0.001$ to $0.1$ s), the optimizer converges reliably.

### Zero and Pole Placement

The optimal design typically places:
- **Zero** $\omega_z$: Near or slightly above the baseline crossover
- **Pole** $\omega_p = \alpha\omega_z$: Much higher (factor of $\alpha$)

This provides phase lead in the critical frequency region while maintaining unity DC gain.

---

## 9. Implementation

This derivation is implemented in [`lead_lag_design.py`](https://github.com/luchp/sensemagic/blob/main/app/pages/lead_lag_compensator/lead_lag_design.py):

```python
from lead_lag_design import LeadCompensatorDesign

# Design for 20ms delay, 45° phase margin
design = LeadCompensatorDesign(tau=0.02, phi_m_deg=45)

print(f"Baseline gain: A1 = {design.A1:.2f}")
print(f"Optimized gain: A = {design.A:.2f}")
print(f"Improvement: {design.A/design.A1:.2f}x")
print(f"Lead zero: {design.wz:.2f} rad/s")
print(f"Lead pole: {design.wp:.2f} rad/s")
print(f"Alpha: {design.alpha:.2f}")
```

---

## References

- Franklin, Powell, Emami-Naeini: "Feedback Control of Dynamic Systems"
- Åström, Murray: "Feedback Systems: An Introduction for Scientists and Engineers"

---

## See Also

Related articles in this series:

- [Control Loop Bandwidth](/app_articles/control-loops/control-loop-bandwidth) - Why delay limits your bandwidth (the 1/10 rule)
- [Improving Control Loop Bandwidth](/app_articles/control-loops/improving-control-loop-bandwidth) - Cascaded loops and lead compensation overview
- [Lead-Lag Compensation](/app_articles/control-loops/lead-lag-compensation) - Practical design guide with examples

---

*Last updated: January 2026*
