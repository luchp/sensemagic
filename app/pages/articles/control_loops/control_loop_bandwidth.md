# 🎯 Is Your Digital Control Loop Bandwidth SMALLER Than You Think?

Here's a number that might ruin your day: your 1 kHz control loop probably can't do better than 100 Hz bandwidth. And it gets worse.

Let me show you why—with a scenic tour through the Mountains of Gain! ⛰️

## The One Delay You Can't Escape

Every digital control loop has at least one sample time of delay. It's physics, not a bug. This delay puts a hard ceiling on your bandwidth.

The surprising part? We can calculate this ceiling with high-school math.

---
## The Topography of Stability

![Control Loop Diagram](/static/images/blog/control_loop.svg)

To analyze stability, we plot the open-loop gain. The rule is simple but unforgiving:

> ⚠️ **When phase shift hits 180°, gain MUST be below 1**
> Otherwise? Your controller becomes an oscillator.

At crossover (gain = 1), we want breathing room—a "phase margin" of 45° to 60°.

Picture the gain plot as a mountain range:

- 🏔️ **Peak of Integration** — steep slopes, hunting for zero steady-state error
- 🏕️ **Valley of Proportionality** — the flat middle ground
- ⛰️ **Hill of Differentiation** — that little bump for speed
- 🎿 **Slope of Stability** — the mandatory descent to gain = 1

- ![PID Mountain](/static/images/blog/pid_mountain.svg)

---
## The Math (It's Simple)

Delay adds phase shift that increases linearly with frequency:

$$\varphi_{delay} = 360° \times \frac{f}{f_s}$$

At sample frequency $f_s$, you've accumulated 360° of phase shift.

Assume a first-order rolloff (the gentlest descent down the Slope of Stability). Often the plant provides this naturally—think motor inertia or thermal time constants. This rolloff contributes ~90° of phase shift.

Total phase at the critical point:

$$\varphi_{total} = \varphi_{plant} + \varphi_{delay} = 180° - \theta$$

Where θ is our phase margin. Solving:

$$90° + 360° \times \frac{f}{f_s} = 180° - \theta$$

$$f = f_s \times \frac{90° - \theta}{360°}$$

For θ = 60° → **f = f_s / 10**

---
## The Reality Check 💥

Your 1 kHz control loop? Maximum bandwidth: **100 Hz**.

But wait—there's more bad news!

Real systems have extra delays:

| Source | Delay |
|--------|-------|
| 📡 Communication bus | ~10 ms |
| 📉 Anti-aliasing filter | ~5 ms |
| ⏱️ Sample time at 1 kHz | 1 ms |

**Total delay: 10 + 5 + 1 = 16 ms**

### 🧠 Pop Quiz: What's the maximum bandwidth now?

...

**Maximum bandwidth = 1 / (16 ms × 10) = 6.25 Hz**

That's not a typo. Six hertz. From a kilohertz loop. 😱

---
## The Takeaway

Before designing your next control loop, add up ALL delays:

- ✓ Sample time
- ✓ Communication latency
- ✓ Filter delays
- ✓ Computation time

Then divide by 10. That's your real bandwidth ceiling.

*Surprised? How do YOU handle delay in your control systems?*

**#ControlSystems #Embedded #Engineering #DSP #RealTimeControl**

