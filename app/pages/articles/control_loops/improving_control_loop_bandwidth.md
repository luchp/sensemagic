# 🚀 Want More Bandwidth From Your Control Loop? Start Here.

*This is part 2 of my series on control loop delays. All examples come from real projects I've worked on.*

Last time we learned a sobering truth: max bandwidth ≈ **1 / (10 × delay)**.

In digital control, you always have at least one sample of delay—so faster sampling helps. But there's more you can do.

---

## 1️⃣ The Simple Wins: Reduce Delay Directly

Start with the obvious:

- ⚡ **Faster communication bus** — CAN-FD, EtherCAT, or direct SPI instead of slow serial
- 🖥️ **Faster processor** — less computation time = less delay
- 🔧 **Optimize your code** — sometimes a few microseconds matter

These are low-hanging fruit. Grab them first.

---

## 2️⃣ Loop Cascading: The Secret Weapon

When your sensor is slow, don't fight it—**work around it**.

**Real example:** An air pressure controller with a sluggish pressure sensor.

![Cascaded Control Loop with Feed-Forward](/static/images/blog/cascaded_loop.svg){: style="width: 80%; display: block; margin: 0 auto;"}

**The solution:**
1. **Inner loop** — controls torque and speed of the fan (fast!)
2. **Outer loop** — controls pressure (slow sensor is fine here)

But here's where it gets clever: add **feed-forward**.

Based on operating conditions and desired pressure, a lookup table calculates the required torque *directly*. The response becomes nearly instantaneous.
The slow pressure sensor? It only needs to clean up the residual error.

> 💡 **Key insight:** Cascading lets you hide slow sensors behind fast inner loops.


## 3️⃣ Better Sensors (Or Smarter Sensor Fusion)

Sometimes you need a faster sensor. But what if faster sensors don't exist—or cost too much?

**Combine sensors.**

**Real example:** Indoor drone tracking using optical image recognition.

**The problem:**

- ❌ Optical tracking was slow
- ❌ Sometimes produced outliers (lost track, wrong detection)

**The solution:** Fuse optical data with a fast IMU using a **Kalman filter**.

**Results:**

- ✅ Much lower effective delay
- ✅ Built-in outlier rejection using Mahalanobis distance
- ✅ Smooth tracking even when optical updates are sparse

> 💡 **Bonus:** Sensor fusion often improves accuracy *and* speed simultaneously.

---

## 🎯 The Takeaway

When you hit the bandwidth wall, you have options:

| Strategy | When to use |
|----------|-------------|
| **Reduce delay** | Always try this first |
| **Loop cascading** | Slow outer sensor, fast inner dynamics available |
| **Sensor fusion** | Fast sensor exists, but lacks accuracy alone |


*Which of these have you tried? What worked (or didn't) in your systems?*

*Next up in this series: Reduce delay with a lead-lag filter.*

**#ControlSystems #Embedded #Engineering #DSP #RealTimeControl #Kalman**


