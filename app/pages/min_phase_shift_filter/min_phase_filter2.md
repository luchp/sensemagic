# Minimum-Delay FIR Design from Piecewise-Linear Log-Magnitude

## 1. Goal

We want to design a **minimum-delay FIR filter** that:

- Satisfies **magnitude constraints** (passband ripple, stopband attenuation),
- Has **low group delay** and **short impulse response**,
- Is constructed in a **principled, analytic way**.

The pipeline:

1. Design a **piecewise-linear log-magnitude** \(L(\omega)\).
2. Compute the **minimum-phase spectrum** from \(L(\omega)\).
3. Inverse FFT to obtain the **minimum-phase impulse response** \(h[n]\).
4. Compute an **optimal smoothing window** \(w[n]\) by minimizing a curvature functional of the complex frequency response.
5. Apply the window and **truncate** the impulse response when it decays below a threshold.

---

## 2. Piecewise-Linear Log-Magnitude

We work on a dense frequency grid \(\omega_k \in [0,\pi]\), \(k=0,\dots,K/2\).

For a lowpass with passband edge \(\omega_p\) and stopband edge \(\omega_s\):

- **Passband** \([0,\omega_p]\): approximately flat at level \(L_p\) (in dB), optionally with a small tilt.
- **Stopband** \([\omega_s,\pi]\): flat at level \(L_s\) (in dB, negative).
- **Transition** \([\omega_p,\omega_s]\): linear in dB between \(L_p\) and \(L_s\).

Let \(L_{\text{dB}}(\omega)\) be this piecewise-linear curve. We convert to **natural log-magnitude**:

\[
|H(\omega)| = 10^{L_{\text{dB}}(\omega)/20}, \quad
L(\omega) = \ln |H(\omega)|.
\]

This \(L(\omega)\) is defined on the one-sided grid \(\omega \in [0,\pi]\).

---

## 3. Minimum-Phase Reconstruction via Real Cepstrum

Given the one-sided log-magnitude \(L(\omega)\), we compute a **minimum-phase spectrum** using the real cepstrum.

1. Interpret \(L(\omega)\) as the log-magnitude of a real, even spectrum on \([0,2\pi)\).
2. Compute the **real cepstrum**:

\[
c[n] = \mathcal{F}^{-1}\{L(\omega)\}, \quad n = 0,\dots,N-1.
\]

3. Enforce **minimum-phase cepstrum**:

\[
c_{\text{min}}[0] = c[0], \quad
c_{\text{min}}[n] = 2c[n] \ \text{for } 1 \le n \le N/2-1, \quad
c_{\text{min}}[n] = 0 \ \text{for } n > N/2.
\]

4. Reconstruct the log-spectrum:

\[
\log H_{\text{min}}(\omega) = \mathcal{F}\{c_{\text{min}}\}.
\]

5. Minimum-phase spectrum:

\[
H_{\text{min}}(\omega) = \exp\big(\log H_{\text{min}}(\omega)\big).
\]

6. Impulse response:

\[
h[n] = \mathcal{F}^{-1}\{H_{\text{min}}(\omega)\}.
\]

This \(h[n]\) is **real** and **minimum-phase**.

---

## 4. Windowing as Frequency-Domain Smoothing

We apply a real window \(w[n]\) to the minimum-phase impulse response:

\[
\tilde{h}[n] = h[n]\,w[n].
\]

In frequency:

\[
\tilde{H}(\omega) = (H_{\text{min}} * W)(\omega),
\]

so windowing acts as a **smoothing operator** on both magnitude and phase. This:

- Reduces curvature in \(\ln |H(\omega)|\),
- Suppresses group delay spikes,
- Shortens the effective impulse response.

We want to choose \(w[n]\) to **minimize curvature** of the complex response.

---

## 5. Quadratic Curvature Objective for the Window

Let \(h[n]\) be fixed (minimum-phase prototype), length \(N\).  
We evaluate the windowed response on a dense grid \(\omega_k = 2\pi k / K\), \(k=0,\dots,K-1\):

\[
\tilde{H}(\omega_k) = \sum_{n=0}^{N-1} h[n]\,w[n]\,e^{-j\omega_k n}.
\]

Define the matrix:

\[
B_{k,n} = h[n] e^{-j\omega_k n}, \quad B \in \mathbb{C}^{K \times N}.
\]

Stack real and imaginary parts into a real matrix:

\[
A =
\begin{bmatrix}
\Re(B) \\
\Im(B)
\end{bmatrix}
\in \mathbb{R}^{2K \times N}.
\]

Define a **second-difference operator** \(D\) over frequency:

\[
(Dx)_k = x_{k+1} - 2x_k + x_{k-1}, \quad k = 1,\dots,K-2.
\]

We apply \(D\) to both real and imaginary parts (block-diagonal). The curvature of the windowed response is:

\[
\text{curv}(w) = D A w.
\]

We define the curvature energy:

\[
J(w) = \|D A w\|_2^2 = w^\top Q w, \quad Q = A^\top D^\top D A.
\]

---

## 6. Regularization and Optimal Window

To avoid pathological windows, we add **Tikhonov regularization**:

\[
J_\lambda(w) = w^\top (Q + \lambda I) w.
\]

We impose a normalization constraint, e.g.:

\[
\|w\|_2 = 1.
\]

The optimal window is the **eigenvector** of \(Q + \lambda I\) corresponding to the **smallest eigenvalue**:

\[
(Q + \lambda I) w_\star = \lambda_{\min} w_\star.
\]

We then normalize \(w_\star\) and use it as the smoothing window.

---

## 7. Truncation of the Impulse Response

The windowed impulse response:

\[
h_{\text{win}}[n] = h[n]\,w[n]
\]

is minimum-phase-like and decays rapidly. We truncate it when:

\[
|h_{\text{win}}[n]| < \varepsilon,
\]

for some small threshold \(\varepsilon\). The resulting FIR:

\[
h_{\text{final}}[n], \quad n = 0,\dots,N_{\text{eff}}-1
\]

is a **short, approximately minimum-delay filter** that respects the designed magnitude constraints (within design slack) and has reduced group delay and curvature.

---

## 8. Summary

- **Log-magnitude design**: piecewise-linear, flat in bands, linear in transition.
- **Minimum-phase reconstruction**: via real cepstrum folding.
- **Window optimization**: quadratic curvature minimization in frequency, solved as an eigenproblem.
- **Truncation**: based on amplitude threshold.

This yields a **principled, tunable minimum-delay FIR design** that directly encodes your intuition:

> Smooth log-magnitude + minimum phase + curvature-aware smoothing  
> ⇒ short impulse response and low maximum group delay.