# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:04:49 2016

@author: Luc
"""

from math import copysign

import numpy as np

import scipy.signal as ssig
from scipy import linalg

from . import numutil


# Stability is based on counting the number of times we encircle the (-1,0) point
# in the s-diagram. Instead of encirclements we can also count the times we cross the
# ray from (-1,0) to (-inf,0).
#
# We must follow the openloop gain along a contour, this countour goes from s=(0,-inf)
# to s=(0,inf) en then closes clockwise (the curve has a D shape).
# The right semicircle of the curve can be ignored because abs(s) goes to infinity
# and we assume that the open loop gain goes to zero when s goes to infinity.
# Therefore, there can be no important crossings for this part of the contour.
#
# Special care must be taken when we have poles on the imaginary axis in which case we need
# to follow a contour around that pole, which again can be a semicircle, with very small diameter
#
def build_contour(logww, wpoles=[], plot=False):
    """Build a contour for evaluating stability in the s domain.
    wpoles are poles on the imaginary axis and can be positive or negative real numbers.
    For instance, if you have an openloop gain with integrators, then 0 will be
    a pole on the imaginary axis.
    """
    ww = np.abs(logww)
    if ww[0] == 0.0:
        ww = ww[1:]
    wwex = np.hstack((-ww[::-1], ww))  # extend with negative frequencies
    wp = np.sort(wpoles)
    if wp.size:
        # make a little semicircle loop to avoid the pole on the imaginary axis
        angles = np.linspace(-90, 90, ww.size) * (1j * np.pi / 180)
        loop = 0.5 * ww[0] * np.exp(angles)
        # insert the loops where the poles are
        prev, sparts = 0, []
        for k, px in enumerate(np.searchsorted(wwex, wp)):
            sparts.append(1j * wwex[prev:px])
            sparts.append(1j * wp[k] + loop)
            prev = px if wwex[px] != wp[k] else px + 1
        sparts.append(1j * wwex[prev:])
        contour = np.hstack(sparts)
    else:
        contour = 1j * wwex
    if plot:
        from . import numplot
        import matplotlib.pyplot as plt

        fax = numplot.plotnyquist(contour, contour, fannon=True)
        fax[1].scatter(np.zeros_like(wp), wp, marker="*", color="red", s=60)
        plt.show()
    return contour


# Evaluate stability where the open loop gain hol is evaluated on the contour
# defined above.
# The numbers returned are up_crossing, down_crossings, worst crossing to the
# right of -1.  From these numbers a margin is calculated. This margin is
# only valid for systems where the closed loop gain has no unstable poles and with
# only one region of stability. If you have unstable poles, or multiple stability
# regions, then don't use this margin but calculate your own.
def contour_stability(hol, center=-1):
    x, y = hol.real, hol.imag
    # remove zero's, having zero's should be very unlikely
    b_nz = y != 0.0
    xx, yy = x[b_nz], y[b_nz]
    # find indices of real axis crossings
    i_c1 = np.nonzero(np.diff(np.sign(yy)) != 0)[0]  # before the cross
    if i_c1.size == 0:
        xm_up, xm_down, xm_right = np.array([]), np.array([]), 0
    else:
        i_c2 = i_c1 + 1  # after the cross
        # interpolate to find x-coordinate of the crossing point
        xm = (xx[i_c2] * yy[i_c1] - xx[i_c1] * yy[i_c2]) / (yy[i_c1] - yy[i_c2])
        # to evaluate stability, only keep crossing points that are left from the (-1,0) point
        b_cross = xm <= center
        b_upcross = yy[i_c1[b_cross]] < yy[i_c2[b_cross]]
        # interpolated points at up- and down-crossing
        xm_up = xm[b_cross][b_upcross]
        xm_down = xm[b_cross][np.logical_not(b_upcross)]
        # Next we find the worst crossing to the right from the -1 point.
        # These can be used as a gain margin
        try:
            xm_right = np.amin(xm[xm >= center])
        except:
            xm_right = 0.0

    # A margin number that is suitable for optimization purposes when we
    # assume that the closed loop gain has no unstable poles.
    # Note that this is not a correct estimate of the true gain margins.
    # Both lowering and increasing gain can lead to a changes in stability.
    # These margins can be calculated from the returned values when needed
    stable = xm_up.size - xm_down.size == 0
    if stable:
        margin = -1 / xm_right if xm_right < 0 else np.inf
    else:
        m1 = -1 / np.min(xm_up) if xm_up.size else 0
        m2 = -1 / np.min(xm_down) if xm_down.size else 0
        margin = max(m1, m2)

    return xm_up, xm_down, xm_right, margin


# It can be proven that the Nyquist contour hol has the property that
# Z = P + N
# where:
# Z = the number of unstable poles of the closed loop tranfer
# P = the number of poles of the open loop gain. This  must be know in
#     advance and given by the parameter open_loop_pole_count
# N = number of clockwise encirclements  of (-1,0)
#        minus number of counter clockwise encirclements  of (-1,0)
#   = number of upcrossings minus number of downcrossings
#   = length(xup)-length(xdown)
#
# Note that Z and P are positive numbers by definition!
#
# Now there are three options:
# 1) Z=0, The system is stable
# 2) Z>0, The system is unstable
# 3) Z<0, You have got your openloop pole count wrong!
#    It is at least abs(Z) more than you thought
#
# To calculate the margins we go about as follows
#
# In case of a stable system it is simply 1/xright
#
# In case of an unstable system we need to shrink hol until we have a
# set of crossings left of (-1,0) such that Z=0
# Is there no such set, the we return 0
def evaluate_contour_stability(xup, xdown, xright, open_loop_pole_count):
    a, b = xup.size, xdown.size
    Z = a - b + open_loop_pole_count
    if Z < 0:
        margin = -1 / min(min(xup), min(xdown))
        print("Open loop pole count wrong, must be at least %d", open_loop_pole_count + abs(Z))
    elif Z == 0:
        margin = 1 / xright if xright > 0 else np.inf
    else:
        # remove crossings one at the time unil Z=0 or both arrays are depleted
        x = 0
        while (a > 0 or b > 0) and Z != 0:
            if a != 0 and (b == 0 or xup[a - 1] > xdown[b - 1]):
                x = xup[a - 1]
                a = a - 1
            else:
                x = xdown[b - 1]
                b = b - 1
            Z = a - b + open_loop_pole_count
        margin = -1 / x if Z == 0 else 0.0
    return margin


# margins from openloop gain
def margins(ol):
    # points where gain crosses 1 and phase crosses pi
    aol, pol = np.abs(ol), np.abs(np.unwrap(np.angle(ol)))
    # phase margin is the distance to 180 degrees where gain is crossing one
    (idx,) = np.diff(aol > 1.0).nonzero()
    if len(idx) == 0:
        idx_phase = None
        phasemax = np.amax(pol) if aol[1] > 1 else 0.0
    else:
        idx_phase = idx[np.argmax(pol[idx])]
        phasemax = pol[idx_phase]
    # gain margin is the gain where phase is crossing 180 degrees
    (idx,) = np.diff(pol > np.pi).nonzero()
    if len(idx) == 0:
        idx_gain = None
        gainmax = np.amax(aol) if pol[1] > np.pi else aol[-1]
    else:
        idx_gain = idx[np.argmax(aol[idx])]
        gainmax = aol[idx_gain]
    return idx_gain, gainmax, idx_phase, phasemax


def group_delay_sos(sos, ww):
    res = np.zeros_like(ww, dtype=float)
    for bq in sos:
        _, gd = ssig.group_delay((bq[:3], bq[3:]), ww)
        res += gd
    return res


def insert_dc_nyquist(hh, dc=None, nq=None):
    if dc is None:
        dc = copysign(abs(hh[0]), np.real(hh[0]))  # extend dc from first line
    if nq is None:
        nq = copysign(abs(hh[-1]), np.real(hh[-1]))  # extend nyquist from last line
    return np.hstack((dc, hh, nq))


def insert_dc(hh, dc=None):
    if dc is None:
        dc = copysign(abs(hh[0]), np.real(hh[0]))  # extend dc from first line
    hhx = np.hstack((dc, hh))
    hhx[-1] = copysign(abs(hh[-1]), np.real(hh[-1]))  # make nyquist real
    return hhx


def step_response(hh):
    imp = np.fft.irfft(hh)
    n2 = len(imp) // 2
    return np.roll(ssig.lfilter([1.0], [1, -1], np.roll(imp, n2)), -n2)


def step_error(dt, hh):
    st = step_response(hh)
    st = st[: len(st) // 4]
    idx_half = np.argmin(np.abs(st - 0.5))
    st_delay = idx_half * dt
    st_left = st[:idx_half]
    st_right = st[idx_half:] - 1
    st_error = np.linalg.norm(st_left) + np.linalg.norm(st_right)
    st_0 = st[st < 0]
    st_1 = st[st > 1]
    st_min = min(st_0) if len(st_0) else 0
    st_max = max(st_1) - 1 if len(st_1) else 0
    st_ripple = st_max - st_min
    return st, st_delay, st_ripple, st_error


def spec_rms(spec):
    """Calculate rms in frequency domain"""
    nt = (len(spec) - 1) * 2
    sa = np.abs(spec) ** 2
    sq = sa[0] + 2 * np.sum(sa[1:-1]) + sa[-1]
    return np.sqrt(sq) / nt


# Utility function to unwrap an angle measurement
def unwrap(angle, period=2 * np.pi, rewrap=None):
    """Unwrap a phase angle to give a continuous curve
    Parameters
    ----------
    angle : array_like
        Array of angles to be unwrapped
    period : float, optional
        Period (defaults to `2*pi`)
    rewrap : float, optional
        Phase is rewrapped on this interval
    Returns
    -------
    angle_out : array_like
        Output array, with jumps of period/2 eliminated
    """
    dangle = np.diff(angle)
    dangle_desired = (dangle + period / 2.0) % period - period / 2.0
    correction = np.cumsum(dangle_desired - dangle)
    angle[1:] += correction
    if rewrap is not None:
        angle = angle % rewrap
    return angle


def savgol_window(order, fmax, dt):
    # See:  "On the Frequency-Domain Properties of Savitzky-Golay Filters" "by "Ronald W. Schafer", "HPL-2010-109"
    # 3dB bandwidth of Savitsky-Golay filter is approximately: fc = (N+1)/(3.2*M -4.6) fnyq
    # where N is the order and M the window size
    # The window should  be at least twice as large than the order, which limits the maximum cutoff frequency
    fnyq = 0.5 / dt
    fmaxgol = min(fmax, fnyq * (order + 1) / (3.2 * 2 * order - 4.6))
    nsavgolwin = int(np.ceil((4.6 + (order + 1) * (fnyq / fmaxgol)) / 3.2))
    if nsavgolwin % 2 == 0:
        nsavgolwin += 1  # make it odd
    return nsavgolwin, fmaxgol


def model_response(hh, sig):
    """calculate response from model hh and signal sig"""
    nt = numutil.ceil_power2(2 * (hh.shape[0] - 1))
    n2 = nt // 2
    fir = np.fft.irfft(hh, nt)
    np.roll(fir, n2)
    return ssig.fftconvolve(sig, hh, mode="same")


def make_fir_from_ir(ir, nx=None, linphase=True):
    # import matplotlib.pyplot as plt
    """Make a fir filter from an impuls response"""
    nt = ir.shape[0]
    if nx is None:
        nx = nt
    n2 = min(nt, nx) // 2
    # fig, ax = plt.subplots(2)
    fir = np.zeros(nx)
    window = np.hanning(2 * n2)
    ir_mean = np.mean(ir)
    ir_norm = linalg.norm(ir)
    irx = ir - ir_mean  # substract mean
    fir[0:n2] = irx[0:n2] * window[-n2:]
    fir[-n2:] = irx[-n2:] * window[0:n2]
    # re-scale after windowing
    x = float(nt) / float(nx)
    fir *= linalg.norm(ir) / linalg.norm(fir)
    fir += x * ir_mean
    if linphase:
        fir = np.roll(fir, n2)
    return fir


def spectrum_smoother(hh, decimate=1, extend=1):
    """Smooth the spectrum by windowing (and decimating) the impuls response"""
    nt = 2 * (hh.shape[0] - 1)
    n2 = nt // decimate
    nx = numutil.ceil_power2(n2 * extend) if extend > 1 else n2
    fir = np.fft.irfft(hh)
    firx = make_fir_from_ir(fir, nx, False)
    return np.fft.rfft(firx)


def clip_relative(hh, cliprel):
    """Clip a spectrum at the bottom if cliprel < 1
    Clip a spectrum at the top if cliprel > 1
    """
    if cliprel < 0:
        raise ValueError("cliprel must be larger then zero")
    if cliprel == 1:
        return np.ones_like(hh)
    ret = np.nan_to_num(hh)
    mag = np.abs(ret)
    if cliprel < 1:
        mx = np.amax(mag)
        clip = mx * cliprel
        ret[mag < clip] = clip
    else:
        mx = np.amin(mag)
        clip = mx * cliprel
        ret[mag > clip] = clip

    return ret


def min_phase(hh, db=-100, extend=256, decimate=256):
    """Convert halfsided complex spectrum hh into a minimal phase system.
    using the cepstrum method
    """
    # to avoid aliasing, clip and smooth the spectrum
    clip_relative(hh, 10 ** (db / 20.0))
    if extend > 1:
        hh = spectrum_smoother(hh, extend=extend)
    # calculate complex cepstrum
    cep = np.fft.irfft(np.log(hh))
    # fold the ceptrum to make a minimal phase
    nt = len(cep)
    n2 = nt // 2
    cep[1:n2] += np.conj(cep[-1:-n2:-1])  # fold
    cep[-1:-n2:-1] = 0.0
    # back to the spectrum, and decimate it to a reasonable length
    hcep = np.exp(np.fft.rfft(cep))
    if decimate > 1:
        hcep = spectrum_smoother(hcep, decimate=decimate)
    return hcep


def leja(x):
    """Reorders the complex roots of a polynomial (found with np.roots)
    to minimize the error in np.poly
    """

    def swap(a, c1, c2):
        if c1 != c2:
            tmp = a[:, c1].copy()
            a[:, c1] = a[:, c2]
            a[:, c2] = tmp

    #
    n = len(x)
    # x = x_in(:).'; n = length(x);
    a = np.tile(x, (n + 1, 1))
    a[0, :] = np.abs(a[0, :])

    ind = np.argmax(a[0, :])
    swap(a, 0, ind)

    y = a[n - 1, 0]
    a[1, 1:] = np.abs(a[1, 1:] - y)

    for l in range(1, n - 1):
        ind = l + np.argmax(np.prod(a[0 : l + 1, l:], axis=0))
        # print(ind+1)
        swap(a, l, ind)
        y = a[n - 1, l]
        a[l + 1, l + 1 :] = np.abs(a[l + 1, l + 1 :] - y)
    return a[n, :]


def polystab(a):
    """stabilizes the polynomial with respect to the unit circle;
    roots whose magnitudes are greater than one are reflected
    inside the unit circle.
    """
    if len(a) == 1:
        return a
    if not np.any(a != 0):
        return a
    v = np.roots(a)
    idx = v != 0
    vnz = v[idx]  # only flip non zero candidates
    vnz[np.abs(vnz) > 1.0] = 1.0 / vnz[np.abs(vnz) > 1.0].conj()  # reflect the roots
    v[idx] = vnz
    idx = np.argmax(a != 0)
    w = leja(v)
    b = a[idx] * np.poly(w)

    # Return only real coefficients if input was real:
    if not np.any(np.iscomplex(a)):
        b = np.real(b)
    return b
