import datetime
import os
import struct
from pathlib import Path
import typing
from math import sqrt, copysign

import numpy as np
import scipy.interpolate as sipol
import scipy.optimize as sopt
import scipy.signal as ssig
import scipy.stats as sstat
from scipy import linalg
from .ureg import Q_

import numpy.polynomial.polynomial as npoly


def philips_date(dt: datetime.datetime | None = None) -> str:
    import datetime

    c = (dt or datetime.datetime.now()).isocalendar()
    return f"{str(c.year)[-2:]}{c.week:02d}{c.weekday}"


def module_from_file(py_file_path, module_name=None):
    if module_name is None:
        module_name = Path(py_file_path).stem

    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, py_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def float_format(v: float, width: int) -> str:
    if v is None:
        return ""
    if not np.isfinite(v):
        return str(v)
    decimals = max(0, width - len(str(int(v))))
    return f"{v:.{decimals}f}"


def q_format(v: Q_, width: int) -> str:
    if v is None:
        return ""

    # ff = [f"{float_format(q.magnitude, width)} [{q.units}]" for q in (v, v.to_compact())]
    # return min(ff, key=len)
    vc = v.to_compact()
    return f"{float_format(vc.magnitude, width)} [{vc.units}]"

def pretty_float(v: float) -> str:
    if v is None:
        return ""
    if np.isnan(v):
        return "nan"
    av = abs(v)
    if not (av and not (1_000_000 > av > 0.001)):
        if av > 1000 or abs(v - round(v)) < 0.00001:
            return f"{v:.0f}"
        else:
            s = f"{v:.4f}"
            if s[-3:] == "000":
                return s[:-3]
            if s[-2:] == "00":
                return s[:-2]
            if s[-1:] == "0":
                return s[:-1]
            return s
    return f"{v:.2e}"


def make_list(v):
    if isinstance(v, list):
        return v
    elif isinstance(v, tuple):
        return [x for x in v]
    else:
        return [v]


def autoscale_factor(x):
    scaler = 1.0
    mx = np.max(np.abs(x))
    if np.isfinite(mx) and mx != 0:
        while mx > 10:
            mx *= 0.1
            scaler *= 0.1
        while mx < 1:
            mx *= 10.0
            scaler *= 10.0
    return scaler


def to_bool(sb):
    b = str(sb).lower().strip()
    if b in ["true", "t", "1"]:
        return True
    if b in ["false", "f", "0"]:
        return False
    raise ValueError(f"Not a bool {sb}")


def first_k(ar, k):
    x = ar.flatten()
    if k < 0:
        return np.array([])
    if x.size < k + 1:
        return np.sort(x)
    return np.partition(x, k - 1)[:k]


def approx_equal(a, b, relerr=np.finfo(float).eps, abserr=0):
    ab = rms(a - b)
    if ab <= abserr:
        return True
    return 2.0 * ab <= relerr * (rms(a) + rms(b))


# function log(1 + exp(x))
def log1exp(x):
    ret = np.atleast_1d(x).copy()
    idx = ret < 30
    ret[idx] = np.log1p(np.exp(ret[idx]))
    return ret


def ceil125(x):
    """Round up to the nearest pleasing number"""
    val125 = (10, 20, 25, 50, 100)
    if x == 0:
        return 0
    if x < 0:
        return -floor125(-x)
    p10 = 10 ** (np.floor(np.log10(x)) - 1)
    y = x / p10
    i = 0
    while val125[i] < y:
        i += 1
    return val125[i] * p10


def floor125(x):
    """Round down to the nearest pleasing number"""
    val125 = (10, 20, 25, 50, 100)
    if x == 0:
        return 0
    if x < 0:
        return -ceil125(-x)
    p10 = 10 ** (np.floor(np.log10(x)) - 1)
    y = x / p10
    i = 0
    while val125[i] <= y:
        i += 1
    return val125[i - 1] * p10


def arange125(a, b, n):
    """Arrange a list of numbers between boundaries a and b that are pleasing to use in a plot scale
    A number is considered pleasing if its significant digits are a divisor of 100
    """
    assert n > 1
    if a < 0 and b < 0:
        ret_inv = arange125(-a, -b, n)
        ret = [-r for r in ret_inv[::-1]]
    else:
        if a > b:
            a, b = b, a
        elif a == b:
            b = a + 1 if a == 0 else a * 1.1
        #
        dd = ceil125((b - a) / n)
        ret = []
        if a < 0 and b > 0:
            # make sure 0 is in the list
            x = -dd
            while x > a:
                ret.insert(0, x)
                x -= dd
            ret.insert(0, x)
            x = 0.0  # add 0
            while x < b:
                ret.append(x)
                x += dd
            ret.append(x)
        else:
            x = dd * np.floor(a / dd)
            while x < b:
                ret.append(x)
                x += dd
            ret.append(x)
    return ret


def prime_factors(n):
    """Find prime factors of a positive integer
    Brute force algorithm, only use for small numbers"""
    if n < 2:
        raise ValueError("Invalid argument")
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def find_divisor(a, b):
    """Find largest divisor c from a such that a/c >= b"""
    """ Brute force algorithm, only use for small numbers"""
    if a < 2 or b > a:
        raise ValueError("Invalid argument")
    for c in range(a, 1, -1):
        if a % c == 0 and a / c >= b:
            return c
    return 1


def make_grad(func, p, h):
    """Computes the gradient of func at point p with stepsize h
    This function is slow and prone to numerical problems,
    but very useful to check your analytical calculated gradient.
    """
    h2 = 0.5 / h
    g = np.zeros_like(p)
    for k in range(len(p)):
        x = p[k]
        p[k] = x - h
        y1 = func(p)
        p[k] = x + h
        y2 = func(p)
        p[k] = x
        g[k] = (y2 - y1) * h2
    return g


def make_filename(inFilename):
    sourcedirname = os.path.dirname(os.path.abspath("__file__"))
    return os.path.join(sourcedirname, inFilename)


def savedebug(ar, name="debug.npy"):
    fname = name if os.path.splitext(name)[1] else name + ".npy"
    np.save(make_filename(fname), ar)


def save_raw(x, name):
    arx = np.asarray(x, dtype=float)
    assert len(arx.shape) == 1
    with open(name, "wb") as f:
        f.write(struct.pack("i", 1729))  # magic cookie
        f.write(struct.pack("i", arx.shape[0]))
        arx.tofile(f)


def read_raw(name):
    with open(name, "rb") as f:
        if 1729 != struct.unpack("i", f.read(4))[0]:  # magic cookie
            raise ValueError()
        xlen = struct.unpack("i", f.read(4))[0]
        arx = np.fromfile(f, dtype=float)
        if len(arx.shape) != 1 or arx.shape[0] != xlen:
            raise ValueError()
        return arx


def is_power2(inX):
    """true if inX is a power of 2"""
    x = abs(int(inX))
    return (x - 1) & x == 0


def ceil_power2(inX):
    """
    rounds up to the next power of 2
    if inX is a power of 2 it returns inX
    """
    x = int(inX)
    if x == 0:
        return 0
    assert x > 0
    ret = 1
    while ret < x:
        ret <<= 1
    return ret


def next_power2(inX):
    """
    returns the next power of 2
    if inX is a power of 2 it returns 2*inX
    """
    return ceil_power2(2 * int(inX))


def floor_power2(inX):
    """
    rounds up to the previous power of 2
    if inX is a power of 2 it returns inX
    """
    x = ceil_power2(inX)
    return inX if x == inX else x // 2


def rolling_stats(x, count, axis: typing.Optional[int] = None):
    """compute cumulative statistics over a range of length count"""
    x = np.atleast_1d(x)
    if axis is None:
        assert x.ndim == 1
        axis = 0
    ret = np.zeros([4] + list(x.shape))
    # start with the front edge: we do it brute force
    sl_x = [slice(None)] * x.ndim
    sl_r = [slice(None)] * x.ndim
    for t in range(min(count, x.shape[axis])):
        sl_x[axis] = slice(0, t + 1)
        sl_r[axis] = t
        ret[tuple([0] + sl_r)] = np.amin(x[tuple(sl_x)], axis=axis)
        ret[tuple([1] + sl_r)] = np.amax(x[tuple(sl_x)], axis=axis)
        ret[tuple([2] + sl_r)] = np.mean(x[tuple(sl_x)], axis=axis)
    # now the bulk using stride_tricks and cumsum
    if x.shape[axis] > count:
        # non-linear statistics with stride_tricks
        sl_r[axis] = slice(count - 1, None)
        shape = x.shape[:axis] + (x.shape[axis] - count + 1, count) + x.shape[axis + 1 :]
        strides = x.strides[:axis] + (x.strides[axis], x.strides[axis]) + x.strides[axis + 1 :]
        rolling = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        ret[tuple([0] + sl_r)] = np.amin(rolling, axis=axis + 1)
        ret[tuple([1] + sl_r)] = np.amax(rolling, axis=axis + 1)
        # linear statistics with cumsum (only mean for now)
        sl_r[axis] = slice(count, None)
        sl_x2 = [slice(None)] * x.ndim
        sl_x[axis] = slice(count, None)
        sl_x2[axis] = slice(None, -count)
        cs1 = np.cumsum(x, axis=axis)
        ret[tuple([2] + sl_r)] = (cs1[tuple(sl_x)] - cs1[tuple(sl_x2)]) / count
    return ret


def monotonic_envelope(x, y):
    """Computes the monotone envelope of a set of 2D points
    and estimates an exponential decay through it
    """
    mi = np.argmax(y)
    y1 = np.maximum.accumulate(y[0:mi])
    y2 = np.maximum.accumulate(y[mi:][::-1])[::-1]
    func = lambda x, a, b, c: np.exp(-a * x) * b + c
    popt, pcov = sopt.curve_fit(func, x[mi:], y2)
    return x, np.hstack((y1, y2)), popt, func


def contiguous_regions(x):
    """Finds contiguous regions of the array "x".
    Returns a 2D array where the first column is the start index of the region and the
    second column is the end index.
    """
    if len(x) == 0:
        return np.empty((0, 2), dtype=int)
    xd = np.nonzero(np.diff(x))[0] + 1
    return np.vstack((np.hstack((0, xd)), np.hstack((xd, len(x)))))


def contiguous_true_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index.
    """
    # Find the indicies of changes in "condition"
    if len(condition) == 0:
        return np.empty((0, 2), dtype=int)
    idx = np.diff(condition).nonzero()[0] + 1
    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]
    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]  # Edit
    # Reshape the result into two columns where each row is a range
    return np.reshape(idx, (-1, 2))


def normalized_hanning(blocksize):
    """Hanning window normalized to 1"""
    window = np.hanning(blocksize)
    window /= linalg.norm(window)
    return window


def butter_taper(ff, fc, order=1, dtype="lowpass", q=5):
    """Creates a taper following a butterworth curve"""
    p = 2 * order
    g = ff / float(fc)
    if dtype == "highpass":
        g = 1 / g
    elif dtype == "bandpass":
        g = q * (g - 1 / g)
    taper = np.sqrt(1.0 / (1.0 + (g) ** p))
    return taper


def gauss_taper(ff, fc, attenuation):
    """Creates a taper following a gaussian curve"""
    assert attenuation > np.sqrt(2)
    ret = np.ones_like(ff)
    tail = ff >= fc
    if np.any(tail):
        e1, e2 = np.sqrt(0.5 * np.log(2)), np.sqrt(np.log(attenuation))
        invsigma = (e2 - e1) / (ff[-1] - fc)
        mu = fc - e1 / invsigma
        gauss = np.exp(-(((ff - mu) * invsigma) ** 2))
        ret[tail] = gauss[tail]
    return ret


def region_slice(ff, ffx):
    """Given a range ff select the same range from ffx"""
    m1 = np.argmin(ffx <= ff[0])
    m2 = np.argmax(ffx > ff[-1])
    return slice(m1, m2)


def rms(signal):
    s = np.atleast_1d(signal)
    return linalg.norm(s) / sqrt(s.size)


def abs_signed(x):
    return copysign(abs(x), x.real) if isinstance(x, complex) else abs(x)


def log_index(n, m):
    """
    Select m indices from range(n) in an exponential progression
    """
    if m > n or n < 2:
        return np.arange(n)

    # exp(k*(m-1)) - 1 == n - 1 + delta
    k = np.log(n) / (m - 1)
    x = np.arange(m)
    y = np.round(np.exp(k * x) - 1)
    idx = np.where(y < x)
    y[idx] = x[idx]
    return y.astype(int)


def gdelay(df, hh, window_length=3, polyorder=2):
    """Estimate groupdelay from a transfer"""
    up = -np.unwrap(np.angle(hh))
    dphase = ssig.savgol_filter(up, window_length=window_length, polyorder=polyorder, deriv=1)
    df = np.atleast_1d(df)
    if len(df) == 1:
        domega = (2 * np.pi) * df[0]
    else:
        domega = (2 * np.pi) * ssig.savgol_filter(df, window_length=window_length, polyorder=polyorder, deriv=1)
    return dphase / domega


def trend(signal, method, ndetrend):
    """Detrend the signal by one of several methods"""
    if method is None:
        return np.zeros_like(signal)
    elif (
        method == "mean"
        or method == "power"
        and ndetrend <= 0
        or method == "spline"
        and ndetrend <= 1
        or method == "fft"
        and ndetrend <= 1
    ):
        return np.zeros_like(signal) + np.mean(signal)
    elif method == "linear" or method == "power" and ndetrend == 1:
        nt = len(signal)
        tvec = np.arange(nt, dtype=float) / nt
        slope, intercept, _, _, _ = sstat.linregress(tvec, signal)
        return intercept + slope * tvec
    elif method == "power":
        nt = len(signal)
        tvec = np.arange(nt, dtype=float) / nt
        coefs = npoly.polyfit(tvec, signal, ndetrend)
        return npoly.polyval(tvec, coefs)
    elif method == "fft":
        # we remove frequency content in the first ndetrend lines.
        nt = len(signal)
        spec = np.fft.rfft(signal)
        spec[ndetrend:] = 0.0
        return np.fft.irfft(spec)
    elif method == "spline":
        # We fit the spline such that the norm of the residue after detrending is
        # approx equal to the norm of the spectrum above ndetrend lines.
        nt = len(signal)
        tvec = np.arange(nt) / nt
        spec = np.fft.rfft(signal)
        shigh = (2.0 / nt) * linalg.norm(spec[ndetrend:]) ** 2
        spl = sipol.UnivariateSpline(tvec, signal, s=shigh)
        return spl(tvec)
    else:
        raise ValueError(f"unknown method {method} for trend")


def detrend(signal, method, ndetrend):
    td = trend(signal, method, ndetrend)
    return signal - td
