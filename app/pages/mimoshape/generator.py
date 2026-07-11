"""Stateless SISO block generator for the web page.

Everything is deterministic given a ``GeneratorParams``: the download
endpoint simply re-runs :func:`synthesize` from the query parameters, so no
server-side state is needed.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from mimoshape import CrestTarget, MomentTarget, SynthesisProblem, MimoShaper
from mimoshape import moments

NT_CHOICES = (1024, 2048, 4096, 8192, 16384)

# beta continuation schedules for the crest surrogate
BETA_QUICK = (5, 10, 20, 40, 80)
BETA_HIGH = (5, 10, 20, 40, 80, 160, 320)

# an unreachably low kurtosis target: CCSAQ settles smoothly onto the
# feasibility floor, which is exactly the minimum-kurtosis signal
KURTOSIS_FLOOR_TARGET = 1.0

MAX_TIME_PER_STAGE = 5.0  # seconds, resource cap for the shared server


@dataclass(frozen=True)
class GeneratorParams:
    nt: int = 4096
    band: float = 0.5  # upper band edge as fraction of Nyquist (zero tail above)
    taper: float = 0.1  # raised-cosine roll-off width at the band edge (fraction of Nyquist)
    objective: Literal["min_kurtosis", "min_crest", "target_kurtosis"] = "min_crest"
    kurtosis: float = 5.0  # only for objective == "target_kurtosis"
    skewness: float = 0.0
    seed: int = 0
    high_quality: bool = False

    def __post_init__(self):
        if self.nt not in NT_CHOICES:
            raise ValueError(f"nt must be one of {NT_CHOICES}")
        if not 0.05 <= self.band <= 1.0:
            raise ValueError("band must be in [0.05, 1.0]")
        if not 0.0 <= self.taper <= 0.5 * self.band:
            raise ValueError("taper must be in [0, band/2]")
        if not -2.0 <= self.skewness <= 2.0:
            raise ValueError("skewness must be in [-2, 2]")
        if not 1.0 <= self.kurtosis <= 30.0:
            raise ValueError("kurtosis must be in [1, 30]")
        if not 0 <= self.seed <= 2**31:
            raise ValueError("seed must be a non-negative 32-bit integer")


@dataclass(frozen=True)
class GeneratorResult:
    x: np.ndarray  # the block, shape (nt,)
    sampled_crest: float
    physical_crest: float  # on an 8x zero-padded reconstruction
    skewness: float
    kurtosis: float


def _band_H(nt: int, band: float, taper: float) -> np.ndarray:
    """Flat band with a raised-cosine roll-off at the edge.

    A smooth (tapered) edge avoids sinc-like ringing in the reconstructed
    waveform and measurably lowers the achievable physical crest factor.
    """
    nf = nt // 2 + 1
    edge = int(round(band * (nf - 1)))
    H = np.zeros((1, 1, nf), dtype=complex)
    H[0, 0, 1:edge] = 1.0
    w = int(round(taper * (nf - 1)))
    if w > 0:
        k = np.arange(w)
        H[0, 0, edge - w : edge] = 0.5 * (1 + np.cos(np.pi * k / w))
    return H


def synthesize(params: GeneratorParams) -> GeneratorResult:
    H = _band_H(params.nt, params.band, params.taper)
    # high weight: an infeasible kurtosis floor target must not be allowed to
    # trade the (feasible) skewness target away
    skew = MomentTarget((0, 0, 0), params.skewness, weight=100.0)
    rng = np.random.default_rng(params.seed)

    if params.objective == "min_crest":
        betas = BETA_HIGH if params.high_quality else BETA_QUICK
        start = rng.uniform(-np.pi, np.pi, params.nt // 2 - 1)
        for beta in betas:
            problem = SynthesisProblem(
                H, targets=[skew], crests=[CrestTarget(0, beta=beta)]
            )
            shaper = MimoShaper(
                problem, max_time=MAX_TIME_PER_STAGE, ftol_rel=1e-7, xtol_rel=1e-9
            )
            x = shaper.make_block(start=start)
            start = shaper.last_phase
    else:
        kurt_value = (
            KURTOSIS_FLOOR_TARGET
            if params.objective == "min_kurtosis"
            else params.kurtosis
        )
        problem = SynthesisProblem(
            H, targets=[skew, MomentTarget((0, 0, 0, 0), kurt_value)]
        )
        max_time = 4 * MAX_TIME_PER_STAGE if params.high_quality else MAX_TIME_PER_STAGE
        shaper = MimoShaper(problem, max_time=max_time, stop_loss=1e-10, rng=rng)
        x = shaper.make_block()

    return GeneratorResult(
        x=x[0],
        sampled_crest=float(np.max(np.abs(x[0])) / np.sqrt(np.mean(x[0] ** 2))),
        physical_crest=float(moments.oversampled_crest(x, 0)),
        skewness=float(moments.normalized_moment(x, (0, 0, 0))),
        kurtosis=float(moments.normalized_moment(x, (0, 0, 0, 0))),
    )


def to_wav_bytes(x: np.ndarray, fs: int = 48000) -> bytes:
    """16-bit mono wav, normalised to 90% full scale."""
    import io
    import wave

    scaled = np.round(x / np.max(np.abs(x)) * 0.9 * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(scaled.tobytes())
    return buf.getvalue()


def to_csv_bytes(x: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    np.savetxt(buf, x, fmt="%.9g")
    return buf.getvalue()


def to_npy_bytes(x: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    np.save(buf, x)
    return buf.getvalue()
