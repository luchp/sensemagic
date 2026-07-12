"""Upload analysis + MIMO reconstruction for the web page.

Single-shot and stateless: one request parses the uploaded record, estimates
the multitaper CSD and moment targets, synthesizes matching blocks, and
returns everything (including download payloads) in the response.
"""

import io
import wave
from dataclasses import dataclass

import numpy as np

from mimoshape import (
    EndpointTarget,
    MomentTarget,
    SynthesisProblem,
    MimoShaper,
    estimate,
    moments,
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_CHANNELS = 4
NFFT_CHOICES = (512, 1024, 2048, 4096, 8192)
NW_CHOICES = (2.0, 3.0, 4.0, 6.0, 8.0)
MAX_BLOCKS = 8
MAX_SECTIONS = 8
MAX_TOTAL_BLOCKS = 16
MERGE_CHOICES = ("crossfade", "c1", "zero")
MAX_TIME_PER_BLOCK = 5.0
ENDPOINT_WEIGHT = 10.0  # scaled by 1/variance to make it dimensionless


class UploadError(ValueError):
    """User-facing problem with the uploaded file or settings."""


def parse_upload(filename: str, data: bytes, fs_field: float) -> tuple[np.ndarray, float]:
    """Parse wav/csv/npy bytes into a (channels, samples) float record + fs.

    Channels beyond MAX_CHANNELS are dropped. fs comes from the wav header,
    or from ``fs_field`` for csv/npy.
    """
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError(f"file exceeds {MAX_UPLOAD_BYTES // 2**20} MB limit")
    name = filename.lower()
    if name.endswith(".wav"):
        record, fs = _parse_wav(data)
    elif name.endswith(".csv") or name.endswith(".txt"):
        record, fs = _parse_csv(data), fs_field
    elif name.endswith(".npy"):
        record, fs = _parse_npy(data), fs_field
    else:
        raise UploadError("supported formats: .wav, .csv/.txt, .npy")
    if fs <= 0:
        raise UploadError("sample rate must be positive")
    record = np.atleast_2d(np.asarray(record, dtype=float))
    if record.shape[0] > record.shape[1]:
        record = record.T  # rows = channels
    if record.shape[0] > MAX_CHANNELS:
        record = record[:MAX_CHANNELS]
    record = record - np.mean(record, axis=1, keepdims=True)
    if np.any(np.std(record, axis=1) == 0):
        raise UploadError("a channel is constant; cannot normalise")
    return record, float(fs)


def _parse_wav(data: bytes) -> tuple[np.ndarray, float]:
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            fs = w.getframerate()
            nch = w.getnchannels()
            width = w.getsampwidth()
            frames = w.readframes(w.getnframes())
    except wave.Error as ex:
        raise UploadError(f"could not read wav file: {ex}")
    if width == 2:
        x = np.frombuffer(frames, dtype="<i2").astype(float)
    elif width == 4:
        x = np.frombuffer(frames, dtype="<i4").astype(float)
    elif width == 3:
        b = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        x = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int8).astype(np.int32) << 16)
        ).astype(float)
    else:
        raise UploadError(f"unsupported wav sample width {width * 8} bit (use 16/24/32)")
    return x.reshape(-1, nch).T, float(fs)


def _parse_csv(data: bytes) -> np.ndarray:
    try:
        x = np.loadtxt(io.BytesIO(data), delimiter=None, ndmin=2)
    except ValueError:
        try:
            x = np.loadtxt(io.BytesIO(data), delimiter=",", ndmin=2)
        except ValueError as ex:
            raise UploadError(f"could not parse csv: {ex}")
    return x


def _parse_npy(data: bytes) -> np.ndarray:
    try:
        x = np.load(io.BytesIO(data), allow_pickle=False)
    except ValueError as ex:
        raise UploadError(f"could not parse npy: {ex}")
    if x.ndim not in (1, 2) or not np.issubdtype(x.dtype, np.number):
        raise UploadError("npy must be a 1-D or 2-D numeric array")
    return x.astype(float)


@dataclass(frozen=True)
class AnalysisParams:
    nfft: int = 4096
    nw: float = 4.0
    match_skewness: bool = True
    match_kurtosis: bool = True
    match_cokurtosis: bool = True
    match_coskewness: bool = False
    num_blocks: int = 4
    num_sections: int = 1  # 1 = whole file; >1 = piecewise (multimodel)
    merge: str = "crossfade"  # how consecutive blocks are joined
    seed: int = 0

    def __post_init__(self):
        if self.nfft not in NFFT_CHOICES:
            raise UploadError(f"block size must be one of {NFFT_CHOICES}")
        if self.nw not in NW_CHOICES:
            raise UploadError(f"NW must be one of {NW_CHOICES}")
        if not 1 <= self.num_blocks <= MAX_BLOCKS:
            raise UploadError(f"blocks must be 1..{MAX_BLOCKS}")
        if not 1 <= self.num_sections <= MAX_SECTIONS:
            raise UploadError(f"sections must be 1..{MAX_SECTIONS}")
        if self.num_sections * self.num_blocks > MAX_TOTAL_BLOCKS:
            raise UploadError(
                f"sections x blocks must be <= {MAX_TOTAL_BLOCKS} "
                f"(got {self.num_sections} x {self.num_blocks})"
            )
        if self.merge not in MERGE_CHOICES:
            raise UploadError(f"merge must be one of {MERGE_CHOICES}")
        if not 0 <= self.seed <= 2**31:
            raise UploadError("seed must be a non-negative 32-bit integer")


def moment_tuples(num_channels: int, p: AnalysisParams) -> list[tuple]:
    tuples = []
    for k in range(num_channels):
        if p.match_skewness:
            tuples.append((k, k, k))
        if p.match_kurtosis:
            tuples.append((k, k, k, k))
    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            if p.match_cokurtosis:
                tuples.append((i, i, j, j))
            if p.match_coskewness:
                tuples.append((i, i, j))
                tuples.append((i, j, j))
    return tuples


@dataclass(frozen=True)
class AnalysisResult:
    record: np.ndarray  # (Nj, N) as analysed
    fs: float
    G_record: np.ndarray  # multitaper CSD of the record
    G_synth: np.ndarray  # same estimator over the raw synthesized ensemble
    blocks: np.ndarray  # (Nj, total_blocks * nfft), raw (unmerged)
    merged: np.ndarray  # (Nj, ~total), blocks joined with the merge method
    num_segments: int  # per section
    num_tapers: int
    targets: list  # (label, tuple, target, achieved)


def _head_state(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Head value and periodic head slope (unit sample time) per channel."""
    return x[:, 0], 0.5 * (x[:, 1] - x[:, -1])


def _best_shift(tail: np.ndarray, block: np.ndarray) -> int:
    """Circular shift of ``block`` maximising correlation with ``tail``.

    Blocks are periodic, so a rotation is a pure linear phase: it changes
    nothing statistically but lets us splice where the waveforms agree
    (WSOLA-style alignment). The same shift is applied to all channels,
    preserving cross-spectra and cross-moments.
    """
    nt = block.shape[1]
    p = np.zeros_like(block)
    p[:, : tail.shape[1]] = tail
    corr = np.fft.irfft(
        np.conj(np.fft.rfft(p, axis=1)) * np.fft.rfft(block, axis=1), n=nt, axis=1
    )
    return int(np.argmax(np.sum(corr, axis=0)))


def _merge_crossfade(blocks: list, fade: int) -> np.ndarray:
    """Aligned equal-power crossfade: rotate each next block to best match
    the outgoing tail, then fade with cos/sin weights (variance-preserving
    for uncorrelated signals, exact for correlated ones)."""
    theta = np.pi / 2 * (np.arange(fade) + 0.5) / fade
    w_out, w_in = np.cos(theta), np.sin(theta)
    out = blocks[0]
    for block in blocks[1:]:
        shift = _best_shift(out[:, -fade:], block)
        rolled = np.roll(block, -shift, axis=1)
        mix = out[:, -fade:] * w_out + rolled[:, :fade] * w_in
        out = np.concatenate([out[:, :-fade], mix, rolled[:, fade:]], axis=1)
    return out


def analyze_and_reconstruct(
    record: np.ndarray, fs: float, p: AnalysisParams
) -> AnalysisResult:
    nj, n = record.shape
    sec_len = n // p.num_sections
    if sec_len < 2 * p.nfft:
        raise UploadError(
            f"sections too short: {sec_len} samples < 2 x block size {p.nfft}; "
            "use fewer sections or a smaller block size"
        )
    tuples = moment_tuples(nj, p)
    rng = np.random.default_rng(p.seed)
    # endpoint errors are in signal units; scale to make them dimensionless
    var = np.var(record, axis=1)
    ep_w = ENDPOINT_WEIGHT / np.maximum(var, 1e-30)

    blocks = []
    labelled = []
    prev = None  # last synthesized block, for the C1 chain
    for s in range(p.num_sections):
        section = record[:, s * sec_len : (s + 1) * sec_len]
        section = section - np.mean(section, axis=1, keepdims=True)
        if np.any(np.std(section, axis=1) == 0):
            raise UploadError(f"section {s + 1} has a constant channel")
        G = estimate.multitaper_csd(section, nw=p.nw, nfft=p.nfft)
        try:
            H = estimate.csd_to_frf(G, variance=np.var(section, axis=1))
        except np.linalg.LinAlgError:
            raise UploadError(
                f"CSD estimate of section {s + 1} is not positive definite "
                "(too few averages for the channel count); use fewer "
                "sections, a smaller block size or larger NW"
            )
        targets = estimate.estimate_moment_targets(section, tuples)

        sec_blocks = []
        for _ in range(p.num_blocks):
            if p.merge == "zero":
                endpoints = [
                    EndpointTarget(k, 0.0, 0.0, ep_w[k], ep_w[k]) for k in range(nj)
                ]
            elif p.merge == "c1" and prev is not None:
                # periodic continuation of the previous block ends at its own
                # head state: match it for a C1 joint
                head, slope = _head_state(prev)
                endpoints = [
                    EndpointTarget(k, head[k], slope[k], ep_w[k], ep_w[k])
                    for k in range(nj)
                ]
            else:
                endpoints = []
            problem = SynthesisProblem(H, targets=targets, endpoints=endpoints)
            shaper = MimoShaper(
                problem, max_time=MAX_TIME_PER_BLOCK, stop_loss=1e-10, rng=rng
            )
            prev = shaper.make_block()
            sec_blocks.append(prev)
        blocks.extend(sec_blocks)

        sec_tag = f" (sec {s + 1})" if p.num_sections > 1 else ""
        labelled += [
            (
                _tuple_label(t.indices) + sec_tag,
                t.indices,
                t.value,
                float(
                    np.mean(
                        [moments.normalized_moment(b, t.indices) for b in sec_blocks]
                    )
                ),
            )
            for t in targets
        ]

    ensemble = np.hstack(blocks)
    if p.merge == "crossfade" and len(blocks) > 1:
        merged = _merge_crossfade(blocks, fade=p.nfft // 16)
    else:
        merged = ensemble  # zero / c1 joints concatenate as-is
    G_synth = estimate.multitaper_csd(ensemble, nw=p.nw, nfft=p.nfft)
    return AnalysisResult(
        record=record,
        fs=fs,
        G_record=estimate.multitaper_csd(record, nw=p.nw, nfft=p.nfft),
        G_synth=G_synth,
        blocks=ensemble,
        merged=merged,
        num_segments=sec_len // p.nfft,
        num_tapers=int(2 * p.nw - 1),
        targets=labelled,
    )


def _tuple_label(indices: tuple) -> str:
    order = len(indices)
    unique = sorted(set(indices))
    if len(unique) == 1:
        return f"{'skewness' if order == 3 else 'kurtosis'} ch{unique[0]}"
    return f"{'co-skewness' if order == 3 else 'co-kurtosis'} {indices}"
