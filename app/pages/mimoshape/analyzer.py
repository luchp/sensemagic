"""Upload analysis + MIMO reconstruction for the web page.

Single-shot and stateless: one request parses the uploaded record, estimates
the multitaper CSD and moment targets, synthesizes matching blocks, and
returns everything (including download payloads) in the response.
"""

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf

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
MAX_MULTIMODEL_SECTIONS = 1024  # ~2 min of 48 kHz audio at nfft 8192
MERGE_CHOICES = ("crossfade", "c1", "zero")
MAX_TIME_PER_BLOCK = 5.0
MIN_TIME_PER_BLOCK = 0.25
TIME_BUDGET = 300.0  # total optimizer seconds, shared over all blocks
ENDPOINT_WEIGHT = 10.0  # scaled by 1/variance to make it dimensionless


class UploadError(ValueError):
    """User-facing problem with the uploaded file or settings."""


def parse_upload(filename: str, data: bytes, fs_field: float) -> tuple[np.ndarray, float]:
    """Parse audio/csv/npy bytes into a (channels, samples) float record + fs.

    Anything that is not csv/npy is handed to libsndfile (wav, flac, ogg,
    mp3, ...) -- compressed audio uploads much faster. Channels beyond
    MAX_CHANNELS are dropped. fs comes from the audio header, or from
    ``fs_field`` for csv/npy.
    """
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError(f"file exceeds {MAX_UPLOAD_BYTES // 2**20} MB limit")
    name = filename.lower()
    if name.endswith(".csv") or name.endswith(".txt"):
        record, fs = _parse_csv(data), fs_field
    elif name.endswith(".npy"):
        record, fs = _parse_npy(data), fs_field
    else:
        record, fs = _parse_audio(data)
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


def _parse_audio(data: bytes) -> tuple[np.ndarray, float]:
    try:
        x, fs = sf.read(io.BytesIO(data), always_2d=True, dtype="float64")
    except Exception as ex:
        raise UploadError(
            f"could not read audio file ({ex}); supported formats: any "
            "libsndfile audio (wav, flac, ogg, mp3, ...), .csv/.txt, .npy"
        )
    return x.T, float(fs)


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
    multimodel: bool = False  # piecewise: one spectral model per block length
    merge: str = "crossfade"  # how consecutive blocks are joined
    seed: int = 0

    def __post_init__(self):
        if self.nfft not in NFFT_CHOICES:
            raise UploadError(f"block size must be one of {NFFT_CHOICES}")
        if self.nw not in NW_CHOICES:
            raise UploadError(f"NW must be one of {NW_CHOICES}")
        if not 1 <= self.num_blocks <= MAX_BLOCKS:
            raise UploadError(f"blocks must be 1..{MAX_BLOCKS}")
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
    num_sections: int
    num_segments: int  # per section
    num_tapers: int
    targets: list  # (label, tuple, target, achieved); multimodel: section avg


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
    record: np.ndarray, fs: float, p: AnalysisParams, progress=None
) -> AnalysisResult:
    """``progress(done_blocks, total_blocks)`` is called after each block."""
    nj, n = record.shape
    if p.multimodel:
        # one spectral model per block length: the output tracks the record
        num_sections = n // p.nfft
        sec_len = p.nfft
        blocks_per_sec = 1
        if num_sections < 2:
            raise UploadError(
                f"record too short for multimodel: {n} samples < "
                f"2 x block size {p.nfft}; untick multimodel or use a "
                "smaller block size"
            )
        if num_sections > MAX_MULTIMODEL_SECTIONS:
            raise UploadError(
                f"too many sections: {num_sections} > "
                f"{MAX_MULTIMODEL_SECTIONS}; use a larger block size or a "
                "shorter record"
            )
    else:
        num_sections = 1
        sec_len = n
        blocks_per_sec = p.num_blocks
        if n < 2 * p.nfft:
            raise UploadError(
                f"record too short: {n} samples < 2 x block size {p.nfft}"
            )
    total_blocks = num_sections * blocks_per_sec
    max_time = max(MIN_TIME_PER_BLOCK, min(MAX_TIME_PER_BLOCK, TIME_BUDGET / total_blocks))
    tuples = moment_tuples(nj, p)
    rng = np.random.default_rng(p.seed)
    # endpoint errors are in signal units; scale to make them dimensionless
    var = np.var(record, axis=1)
    ep_w = ENDPOINT_WEIGHT / np.maximum(var, 1e-30)

    blocks = []
    labelled = {}  # label -> (indices, [targets], [achieved])
    prev = None  # last synthesized block, for the C1 chain
    done = 0
    for s in range(num_sections):
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
                "(too few averages for the channel count); use a larger NW "
                "or block size"
            )
        targets = estimate.estimate_moment_targets(section, tuples)

        sec_blocks = []
        for _ in range(blocks_per_sec):
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
                problem, max_time=max_time, stop_loss=1e-10, rng=rng
            )
            prev = shaper.make_block()
            sec_blocks.append(prev)
            done += 1
            if progress is not None:
                progress(done, total_blocks)
        blocks.extend(sec_blocks)

        for t in targets:
            label = _tuple_label(t.indices)
            ach = float(
                np.mean([moments.normalized_moment(b, t.indices) for b in sec_blocks])
            )
            entry = labelled.setdefault(label, (t.indices, [], []))
            entry[1].append(t.value)
            entry[2].append(ach)

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
        num_sections=num_sections,
        num_segments=sec_len // p.nfft,
        num_tapers=int(2 * p.nw - 1),
        targets=[
            (label, idx, float(np.mean(tgts)), float(np.mean(achs)))
            for label, (idx, tgts, achs) in labelled.items()
        ],
    )


def _tuple_label(indices: tuple) -> str:
    order = len(indices)
    unique = sorted(set(indices))
    if len(unique) == 1:
        return f"{'skewness' if order == 3 else 'kurtosis'} ch{unique[0]}"
    return f"{'co-skewness' if order == 3 else 'co-kurtosis'} {indices}"
