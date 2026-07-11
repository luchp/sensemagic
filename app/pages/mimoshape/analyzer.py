"""Upload analysis + MIMO reconstruction for the web page.

Single-shot and stateless: one request parses the uploaded record, estimates
the multitaper CSD and moment targets, synthesizes matching blocks, and
returns everything (including download payloads) in the response.
"""

import io
import wave
from dataclasses import dataclass

import numpy as np

from mimoshape import MomentTarget, SynthesisProblem, MimoShaper, estimate, moments

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_CHANNELS = 4
NFFT_CHOICES = (512, 1024, 2048, 4096, 8192)
NW_CHOICES = (2.0, 3.0, 4.0, 6.0, 8.0)
MAX_BLOCKS = 8
MAX_TIME_PER_BLOCK = 5.0


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
    seed: int = 0

    def __post_init__(self):
        if self.nfft not in NFFT_CHOICES:
            raise UploadError(f"block size must be one of {NFFT_CHOICES}")
        if self.nw not in NW_CHOICES:
            raise UploadError(f"NW must be one of {NW_CHOICES}")
        if not 1 <= self.num_blocks <= MAX_BLOCKS:
            raise UploadError(f"blocks must be 1..{MAX_BLOCKS}")
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
    G_synth: np.ndarray  # same estimator over the synthesized ensemble
    blocks: np.ndarray  # (Nj, num_blocks * nfft)
    num_segments: int
    num_tapers: int
    targets: list  # (label, tuple, target, achieved)


def analyze_and_reconstruct(
    record: np.ndarray, fs: float, p: AnalysisParams
) -> AnalysisResult:
    nj, n = record.shape
    if n < 2 * p.nfft:
        raise UploadError(
            f"record too short: {n} samples < 2 x block size {p.nfft}"
        )
    G = estimate.multitaper_csd(record, nw=p.nw, nfft=p.nfft)
    try:
        H = estimate.csd_to_frf(G, variance=np.var(record, axis=1))
    except np.linalg.LinAlgError:
        raise UploadError(
            "CSD estimate is not positive definite at some frequency "
            "(too few averages for the channel count); use a smaller "
            "block size or larger NW"
        )
    tuples = moment_tuples(nj, p)
    targets = estimate.estimate_moment_targets(record, tuples)

    problem = SynthesisProblem(H, targets=targets)
    shaper = MimoShaper(
        problem,
        max_time=MAX_TIME_PER_BLOCK,
        stop_loss=1e-10,
        rng=np.random.default_rng(p.seed),
    )
    blocks = [shaper.make_block() for _ in range(p.num_blocks)]
    ensemble = np.hstack(blocks)

    achieved = [
        (
            t.indices,
            t.value,
            float(np.mean([moments.normalized_moment(b, t.indices) for b in blocks])),
        )
        for t in targets
    ]
    G_synth = estimate.multitaper_csd(ensemble, nw=p.nw, nfft=p.nfft)
    num_tapers = int(2 * p.nw - 1)
    return AnalysisResult(
        record=record,
        fs=fs,
        G_record=G,
        G_synth=G_synth,
        blocks=ensemble,
        num_segments=n // p.nfft,
        num_tapers=num_tapers,
        targets=[
            (_tuple_label(idx), idx, tgt, ach) for idx, tgt, ach in achieved
        ],
    )


def _tuple_label(indices: tuple) -> str:
    order = len(indices)
    unique = sorted(set(indices))
    if len(unique) == 1:
        return f"{'skewness' if order == 3 else 'kurtosis'} ch{unique[0]}"
    return f"{'co-skewness' if order == 3 else 'co-kurtosis'} {indices}"
