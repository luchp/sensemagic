"""Router for the mimoshape record analyzer / MIMO reconstruction page.

Single-shot and stateless: one POST parses the uploaded record (wav/csv/npy),
estimates the Slepian multitaper CSD and moment targets, synthesizes matching
MIMO blocks, and embeds plots and downloads (base64 data URLs) in the
response.  Nothing is stored server-side.
"""

import base64
import io
import json
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np

from pages.mimoshape.analyzer import (
    MAX_BLOCKS,
    MERGE_CHOICES,
    NFFT_CHOICES,
    NW_CHOICES,
    AnalysisParams,
    UploadError,
    analyze_and_reconstruct,
    parse_upload,
)
from pages.mimoshape.generator import to_wav_bytes

prefix = Path(__file__).stem  # "app_mimoshape_file"
router = APIRouter(prefix=f"/{prefix}", tags=["mimoshape"])
router.description = (
    "Upload a multichannel record (audio, csv or numpy) and synthesize new "
    "signals matching its cross-spectral density and higher-order moments"
)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _render(request, context):
    """TemplateResponse across starlette versions."""
    context["request"] = request
    try:
        return templates.TemplateResponse(
            request, "mimoshape/analyze.html", context
        )
    except TypeError:
        return templates.TemplateResponse("mimoshape/analyze.html", context)


def _form_context(params: AnalysisParams, fs: float = 48000.0):
    return {
        "params": params,
        "fs": fs,
        "nfft_choices": NFFT_CHOICES,
        "nw_choices": NW_CHOICES,
        "max_blocks": MAX_BLOCKS,
        "merge_choices": MERGE_CHOICES,
        "error": None,
        "result": None,
        "plot_data": None,
        "downloads": None,
    }


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, standalone: bool = True):
    ctx = _form_context(AnalysisParams())
    ctx["standalone"] = standalone
    return _render(request, ctx)


@router.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    standalone: bool = True,
    file: UploadFile = File(...),
    fs: float = Form(default=48000.0),
    nfft: int = Form(default=4096),
    nw: float = Form(default=4.0),
    match_skewness: bool = Form(default=False),
    match_kurtosis: bool = Form(default=False),
    match_cokurtosis: bool = Form(default=False),
    match_coskewness: bool = Form(default=False),
    num_blocks: int = Form(default=4),
    multimodel: bool = Form(default=False),
    merge: str = Form(default="crossfade"),
    seed: int = Form(default=0),
):
    try:
        params = AnalysisParams(
            nfft=nfft,
            nw=nw,
            match_skewness=match_skewness,
            match_kurtosis=match_kurtosis,
            match_cokurtosis=match_cokurtosis,
            match_coskewness=match_coskewness,
            num_blocks=num_blocks,
            multimodel=multimodel,
            merge=merge,
            seed=seed,
        )
        data = await file.read()
        record, fs_used = parse_upload(file.filename or "", data, fs)
        result = analyze_and_reconstruct(record, fs_used, params)
    except UploadError as ex:
        ctx = _form_context(_safe_params(nfft, nw, num_blocks, seed), fs)
        ctx["standalone"] = standalone
        ctx["error"] = str(ex)
        return _render(request, ctx)

    ctx = _form_context(params, fs_used)
    ctx["standalone"] = standalone
    ctx.update(_result_context(result, params))
    return _render(request, ctx)


def _safe_params(nfft, nw, num_blocks, seed):
    """Best-effort params for re-rendering the form after an UploadError."""
    try:
        return AnalysisParams(nfft=nfft, nw=nw, num_blocks=num_blocks, seed=seed)
    except UploadError:
        return AnalysisParams()


def _data_url(payload: bytes, media_type: str) -> str:
    return f"data:{media_type};base64,{base64.b64encode(payload).decode()}"


def _result_context(result, params: AnalysisParams):
    nj = result.record.shape[0]
    nfft = params.nfft
    ff = (np.fft.rfftfreq(nfft) * result.fs).tolist()

    def csd_curves(G):
        psd = [np.abs(G[k, k]).tolist() for k in range(nj)]
        coh, phase, pair_labels = [], [], []
        for i in range(nj):
            for j in range(i + 1, nj):
                denom = np.abs(G[i, i]) * np.abs(G[j, j])
                coh.append((np.abs(G[i, j]) ** 2 / np.maximum(denom, 1e-30)).tolist())
                phase.append(np.angle(G[i, j]).tolist())
                pair_labels.append(f"{i}-{j}")
        return psd, coh, phase, pair_labels

    psd_r, coh_r, phase_r, pairs = csd_curves(result.G_record)
    psd_s, coh_s, phase_s, _ = csd_curves(result.G_synth)

    step = max(1, result.merged.shape[1] // 8192)
    trace = result.merged[:, ::step]

    # downloads: the merged synthesized signal (float32 keeps data URLs lean)
    npy_buf = io.BytesIO()
    np.save(npy_buf, result.merged.astype(np.float32))
    downloads = [
        ("synth.npy", _data_url(npy_buf.getvalue(), "application/octet-stream")),
    ]
    if result.merged.size <= 2_000_000:  # csv is ~10x bigger; skip when large
        csv_buf = io.BytesIO()
        np.savetxt(csv_buf, result.merged.T, fmt="%.9g", delimiter=",")
        downloads.append(("synth.csv", _data_url(csv_buf.getvalue(), "text/csv")))
    if nj == 1:
        downloads.append(
            (
                "synth.wav",
                _data_url(
                    to_wav_bytes(result.merged[0], fs=int(result.fs)), "audio/wav"
                ),
            )
        )

    return {
        "result": {
            "num_channels": nj,
            "num_samples": result.record.shape[1],
            "num_segments": result.num_segments,
            "num_tapers": result.num_tapers,
            "num_averages": result.num_segments * result.num_tapers,
            "num_sections": result.num_sections,
            "merge": params.merge,
            "merged_samples": result.merged.shape[1],
            "df": result.fs / nfft,
            "targets": [
                (label, f"{tgt:.3f}", f"{ach:.3f}")
                for label, _idx, tgt, ach in result.targets
            ],
        },
        "downloads": downloads,
        "plot_data": json.dumps(
            {
                "f": ff[1:],
                "psd_record": [p[1:] for p in psd_r],
                "psd_synth": [p[1:] for p in psd_s],
                "coh_record": [c[1:] for c in coh_r],
                "coh_synth": [c[1:] for c in coh_s],
                "phase_record": [p[1:] for p in phase_r],
                "phase_synth": [p[1:] for p in phase_s],
                "pairs": pairs,
                "trace_t": np.arange(0, result.merged.shape[1], step).tolist(),
                "trace": trace.tolist(),
            }
        ),
    }
