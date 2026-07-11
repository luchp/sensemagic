"""Router for the mimoshape block generator page.

Follows the sensemagic conventions: ``prefix`` from the filename, a
``router`` variable, ``router.description`` for WordPress sync.  Downloads
are stateless: synthesis is deterministic given the query parameters, so the
download endpoint just re-runs it.
"""

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
import numpy as np

from pages.mimoshape.generator import (
    NT_CHOICES,
    GeneratorParams,
    synthesize,
    to_csv_bytes,
    to_npy_bytes,
    to_wav_bytes,
)

prefix = Path(__file__).stem  # "app_mimoshape"
router = APIRouter(prefix=f"/{prefix}", tags=["mimoshape"])
router.description = (
    "Generate synthetic test signals with shaped statistics: minimum crest "
    "factor, minimum or target kurtosis, and prescribed skewness"
)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

OBJECTIVES = {
    "min_crest": "minimum crest factor",
    "min_kurtosis": "minimum kurtosis",
    "target_kurtosis": "target kurtosis",
}


def query_params(
    nt: int = Query(default=4096),
    band: float = Query(default=0.5, ge=0.05, le=1.0),
    objective: str = Query(default="min_crest"),
    kurtosis: float = Query(default=5.0, ge=1.0, le=30.0),
    skewness: float = Query(default=0.0, ge=-2.0, le=2.0),
    seed: int = Query(default=0, ge=0, le=2**31),
    high_quality: bool = Query(default=False),
) -> GeneratorParams:
    if objective not in OBJECTIVES:
        raise HTTPException(422, f"objective must be one of {list(OBJECTIVES)}")
    try:
        return GeneratorParams(
            nt=nt, band=band, objective=objective, kurtosis=kurtosis,
            skewness=skewness, seed=seed, high_quality=high_quality,
        )
    except ValueError as ex:
        raise HTTPException(422, str(ex))


def _render(request, context):
    """TemplateResponse across starlette versions: new (request, name, ctx)
    signature first, old (name, ctx-with-request) as fallback."""
    context["request"] = request
    try:
        return templates.TemplateResponse(
            request, "mimoshape/index.html", context
        )
    except TypeError:
        return templates.TemplateResponse("mimoshape/index.html", context)


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, standalone: bool = True):
    """Form with defaults; no synthesis until the user hits Generate."""
    return _render(request, _context(standalone, GeneratorParams(), result=None))


@router.get("/generate", response_class=HTMLResponse)
async def generate(
    request: Request,
    standalone: bool = True,
    params: GeneratorParams = Depends(query_params),
):
    """Synthesize a block and render form + results (shareable URL)."""
    result = synthesize(params)
    return _render(request, _context(standalone, params, result))


@router.get("/download")
async def download(
    fmt: str = Query(default="npy"),
    fs: int = Query(default=48000, ge=1000, le=192000),
    params: GeneratorParams = Depends(query_params),
):
    """Stateless download: re-runs the deterministic synthesis."""
    media = {
        "npy": ("application/octet-stream", to_npy_bytes),
        "csv": ("text/csv", to_csv_bytes),
        "wav": ("audio/wav", lambda x: to_wav_bytes(x, fs=fs)),
    }
    if fmt not in media:
        raise HTTPException(422, "fmt must be npy, csv or wav")
    result = synthesize(params)
    media_type, encode = media[fmt]
    stem = f"mimoshape_{params.objective}_nt{params.nt}_seed{params.seed}"
    return Response(
        encode(result.x),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{stem}.{fmt}"'},
    )


def _context(standalone, params, result):
    ctx = {
        "standalone": standalone,
        "params": params,
        "nt_choices": NT_CHOICES,
        "objectives": OBJECTIVES,
        "result": None,
        "plot_data": None,
    }
    if result is not None:
        x = result.x
        nt = len(x)
        psd = (np.abs(np.fft.rfft(x)) ** 2) * (2.0 / nt**2)
        ff = np.fft.rfftfreq(nt)
        hist, edges = np.histogram(x / np.std(x), bins=80, density=True)
        ctx["result"] = {
            "sampled_crest": f"{result.sampled_crest:.3f}",
            "physical_crest": f"{result.physical_crest:.3f}",
            "skewness": f"{result.skewness:.3f}",
            "kurtosis": f"{result.kurtosis:.3f}",
        }
        step = max(1, nt // 4096)  # cap payload for large blocks
        ctx["plot_data"] = json.dumps(
            {
                "trace_t": np.arange(0, nt, step).tolist(),
                "trace": x[::step].tolist(),
                "psd_f": ff[1:].tolist(),
                "psd": np.maximum(psd[1:], 1e-20).tolist(),
                "hist_x": ((edges[:-1] + edges[1:]) / 2).tolist(),
                "hist_y": hist.tolist(),
            }
        )
    return ctx
