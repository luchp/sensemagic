"""Router for developer tools: Philips date code and password generator"""

from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from numlib.numutil import philips_date
from pages.utils.password_generator import generate_password

prefix = Path(__file__).stem  # "app_tools"
router = APIRouter(prefix=f"/{prefix}", tags=["tools"])

# Application description for WordPress sync
router.description = "Handy developer tools: Philips date code and strong password generator"

# Setup templates
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@router.get("/", response_class=HTMLResponse)
async def tools_index(request: Request, standalone: bool = True):
    """Tools page with Philips date and password generator"""
    pd = philips_date()
    pwd = generate_password(16)
    return templates.TemplateResponse(
        "tools/index.html",
        {
            "request": request,
            "standalone": standalone,
            "philips_date": pd,
            "password": pwd,
            "password_length": 16,
        },
    )


@router.get("/api/philips-date", response_class=JSONResponse)
async def api_philips_date():
    """Return the current Philips date code"""
    return {"philips_date": philips_date()}


@router.get("/api/password", response_class=JSONResponse)
async def api_password(length: int = Query(default=16, ge=8, le=128)):
    """Generate a new password with the given length"""
    return {"password": generate_password(length)}

