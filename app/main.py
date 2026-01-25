from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from discover import Discover
import platform
from pathlib import Path

disc = Discover()
disc.get_routers()

app = FastAPI()

# Mount static files for CSS, JS, images and other assets
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Register routers
for prefix, router in disc.routers.items():
    app.include_router(router)

# Convenience route for print-style articles (LinkedIn carousel PDF)
@app.get("/print/articles/{slug}")
async def print_article(slug: str):
    """Redirect to article with print=linkedin mode for PDF export"""
    return RedirectResponse(url=f"/app_articles/{slug}?print=linkedin&standalone=true")

# Add root router on Windows to show all available routes
if platform.system() == "Windows":
    from routers.main_router import create_main_router
    main_router = create_main_router(disc)
    app.include_router(main_router)

