from fastapi import APIRouter
from fastapi.responses import HTMLResponse
import platform


def create_main_router(disc):
    """
    Create and return a router for the root path that displays all discovered routers.

    Args:
        disc: Discover instance with routers already loaded

    Returns:
        APIRouter configured with root path handler
    """
    router = APIRouter(tags=["Home"])

    @router.get("/", response_class=HTMLResponse)
    def root():
        """Display a list of all discovered routers"""
        routers_html = ""
        for prefix, router_obj in disc.routers.items():
            if not router_obj.routes:
                continue
            route = router_obj.routes[0]
            routers_html += f"""
            <div class="router-card">
                <a href='{prefix}/?standalone=true'>{route.path}</a></li>
            </div>
            """

        # Get platform info for display
        platform_name = platform.system()
        python_version = platform.python_version()
        router_count = len(disc.routers)

        return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SenseMagic API - Router Overview</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #555;
                margin-top: 30px;
            }}
            .router-card {{
                background-color: #f9f9f9;
                border-left: 4px solid #2196F3;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            }}
            .router-card h3 {{
                margin-top: 0;
                color: #2196F3;
            }}
            ul {{
                list-style: none;
                padding: 0;
            }}
            li {{
                padding: 8px 0;
                border-bottom: 1px solid #eee;
            }}
            li:last-child {{
                border-bottom: none;
            }}
            code {{
                background-color: #e7f3fe;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 12px;
                color: #0066cc;
                margin-right: 10px;
                min-width: 80px;
                display: inline-block;
            }}
            a {{
                color: #2196F3;
                text-decoration: none;
                font-weight: 500;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            .info {{
                background-color: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 12px;
                margin: 20px 0;
                border-radius: 5px;
            }}
            .footer {{
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                text-align: center;
                color: #777;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 SenseMagic API - Router Overview</h1>

            <div class="info">
                <strong>ℹ️ Development Mode:</strong> This index page is only shown when running on Windows.
                Found <strong>{router_count}</strong> router(s) automatically discovered.
            </div>

            <h2>📡 Available Routers</h2>
            {routers_html}

            <div class="footer">
                <p>Platform: {platform_name} | Python {python_version}</p>
                <p>FastAPI Application - Auto-discovered Routers</p>
            </div>
        </div>
    </body>
    </html>
    """

    return router
