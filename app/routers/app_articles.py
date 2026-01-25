"""Router for serving blog posts and technical articles"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from shared.article_utils import (
    discover_articles,
    articles_to_tree,
    get_article_file_by_slug,
    render_markdown_to_html,
    extract_metadata_from_markdown,
    ARTICLES_DIR
)

prefix = Path(__file__).stem  # Gets "app_articles" from filename
router = APIRouter(prefix=f"/{prefix}", tags=["articles"])

# Application description for WordPress sync
router.description = "Read our technical articles about algorithms, software, electronics, and engineering topics"

# Setup templates
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@router.get("/", response_class=HTMLResponse)
async def blog_index(request: Request, standalone: bool = True):
    """List of available articles - automatically discovered from pages/articles/ directory"""

    posts = discover_articles(include_private=False)
    article_tree = articles_to_tree(posts)

    return templates.TemplateResponse(
        "blog/index.html",
        {
            "request": request,
            "posts": posts,
            "article_tree": article_tree,
            "standalone": standalone
        }
    )


@router.get("/{slug:path}", response_class=HTMLResponse)
async def blog_post(request: Request, slug: str, standalone: bool = True, print: str = None):
    """
    Serve any article by slug. Supports both flat slugs and category paths.

    Examples:
        - /app_articles/architecture-blog-post
        - /app_articles/control-loops/control-loop-bandwidth

    Parameters:
        standalone: If True, render with full page layout
        print: If 'linkedin', render optimized for Print to PDF as LinkedIn carousel
    """

    # Find the article file by slug using shared utility
    md_file = get_article_file_by_slug(slug)

    if not md_file or not md_file.exists():
        return HTMLResponse(
            content=f'<h1>Error</h1><p>Article not found: {slug}</p>',
            status_code=404
        )

    # Read the file content
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return HTMLResponse(
            content=f'<h1>Error</h1><p>Failed to read article: {str(e)}</p>',
            status_code=500
        )

    # Extract metadata and render markdown
    metadata = extract_metadata_from_markdown(content)
    result = render_markdown_to_html(content)

    if not result['success']:
        return HTMLResponse(content=result['html'], status_code=500)

    return templates.TemplateResponse(
        "blog/post.html",
        {
            "request": request,
            "title": metadata['title'],
            "content": result['html'],
            "toc": result.get('toc'),
            "standalone": standalone,
            "date": metadata['date'],
            "author": "SenseMagic Engineering",
            "print_mode": print  # 'linkedin' for carousel PDF mode
        }
    )

