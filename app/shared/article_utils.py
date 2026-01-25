"""
Shared utilities for article management.
Used by both app_articles.py (FastAPI) and wordpress_sync.py (WordPress sync).
"""

from pathlib import Path
from typing import List, Dict, Optional
import markdown
import re


# Base path for articles content
ARTICLES_DIR = Path(__file__).parent.parent / "pages" / "articles"


def extract_metadata_from_markdown(md_content: str) -> Dict[str, str]:
    """
    Extract metadata from markdown file.
    Looks for title (first # heading), description (first paragraph), and date.

    Args:
        md_content: Raw markdown content

    Returns:
        dict with title, description, date
    """
    lines = md_content.split('\n')
    title = None
    description = None
    date = None

    # Find first heading
    for line in lines:
        if line.strip().startswith('# '):
            title = line.strip()[2:].strip()
            break

    # Find first paragraph (non-empty, non-heading line)
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#') and not stripped.startswith('*') and len(stripped) > 20:
            description = stripped[:200] + ('...' if len(stripped) > 200 else '')
            break

    # Look for date in italic text at the end (markdown convention: *December 30, 2025*)
    for line in reversed(lines[-10:]):  # Check last 10 lines
        if line.strip().startswith('*') and line.strip().endswith('*'):
            date_text = line.strip()[1:-1]
            # Simple date extraction - you can make this more sophisticated
            if any(month in date_text for month in ['January', 'February', 'March', 'April', 'May', 'June',
                                                      'July', 'August', 'September', 'October', 'November', 'December']):
                date = date_text
                break

    if not date:
        # Default to current date if not found
        from datetime import datetime
        date = datetime.now().strftime('%B %d, %Y')

    return {
        'title': title or 'Untitled',
        'description': description or 'No description available.',
        'date': date
    }


def discover_articles(include_private: bool = False) -> List[Dict[str, str]]:
    """
    Scan the articles directory recursively for markdown files and extract metadata.
    Subdirectories create categories in the tree structure.

    Args:
        include_private: Whether to include files starting with underscore

    Returns:
        List of dicts with slug, title, description, date, filename, category, category_path
    """
    if not ARTICLES_DIR.exists():
        return []

    articles = []
    # Use rglob for recursive search
    for md_file in ARTICLES_DIR.rglob("*.md"):
        # Skip private files unless explicitly requested
        if not include_private and md_file.name.startswith('_'):
            continue

        # Skip files in __pycache__ or similar
        if '__pycache__' in str(md_file):
            continue

        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            metadata = extract_metadata_from_markdown(content)

            # Get relative path from articles directory
            rel_path = md_file.relative_to(ARTICLES_DIR)

            # Extract category from subdirectory path
            if len(rel_path.parts) > 1:
                # File is in a subdirectory
                category_parts = rel_path.parts[:-1]  # All parts except filename
                category = category_parts[-1]  # Immediate parent folder name
                category_path = '/'.join(category_parts)  # Full path for nested categories
                # Create slug with category prefix: control_loops/article.md -> control-loops/article
                slug = category_path.lower().replace('_', '-') + '/' + md_file.stem.lower().replace('_', '-')
            else:
                # File is in root articles directory
                category = None
                category_path = None
                slug = md_file.stem.lower().replace('_', '-')

            # Format category for display: control_loops -> Control Loops
            category_display = category.replace('_', ' ').replace('-', ' ').title() if category else None

            articles.append({
                'slug': slug,
                'title': metadata['title'],
                'description': metadata['description'],
                'date': metadata['date'],
                'filename': md_file.name,
                'filepath': str(rel_path),  # Relative path for file operations
                'category': category_display,
                'category_path': category_path,
                'category_raw': category  # Original folder name
            })
        except Exception as e:
            print(f"Error reading article {md_file}: {e}")
            continue

    # Sort by date (newest first) - simple string comparison works for most date formats
    articles.sort(key=lambda x: x['date'], reverse=True)

    return articles


def articles_to_tree(articles: List[Dict]) -> Dict:
    """
    Organize flat list of articles into a hierarchical tree structure.

    Args:
        articles: List of article dicts from discover_articles()

    Returns:
        Dict with structure:
        {
            'uncategorized': [articles without category],
            'categories': {
                'Category Name': {
                    'articles': [articles in this category],
                    'subcategories': {...}  # For nested categories
                }
            }
        }
    """
    tree = {
        'uncategorized': [],
        'categories': {}
    }

    for article in articles:
        if article.get('category') is None:
            tree['uncategorized'].append(article)
        else:
            category = article['category']
            if category not in tree['categories']:
                tree['categories'][category] = {
                    'articles': [],
                    'category_raw': article.get('category_raw'),
                    'category_path': article.get('category_path')
                }
            tree['categories'][category]['articles'].append(article)

    # Sort categories alphabetically
    tree['categories'] = dict(sorted(tree['categories'].items()))

    return tree


def get_all_categories() -> List[Dict]:
    """
    Get list of all article categories (subdirectories).

    Returns:
        List of dicts with name, slug, article_count
    """
    articles = discover_articles(include_private=False)
    categories = {}

    for article in articles:
        cat = article.get('category')
        if cat:
            if cat not in categories:
                categories[cat] = {
                    'name': cat,
                    'slug': article.get('category_raw', '').lower().replace('_', '-'),
                    'path': article.get('category_path'),
                    'article_count': 0
                }
            categories[cat]['article_count'] += 1

    return list(categories.values())


def get_article_file_by_slug(slug: str) -> Optional[Path]:
    """
    Find an article file by its slug.
    Supports both flat slugs (architecture-blog-post) and
    category slugs (control-loops/bandwidth-article).

    Args:
        slug: Article slug (e.g., "architecture-blog-post" or "control-loops/article")

    Returns:
        Path to the markdown file, or None if not found
    """
    if not ARTICLES_DIR.exists():
        return None

    # Check if slug contains category path
    if '/' in slug:
        parts = slug.split('/')
        category_path = '/'.join(parts[:-1])
        file_slug = parts[-1]

        # Convert category path back to directory: control-loops -> control_loops
        dir_path = ARTICLES_DIR / category_path.replace('-', '_')

        if dir_path.exists():
            # Try uppercase version
            filename = file_slug.upper().replace('-', '_') + '.md'
            md_file = dir_path / filename
            if md_file.exists():
                return md_file

            # Try lowercase version
            filename = file_slug.replace('-', '_') + '.md'
            md_file = dir_path / filename
            if md_file.exists():
                return md_file

            # Try matching any file in the directory
            for file in dir_path.glob("*.md"):
                file_slug_check = file.stem.lower().replace('_', '-')
                if file_slug_check == file_slug:
                    return file

    # Flat slug (no category) - original behavior
    # Convert slug to filename: architecture-blog-post -> ARCHITECTURE_BLOG_POST.md
    filename = slug.upper().replace('-', '_') + '.md'
    md_file = ARTICLES_DIR / filename

    # If uppercase conversion doesn't find it, try lowercase with underscores
    if not md_file.exists():
        filename = slug.replace('-', '_') + '.md'
        md_file = ARTICLES_DIR / filename

    # If still not found, try to find by matching slug pattern (in root or subdirs)
    if not md_file.exists():
        for file in ARTICLES_DIR.rglob("*.md"):
            if '__pycache__' in str(file):
                continue
            # Check if this file's slug matches (considering its path)
            rel_path = file.relative_to(ARTICLES_DIR)
            if len(rel_path.parts) > 1:
                # File in subdirectory
                category_path = '/'.join(rel_path.parts[:-1])
                file_slug = category_path.lower().replace('_', '-') + '/' + file.stem.lower().replace('_', '-')
            else:
                file_slug = file.stem.lower().replace('_', '-')

            if file_slug == slug:
                return file
        return None

    return md_file if md_file.exists() else None


def render_markdown_to_html(md_content: str, strip_title: bool = True) -> Dict[str, any]:
    """
    Render markdown content to HTML with all extensions enabled.
    Supports LaTeX (via MathJax), code highlighting, tables, etc.

    Args:
        md_content: Raw markdown content
        strip_title: If True, removes the first H1 heading (since title is shown in template header)

    Returns:
        dict with 'html', 'toc', and 'success' keys
    """
    try:
        # Strip the first H1 heading if requested (to avoid duplicate title)
        if strip_title:
            # Remove the first line that starts with "# " (H1 heading)
            lines = md_content.split('\n')
            new_lines = []
            title_stripped = False
            for line in lines:
                if not title_stripped and line.strip().startswith('# '):
                    title_stripped = True
                    # Also skip any empty lines immediately after the title
                    continue
                # Skip empty lines right after title was stripped
                if title_stripped and not new_lines and not line.strip():
                    continue
                new_lines.append(line)
            md_content = '\n'.join(new_lines)

        # Configure markdown with extensions
        md = markdown.Markdown(extensions=[
            'tables',           # GitHub-style tables
            'fenced_code',      # ```code blocks```
            'codehilite',       # Syntax highlighting
            'toc',              # Table of contents
            'attr_list',        # {: style=""} syntax for images
        ], extension_configs={
            'codehilite': {
                'css_class': 'highlight',
                'linenums': False,
                'guess_lang': True
            }
        })

        html = md.convert(md_content)

        return {
            'html': html,
            'toc': md.toc if hasattr(md, 'toc') else None,
            'success': True
        }

    except Exception as e:
        return {
            'html': f'<h1>Error</h1><p>Failed to render markdown: {str(e)}</p>',
            'toc': None,
            'success': False
        }

