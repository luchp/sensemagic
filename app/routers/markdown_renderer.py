"""Utility for rendering markdown content with LaTeX support for FastAPI templates"""

import markdown
import re
from pathlib import Path
from typing import Optional, Dict


class MarkdownRenderer:
    """
    Renders markdown files to HTML with LaTeX preservation and link rewriting.

    Features:
    - Preserves LaTeX expressions ($...$ and $$...$$)
    - Rewrites markdown links to proper routes
    - Handles file loading and error cases
    - Generates standalone or iframe-embedded HTML
    """

    def __init__(self, base_path: Optional[Path] = None, route_prefix: str = ""):
        """
        Initialize the markdown renderer.

        Args:
            base_path: Base directory for markdown files (default: pages/rectifier)
            route_prefix: URL prefix for routes (e.g., '/app_rectifier')
        """
        if base_path is None:
            # Default to pages/rectifier
            base_path = Path(__file__).parent.parent / "pages" / "rectifier"
        self.base_path = base_path
        self.route_prefix = route_prefix

    def render_markdown(self, md_content: str, include_toc: bool = False, rewrite_links: bool = True, standalone: bool = True) -> dict:
        """
        Convert markdown to HTML with LaTeX support preserved.

        Args:
            md_content: Markdown content string
            include_toc: Whether to generate a table of contents
            rewrite_links: Whether to rewrite markdown links to route URLs
            standalone: Whether links should preserve standalone mode

        Returns:
            dict with 'html' and optionally 'toc' keys
        """
        # Protect LaTeX expressions from markdown processing
        latex_display_blocks = []
        latex_inline_blocks = []

        # Store display math ($$...$$)
        def store_display_math(match):
            latex_display_blocks.append(match.group(0))
            return f"<!--LATEXDISPLAY{len(latex_display_blocks)-1}-->"

        # Store inline math ($...$)
        def store_inline_math(match):
            latex_inline_blocks.append(match.group(0))
            return f"<!--LATEXINLINE{len(latex_inline_blocks)-1}-->"

        # Replace LaTeX with placeholders
        # First handle display math ($$...$$) - must be done before inline to avoid conflicts
        content = re.sub(r'\$\$(.+?)\$\$', store_display_math, md_content, flags=re.DOTALL)

        # Then handle inline math ($...$) - match anything except newline or dollar sign
        content = re.sub(r'(?<!\$)\$(?!\$)([^\$\n]+?)\$(?!\$)', store_inline_math, content)

        # Rewrite markdown links if requested
        if rewrite_links:
            content = self._rewrite_links(content, standalone=standalone)

        # Configure markdown extensions
        extensions = [
            'tables',
            'fenced_code',
            'codehilite',
            'nl2br',
            'attr_list',  # Allows {: style="..." } attribute syntax
        ]

        if include_toc:
            extensions.append('toc')

        extension_configs = {
            'codehilite': {
                'css_class': 'highlight',
                'linenums': False
            }
        }

        # Convert markdown to HTML
        md = markdown.Markdown(extensions=extensions, extension_configs=extension_configs)
        html = md.convert(content)

        # Restore LaTeX expressions
        for i, latex in enumerate(latex_display_blocks):
            html = html.replace(f"<!--LATEXDISPLAY{i}-->", latex)

        for i, latex in enumerate(latex_inline_blocks):
            html = html.replace(f"<!--LATEXINLINE{i}-->", latex)

        result = {'html': html}

        if include_toc and hasattr(md, 'toc'):
            result['toc'] = md.toc

        return result

    def _rewrite_links(self, content: str, standalone: bool = True) -> str:
        """
        Rewrite markdown links and images to proper route URLs.

        Converts:
        - [text](file.md) -> [text](/app_rectifier/route?standalone=...)
        - ![alt](image.jpg) -> ![alt](/static/rectifier/image.jpg)
        - [text](http://...) -> unchanged (external links)
        - [text](#anchor) -> unchanged (internal anchors)

        Args:
            content: Markdown content
            standalone: Whether to preserve standalone mode in links

        Returns:
            Content with rewritten links and images
        """
        def rewrite_link(match):
            text = match.group(1)
            url = match.group(2)

            # Don't rewrite external links or anchors
            if url.startswith(('http://', 'https://', '#', '/', 'data:')):
                return match.group(0)

            # Convert markdown file references to routes
            if url.endswith('.md'):
                # Extract just the filename without path or extension
                filename = Path(url).stem

                # Map common filenames to routes (case-insensitive)
                route_map = {
                    'rectifier_user_guide': f'{self.route_prefix}/guide',
                    'rectifier_math_clean': f'{self.route_prefix}/math',
                    'rectifier_mathematics': f'{self.route_prefix}/math',  # Alias for math page
                    'rectifier_math': f'{self.route_prefix}/math',
                    'wordpress_page_content': f'{self.route_prefix}/wordpress',
                }

                # Check if we have a mapping (case-insensitive lookup)
                route = route_map.get(filename.lower())
                if route:
                    return f'[{text}]({route}?standalone={str(standalone).lower()})'

                # For other .md files, try to construct a route from the filename
                # Convert SOME_FILE.md -> /app_rectifier/some-file
                route_name = filename.lower().replace('_', '-')
                return f'[{text}]({self.route_prefix}/{route_name}?standalone={str(standalone).lower()})'

            # Leave other links unchanged
            return match.group(0)

        def rewrite_image(match):
            alt_text = match.group(1)
            url = match.group(2)

            # Don't rewrite external images, absolute paths, or data URLs
            if url.startswith(('http://', 'https://', '/', 'data:')):
                return match.group(0)

            # Convert relative image paths to static URLs
            # Extract the feature name from route_prefix (e.g., /app_rectifier -> rectifier)
            feature_name = self.route_prefix.split('_')[-1] if '_' in self.route_prefix else 'default'

            # Convert relative path to static URL - images are now in static/images/
            static_url = f'/static/images/{feature_name}/{url}'

            return f'![{alt_text}]({static_url})'

        # Match markdown links: [text](url)
        content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', rewrite_link, content)

        # Match markdown images: ![alt](url)
        # Note: attr_list extension handles {: style=""} separately after markdown processing
        content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', rewrite_image, content)

        return content

    def _rewrite_html_images(self, html: str) -> str:
        """
        Rewrite HTML <img src="..."> tags to use static URLs for local files.

        Args:
            html: HTML content that may contain img tags

        Returns:
            HTML with rewritten image sources
        """
        def rewrite_img_src(match):
            # Get the full img tag and the src value
            full_tag = match.group(0)
            src = match.group(1)

            # Don't rewrite external URLs or absolute paths
            if src.startswith(('http://', 'https://', '/', 'data:')):
                return full_tag

            # Convert relative path to static URL
            feature_name = self.route_prefix.split('_')[-1] if '_' in self.route_prefix else 'default'
            static_url = f'/static/images/{feature_name}/{src}'

            # Replace src in the tag
            return full_tag.replace(f'src="{src}"', f'src="{static_url}"')

        # Match <img src="..."> tags
        html = re.sub(r'<img\s+[^>]*src="([^"]+)"[^>]*>', rewrite_img_src, html)

        return html


    def load_and_render(self, filename: str, include_toc: bool = False, standalone: bool = True) -> Dict[str, str]:
        """
        Load a markdown file and render it to HTML.

        Args:
            filename: Name of the markdown file (e.g., 'RECTIFIER_USER_GUIDE.md')
            include_toc: Whether to generate table of contents
            standalone: Whether to preserve standalone mode in links

        Returns:
            dict with 'html', 'success' (bool), and optional 'error' keys
        """
        md_file = self.base_path / filename

        if not md_file.exists():
            return {
                'html': f'<h1>Error</h1><p>Documentation file not found: {filename}</p>',
                'success': False,
                'error': f'File not found: {md_file}'
            }

        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                md_content = f.read()

            result = self.render_markdown(md_content, include_toc=include_toc, standalone=standalone)
            result['success'] = True
            return result

        except Exception as e:
            return {
                'html': f'<h1>Error</h1><p>Failed to read file: {str(e)}</p>',
                'success': False,
                'error': str(e)
            }

    def render_to_page(self, filename: str, title: str, standalone: bool = True, include_toc: bool = False) -> str:
        """
        Load markdown file and render as complete HTML page.

        Args:
            filename: Name of the markdown file
            title: Page title
            standalone: Whether to include MathJax (True) or iframe mode (False)
            include_toc: Whether to generate table of contents

        Returns:
            Complete HTML page as string
        """
        result = self.load_and_render(filename, include_toc=include_toc, standalone=standalone)
        return self.wrap_in_template(result['html'], title=title, standalone=standalone)

    def wrap_in_template(self, html_content: str, title: str = "Documentation", standalone: bool = True) -> str:
        """
        Wrap HTML content in a basic template structure.

        Args:
            html_content: The HTML content to wrap
            title: Page title
            standalone: Whether to include MathJax (True) or iframe communication (False)

        Returns:
            Complete HTML page as string
        """
        mathjax_script = """
    <script>
    window.MathJax = {
        tex: {
            inlineMath: [['$', '$']],
            displayMath: [['$$', '$$']],
            processEscapes: false,
            processEnvironments: true
        },
        options: {
            skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
        }
    };
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
    """ if standalone else ""

        iframe_script = """
    <script>
    function sendHeight() {
        const height = document.body.scrollHeight;
        parent.postMessage({iframeHeight: height}, "*");
    }
    window.onload = sendHeight;
    window.onresize = sendHeight;
    if (window.MutationObserver) {
        const observer = new MutationObserver(sendHeight);
        observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true
        });
    }
    </script>
    """ if not standalone else ""

        # Logo section for standalone mode
        logo_html = """
    <div class="logo-container">
        <a href="https://www.sensemagic.nl" target="_top">
            <img src="https://www.sensemagic.nl/wp-content/uploads/2017/05/logo_klein.png" alt="SenseMagic Logo">
        </a>
    </div>
    """ if standalone else ""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="/static/css/base.css">
    <link rel="stylesheet" href="/static/css/standalone.css">
    <style>
        /* MathJax-specific styling */
        .mjx-chtml {{
            font-size: 1.1em !important;
        }}
    </style>
    {mathjax_script}
</head>
<body>
    {logo_html}
    <div class="container">
        {html_content}
    </div>
    {iframe_script}
</body>
</html>
"""


# Backward compatibility functions
def render_markdown(md_content: str, include_toc: bool = False) -> dict:
    """Legacy function for backward compatibility"""
    renderer = MarkdownRenderer()
    return renderer.render_markdown(md_content, include_toc=include_toc)


def wrap_in_template(html_content: str, title: str = "Documentation", standalone: bool = True) -> str:
    """Legacy function for backward compatibility"""
    renderer = MarkdownRenderer()
    return renderer.wrap_in_template(html_content, title=title, standalone=standalone)


