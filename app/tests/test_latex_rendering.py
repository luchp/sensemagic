#!/usr/bin/env python
"""Test script to verify markdown rendering with LaTeX preservation"""

from pathlib import Path
from routers.markdown_renderer import render_markdown
import re

# Test with the actual markdown file
md_file = Path('pages/rectifier/RECTIFIER_MATH_CLEAN.md')
content = md_file.read_text(encoding='utf-8')

print("Testing RECTIFIER_MATH_CLEAN.md")
print("=" * 70)

# Count LaTeX in source
inline_src = len(re.findall(r'(?<!\$)\$(?!\$)[^\$\n]+\$(?!\$)', content))
display_src = len(re.findall(r'\$\$[^\$]+\$\$', content, re.DOTALL))
print(f"Source markdown:")
print(f"  Inline LaTeX expressions: {inline_src}")
print(f"  Display LaTeX expressions: {display_src}")
print()

# Render markdown
result = render_markdown(content)
html = result['html']

# Count LaTeX in HTML
inline_html = len(re.findall(r'(?<!\$)\$(?!\$)[^\$\n]+\$(?!\$)', html))
display_html = len(re.findall(r'\$\$.+?\$\$', html, re.DOTALL))
print(f"Rendered HTML:")
print(f"  Inline LaTeX expressions: {inline_html}")
print(f"  Display LaTeX expressions: {display_html}")
print()

# Show preservation status
if inline_src == inline_html and display_src == display_html:
    print("✓ All LaTeX expressions preserved correctly!")
else:
    print("✗ LaTeX preservation issue detected!")
    print(f"  Inline: {inline_src} → {inline_html} (lost {inline_src - inline_html})")
    print(f"  Display: {display_src} → {display_html} (lost {display_src - display_html})")

print()
print("Sample of rendered HTML (first 1000 chars):")
print("-" * 70)
print(html[:1000])
print("-" * 70)
print()

# Show specific examples
print("Specific LaTeX examples in HTML:")
inline_matches = re.findall(r'(?<!\$)\$(?!\$)[^\$\n]+\$(?!\$)', html)[:5]
for i, match in enumerate(inline_matches, 1):
    print(f"  {i}. {match}")

print()
display_matches = re.findall(r'\$\$.+?\$\$', html, re.DOTALL)[:3]
for i, match in enumerate(display_matches, 1):
    print(f"  {i}. {match[:60]}...")

