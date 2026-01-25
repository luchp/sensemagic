#!/usr/bin/env python
"""Test script to verify link rewriting in markdown"""

from pathlib import Path
from routers.markdown_renderer import MarkdownRenderer

# Create renderer instance
mr = MarkdownRenderer(
    base_path=Path('pages/rectifier'),
    route_prefix='/app_rectifier'
)

# Test cases for link rewriting
test_cases = [
    # Markdown file references
    ('[User Guide](RECTIFIER_USER_GUIDE.md)', '[User Guide](/app_rectifier/guide?standalone=true)'),
    ('[Math Details](RECTIFIER_MATH_CLEAN.md)', '[Math Details](/app_rectifier/math?standalone=true)'),
    ('[WordPress](WORDPRESS_PAGE_CONTENT.md)', '[WordPress](/app_rectifier/wordpress?standalone=true)'),

    # External links (should not change)
    ('[Google](https://google.com)', '[Google](https://google.com)'),
    ('[Local](http://localhost:8000)', '[Local](http://localhost:8000)'),

    # Anchors (should not change)
    ('[Section](#my-section)', '[Section](#my-section)'),

    # Absolute paths (should not change)
    ('[Root](/some/path)', '[Root](/some/path)'),
]

print("Testing Link Rewriting")
print("=" * 70)

all_passed = True
for input_md, expected in test_cases:
    result = mr._rewrite_links(input_md)
    passed = result == expected
    all_passed = all_passed and passed

    status = '✓' if passed else '✗'
    print(f"{status} {input_md[:40]:<40}")
    if not passed:
        print(f"  Expected: {expected}")
        print(f"  Got:      {result}")

print("=" * 70)
if all_passed:
    print("✓ All link rewriting tests passed!")
else:
    print("✗ Some tests failed")
print()

# Test with actual markdown content
print("Testing with Sample Markdown")
print("=" * 70)

sample_md = """
# Rectifier Calculator

For more details, see:
- [User Guide](RECTIFIER_USER_GUIDE.md)
- [Mathematical Details](RECTIFIER_MATH_CLEAN.md)
- [External Reference](https://example.com)

Also see [another section](#details) below.
"""

result = mr.render_markdown(sample_md, include_toc=False, rewrite_links=True)
html = result['html']

# Check if links were rewritten correctly
checks = [
    ('/app_rectifier/guide?standalone=true' in html, 'User guide link rewritten'),
    ('/app_rectifier/math?standalone=true' in html, 'Math link rewritten'),
    ('https://example.com' in html, 'External link preserved'),
    ('#details' in html, 'Anchor preserved'),
]

for passed, description in checks:
    status = '✓' if passed else '✗'
    print(f"{status} {description}")

print()
print("Sample HTML output:")
print(html[:500])

