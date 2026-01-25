#!/usr/bin/env python3
"""
Diagnostic script to test article discovery
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from shared.article_utils import discover_articles, ARTICLES_DIR

print("="*70)
print("Article Discovery Diagnostic")
print("="*70)
print()

print(f"ARTICLES_DIR: {ARTICLES_DIR}")
print(f"ARTICLES_DIR exists: {ARTICLES_DIR.exists()}")
print()

if ARTICLES_DIR.exists():
    print("Files in ARTICLES_DIR:")
    for file in sorted(ARTICLES_DIR.glob("*")):
        print(f"  - {file.name} {'(directory)' if file.is_dir() else ''}")
    print()

    print("Markdown files:")
    for file in sorted(ARTICLES_DIR.glob("*.md")):
        print(f"  - {file.name}")
    print()
else:
    print("ERROR: ARTICLES_DIR does not exist!")
    print()

print("Discovering articles...")
articles = discover_articles(include_private=False)
print(f"Found {len(articles)} articles:")
print()

for article in articles:
    print(f"  Title: {article['title']}")
    print(f"  Slug: {article['slug']}")
    print(f"  Description: {article['description'][:100]}...")
    print(f"  File: {article['filename']}")
    print()

print("="*70)
print(f"Total: {len(articles)} articles discovered")
print("="*70)

