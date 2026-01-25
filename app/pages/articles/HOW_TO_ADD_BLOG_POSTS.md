# Blog System - Auto-Discovery Guide

## Overview

The blog system automatically discovers and serves markdown files from `app/pages/blog/` directory. Just drop a `.md` file in the folder and it's instantly available!

---

## ✅ Adding a New Blog Post (Automatic!)

### Step 1: Create Markdown File

Create a new `.md` file in `app/pages/blog/`:

```
app/pages/blog/MY_NEW_POST.md
```

### Step 2: Write Your Content

```markdown
# My Awesome New Post Title

This is the first paragraph that will be used as the description in the blog index.
Keep it concise and informative - the first 200 characters are used.

## Introduction

Your content goes here...

## Section 2

More content...

![Diagram](/static/images/blog/my-diagram.svg)

---

*December 30, 2025*
```

### Step 3: That's It!

**The post is automatically available at:**
```
https://www.sensemagic.nl/app_blog/my-new-post
```

No code changes needed! The system:
- ✅ Discovers the file automatically
- ✅ Extracts title from first `#` heading
- ✅ Extracts description from first paragraph
- ✅ Extracts date from italicized text at end
- ✅ Creates slug from filename (MY_NEW_POST → my-new-post)
- ✅ Lists it on the blog index page

---

## 📝 Markdown File Conventions

### Title (Required)
First `#` heading becomes the post title:
```markdown
# This Becomes The Title
```

### Description (Auto-extracted)
First substantial paragraph (20+ chars) becomes the description:
```markdown
# Title

This paragraph becomes the description in the blog index.
It's shown as a preview to entice readers...
```

### Date (Optional)
Add date in italics at the end:
```markdown
---

*December 30, 2025*
```

If not provided, uses current date.

### Images
Reference images in `/static/images/blog/`:
```markdown
![Architecture Diagram](/static/images/blog/architecture.svg)
```

---

## 🔄 Slug Generation

The filename is converted to a URL slug:

| Filename | URL Slug |
|----------|----------|
| `MY_NEW_POST.md` | `/app_blog/my-new-post` |
| `ARCHITECTURE_BLOG_POST.md` | `/app_blog/architecture-blog-post` |
| `python_tips_2025.md` | `/app_blog/python-tips-2025` |

Underscores → hyphens, uppercase → lowercase

---

## 📁 File Organization

```
app/
  pages/
    blog/
      ARCHITECTURE_BLOG_POST.md       ← Existing post
      MY_NEW_POST.md                  ← Your new post
      PYTHON_TIPS_2025.md             ← Another post
  static/
    images/
      blog/
        architecture-*.svg
        my-diagram.png                ← Your images
```

---

## 🎨 Markdown Features Supported

### Basic Formatting
- **Bold**, *italic*, `code`
- Headings (##, ###, ####)
- Lists (ordered and unordered)
- Links: `[text](url)`
- Images: `![alt](url)`

### Code Blocks
```python
def hello():
    print("Hello, world!")
```

### Tables
| Column 1 | Column 2 |
|----------|----------|
| Value 1  | Value 2  |

### Blockquotes
> This is a quote

### Horizontal Rules
---

### Table of Contents (Optional)
The system generates TOC automatically from headings if you want to display it.

---

## 🚀 Deployment

### Automatic Discovery
When you push a new `.md` file:

```bash
git add app/pages/blog/MY_NEW_POST.md
git commit -m "Add new blog post about X"
git push origin main
```

Then on the server:
```bash
cd /home/projects/sensemagic/app
git pull --rebase origin main
sudo supervisorctl restart fastapi
```

The new post appears immediately on:
- Blog index: `https://www.sensemagic.nl/app_blog/`
- Post URL: `https://www.sensemagic.nl/app_blog/my-new-post`

---

## 📊 Example: Complete Blog Post

**Filename:** `app/pages/blog/PYTHON_BEST_PRACTICES.md`

```markdown
# Python Best Practices for 2025

Modern Python development has evolved significantly. This guide covers the essential best practices every Python developer should follow in 2025, from type hints to async patterns.

## Type Hints

Python's type system has matured...

```python
def greet(name: str) -> str:
    return f"Hello, {name}"
```

## Async/Await

Modern Python is async-first...

```python
async def fetch_data():
    async with aiohttp.ClientSession() as session:
        ...
```

## Testing

Use pytest with fixtures...

## Conclusion

Following these practices will make your code more maintainable...

---

*December 30, 2025*
```

**Automatic Results:**
- **URL:** `/app_blog/python-best-practices`
- **Title:** "Python Best Practices for 2025"
- **Description:** "Modern Python development has evolved significantly. This guide covers the essential best practices..."
- **Date:** "December 30, 2025"

---

## 🔧 Advanced: Custom Metadata

If you need more control, you can add YAML frontmatter (future enhancement):

```markdown
---
title: Custom Title
description: Custom description
date: January 1, 2026
author: Your Name
tags: [python, tutorial]
---

# Post Content Starts Here
```

*(This is not implemented yet but could be added easily)*

---

## ✅ Benefits of Auto-Discovery

1. **Zero Configuration** - Just add `.md` file
2. **Instant Publishing** - No code changes needed
3. **Consistent URLs** - Automatic slug generation
4. **SEO Friendly** - Proper titles and descriptions
5. **Scalable** - Add 1 or 100 posts, works the same
6. **Simple** - Writers focus on content, not code

---

## 📝 Writing Tips

### Good First Paragraph
```markdown
# Post Title

This is a great first paragraph because it's concise, explains what the post is about, 
and makes people want to read more. Keep it under 200 characters for best results.
```

### Images
Place images in `/static/images/blog/` and reference them:
```markdown
![Descriptive Alt Text](/static/images/blog/diagram.svg)
```

### Code Snippets
Always specify the language for syntax highlighting:
````markdown
```python
# Your code here
```
````

### Headers
Use proper hierarchy:
```markdown
# Title (H1) - Only one per post
## Main Sections (H2)
### Subsections (H3)
#### Details (H4)
```

---

## 🎯 Quick Reference

| Task | Action |
|------|--------|
| Add new post | Drop `.md` file in `app/pages/blog/` |
| Add image | Put in `app/static/images/blog/` |
| Update post | Edit the `.md` file |
| Delete post | Remove the `.md` file |
| View post | `/app_blog/{slug}` |
| Blog index | `/app_blog/` |

---

**Status:** ✅ Fully Automatic  
**Configuration:** Zero  
**Files to Edit:** Just your markdown content  
**Discovery:** Instant on restart  

🎉 **Just write markdown and publish!**

