#!/usr/bin/env python3
"""Generate per-post HTML shells with Open Graph meta tags.

Why this exists: GitHub Pages serves static HTML, and crawlers (Twitter,
LinkedIn, etc.) don't execute JavaScript — so a SPA with hash routing has no
way to expose post-specific OG tags. This script writes one /blog/<slug>/index.html
per post, each containing the right meta block in <head>, by templating off
index.html. Re-run after adding/editing a post or after editing index.html.

Usage:
    python3 build.py

Workflow:
    1. Drop my-post.md into blogs/
    2. Add "my-post.md" to blogs/manifest.json
    3. Run: python3 build.py
    4. Commit: git add . && git commit -m "post: my post"
"""

import html
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SITE_URL = "https://atakantekparmak.github.io"
DEFAULT_IMAGE = "/twitter_avatar.jpg"
TWITTER_HANDLE = "@AtakanTekparmak"
SITE_NAME = "Atakan Tekparmak"

META_START = "<!-- POST_META_START -->"
META_END = "<!-- POST_META_END -->"


def parse_frontmatter(text: str):
    m = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?(.*)$", text, re.DOTALL)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).split("\n"):
        kv = re.match(r"^([A-Za-z_][\w-]*)\s*:\s*(.*)$", line)
        if kv:
            value = kv.group(2).strip()
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            meta[kv.group(1)] = value
    return meta, m.group(2)


def og_block(*, title, description, url, image, kind, date=None, authors=None):
    """Render the meta block that goes between POST_META markers."""
    parts = [
        f"<title>{html.escape(title)}</title>",
        f'<meta name="description" content="{html.escape(description)}">',
        "",
        f'<meta property="og:type" content="{kind}">',
        f'<meta property="og:site_name" content="{SITE_NAME}">',
        f'<meta property="og:title" content="{html.escape(title)}">',
        f'<meta property="og:description" content="{html.escape(description)}">',
        f'<meta property="og:url" content="{url}">',
        f'<meta property="og:image" content="{image}">',
        "",
        '<meta name="twitter:card" content="summary_large_image">',
        f'<meta name="twitter:site" content="{TWITTER_HANDLE}">',
        f'<meta name="twitter:creator" content="{TWITTER_HANDLE}">',
        f'<meta name="twitter:title" content="{html.escape(title)}">',
        f'<meta name="twitter:description" content="{html.escape(description)}">',
        f'<meta name="twitter:image" content="{image}">',
    ]
    if kind == "article":
        if date:
            parts.append(f'<meta property="article:published_time" content="{date}">')
        for a in authors or []:
            parts.append(f'<meta property="article:author" content="{html.escape(a)}">')
    return "\n    ".join(parts)


def replace_meta(template: str, new_meta: str) -> str:
    pattern = re.compile(
        re.escape(META_START) + r".*?" + re.escape(META_END), re.DOTALL
    )
    replacement = f"{META_START}\n    {new_meta}\n    {META_END}"
    out, count = pattern.subn(replacement, template)
    if count == 0:
        sys.exit(
            f"index.html is missing the {META_START} / {META_END} markers — "
            "add them around the OG meta block first."
        )
    return out


def absolute(url_path: str) -> str:
    if url_path.startswith(("http://", "https://")):
        return url_path
    if not url_path.startswith("/"):
        url_path = "/" + url_path
    return SITE_URL + url_path


def slug_for(filename: str, frontmatter: dict) -> str:
    return frontmatter.get("slug") or re.sub(
        r"\.(md|html?)$", "", filename, flags=re.IGNORECASE
    )


def build():
    template_path = ROOT / "index.html"
    template = template_path.read_text()

    manifest_path = ROOT / "blogs" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    blog_dir = ROOT / "blog"
    blog_dir.mkdir(exist_ok=True)

    written = []

    # 1) Per-post pages: /blog/<slug>/index.html
    active_slugs = set()
    for fname in manifest:
        post_path = ROOT / "blogs" / fname
        if not post_path.exists():
            print(f"!! manifest references missing file: {fname}", file=sys.stderr)
            continue
        meta, _ = parse_frontmatter(post_path.read_text())
        slug = slug_for(fname, meta)
        active_slugs.add(slug)

        title = meta.get("title", slug)
        excerpt = meta.get("excerpt", "")
        date = meta.get("date", "")
        authors_str = meta.get("authors", meta.get("author", "")) or ""
        authors = [a.strip() for a in authors_str.split(",") if a.strip()]
        image = absolute(meta.get("image") or DEFAULT_IMAGE)

        post_meta = og_block(
            title=f"{title} — {SITE_NAME}",
            description=excerpt or title,
            url=f"{SITE_URL}/blog/{slug}/",
            image=image,
            kind="article",
            date=date,
            authors=authors,
        )
        post_dir = blog_dir / slug
        post_dir.mkdir(parents=True, exist_ok=True)
        (post_dir / "index.html").write_text(replace_meta(template, post_meta))
        written.append(f"blog/{slug}/index.html")

    # 2) Blog list page: /blog/index.html
    list_meta = og_block(
        title=f"Blog — {SITE_NAME}",
        description="Notes on LLM agents, function calling, memory systems, and research.",
        url=f"{SITE_URL}/blog/",
        image=absolute(DEFAULT_IMAGE),
        kind="website",
    )
    (blog_dir / "index.html").write_text(replace_meta(template, list_meta))
    written.append("blog/index.html")

    # 3) 404 fallback (SPA): copy index.html so unknown paths still render.
    (ROOT / "404.html").write_text(template)
    written.append("404.html")

    # 4) Clean up stale post directories.
    removed = []
    for child in blog_dir.iterdir():
        if child.is_dir() and child.name not in active_slugs:
            shutil.rmtree(child)
            removed.append(f"blog/{child.name}/")

    print("wrote:")
    for w in written:
        print(f"  + {w}")
    if removed:
        print("removed (stale):")
        for r in removed:
            print(f"  - {r}")


if __name__ == "__main__":
    build()
