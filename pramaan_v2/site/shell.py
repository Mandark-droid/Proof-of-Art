"""Shared <head>, header and footer for every v2 page (markup ported from the design boards)."""
from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from html import escape
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "static"

ROUTE_TARGETS = {
    "home": "/", "replays": "/replays", "leaderboard": "/leaderboard", "try": "/try",
    "build": "/build", "submit": "/submit", "traditions": "/traditions", "playground": "/playground",
}
PRIMARY = (("replays", "Replays"), ("leaderboard", "Leaderboard"), ("try", "Try it"), ("build", "Build with it"))

FONT_SANS = "'Anek Latin', 'Segoe UI', system-ui, sans-serif"
FONT_MONO = "'JetBrains Mono', ui-monospace, Menlo, monospace"
MARK = ('<svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true"><path d="M6 6h.01M16 6h.01M6 16h.01M16 16h.01" '
        'stroke="#F4B63F" stroke-width="4.5" stroke-linecap="round" fill="none"></path></svg>')


V2_SPACE = "https://kshitijthakkar-pramaan-proof-of-art-v2.hf.space"


def public_base() -> str:
    """The Space origin for code snippets, from configuration (never request headers).

    ``PUBLIC_BASE_URL``, then the ``SPACE_HOST`` Hugging Face sets, then the public v2 Space.
    """
    configured = os.environ.get("PUBLIC_BASE_URL", "").strip().rstrip("/")
    if configured.startswith(("https://", "http://")):
        return configured
    space_host = os.environ.get("SPACE_HOST", "").strip()
    if space_host and re.fullmatch(r"[a-z0-9.-]+", space_host):
        return f"https://{space_host}"
    return V2_SPACE


def json_for_script(value, **dumps) -> str:
    """JSON safe inside a <script> element: every "<" becomes the escape \\u003c, so no "</script>" or "<!--" can occur."""
    return json.dumps(value, **dumps).replace("<", "\\u003c")


def esc(value) -> str:
    return escape("" if value is None else str(value), quote=True)


@lru_cache(maxsize=1)
def bundle_css() -> bytes:
    """The one stylesheet: every fragment in static/css, in file-name order."""
    return b"\n".join(path.read_bytes() for path in sorted((STATIC / "css").glob("*.css")))


@lru_cache(maxsize=1)
def bundle_js() -> bytes:
    """The one page script: every fragment in static/js, in file-name order, each self-guarded."""
    return b"\n".join(path.read_bytes() for path in sorted((STATIC / "js").glob("*.js")))


@lru_cache(maxsize=32)
def asset_url(name: str) -> str:
    """Static URL with a content hash so files can be cached forever."""
    data = {"v2.js": bundle_js, "v2.css": bundle_css}.get(name, lambda: (STATIC / name).read_bytes())()
    return f"/static/v2/{name}?v={hashlib.sha256(data).hexdigest()[:10]}"


def header(current: str | None, *, submit_label: str = "Submit a run") -> str:
    links = []
    for key, label in PRIMARY:
        here = key == current
        links.append(
            f'<a href="{ROUTE_TARGETS[key]}"{" aria-current=" + chr(34) + "page" + chr(34) if here else ""} '
            f'style="display: flex; align-items: center; min-height: 44px; padding: 0 14px; '
            f'color: {"#F4B63F" if here else "#F4EFE6"}; text-decoration: none; font-weight: {600 if here else 500}">'
            f'{label}</a>')
    return (
        '<header class="v2-header" style="display: flex; flex-wrap: wrap; align-items: center; gap: 12px 24px; '
        'padding: 16px 0; border-bottom: 1px solid #2E2621">'
        f'<a href="/" class="v2-brand" style="display: flex; align-items: center; gap: 10px; min-height: 44px; '
        f'text-decoration: none; color: #F4EFE6">{MARK}'
        '<span style="font-size: 21px; font-weight: 700; font-stretch: 112%">Pramaan</span>'
        "<span lang=\"hi\" style=\"font-family: 'Anek Devanagari', 'Nirmala UI', sans-serif; font-size: 18px; "
        'font-weight: 500; color: #B8ADA1">प्रमाण</span></a>'
        '<button type="button" class="v2-menu" aria-expanded="false" aria-controls="v2-nav" '
        'aria-label="Menu"><span aria-hidden="true"></span></button>'
        '<div id="v2-nav" class="v2-navwrap">'
        '<nav aria-label="Primary" style="display: flex; flex-wrap: wrap; gap: 4px; margin-left: auto">'
        + "".join(links) + "</nav>"
        f'<a href="{ROUTE_TARGETS["submit"]}" class="v2-submit"'
        f'{" aria-current=" + chr(34) + "page" + chr(34) if current == "submit" else ""} '
        'style="display: flex; align-items: center; min-height: 44px; padding: 0 18px; border: 1px solid #F4B63F; '
        f'border-radius: 22px; color: #F4B63F; text-decoration: none; font-weight: 600">{esc(submit_label)}</a>'
        "</div></header>"
    )


def footer(links: list[tuple[str, str]], right: str, *, mono: bool = True, margin_top: int = 88,
           border: bool = True) -> str:
    """Page footer. Boards differ: right text is mono on replay pages, sans elsewhere."""
    items = "".join(
        f'<a href="{esc(href)}" style="display: inline-flex; align-items: center; min-height: 44px; '
        f'padding-right: 16px">{esc(label)}</a>' for label, href in links)
    right_style = (f"margin-left: auto; font-family: {FONT_MONO}; font-size: 12px; color: #948980" if mono
                   else "margin-left: auto; color: #948980")
    return (
        '<footer style="display: flex; flex-wrap: wrap; gap: 4px 8px; align-items: center; '
        f'{f"margin-top: {margin_top}px; " if margin_top else ""}padding: 20px 0 56px; '
        f'{"border-top: 1px solid #2E2621; " if border else ""}font-size: 15px">'
        f'{items}<span style="{right_style}">{esc(right)}</span></footer>'
    )


def page(*, title: str, description: str, path: str, body: str, footer: str, current: str | None,
         base_url: str, og_image: str | None = "/og/home.png", head_extra: str = "") -> str:
    base = base_url.rstrip("/")
    image = f"{base}{og_image}" if og_image else None
    meta = [
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{esc(title)}</title>",
        f'<meta name="description" content="{esc(description)}">',
        f'<link rel="canonical" href="{esc(base + path)}">',
        f'<meta property="og:type" content="website">',
        f'<meta property="og:site_name" content="Pramaan">',
        f'<meta property="og:title" content="{esc(title)}">',
        f'<meta property="og:description" content="{esc(description)}">',
        f'<meta property="og:url" content="{esc(base + path)}">',
        '<meta name="twitter:card" content="summary_large_image">',
        f'<meta name="twitter:title" content="{esc(title)}">',
        f'<meta name="twitter:description" content="{esc(description)}">',
        '<meta name="theme-color" content="#15110F">',
        f'<link rel="icon" href="{asset_url("favicon.svg")}" type="image/svg+xml">',
        f'<link rel="preload" href="{asset_url("fonts/anek-latin-latin-wdth-normal.woff2")}" as="font" '
        'type="font/woff2" crossorigin>',
        f'<link rel="stylesheet" href="{asset_url("v2.css")}">',
    ]
    if image:
        meta += [f'<meta property="og:image" content="{esc(image)}">',
                 '<meta property="og:image:width" content="1200">', '<meta property="og:image:height" content="630">',
                 f'<meta name="twitter:image" content="{esc(image)}">']
    return (
        "<!doctype html>\n<html lang=\"en\"><head>" + "".join(meta) + head_extra + "</head><body>"
        f'<div class="v2-root" style="background: #15110F; color: #F4EFE6; font-family: {FONT_SANS}; '
        'font-size: 17px; line-height: 1.5">'
        '<div class="v2-wrap" style="max-width: 1296px; margin: 0 auto; padding: 0 24px; box-sizing: border-box">'
        f"{header(current)}<main>{body}</main>{footer}</div></div>"
        f'<script src="{asset_url("v2.js")}" defer></script></body></html>'
    )
