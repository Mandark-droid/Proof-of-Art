"""The v2 404 page."""
from __future__ import annotations

from ..shell import footer, page


def render(base_url: str, path: str = "/404") -> str:
    body = (
        '<section style="padding: 88px 0 24px">'
        "<p style=\"margin: 0 0 12px; font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace; font-size: 12px; "
        'letter-spacing: 0.12em; color: #948980">NOT FOUND</p>'
        '<h1 style="margin: 0 0 20px; font-size: 58px; line-height: 0.98; font-weight: 700; font-stretch: 118%; '
        'text-wrap: balance">That page does not exist.</h1>'
        '<p style="margin: 0 0 28px; font-size: 18px; color: #B8ADA1">The link may be from an older version of '
        "the site, or the run it names is not in the published results.</p>"
        '<div style="display: flex; flex-wrap: wrap; gap: 12px">'
        '<a href="/" style="display: inline-flex; align-items: center; min-height: 48px; padding: 0 28px; '
        'background: #F4B63F; color: #15110F; border-radius: 24px; text-decoration: none; font-weight: 700">Home</a>'
        '<a href="/replays" style="display: inline-flex; align-items: center; min-height: 44px; padding: 0 22px; '
        'border: 1px solid #F4EFE6; border-radius: 22px; color: #F4EFE6; text-decoration: none; font-weight: 600">'
        "Replays</a></div></section>"
    )
    return page(title="Not found · Pramaan", description="That page does not exist.", path=path, body=body,
                footer=footer([("Home", "/"), ("Replays", "/replays")], ""), current=None, base_url=base_url)
