"""Compatibility redirects for Hugging Face OAuth callbacks at the Space root."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import RedirectResponse


ROOT_OAUTH_TARGETS = {
    "/login/callback": "/studio/login/callback",
    "/login/huggingface": "/studio/login/huggingface",
    "/logout": "/studio/logout",
}


def root_oauth_redirect(target: str, request: Request) -> RedirectResponse:
    """Redirect a root OAuth route into Studio without losing OAuth state."""
    query = request.scope.get("query_string", b"")
    if query:
        target += "?" + query.decode("latin-1")
    return RedirectResponse(target, status_code=307)
