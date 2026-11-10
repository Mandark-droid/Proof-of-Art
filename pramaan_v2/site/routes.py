"""Plain FastAPI routes for the read-only pages."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import threading
import time
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from . import pagecache
from .data import site_data

router = APIRouter()
HTML_CACHE = {"Cache-Control": "public, max-age=300"}


_LOCAL_HOST = re.compile(r"(localhost|127[.]0[.]0[.]1)(:[0-9]{1,5})?")


def base_url(request: Request) -> str:
    """The public origin for canonical and link-preview URLs.

    Taken from configuration (``PUBLIC_BASE_URL``, or the ``SPACE_HOST`` Hugging Face sets),
    never from request headers, so a cached page cannot carry a forged host. Without
    configuration (local development) only a loopback Host header is accepted.
    """
    configured = os.environ.get("PUBLIC_BASE_URL", "").strip().rstrip("/")
    if configured.startswith(("https://", "http://")):
        return configured
    space_host = os.environ.get("SPACE_HOST", "").strip()
    if space_host and re.fullmatch(r"[a-z0-9.-]+", space_host):
        return f"https://{space_host}"
    host = request.headers.get("host", "")
    return f"http://{host}" if _LOCAL_HOST.fullmatch(host) else "http://localhost"


def html(content: str, status: int = 200) -> HTMLResponse:
    return HTMLResponse(content, status_code=status, headers=HTML_CACHE if status == 200 else None)


def _serve(request: Request, key: str | None, render) -> Response:
    """A cached page: rendered once, stored gzip-compressed, sent as stored bytes.

    ``key`` is built from canonical values only; ``None`` renders without caching, so
    arbitrary query strings can never fill the cache or push warmed pages out.
    """
    if key is None:
        content = render()
        if content is None:
            raise HTTPException(status_code=404)
        return html(content)
    data = pagecache.get_or_render(f"{base_url(request)}|{site_data().root}|{key}", render)
    if data is None:
        raise HTTPException(status_code=404)
    if "gzip" in request.headers.get("accept-encoding", ""):
        return Response(data, media_type="text/html; charset=utf-8",
                        headers={**HTML_CACHE, "Content-Encoding": "gzip", "Vary": "Accept-Encoding"})
    return HTMLResponse(gzip.decompress(data).decode("utf-8"), headers={**HTML_CACHE, "Vary": "Accept-Encoding"})


MAX_TRIES = 3
WARM_PAUSE_SECONDS = 0.02


def _grammar(request: Request) -> str | None:
    """A known grammar id, or None: unknown values render the unfiltered page."""
    from ..constants import GRAMMAR_ORDER

    value = request.query_params.get("grammar")
    return value if value in GRAMMAR_ORDER else None


def _query_key(path: str, query: dict) -> str:
    return path + ("?" + urlencode(sorted(query.items())) if query else "")


@router.get("/healthz", include_in_schema=False)
def healthz() -> JSONResponse:
    return JSONResponse({"ok": True, "revision": site_data().meta["revision"]})


def _try_number(raw: str | None) -> int | None:
    raw = raw or ""
    # ASCII digits only, at most two: "²" and "①" pass isdigit() but not int(), and a long string is work.
    return int(raw) if raw.isascii() and raw.isdigit() and len(raw) <= 2 else None


def _home(base: str) -> str:
    from .render.home import render_home

    return render_home(site_data(), base)


def _brief(task_id: str, base: str, path: str) -> str | None:
    from .render.replays import render_replays

    return render_replays(site_data(), task_id, base, path=path)


def _run(task_id: str, model_key: str, try_n: int | None, base: str) -> str | None:
    from .render.replay import render_run

    return render_run(site_data(), task_id, model_key, try_n, base)


def _leaderboard(base: str, query: dict) -> str:
    from .render.leaderboard import render_leaderboard

    return render_leaderboard(site_data(), base, query)


def _traditions(code: str | None, base: str, grammar: str | None) -> str | None:
    from .render.traditions import render_traditions

    return render_traditions(site_data(), code, base, grammar=grammar)


def _build(base: str) -> str:
    from .render.build import render_build

    return render_build(site_data(), base)


def _community_fingerprint() -> str:
    from . import community

    rows = community.community_snapshot().get("rows") or []
    return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


@router.get("/", include_in_schema=False)
def home_page(request: Request) -> Response:
    base = base_url(request)
    return _serve(request, "/", lambda: _home(base))


@router.get("/replays", include_in_schema=False)
def replays_default(request: Request) -> Response:
    from .render.replays import DEFAULT_TASK

    base = base_url(request)
    return _serve(request, "/replays", lambda: _brief(DEFAULT_TASK, base, "/replays"))


@router.get("/replays/{task_id}", include_in_schema=False)
def replays_page(task_id: str, request: Request) -> Response:
    base = base_url(request)
    if task_id not in set(site_data().task_ids):
        raise HTTPException(status_code=404)
    return _serve(request, f"/replays/{task_id}", lambda: _brief(task_id, base, f"/replays/{task_id}"))


@router.get("/replays/{task_id}/{model_key}", include_in_schema=False)
def run_page(task_id: str, model_key: str, request: Request) -> Response:
    data = site_data()
    if task_id not in set(data.task_ids) or model_key not in data.model_index:
        raise HTTPException(status_code=404)
    base = base_url(request)
    try_n = _try_number(request.query_params.get("try"))
    if try_n is not None and not 1 <= try_n <= MAX_TRIES:
        try_n = None  # the renderer falls back to try 1; one canonical key for every out-of-range value
    key = f"/replays/{task_id}/{model_key}?try={try_n or ''}"
    return _serve(request, key, lambda: _run(task_id, model_key, try_n, base))


@router.get("/og/home.png", include_in_schema=False)
def og_home_image() -> Response:
    from .render.og import og_home

    return Response(og_home(), media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@router.get("/og/replay/{task_id}/{model_key}.png", include_in_schema=False)
def og_replay_image(task_id: str, model_key: str) -> Response:
    from .render.og import og_replay

    data = og_replay(task_id, model_key)
    if data is None:
        raise HTTPException(status_code=404)
    return Response(data, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@router.get("/site/wall.png", include_in_schema=False)
def wall_sprite() -> Response:
    data = (site_data().root / "wall.png").read_bytes()
    return Response(data, media_type="image/png", headers={"Cache-Control": "public, max-age=31536000, immutable"})


@router.get("/leaderboard", include_in_schema=False)
def leaderboard_page(request: Request) -> Response:
    base = base_url(request)
    query = dict(request.query_params)
    # Filtered and sorted views are cheap and unbounded in number: only the plain page is cached.
    key = None if query else "/leaderboard#community=" + _community_fingerprint()
    return _serve(request, key, lambda: _leaderboard(base, query))


@router.get("/traditions", include_in_schema=False)
def traditions_index(request: Request) -> Response:
    base = base_url(request)
    grammar = _grammar(request)
    key = _query_key("/traditions", {"grammar": grammar} if grammar else {})
    return _serve(request, key, lambda: _traditions(None, base, grammar))


@router.get("/traditions/{state_code}", include_in_schema=False)
def traditions_state(state_code: str, request: Request) -> Response:
    code = state_code.upper()
    if code not in {state["code"] for state in site_data().meta["states"]}:
        raise HTTPException(status_code=404)
    base = base_url(request)
    grammar = _grammar(request)
    key = _query_key(f"/traditions/{code}", {"grammar": grammar} if grammar else {})
    return _serve(request, key, lambda: _traditions(code, base, grammar))


@router.get("/build", include_in_schema=False)
def build_page(request: Request) -> Response:
    base = base_url(request)
    return _serve(request, "/build", lambda: _build(base))


def warm_pages(base: str) -> int:
    """Render every read-only page into the cache; returns how many pages were rendered."""
    from .render.og import og_home
    from .render.replays import DEFAULT_TASK

    data = site_data()
    jobs = [("/", lambda: _home(base)), ("/leaderboard#community=" + _community_fingerprint(),
                                         lambda: _leaderboard(base, {})),
            ("/build", lambda: _build(base)), ("/traditions", lambda: _traditions(None, base, None)),
            ("/replays", lambda: _brief(DEFAULT_TASK, base, "/replays"))]
    jobs += [(f"/traditions/{s['code']}", (lambda c=s["code"]: _traditions(c, base, None)))
             for s in data.meta["states"]]
    jobs += [(f"/replays/{t}", (lambda t=t: _brief(t, base, f"/replays/{t}"))) for t in data.task_ids]
    jobs += [(f"/replays/{r['task']}/{r['model']}?try=", (lambda r=r: _run(r["task"], r["model"], None, base)))
             for r in data.wall["runs"]]
    done = 0
    for key, render in jobs:
        try:
            if pagecache.get_or_render(f"{base}|{data.root}|{key}", render) is not None:
                done += 1
        except Exception:  # noqa: BLE001 - a page that fails here still renders (and errors) on request
            continue
        # Yield between pages so visitor requests are served while the warm-up runs.
        time.sleep(WARM_PAUSE_SECONDS)
    og_home()
    return done


def configured_base() -> str:
    """The base URL pages are rendered with when no request is at hand (warm-up)."""
    from starlette.requests import Request as _Request

    return base_url(_Request({"type": "http", "headers": [], "path": "/", "query_string": b""}))


def start_warming(base: str | None = None) -> threading.Thread:
    thread = threading.Thread(target=warm_pages, args=(base or configured_base(),),
                              name="pramaan-warm-pages", daemon=True)
    thread.start()
    return thread


@router.get("/api/community.json", include_in_schema=False)
def community_runs() -> JSONResponse:
    from .community import community_snapshot

    return JSONResponse(community_snapshot(), headers={"Cache-Control": "public, max-age=120"})
