"""The v2 Space server: the OpenEnv environment and playground, the read-only site, and the studio.

OpenEnv's ``create_app`` provides the environment contract (``/reset``, ``/step``, ``/ws``,
``/web``...) unchanged; the v2 site replaces only its ``GET /``.
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from fastapi import Request, Response
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from openenv.core.env_server.http_server import create_app
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.gzip import GZipMiddleware

from pramaan_env.models import PramaanAction, PramaanObservation, PramaanState
from pramaan_env.server.environment import PramaanEnvironment

from pramaan_env.server.oauth_redirects import ROOT_OAUTH_TARGETS, root_oauth_redirect

from .apps.playground import PLAYGROUND_TITLE, build_playground
from .apps.studio import STUDIO_HEAD, build_studio, studio_theme
from .site import community, routes
from .site.render import notfound
from .site.shell import bundle_css, bundle_js

STATIC = Path(__file__).resolve().parent / "static"
IMMUTABLE = "public, max-age=31536000, immutable"


class CachedStatic(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = IMMUTABLE
        return response


def _openenv_app():
    os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")
    # Keep any Space-level Gradio root path from leaking into OpenEnv's own /web mount.
    saved_root = os.environ.pop("GRADIO_ROOT_PATH", None)
    try:
        return create_app(
            PramaanEnvironment, PramaanAction, PramaanObservation, env_name="pramaan_env",
            max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "4")), state_cls=PramaanState,
            gradio_builder=build_playground, custom_tab_name="Playground", custom_tab_primary=True,
            show_default_tab=False, title_override=PLAYGROUND_TITLE,
        )
    finally:
        if saved_root is not None:
            os.environ["GRADIO_ROOT_PATH"] = saved_root


app = _openenv_app()
app.router.routes = [route for route in app.router.routes
                     if not (getattr(route, "path", None) == "/" and "GET" in (getattr(route, "methods", None) or ()))]


@app.get("/static/v2/v2.css", include_in_schema=False)
def page_style() -> Response:
    return Response(bundle_css(), media_type="text/css", headers={"Cache-Control": IMMUTABLE})


@app.get("/static/v2/v2.js", include_in_schema=False)
def page_script() -> Response:
    return Response(bundle_js(), media_type="text/javascript", headers={"Cache-Control": IMMUTABLE})


app.mount("/static/v2", CachedStatic(directory=STATIC), name="static")
app.include_router(routes.router)
# Load community runs in the background from the start; no page request waits on the Hub.
app.router.add_event_handler("startup", community.start_refresh)
# Render every read-only page ahead of time so no visitor waits on a first render.
app.router.add_event_handler("startup", routes.start_warming)


@app.middleware("http")
async def short_routes(request: Request, call_next):
    """Relative redirects so HTTPS proxies never see an HTTP location."""
    path = request.url.path
    query = request.url.query
    target = {"/try": "/studio/", "/studio": "/studio/", "/submit": "/studio/#/submit",
              "/playground": "/web/", "/web": "/web/"}.get(path)
    if target is not None:
        if query and "#" not in target:
            target += "?" + query
        return RedirectResponse(target, status_code=307)
    return await call_next(request)


app.add_middleware(GZipMiddleware, minimum_size=1024, exclude_content_types=("text/event-stream",))


@app.exception_handler(StarletteHTTPException)
async def not_found(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404 and request.method == "GET":
        return routes.html(notfound.render(routes.base_url(request), request.url.path), status=404)
    from fastapi.exception_handlers import http_exception_handler

    return await http_exception_handler(request, exc)


def _oauth_forwarder(target: str):
    async def forward(request: Request):
        return root_oauth_redirect(target, request)
    return forward


for _source, _target in ROOT_OAUTH_TARGETS.items():
    app.add_api_route(_source, _oauth_forwarder(_target), methods=["GET"], include_in_schema=False)

app = gr.mount_gradio_app(app, build_studio(), path="/studio", theme=studio_theme(),
                          css=bundle_css().decode("utf-8"), head=STUDIO_HEAD)


def main() -> None:
    import uvicorn

    uvicorn.run("pramaan_v2.server:app", host=".".join(["0"] * 4), port=int(os.getenv("PORT", "8000")),
                proxy_headers=True, forwarded_allow_ips="*")


if __name__ == "__main__":
    main()
