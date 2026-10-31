import os
from pathlib import Path

import gradio as gr
import uvicorn
from fastapi import Request
from fastapi.responses import FileResponse, RedirectResponse
from openenv.core.env_server.http_server import create_app

from pramaan.app import PRAMAAN_CSS, build_demo, create_brand_theme
from pramaan.paths import resolve_project_root
from pramaan_env.models import PramaanAction, PramaanObservation, PramaanState
from pramaan_env.server.environment import PramaanEnvironment
from pramaan_env.server.web_ui import WEB_TITLE, build_pramaan_web

# Hugging Face Spaces sets GRADIO_ROOT_PATH from README base_path. Keep that
# Studio-only prefix from leaking into OpenEnv's independent `/web` Gradio mount.
_space_gradio_root_path = os.environ.pop("GRADIO_ROOT_PATH", None)
try:
    app = create_app(
        PramaanEnvironment,
        PramaanAction,
        PramaanObservation,
        env_name="pramaan_env",
        max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "4")),
        state_cls=PramaanState,
        gradio_builder=build_pramaan_web,
        custom_tab_name="Pramaan RL",
        custom_tab_primary=True,
        show_default_tab=False,
        title_override=WEB_TITLE,
    )
finally:
    if _space_gradio_root_path is not None:
        os.environ["GRADIO_ROOT_PATH"] = _space_gradio_root_path


_favicon_path = resolve_project_root(__file__) / "assets" / "pramaan-mark.png"


@app.get("/favicon.ico", include_in_schema=False)
async def space_favicon():
    return FileResponse(_favicon_path, media_type="image/png")


@app.middleware("http")
async def normalize_public_mount_paths(request: Request, call_next):
    """Keep Space redirects relative so HTTPS reverse proxies cannot emit HTTP URLs."""
    path = request.url.path
    target = {"/": "/studio/", "/studio": "/studio/", "/web": "/web/"}.get(path)
    if target is not None:
        query = request.scope.get("query_string", b"")
        if query:
            target += "?" + query.decode("latin-1")
        return RedirectResponse(target, status_code=307)
    return await call_next(request)


app = gr.mount_gradio_app(
    app,
    build_demo(),
    path="/studio",
    theme=create_brand_theme(),
    css=PRAMAAN_CSS,
    head='<link rel="icon" href="/favicon.ico" type="image/png">',
)


def main() -> None:
    uvicorn.run(
        "server.app:app",
        host=".".join(["0"] * 4),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
