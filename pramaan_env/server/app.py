import os

import gradio as gr
import uvicorn
from openenv.core.env_server.http_server import create_app

from pramaan.app import PRAMAAN_CSS, build_demo, create_brand_theme
from pramaan_env.models import PramaanAction, PramaanObservation, PramaanState
from pramaan_env.server.environment import PramaanEnvironment

app = create_app(
    PramaanEnvironment,
    PramaanAction,
    PramaanObservation,
    env_name="pramaan_env",
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "4")),
    state_cls=PramaanState,
)
app = gr.mount_gradio_app(
    app,
    build_demo(),
    path="/studio",
    theme=create_brand_theme(),
    css=PRAMAAN_CSS,
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
