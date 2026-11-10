FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PORT=8000 \
    SYSTEM=spaces \
    ENABLE_WEB_INTERFACE=true \
    MAX_CONCURRENT_ENVS=4 \
    PRAMAAN_PROJECT_ROOT=/app \
    PYTHONPATH=/app

WORKDIR /app

COPY LICENSE README.md pyproject.toml uv.lock ./
COPY pramaan ./pramaan
COPY pramaan_env ./pramaan_env
COPY pramaan_v2 ./pramaan_v2
COPY benchmarks ./benchmarks
# Read-only v1 modules and examples the studio and community runs import lazily.
COPY examples ./examples
COPY scripts/push_hub_datasets.py ./scripts/push_hub_datasets.py

RUN uv sync --locked --no-dev
# Gradio's sign-in only exists at runtime on the Space, so the build check stops short of mounting the studio.
RUN uv run --no-sync --project . python -c "from pramaan_v2.site.data import site_data; from pramaan_v2.site.render.home import render_home; assert site_data().meta['totals']['runs'] > 0; assert render_home(site_data(), 'https://example.hf.space'); import pramaan.app as v1; assert v1._community_runtime() is not None; assert v1._community_submission_archive_runtime() is not None; assert len(v1.EXAMPLES) > 0"
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app

USER 1000
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "--project", ".", "python", "-m", "pramaan_v2.server"]
