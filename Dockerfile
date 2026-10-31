FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PORT=8000 \
    ENABLE_WEB_INTERFACE=true \
    MAX_CONCURRENT_ENVS=4 \
    PRAMAAN_PROJECT_ROOT=/app \
    PYTHONPATH=/app

WORKDIR /app

COPY LICENSE README.md pyproject.toml uv.lock ./
COPY client.py models.py ./
COPY pramaan ./pramaan
COPY pramaan_env ./pramaan_env
COPY server ./server
COPY examples ./examples
COPY assets ./assets
COPY benchmarks ./benchmarks
COPY scripts/push_hub_datasets.py ./scripts/push_hub_datasets.py

RUN uv sync --locked --no-dev
RUN uv run --no-sync --project . python -c "from pramaan import app; assert app._community_runtime() is not None; assert app._community_model_choices()"
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app

USER 1000
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "--project", ".", "server"]
