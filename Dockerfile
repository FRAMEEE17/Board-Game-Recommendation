# syntax=docker/dockerfile:1.7
# Three stages. deps: the runtime virtualenv. assets: model files, vectors and the game map, the only
# stage with network access and umap-learn. runtime: what ships, non-root, offline.
ARG PY=python:3.12-slim
FROM ghcr.io/astral-sh/uv:0.12 AS uv

# --- deps: the virtualenv from the lock, no project code, so code edits do not reinstall packages ---
FROM ${PY} AS deps
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-default-groups --group app

# --- assets: downloads the int8 model for the target arch, builds vectors and the map ---
FROM deps AS assets
ARG TARGETARCH
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-default-groups --group dev --group app --group map
COPY board_game_reco ./board_game_reco
COPY data ./data
ENV HF_HOME=/opt/hf PYTHONPATH=/app
# buildx runs this stage on the target arch, so Embedder picks the int8 file for it. The check fails
# the build if that file does not match the platform buildx was asked for.
RUN .venv/bin/python -m board_game_reco.build --cache /app/.cache --expect-arch "${TARGETARCH:-$(uname -m)}"

# --- runtime: no umap, no numba, no network needed ---
FROM ${PY} AS runtime
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=deps   /app/.venv          /app/.venv
COPY --from=assets /opt/hf             /opt/hf
COPY --from=assets /app/.cache/vectors /app/.cache/vectors
COPY --from=assets /app/.cache/map     /app/.cache/map
COPY pyproject.toml uv.lock ./
COPY board_game_reco ./board_game_reco
COPY data ./data
COPY app ./app
COPY .streamlit ./.streamlit
COPY evals ./evals
COPY tests ./tests
COPY docker/entrypoint.sh /usr/local/bin/entrypoint
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    BGR_THREADS=2 \
    BGR_IMAGE=1 \
    BGR_RESULTS=/tmp/results \
    MPLCONFIGDIR=/tmp/matplotlib \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
# No key may reach the image. .dockerignore drops .env; this fails the build if one slips through.
# /app and /opt/hf stay root-owned, so the app user cannot write there. A recursive chmod would
# copy the whole virtualenv into a new layer (905 MB) for no gain.
RUN test ! -e /app/.env \
 && chmod 0755 /usr/local/bin/entrypoint \
 && python -c "import board_game_reco, app.state"
USER app
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')"
ENTRYPOINT ["entrypoint"]
CMD ["app"]

# --- test: runtime plus pytest and scikit-learn, for the test and perf modes ---
# Build the app image with `--target runtime`. Test-only packages stay out of it to keep it under 1 GB.
FROM runtime AS test
USER root
COPY --from=uv /uv /usr/local/bin/uv
RUN UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 \
    uv sync --frozen --no-install-project --no-default-groups --group app --group dev
USER app
