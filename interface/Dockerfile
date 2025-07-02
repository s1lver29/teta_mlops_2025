FROM python:3.12-slim-bookworm AS base

RUN apt-get update && apt-get -y install libpq-dev gcc

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/uv


FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app
COPY uv.lock pyproject.toml /app/

RUN uv sync --locked --no-dev --no-install-project --no-editable

ADD . /app

RUN uv sync --locked --no-dev --no-editable

FROM base
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]