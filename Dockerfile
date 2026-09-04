# CPU serving image for the pricing/hedging API.
#
#   docker build -t dhps-api .
#   docker run -p 8000:8000 dhps-api
#
# The full-budget learners are trained at BUILD time (seeded, ~minutes of
# CPU) so the container starts warm; override the budget for smoke builds:
#   docker build --build-arg API_BUDGET=test -t dhps-api:smoke .

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra api --extra cpu --no-install-project

COPY src ./src
COPY scripts ./scripts
RUN uv sync --frozen --extra api --extra cpu

ARG API_BUDGET=full
ENV DHPS_API_BUDGET=${API_BUDGET}
RUN uv run python scripts/export_models.py --budget ${API_BUDGET}

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "dhps.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000"]
