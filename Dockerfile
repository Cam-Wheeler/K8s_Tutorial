# 1. Use a separate build stage to keep the final image clean
FROM nvidia/cuda:13.1.0-runtime-ubuntu24.04 AS builder

# Update apt.
RUN apt-get update && rm -rf /var/lib/apt/lists/*

# Lets grab uv
COPY --from=ghcr.io/astral-sh/uv:0.9.22 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock ./

# Install dependencies into a standard location
RUN uv sync --frozen --no-install-project --no-dev

FROM nvidia/cuda:13.1.0-runtime-ubuntu24.04 AS runtime

WORKDIR /app

# Lets grab just the venv from the builder
COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

# Lets copy of the source now
COPY conf ./conf/
COPY src ./src
COPY main.py .

# CMD ["python", "main.py"] We do not really need to have this unless we are testing locally, we use K8s for the command

FROM builder AS development

WORKDIR /app

# Install some things we might need when debugging or monitoring.
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

COPY --from=builder /app/.venv /app/.venv

COPY conf ./conf/
COPY src ./src
COPY main.py .

# Again no real need for a CMD here, we will be running it in K8s