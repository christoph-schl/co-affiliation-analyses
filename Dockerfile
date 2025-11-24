FROM python:3.12-slim

# Project directory
WORKDIR /app

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# System deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      curl \
      ca-certificates \
      less \
 && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy only Poetry files first (layer caching)
COPY pyproject.toml poetry.lock /app/

# Copy package source code
COPY src /app/src

# Copy misc files
COPY README.md LICENSE mypy.ini /app/

# Create mount points
RUN mkdir -p /app/config /app/data

# Install dependencies AND your package (entrypoints!)
RUN --mount=type=cache,id=pypoetrycache,target=/root/.cache/pypoetry \
    poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi

# Default command (can be overridden)
CMD ["--help"]
