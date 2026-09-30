FROM python:3.11-slim AS build
WORKDIR /build

# Install build dependencies
COPY pyproject.toml README.md ./
COPY vajranow ./vajranow
COPY eval ./eval
COPY adapters ./adapters
COPY tracking ./tracking
COPY nowcast_baseline ./nowcast_baseline
COPY models ./models
COPY calibration ./calibration
COPY alerts ./alerts
COPY risk ./risk
COPY cap ./cap
COPY replay ./replay
COPY api ./api
COPY config ./config

RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV VAJRANOW_CONFIG=/app/config/prototype.yaml

WORKDIR /app
COPY --from=build /install /usr/local
COPY config ./config
COPY web ./web

# Render dynamically sets $PORT environment variable
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn vajranow.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
