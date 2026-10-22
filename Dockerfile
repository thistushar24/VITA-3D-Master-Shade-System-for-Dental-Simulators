# Single-service build for Render: builds the v1 UI, then serves UI + API from one FastAPI process.
FROM node:20-slim AS ui
WORKDIR /app
COPY frontend/src ./frontend/src
COPY versions/v1/package.json versions/v1/package-lock.json ./versions/v1/
RUN cd versions/v1 && npm ci
COPY versions/v1 ./versions/v1
RUN cd versions/v1 && npm run build

FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend backend
COPY data data
COPY --from=ui /app/versions/v1/dist versions/v1/dist
ENV DENTAL_STORAGE=/tmp/dental-storage
WORKDIR /app/backend
# Render provides $PORT; default 8000 for local `docker run`
CMD ["sh", "-c", "uvicorn dental_reg.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
