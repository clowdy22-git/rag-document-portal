# RAG Document Portal — API service
#
# Build (from the project root, so both app/ and api/ are in context):
#   docker build -f docker/api.Dockerfile -t rag-portal-api .
# Run locally:
#   docker run -p 8000:8000 --env-file .env -e PORTAL_API_KEY=... rag-portal-api
# On Render, the PORT env var is injected automatically and the app binds to
# it instead of the hardcoded 8000 (see CMD below) — Render routes external
# traffic to whatever port the container actually listens on.

FROM python:3.12-slim

# tesseract-ocr: needed by app/ingestion/extractor.py's scanned-PDF OCR fallback.
# libgomp1: runtime dependency of faiss-cpu.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements/api.txt ./requirements/api.txt
RUN pip install --no-cache-dir -r requirements/api.txt

COPY app ./app
COPY api ./api

RUN useradd --create-home --uid 1000 appuser
ENV DATA_DIR=/app/data
ENV PORT=8000
RUN mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",8000)}/health', timeout=3)" || exit 1

# Shell form (not exec/JSON form) so $PORT actually expands at container start.
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
