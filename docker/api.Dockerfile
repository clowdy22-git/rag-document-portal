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

# Limits thread-pool sizes for torch/tokenizers, which otherwise default to
# spawning one thread per detected CPU core. On a shared/throttled host
# (e.g. Render's free tier) that default creates more contention and
# per-thread memory overhead than it's worth for a single small model.
ENV OMP_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

COPY requirements/api.txt ./requirements/api.txt
RUN pip install --no-cache-dir -r requirements/api.txt

COPY app ./app
COPY api ./api

# Pre-download the embedding model into the image at BUILD time, not runtime.
# Without this, the first request after every cold start (Render's free tier
# sleeps the container after 15min idle) triggers a live download from
# Hugging Face Hub — slow and unauthenticated, easily slow enough to blow
# past the request timeout and surface as a 502 Bad Gateway to the caller.
# Baking it in means the model is already on disk before uvicorn even starts.
#
# HF_HOME is set explicitly (rather than the default ~/.cache) because this
# RUN executes as root, and root's home dir isn't world-readable — appuser
# (below) wouldn't be able to reach the cache otherwise.
ENV HF_HOME=/app/hf_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

RUN useradd --create-home --uid 1000 appuser
ENV DATA_DIR=/app/data
ENV PORT=8000
# HF_HUB_OFFLINE: belt-and-suspenders — forces sentence-transformers to use
# only the model baked in above and never attempt a network call, so a
# transient Hugging Face outage can't cause a runtime failure either.
ENV HF_HUB_OFFLINE=1
RUN mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",8000)}/health', timeout=3)" || exit 1

# Shell form (not exec/JSON form) so $PORT actually expands at container start.
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
