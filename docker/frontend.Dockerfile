# RAG Document Portal — Frontend service
#
# Build (from the project root):
#   docker build -f docker/frontend.Dockerfile -t rag-portal-frontend .
# Run locally:
#   docker run -p 8501:8501 -e API_BASE=http://api:8000 -e PORTAL_API_KEY=... rag-portal-frontend
# On Render, PORT is injected automatically (see CMD below).

FROM python:3.12-slim

WORKDIR /app

COPY requirements/frontend.txt ./requirements/frontend.txt
RUN pip install --no-cache-dir -r requirements/frontend.txt

COPY frontend ./frontend
COPY .streamlit ./.streamlit

RUN useradd --create-home --uid 1000 appuser
ENV PORT=8501
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",8501)}/_stcore/health', timeout=3)" || exit 1

# Shell form (not exec/JSON form) so $PORT actually expands at container start.
CMD streamlit run frontend/app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}
