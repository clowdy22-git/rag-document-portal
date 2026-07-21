"""
Phase 5/6 — FastAPI backend for the RAG Document Portal.

Wraps the existing app/ pipeline (ingestion, retrieval, generation, chat,
compare, cache) behind HTTP endpoints:

    GET  /health              — liveness check + basic stats (no auth — needed
                                 for ECS/ALB health checks, which don't send API keys)
    POST /documents/upload    — upload a PDF/DOCX, ingest it into the shared index
    GET  /documents           — list all currently indexed documents
    POST /chat                — ask a question (multi-turn, session-based)
    POST /compare              — compare 2+ documents on a topic

All routes except /health require an `X-API-Key` header matching the
PORTAL_API_KEY environment variable (see api/auth.py), and are rate-limited
per API key (see the @limiter.limit(...) decorators below) so a single leaked
key can't run up the Groq bill or hammer the server.

Run locally:
    uvicorn api.main:app --reload --port 8000

Dependencies (in addition to the app/ pipeline's own):
    pip install fastapi uvicorn python-multipart slowapi --break-system-packages
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # must run before importing app.generation.generator, which reads GROQ_API_KEY at call time

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.ingestion.extractor import extract
from app.ingestion.chunker import chunk_pages
from app.ingestion.utils import make_source_id
from app.generation.generator import generate_answer_with_history, condense_question
from app.compare.comparator import compare_documents

from api.state import state, UPLOAD_DIR
from api.auth import verify_api_key
from api.schemas import (
    UploadResponse,
    DocumentInfo,
    ChatRequest,
    ChatResponse,
    CompareRequest,
    CompareResponse,
    HealthResponse,
)

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25MB — generous for text-heavy docs, blocks accidental huge files


def rate_limit_key(request: Request) -> str:
    """Rate-limit per API key rather than per IP. Behind an ALB, many users
    share the load balancer's IP, so IP-based limiting would either be too
    strict (punishing everyone for one bad actor) or too loose (an ALB IP
    might get excluded from limits entirely). The API key is what actually
    identifies a caller here."""
    key = request.headers.get("X-API-Key")
    return key if key else get_remote_address(request)  # unauthenticated requests still get limited, by IP


limiter = Limiter(key_func=rate_limit_key)

app = FastAPI(title="RAG Document Portal API", version="1.0")
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}. Please slow down and try again shortly."},
    )


# Comma-separated list of allowed origins, e.g. "https://portal.example.com,http://localhost:8501".
# Defaults to local Streamlit only — set ALLOWED_ORIGINS explicitly in every
# other environment (especially any real deployment) rather than relying on this default.
_origins_env = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8501")
ALLOWED_ORIGINS = [o.strip() for o in _origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    # Intentionally unauthenticated and unrate-limited: ECS/ALB health checks
    # hit this frequently without credentials. Keep this handler free of
    # anything sensitive.
    return HealthResponse(status="ok", documents_indexed=len(state.documents))


@app.post("/documents/upload", response_model=UploadResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
def upload_document(request: Request, file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix or '(none)'}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    contents = file.file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit.",
        )

    # Persist to disk before hashing — make_source_id derives the ID from
    # file content, and the extractors need a real path to read from too.
    safe_name = Path(file.filename or "upload").name  # strip any path components from the client
    dest_path = UPLOAD_DIR / safe_name
    dest_path.write_bytes(contents)

    source_id = make_source_id(str(dest_path))

    # Same content already indexed (by this filename or another) — skip re-ingesting.
    if state.has_document(source_id):
        info = state.documents[source_id]
        return UploadResponse(document=DocumentInfo(**info), already_indexed=True)

    try:
        pages = extract(str(dest_path), source_id=source_id)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to extract document: {e}")

    chunks = chunk_pages(pages)
    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No extractable text found in this document (empty, corrupted, or unreadable scan).",
        )

    with state.lock:
        state.store.add_chunks(chunks)
        state.register_document(source_id, file.filename or safe_name, len(pages), len(chunks))

    return UploadResponse(
        document=DocumentInfo(**state.documents[source_id]),
        already_indexed=False,
    )


@app.get("/documents", response_model=list[DocumentInfo], dependencies=[Depends(verify_api_key)])
@limiter.limit("60/minute")
def list_documents(request: Request):
    return [DocumentInfo(**d) for d in state.documents.values()]


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("20/minute")
def chat(request: Request, req: ChatRequest):
    unknown = [d for d in req.document_ids if not state.has_document(d)]
    if unknown:
        raise HTTPException(status_code=404, detail=f"Unknown document id(s): {', '.join(unknown)}")

    session = state.sessions.get_or_create(req.session_id, document_ids=req.document_ids)
    history_text = session.history_as_text()

    try:
        standalone_query = condense_question(req.question, history_text)
        with state.lock:
            results = state.store.search(standalone_query, top_k=5)

        if not results:
            answer = "No relevant content found in the indexed documents."
        else:
            answer = generate_answer_with_history(req.question, results, history_text)
    except Exception as e:
        # Covers Groq/Ollama being unreachable, rate-limited, or misconfigured —
        # surfaces as a clean 502 instead of a bare 500 with a stack trace.
        raise HTTPException(status_code=502, detail=f"Generation failed: {e}")

    session.add_user_message(req.question)
    session.add_assistant_message(answer)

    return ChatResponse(session_id=session.session_id, answer=answer)


@app.post("/compare", response_model=CompareResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
def compare(request: Request, req: CompareRequest):
    unknown = [d for d in req.document_ids if not state.has_document(d)]
    if unknown:
        raise HTTPException(status_code=404, detail=f"Unknown document id(s): {', '.join(unknown)}")

    try:
        result = compare_documents(
            state.store, req.document_ids, req.topic, cache=state.cache, store_lock=state.lock
        )
    except ValueError as e:
        # e.g. fewer than 2 valid document_ids reaching compare_documents
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Comparison failed: {e}")

    return CompareResponse(result=result)
