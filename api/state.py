"""
Shared application state for the FastAPI backend.

One VectorStore, one SessionManager, one QueryCache, and one document
registry live here as a single process-wide instance. Because FastAPI can
serve concurrent requests on the same process, all reads/writes to the
FAISS-backed VectorStore go through `state.lock` — FAISS's IndexFlatIP is
not guaranteed safe for a concurrent add() during a search().

Note: this is single-process, in-memory state (same limitation the chat
sessions already had). Restarting the API loses all indexed documents,
sessions, and the document registry (the on-disk query cache survives).
Phase 6 can swap this for persistent storage without changing the routes.
"""

import os
import threading
from pathlib import Path

from app.retrieval.vector_store import VectorStore
from app.chat.manager import SessionManager
from app.cache.cache_store import QueryCache

# All persistent-ish data lives under one directory so a single volume mount
# (Docker -v, or an ECS/EFS mount) covers everything. Defaults to the current
# directory for local dev, matching the original Phase 5 behavior.
DATA_DIR = Path(os.environ.get("DATA_DIR", "."))
DATA_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = DATA_DIR / "uploaded_documents"
UPLOAD_DIR.mkdir(exist_ok=True)

CACHE_PATH = str(DATA_DIR / "cache_store.json")


class AppState:
    def __init__(self):
        self.store = VectorStore()
        self.sessions = SessionManager()
        self.cache = QueryCache(path=CACHE_PATH)
        self.documents: dict[str, dict] = {}  # source_id -> metadata (global registry, all clients)
        self.lock = threading.Lock()  # guards all store reads/writes

        # Per-client document visibility. Keyed by an opaque client_id sent
        # in the X-Client-Id header (one generated per browser/device by the
        # frontend — see frontend/app.py). This is deliberately separate from
        # `documents` above: the underlying content is still deduplicated
        # globally by content hash (two clients uploading the identical file
        # don't re-embed it twice), but *visibility* of what's been uploaded
        # is scoped per client — otherwise every device sharing the one
        # PORTAL_API_KEY sees every other device's document list, which is
        # exactly the cross-device leak this was built to close.
        self.client_documents: dict[str, set[str]] = {}  # client_id -> {source_id, ...}

        # The embedding model (BGE, via sentence-transformers) takes tens of
        # seconds to load its weights on constrained CPU (e.g. Render's free
        # tier). Loading it lazily on first use — inside a request — blocks
        # that whole process long enough to starve the health check, which
        # gets the instance killed as "unhealthy" mid-request. Instead we
        # kick off loading in a background thread at startup (see
        # api/main.py's startup event) and let routes check `model_ready`
        # to fail fast with a clear message instead of hanging into a timeout.
        self.model_ready = False
        self.model_load_error: str | None = None

        # Background upload processing. Large files can take longer to
        # extract+chunk+embed than a hosting platform's own proxy is willing
        # to hold a connection open for (confirmed in practice: a 20MB PDF
        # reliably got its connection dropped mid-request on Render's free
        # tier, well before our own code finished or failed). Instead of one
        # long blocking request, upload now returns immediately with a job
        # id, does the real work in a background thread, and the frontend
        # polls /documents/status/{source_id} until it's done — no single
        # HTTP request stays open long enough to hit any proxy timeout.
        self.upload_jobs: dict[str, dict] = {}  # source_id -> {"status", "error"}
        self.jobs_lock = threading.Lock()  # guards upload_jobs and client_documents

    def has_document(self, source_id: str) -> bool:
        return source_id in self.documents

    def register_document(self, source_id: str, filename: str, num_pages: int, num_chunks: int) -> None:
        self.documents[source_id] = {
            "source_id": source_id,
            "filename": filename,
            "num_pages": num_pages,
            "num_chunks": num_chunks,
        }

    def grant_access(self, client_id: str, source_id: str) -> None:
        """Give a client visibility into a document — called on every
        upload, including when the content was already indexed by someone
        else (dedup case), since this client still needs it in their own list."""
        with self.jobs_lock:
            self.client_documents.setdefault(client_id, set()).add(source_id)

    def client_owns(self, client_id: str, source_id: str) -> bool:
        return source_id in self.client_documents.get(client_id, set())

    def documents_for_client(self, client_id: str) -> list[dict]:
        owned = self.client_documents.get(client_id, set())
        return [self.documents[sid] for sid in owned if sid in self.documents]

    def start_job(self, source_id: str) -> None:
        with self.jobs_lock:
            self.upload_jobs[source_id] = {"status": "processing", "error": None}

    def finish_job(self, source_id: str) -> None:
        with self.jobs_lock:
            self.upload_jobs[source_id] = {"status": "done", "error": None}

    def fail_job(self, source_id: str, error: str) -> None:
        with self.jobs_lock:
            self.upload_jobs[source_id] = {"status": "error", "error": error}

    def job_status(self, source_id: str) -> dict | None:
        with self.jobs_lock:
            return self.upload_jobs.get(source_id)

    def preload_model(self) -> None:
        """Load the embedding model once, synchronously — meant to be called
        from a background thread at startup, not on the event loop."""
        try:
            from app.retrieval.embeddings import get_model

            get_model()
            self.model_ready = True
        except Exception as e:
            self.model_load_error = str(e)


# Single shared instance used by all routes.
state = AppState()
