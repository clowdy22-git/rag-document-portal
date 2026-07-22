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
        self.documents: dict[str, dict] = {}  # source_id -> metadata
        self.lock = threading.Lock()  # guards all store reads/writes

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

    def has_document(self, source_id: str) -> bool:
        return source_id in self.documents

    def register_document(self, source_id: str, filename: str, num_pages: int, num_chunks: int) -> None:
        self.documents[source_id] = {
            "source_id": source_id,
            "filename": filename,
            "num_pages": num_pages,
            "num_chunks": num_chunks,
        }

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
