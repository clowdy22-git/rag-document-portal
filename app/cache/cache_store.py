"""
Query result cache — Cache-Augmented Generation (CAG) for repeated queries.

Skips retrieval + LLM generation entirely when the same question (against
the same set of documents) has been asked before. Persisted to a JSON file
on disk so the cache survives between runs, not just within one session.

This is deliberately simple (exact-match on a normalized key) rather than
semantic caching (e.g. "what was Q3 revenue?" vs "how much did revenue grow
in Q3?" would NOT hit the same cache entry). Semantic caching would need its
own similarity check against past queries — a reasonable next step, but
exact-match already helps a lot for dashboards/demos where the same few
questions get re-asked repeatedly.
"""

import json
import hashlib
import threading
from pathlib import Path
from datetime import datetime, timezone

DEFAULT_CACHE_PATH = "cache_store.json"


class QueryCache:
    def __init__(self, path: str = DEFAULT_CACHE_PATH):
        self.path = Path(path)
        self._lock = threading.Lock()  # guards read-modify-write of the JSON file
        self._data: dict[str, dict] = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}  # corrupted cache file — start fresh rather than crash
        return {}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    @staticmethod
    def make_key(query: str, doc_ids: list[str]) -> str:
        """Build a stable cache key from the normalized query text and the
        (order-independent) set of documents it was run against."""
        normalized_query = query.strip().lower()
        doc_key = ",".join(sorted(doc_ids))
        raw = f"{normalized_query}|{doc_key}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, query: str, doc_ids: list[str]) -> str | None:
        key = self.make_key(query, doc_ids)
        with self._lock:
            entry = self._data.get(key)
            return entry["answer"] if entry else None

    def set(self, query: str, doc_ids: list[str], answer: str) -> None:
        key = self.make_key(query, doc_ids)
        with self._lock:
            self._data[key] = {
                "query": query,
                "doc_ids": doc_ids,
                "answer": answer,
                "cached_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save()

    def clear(self) -> None:
        with self._lock:
            self._data = {}
            self._save()

    def __len__(self) -> int:
        return len(self._data)