"""
Phase 3 — document comparison. Given 2+ indexed documents and a topic,
retrieve relevant chunks from EACH document separately and ask the LLM
to produce a structured side-by-side comparison.
"""

from contextlib import nullcontext

from app.retrieval.vector_store import VectorStore
from app.generation.generator import generate_comparison
from app.cache.cache_store import QueryCache


def compare_documents(
    store: VectorStore,
    source_ids: list[str],
    topic: str,
    top_k_per_doc: int = 3,
    cache: QueryCache | None = None,
    store_lock=None,
) -> str:
    """Compare how multiple documents address the same topic.

    Unlike a normal chat query, this deliberately searches each document
    independently so a strongly-matching document can't crowd out the
    others in the results — every document gets a fair chance to be heard.

    If a cache is provided and this exact topic was already compared across
    this exact set of documents, returns the cached result instantly.

    If store_lock is provided (e.g. when multiple callers share one
    VectorStore, as in the API), it's held only for the retrieval step —
    NOT across the LLM call, which can take several seconds and would
    otherwise block every other request against the shared store.
    """
    if len(source_ids) < 2:
        raise ValueError("Comparison needs at least 2 documents (source_ids).")

    if cache is not None:
        cached = cache.get(topic, source_ids)
        if cached is not None:
            print("(served from cache)")
            return cached

    with (store_lock or nullcontext()):
        grouped = store.search_across_documents(topic, source_ids, top_k_per_doc=top_k_per_doc)

    result = generate_comparison(topic, grouped)

    if cache is not None:
        cache.set(topic, source_ids, result)

    return result