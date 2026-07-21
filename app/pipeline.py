"""
End-to-end RAG pipeline — Phase 1: single document, ask a question, get an
answer grounded in the document with page citations.

Usage:
    python -m app.pipeline path/to/document.pdf "What is this document about?"
"""

import sys

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads .env in the current working directory and loads GROQ_API_KEY etc.

from app.ingestion.extractor import extract
from app.ingestion.chunker import chunk_pages
from app.retrieval.vector_store import VectorStore
from app.generation.generator import generate_answer
from app.cache.cache_store import QueryCache
from app.ingestion.utils import make_source_id


def ingest_document(file_path: str, store: VectorStore) -> str:
    """Extract, chunk, embed, and index a single document. Returns its source_id."""
    source_id = make_source_id(file_path)
    pages = extract(file_path, source_id=source_id)
    chunks = chunk_pages(pages)
    store.add_chunks(chunks)
    print(f"Ingested '{file_path}' as '{source_id}': {len(pages)} pages, {len(chunks)} chunks")
    return source_id


def ask(store: VectorStore, question: str, source_id: str, cache: QueryCache | None = None, top_k: int = 5) -> str:
    """Retrieve relevant chunks and generate a grounded answer.
    If a cache is provided and this exact question was already asked against
    this document, returns the cached answer instantly with no retrieval or
    LLM call at all."""
    if cache is not None:
        cached = cache.get(question, [source_id])
        if cached is not None:
            print("(served from cache)")
            return cached

    results = store.search(question, top_k=top_k)
    if not results:
        return "No relevant content found in the indexed documents."

    answer = generate_answer(question, results)

    if cache is not None:
        cache.set(question, [source_id], answer)

    return answer


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python -m app.pipeline <file_path> "<question>"')
        sys.exit(1)

    file_path, question = sys.argv[1], sys.argv[2]

    store = VectorStore()
    source_id = ingest_document(file_path, store)
    cache = QueryCache()

    print(f"\nQuestion: {question}")
    answer = ask(store, question, source_id, cache=cache)
    print(f"\nAnswer:\n{answer}")