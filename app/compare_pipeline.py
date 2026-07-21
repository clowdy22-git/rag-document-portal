"""
Phase 3 pipeline — ingest 2+ documents, then compare them on one or more topics.

Usage:
    python -m app.compare_pipeline doc1.pdf doc2.pdf
    (then type a topic to compare, e.g. "revenue growth"; type 'exit' to quit)
"""

import sys

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from app.ingestion.extractor import extract
from app.ingestion.chunker import chunk_pages
from app.retrieval.vector_store import VectorStore
from app.compare.comparator import compare_documents
from app.cache.cache_store import QueryCache
from app.ingestion.utils import make_source_id


def ingest_documents(file_paths: list[str], store: VectorStore) -> list[str]:
    source_ids = []
    for file_path in file_paths:
        source_id = make_source_id(file_path)
        pages = extract(file_path, source_id=source_id)
        chunks = chunk_pages(pages)
        store.add_chunks(chunks)
        source_ids.append(source_id)
        print(f"Ingested '{file_path}' as '{source_id}': {len(pages)} pages, {len(chunks)} chunks")
    return source_ids


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m app.compare_pipeline doc1.pdf doc2.pdf [doc3.pdf ...]")
        print("(comparison needs at least 2 documents)")
        sys.exit(1)

    file_paths = sys.argv[1:]

    store = VectorStore()
    source_ids = ingest_documents(file_paths, store)
    cache = QueryCache()

    print(f"\n{len(source_ids)} document(s) indexed: {', '.join(source_ids)}")
    print("Type a topic to compare across all documents (e.g. 'revenue growth').")
    print("Type 'exit' to quit.\n")

    while True:
        topic = input("Compare topic: ").strip()
        if topic.lower() in {"exit", "quit"}:
            break
        if not topic:
            continue

        result = compare_documents(store, source_ids, topic, cache=cache)
        print(f"\n{result}\n")