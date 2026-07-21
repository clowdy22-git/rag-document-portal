"""
Phase 2 pipeline — multiple documents in one shared index, multi-turn chat
with session memory. Follow-up questions get condensed against history
before retrieval, so "what about page 2?" resolves correctly.

Usage:
    python -m app.chat_pipeline doc1.pdf doc2.pdf
    (then type questions interactively; type 'exit' to quit)
"""

import sys

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from app.ingestion.extractor import extract
from app.ingestion.chunker import chunk_pages
from app.retrieval.vector_store import VectorStore
from app.generation.generator import generate_answer_with_history, condense_question
from app.chat.manager import SessionManager
from app.chat.session import ChatSession
from app.ingestion.utils import make_source_id


def ingest_documents(file_paths: list[str], store: VectorStore) -> list[str]:
    """Ingest multiple documents into one shared vector store. Returns their source_ids."""
    source_ids = []
    for file_path in file_paths:
        source_id = make_source_id(file_path)
        pages = extract(file_path, source_id=source_id)
        chunks = chunk_pages(pages)
        store.add_chunks(chunks)
        source_ids.append(source_id)
        print(f"Ingested '{file_path}' as '{source_id}': {len(pages)} pages, {len(chunks)} chunks")
    return source_ids


def chat_turn(store: VectorStore, session: ChatSession, user_question: str, top_k: int = 5) -> str:
    """Handle one turn of a multi-turn conversation: condense the question
    against history, retrieve across all indexed documents, generate an
    answer, and update the session's history."""
    history_text = session.history_as_text()

    standalone_query = condense_question(user_question, history_text)

    results = store.search(standalone_query, top_k=top_k)
    if not results:
        answer = "No relevant content found in the indexed documents."
    else:
        answer = generate_answer_with_history(user_question, results, history_text)

    session.add_user_message(user_question)
    session.add_assistant_message(answer)
    return answer


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m app.chat_pipeline doc1.pdf [doc2.pdf ...]")
        sys.exit(1)

    file_paths = sys.argv[1:]

    store = VectorStore()
    source_ids = ingest_documents(file_paths, store)

    manager = SessionManager()
    session = manager.create_session(document_ids=source_ids)

    print(f"\nSession '{session.session_id}' ready with {len(source_ids)} document(s).")
    print("Type your questions below. Type 'exit' to quit.\n")

    while True:
        user_question = input("You: ").strip()
        if user_question.lower() in {"exit", "quit"}:
            break
        if not user_question:
            continue

        answer = chat_turn(store, session, user_question)
        print(f"\nAssistant: {answer}\n")