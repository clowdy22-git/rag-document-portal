"""
Answer generation. Defaults to Groq's free tier (fast, hosted, needs an API key
from console.groq.com — no payment required). Falls back to a local Ollama
model if no Groq key is set, so the pipeline works fully offline too.

Dependencies:
    pip install groq --break-system-packages   # for Groq
    # For Ollama: install from ollama.com, then `ollama pull llama3.1`
"""

import os
from app.retrieval.vector_store import Chunk

GROQ_MODEL = "llama-3.1-8b-instant"
OLLAMA_MODEL = "llama3.1"

SYSTEM_PROMPT = (
    "You are a document assistant. Answer the user's question using ONLY the "
    "provided context. If the context doesn't contain the answer, say so clearly "
    "instead of guessing. Always cite the source page number(s) you used."
)


def build_context(chunks_with_scores: list[tuple[Chunk, float]]) -> str:
    """Format retrieved chunks into a context block the LLM can cite from."""
    parts = []
    for chunk, _score in chunks_with_scores:
        parts.append(f"[Source: {chunk.source_id}, page {chunk.page_number}]\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, chunks_with_scores: list[tuple[Chunk, float]]) -> str:
    """Generate an answer grounded in retrieved chunks (single-turn, no history)."""
    context = build_context(chunks_with_scores)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"
    return _call_llm(SYSTEM_PROMPT, user_message)


def generate_answer_with_history(
    query: str,
    chunks_with_scores: list[tuple[Chunk, float]],
    history_text: str,
) -> str:
    """Generate an answer grounded in retrieved chunks, aware of prior chat turns
    so the response can reference earlier parts of the conversation naturally."""
    context = build_context(chunks_with_scores)
    history_block = f"Conversation so far:\n{history_text}\n\n" if history_text else ""
    user_message = f"{history_block}Context:\n{context}\n\nQuestion: {query}"
    return _call_llm(SYSTEM_PROMPT, user_message)


CONDENSE_SYSTEM_PROMPT = (
    "Rewrite the user's latest question as a standalone question that makes "
    "sense without the conversation history, by resolving pronouns and implicit "
    "references (e.g. 'what about page 2?' becomes a fully specified question). "
    "If the question is already standalone, return it unchanged. "
    "Respond with ONLY the rewritten question, nothing else."
)


def condense_question(query: str, history_text: str) -> str:
    """Rewrite a follow-up question into a standalone one, using chat history.
    This standalone version is what gets embedded and searched — the raw
    follow-up ('what about that?') would retrieve poorly on its own."""
    if not history_text:
        return query  # first turn — nothing to condense against

    user_message = f"Conversation so far:\n{history_text}\n\nLatest question: {query}"
    return _call_llm(CONDENSE_SYSTEM_PROMPT, user_message).strip()

COMPARISON_SYSTEM_PROMPT = (
    "You are a document comparison assistant. You will be given content from "
    "MULTIPLE documents on the same topic, grouped by source. Compare them: "
    "call out concrete similarities and differences (numbers, claims, dates, "
    "conclusions). If a document doesn't address the topic at all, say so "
    "explicitly rather than guessing. Structure your answer with one short "
    "section per document, then a final summary of key differences. Always "
    "cite the source ID and page number for each claim."
)


def build_comparison_context(grouped_chunks: dict[str, list[tuple[Chunk, float]]]) -> str:
    """Format per-document chunk groups into a labeled context block for comparison."""
    sections = []
    for source_id, chunks_with_scores in grouped_chunks.items():
        if not chunks_with_scores:
            sections.append(f"=== Document: {source_id} ===\n(No relevant content found for this topic.)")
            continue
        body = "\n\n".join(
            f"[page {chunk.page_number}]\n{chunk.text}" for chunk, _score in chunks_with_scores
        )
        sections.append(f"=== Document: {source_id} ===\n{body}")
    return "\n\n".join(sections)


def generate_comparison(topic: str, grouped_chunks: dict[str, list[tuple[Chunk, float]]]) -> str:
    """Generate a structured comparison across multiple documents on a given topic."""
    context = build_comparison_context(grouped_chunks)
    user_message = f"Topic to compare: {topic}\n\n{context}"
    return _call_llm(COMPARISON_SYSTEM_PROMPT, user_message)


def _call_llm(system_prompt: str, user_message: str) -> str:
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        return _call_groq(system_prompt, user_message, groq_key)
    return _call_ollama(system_prompt, user_message)


def _call_groq(system_prompt: str, user_message: str, api_key: str) -> str:
    from groq import Groq

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def _call_ollama(system_prompt: str, user_message: str) -> str:
    import requests

    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]