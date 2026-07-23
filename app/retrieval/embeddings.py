"""
Embedding generation using a free, local HuggingFace model (BGE).
No API key or paid service required — runs on CPU (slower) or GPU if available.

Dependencies:
    pip install sentence-transformers --break-system-packages
"""

import os
from sentence_transformers import SentenceTransformer
import numpy as np

# bge-small is fast and good enough for most RAG use cases.
# Swap to bge-base or bge-large if retrieval quality matters more than speed.
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# sentence-transformers defaults to batch_size=32 internally, meaning up to
# 32 chunks get embedded simultaneously in memory. On a memory-constrained
# host (e.g. Render's free 512MB tier, already carrying FastAPI + FAISS +
# PyMuPDF + the loaded model itself), that spike is large enough to get the
# whole process OOM-killed — specifically during document upload (which
# embeds many chunks at once), while /health and chat (a single short query)
# stay unaffected. Capping this explicitly trades a little speed for a much
# smaller, safer peak memory footprint. Override via env var if you deploy
# somewhere with more headroom and want faster ingestion.
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "8"))

_model_cache: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it (loading is slow)."""
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer(MODEL_NAME)
    return _model_cache


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts. Returns a (n, dim) float32 array."""
    model = get_model()
    # BGE models recommend a query instruction prefix for queries specifically,
    # but plain encoding works fine for documents.
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=EMBEDDING_BATCH_SIZE,
    )
    return np.asarray(embeddings, dtype="float32")


def embed_query(query: str) -> np.ndarray:
    """Embed a single search query. BGE recommends prefixing queries
    (not documents) with an instruction for better retrieval quality."""
    instructed = f"Represent this sentence for searching relevant passages: {query}"
    return embed_texts([instructed])[0]


if __name__ == "__main__":
    vecs = embed_texts(["hello world", "another sentence"])
    print(vecs.shape)
