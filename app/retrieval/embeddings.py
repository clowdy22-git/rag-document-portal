"""
Embedding generation using a free, local HuggingFace model (BGE).
No API key or paid service required — runs on CPU (slower) or GPU if available.

Dependencies:
    pip install sentence-transformers --break-system-packages
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# bge-small is fast and good enough for most RAG use cases.
# Swap to bge-base or bge-large if retrieval quality matters more than speed.
MODEL_NAME = "BAAI/bge-small-en-v1.5"

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
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype="float32")


def embed_query(query: str) -> np.ndarray:
    """Embed a single search query. BGE recommends prefixing queries
    (not documents) with an instruction for better retrieval quality."""
    instructed = f"Represent this sentence for searching relevant passages: {query}"
    return embed_texts([instructed])[0]


if __name__ == "__main__":
    vecs = embed_texts(["hello world", "another sentence"])
    print(vecs.shape)
