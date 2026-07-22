"""
FAISS-backed vector store. Free, local, no hosted service required.
Stores chunk text + metadata alongside the FAISS index so search results
can be traced back to source_id and page_number for citations.

Dependencies:
    pip install faiss-cpu --break-system-packages
"""

import faiss
import numpy as np
import pickle
from pathlib import Path

from app.ingestion.chunker import Chunk
from app.retrieval.embeddings import embed_texts, embed_query


class VectorStore:
    def __init__(self, dim: int = 384):
        # 384 = embedding dim for bge-small-en-v1.5. Change if you swap models.
        self.index = faiss.IndexFlatIP(dim)  # inner product == cosine similarity
        # since embeddings are normalized in embeddings.py
        self.chunks: list[Chunk] = []  # parallel array: index i <-> self.chunks[i]

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vectors = embed_texts(texts)
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(
        self, query: str, top_k: int = 5, allowed_source_ids: list[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        """Return the top_k most similar chunks with their similarity scores.

        If allowed_source_ids is given, only chunks from those documents are
        eligible — used to scope a search to the documents a particular
        session/user actually selected, rather than the entire shared index.
        """
        if self.index.ntotal == 0:
            return []

        query_vec = embed_query(query).reshape(1, -1)

        if allowed_source_ids is None:
            scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                results.append((self.chunks[idx], float(score)))
            return results

        # FAISS's IndexFlatIP has no native metadata filtering, so we fetch a
        # wider candidate set from the whole index, then filter down to just
        # the allowed documents and truncate to top_k. Fetching everything is
        # fine here — this app's collections are small (single-user document
        # uploads), not a large-scale index where over-fetching would matter.
        allowed = set(allowed_source_ids)
        scores, indices = self.index.search(query_vec, self.index.ntotal)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            if chunk.source_id in allowed:
                results.append((chunk, float(score)))
                if len(results) >= top_k:
                    break
        return results

        
    def search_across_documents(
        self, query: str, source_ids: list[str], top_k_per_doc: int = 3
    ) -> dict[str, list[tuple[Chunk, float]]]:
        """Retrieve the top_k_per_doc most relevant chunks from EACH of the
        given documents, for the same query. Used for comparison — a plain
        top-k search would likely return all chunks from whichever single
        document scores highest, starving the others."""
        if self.index.ntotal == 0:
            return {sid: [] for sid in source_ids}

        all_results = self.search(query, top_k=self.index.ntotal)

        grouped: dict[str, list[tuple[Chunk, float]]] = {sid: [] for sid in source_ids}
        for chunk, score in all_results:
            bucket = grouped.get(chunk.source_id)
            if bucket is not None and len(bucket) < top_k_per_doc:
                bucket.append((chunk, score))

        return grouped


    def save(self, dir_path: str) -> None:
        """Persist the index and chunk metadata to disk."""
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, dir_path: str) -> "VectorStore":
        path = Path(dir_path)
        index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        store = cls(dim=index.d)
        store.index = index
        store.chunks = chunks
        return store


if __name__ == "__main__":
    from app.ingestion.chunker import Chunk

    store = VectorStore()
    store.add_chunks([
        Chunk(chunk_id="1", text="The cat sat on the mat.", source_id="doc1", page_number=1, chunk_index=0),
        Chunk(chunk_id="2", text="Quarterly revenue grew by 12 percent.", source_id="doc1", page_number=2, chunk_index=1),
    ])
    results = store.search("How did revenue perform?", top_k=2)
    for chunk, score in results:
        print(f"{score:.3f} | {chunk.text}")
