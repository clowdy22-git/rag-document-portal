"""
Chunking for the RAG ingestion pipeline.
Splits extracted page text into overlapping chunks, preserving metadata
(source_id, page_number) needed for citations later.

Dependencies:
    pip install langchain-text-splitters --break-system-packages
"""

from dataclasses import dataclass, field
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.extractor import PageContent


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_id: str
    page_number: int
    chunk_index: int


DEFAULT_CHUNK_SIZE = 800       # characters, not tokens — simple and predictable
DEFAULT_CHUNK_OVERLAP = 150    # keeps context continuity across chunk boundaries


def chunk_pages(
    pages: list[PageContent],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split a list of extracted pages into overlapping text chunks.

    Each chunk keeps a reference back to its source document and page,
    so retrieval results can always be cited.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    global_index = 0

    for page in pages:
        if not page.text.strip():
            continue  # skip empty pages (e.g. failed OCR)

        pieces = splitter.split_text(page.text)
        for piece in pieces:
            chunks.append(
                Chunk(
                    chunk_id=f"{page.source_id}-p{page.page_number}-c{global_index}",
                    text=piece,
                    source_id=page.source_id,
                    page_number=page.page_number,
                    chunk_index=global_index,
                )
            )
            global_index += 1

    return chunks


if __name__ == "__main__":
    # Quick manual test with fake page content
    sample_pages = [
        PageContent(page_number=1, text="This is a long paragraph. " * 40, source_id="test-doc"),
    ]
    result = chunk_pages(sample_pages)
    print(f"Produced {len(result)} chunks")
    print(result[0])
