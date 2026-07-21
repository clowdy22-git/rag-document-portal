# RAG Portal — Backend (Phase 1)

Single-document RAG pipeline: extract → chunk → embed → store → retrieve → generate.

## Setup

```bash
pip install -r requirements.txt --break-system-packages

# For OCR support (scanned PDFs), also install the Tesseract binary:
#   Ubuntu/Debian: apt install tesseract-ocr
#   macOS:         brew install tesseract

# Generation defaults to Groq's free tier. Get a free key at console.groq.com:
export GROQ_API_KEY="your-key-here"

# No key? The pipeline falls back to a local Ollama model instead.
# Install Ollama from ollama.com, then: ollama pull llama3.1
```

## Run it

```bash
python -m app.pipeline path/to/document.pdf "What is this document about?"
```

First run will download the BGE embedding model (~130MB) from HuggingFace —
this requires internet access and only happens once (cached afterward).

## What's implemented (Phase 1)

- `app/ingestion/extractor.py` — PDF (digital + OCR fallback for scanned pages) and DOCX text extraction
- `app/ingestion/chunker.py` — overlapping text chunking with page/source metadata preserved
- `app/retrieval/embeddings.py` — free local embeddings via BGE (`sentence-transformers`)
- `app/retrieval/vector_store.py` — FAISS index with save/load persistence
- `app/generation/generator.py` — Groq (free tier) or local Ollama for grounded, cited answers
- `app/pipeline.py` — orchestrates the full flow end to end

## Verified working

Extraction, chunking, FAISS indexing, similarity search, and save/load were all
tested end to end on a real PDF. The embedding model download itself requires
network access to huggingface.co, which isn't available in this build sandbox —
run it locally and it'll pull the model automatically on first use.

## Not yet built (later phases)

- Multi-document indexing and chat session memory (Phase 2)
- Document comparison (Phase 3)
- Caching layer (Phase 4)
- FastAPI HTTP endpoints + Streamlit UI (Phase 5)
- Docker + CI/CD + AWS deployment (Phase 6)
