"""
Deterministic document ID generation. Using the file's content hash (rather
than a random UUID) means the same file always gets the same source_id
across separate runs — which the query cache in app/cache/cache_store.py
depends on to actually produce cache hits between runs, not just within one.
"""

import hashlib
from pathlib import Path


def make_source_id(file_path: str) -> str:
    """Generate a stable ID like 'report-a1b2c3' from a file's content hash."""
    content = Path(file_path).read_bytes()
    digest = hashlib.sha256(content).hexdigest()[:8]
    return f"{Path(file_path).stem}-{digest}"