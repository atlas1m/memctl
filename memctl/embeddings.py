"""Embedding layer for memctl — local embeddings via fastembed."""

import struct
from typing import List, Optional

_model = None
_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # ~24MB, fast, good quality
EMBEDDING_DIM = 384


def _get_model():
    """Lazy-load the embedding model."""
    global _model
    if _model is None:
        try:
            from fastembed import TextEmbedding
            _model = TextEmbedding(_MODEL_NAME)
        except ImportError:
            return None
    return _model


def embed_text(text: str) -> Optional[List[float]]:
    """Generate an embedding for a text string. Returns None if fastembed unavailable."""
    model = _get_model()
    if model is None:
        return None
    try:
        embeddings = list(model.embed([text]))
        return embeddings[0].tolist()
    except Exception:
        return None


def embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    """Batch embed multiple texts."""
    model = _get_model()
    if model is None:
        return None
    try:
        embeddings = list(model.embed(texts))
        return [e.tolist() for e in embeddings]
    except Exception:
        return None


def pack_embedding(vec: List[float]) -> bytes:
    """Pack a float32 embedding to bytes for SQLite storage."""
    return struct.pack(f"{len(vec)}f", *vec)


def unpack_embedding(blob: bytes) -> List[float]:
    """Unpack bytes to float32 embedding."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
