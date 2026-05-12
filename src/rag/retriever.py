"""FAISS-backed retrieval for the curated freight/trade corpus."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_CACHED_INDEX = None
_CACHED_CHUNKS_DF: pd.DataFrame | None = None
_CACHED_EMBED_MODEL: Any | None = None
_CACHED_INDEX_DIR: Path | None = None
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_index_dir(index_dir: str | Path = "data/index") -> Path:
    path = Path(index_dir)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def load_index(index_dir: str | Path = "data/index") -> tuple[faiss.Index, pd.DataFrame, Any]:
    """Load the FAISS index, chunk metadata, and embedding model with module caching."""
    global _CACHED_INDEX, _CACHED_CHUNKS_DF, _CACHED_EMBED_MODEL, _CACHED_INDEX_DIR

    resolved_dir = _resolve_index_dir(index_dir)
    if (
        _CACHED_INDEX is not None
        and _CACHED_CHUNKS_DF is not None
        and _CACHED_EMBED_MODEL is not None
        and _CACHED_INDEX_DIR == resolved_dir
    ):
        return _CACHED_INDEX, _CACHED_CHUNKS_DF, _CACHED_EMBED_MODEL

    index_path = resolved_dir / "faiss.index"
    chunks_path = resolved_dir / "chunks.parquet"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunk metadata not found: {chunks_path}")

    from sentence_transformers import SentenceTransformer

    _CACHED_INDEX = faiss.read_index(str(index_path))
    _CACHED_CHUNKS_DF = pd.read_parquet(chunks_path)
    _CACHED_EMBED_MODEL = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    _CACHED_INDEX_DIR = resolved_dir
    return _CACHED_INDEX, _CACHED_CHUNKS_DF, _CACHED_EMBED_MODEL


def retrieve(
    query: str,
    top_k: int = 3,
    threshold: float = 0.45,
    year_hint: int | None = None,
) -> list[dict]:
    """Retrieve the top matching chunks using cosine similarity on the FAISS index."""
    if not isinstance(query, str) or not query.strip():
        return []

    top_k = max(1, int(top_k))
    index, chunks_df, embed_model = load_index()

    effective_query = query.strip()
    if year_hint is not None:
        effective_query = f"{effective_query} {int(year_hint)}"

    query_vec = embed_model.encode([effective_query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_vec)

    scores, idxs = index.search(query_vec, top_k)
    score_row = scores[0]
    idx_row = idxs[0]

    valid_pairs = [(score, idx) for score, idx in zip(score_row, idx_row) if idx >= 0]
    if not valid_pairs or all(score < threshold for score, _ in valid_pairs):
        return []

    results: list[dict] = []
    for score, idx in valid_pairs:
        row = chunks_df.iloc[int(idx)]
        results.append(
            {
                "text": str(row["text"]),
                "source": str(row["source"]),
                "category": str(row["corpus"]),
                "score": float(score),
            }
        )
    return results
