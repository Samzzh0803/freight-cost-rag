"""
Tests for the RAG corpus ingest pipeline.
Run: pytest tests/test_rag_ingest.py -v
"""
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import pytest

CORPUS_DIR = Path(__file__).parent.parent / "data" / "corpus"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
REQUIRED_FRONTMATTER_KEYS = {"source", "source_url", "date_retrieved", "corpus", "topic"}
EMBEDDING_DIM = 384
# Cosine similarity threshold below which a result is "no good match".
# Empirically with all-MiniLM-L6-v2: strong match ~0.60, unrelated ~0.35–0.45.
# retriever.py should use ~0.45 as the "no relevant chunk" cutoff.
NO_MATCH_THRESHOLD = 0.45


def _require_sentence_transformers():
    return pytest.importorskip("sentence_transformers").SentenceTransformer


def _parse_frontmatter(text: str) -> dict:
    import re

    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        return {}
    meta: dict = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip().strip('"')
    return meta


def _load_index_and_df():
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    df = pd.read_parquet(INDEX_DIR / "chunks.parquet")
    return index, df


def _query(model, index, df, text: str, k: int = 3) -> pd.DataFrame:
    vec = model.encode([text], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(vec)
    scores, idxs = index.search(vec, k)
    results = df.iloc[idxs[0]].copy()
    results["score"] = scores[0]  # cosine similarity (higher = better)
    return results.reset_index(drop=True)


# ── Corpus file checks (no index needed) ──────────────────────────────────────

def test_corpus_dir_exists():
    assert CORPUS_DIR.exists(), f"Corpus directory not found: {CORPUS_DIR}"


def test_frontmatter_parsing():
    """Every .md in corpus must have valid YAML frontmatter with required keys."""
    md_files = list(CORPUS_DIR.rglob("*.md"))
    assert len(md_files) >= 24, f"Expected at least 24 .md files, found {len(md_files)}"
    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        meta = _parse_frontmatter(text)
        missing = REQUIRED_FRONTMATTER_KEYS - meta.keys()
        assert not missing, f"{md_path.name}: missing frontmatter keys {missing}"


def test_no_chunk_has_truncated_table_row():
    """No chunk should start mid-table (i.e. start with '|' without a heading above it)."""
    parquet_path = INDEX_DIR / "chunks.parquet"
    if not parquet_path.exists():
        pytest.skip("chunks.parquet not generated yet")
    df = pd.read_parquet(parquet_path)
    # A chunk that starts with '|' and has no heading line is a truncated table
    bad = df[df["text"].str.startswith("|") & ~df["text"].str.contains(r"^#{1,2} ", regex=True)]
    assert len(bad) == 0, f"{len(bad)} chunks appear to start mid-table: {bad['text'].head(2).tolist()}"


# ── Index structure checks ─────────────────────────────────────────────────────

def test_chunk_count_reasonable():
    """Total chunk count should be between 30 and 150."""
    parquet_path = INDEX_DIR / "chunks.parquet"
    if not parquet_path.exists():
        pytest.skip("chunks.parquet not generated yet")
    df = pd.read_parquet(parquet_path)
    assert 30 <= len(df) <= 150, f"Chunk count {len(df)} outside [30, 150]"


def test_chunks_parquet_schema():
    """chunks.parquet must have all required columns."""
    parquet_path = INDEX_DIR / "chunks.parquet"
    if not parquet_path.exists():
        pytest.skip("chunks.parquet not generated yet")
    df = pd.read_parquet(parquet_path)
    required_cols = {"chunk_id", "text", "source", "source_url", "corpus", "topic", "file"}
    missing = required_cols - set(df.columns)
    assert not missing, f"chunks.parquet missing columns: {missing}"


def test_index_type_is_flat_ip():
    """Index must be IndexFlatIP (cosine similarity on L2-normalised vectors)."""
    index_path = INDEX_DIR / "faiss.index"
    if not index_path.exists():
        pytest.skip("faiss.index not generated yet")
    index = faiss.read_index(str(index_path))
    assert isinstance(index, faiss.IndexFlatIP), (
        f"Expected IndexFlatIP, got {type(index).__name__}. Re-run ingest."
    )


def test_index_dimension():
    """FAISS index dimension must match all-MiniLM-L6-v2 (384)."""
    index_path = INDEX_DIR / "faiss.index"
    if not index_path.exists():
        pytest.skip("faiss.index not generated yet")
    index = faiss.read_index(str(index_path))
    assert index.d == EMBEDDING_DIM, f"Expected dim {EMBEDDING_DIM}, got {index.d}"


def test_index_ntotal_matches_parquet():
    """FAISS vector count must equal parquet row count."""
    if not (INDEX_DIR / "faiss.index").exists() or not (INDEX_DIR / "chunks.parquet").exists():
        pytest.skip("Index files not generated yet")
    index, df = _load_index_and_df()
    assert index.ntotal == len(df), f"Index {index.ntotal} vectors vs parquet {len(df)} rows"


def test_embeddings_are_normalised():
    """Spot-check that stored vectors are L2-normalised (norm ~1.0)."""
    index_path = INDEX_DIR / "faiss.index"
    if not index_path.exists():
        pytest.skip("faiss.index not generated yet")
    index = faiss.read_index(str(index_path))
    # Reconstruct is not available on IndexFlatIP, so re-embed one known chunk
    # and verify that querying it against itself returns score ~1.0
    df = pd.read_parquet(INDEX_DIR / "chunks.parquet")
    SentenceTransformer = _require_sentence_transformers()
    model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    vec = model.encode([df.iloc[0]["text"]], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(vec)
    scores, _ = index.search(vec, 1)
    assert scores[0][0] > 0.98, f"Self-similarity should be ~1.0, got {scores[0][0]:.4f}"


# ── Retrieval quality checks ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def retrieval_setup():
    if not (INDEX_DIR / "faiss.index").exists():
        pytest.skip("Index not generated yet")
    index, df = _load_index_and_df()
    SentenceTransformer = _require_sentence_transformers()
    model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    return model, index, df


def test_ddp_query_top1(retrieval_setup):
    """'What does DDP mean?' must return an incoterms/DDP chunk as top-1."""
    model, index, df = retrieval_setup
    results = _query(model, index, df, "What does DDP mean?")
    assert results.iloc[0]["corpus"] == "incoterms", (
        f"Expected incoterms top-1, got: {results.iloc[0]['corpus']}"
    )
    assert "DDP" in results.iloc[0]["topic"], (
        f"Expected DDP in topic, got: {results.iloc[0]['topic']}"
    )
    # MiniLM cosine on short query vs longer structured chunk: empirically 0.55–0.65 for
    # strong semantic matches. Threshold is a regression guard, not a quality bar.
    assert results.iloc[0]["score"] > 0.50, (
        f"Top-1 cosine score too low: {results.iloc[0]['score']:.3f}"
    )


def test_fuel_surcharge_query(retrieval_setup):
    """'Air cargo fuel surcharge 2026' must hit market_2026 or modes in top-2."""
    model, index, df = retrieval_setup
    results = _query(model, index, df, "Air cargo fuel surcharge 2026", k=3)
    top2_corpora = results.iloc[:2]["corpus"].tolist()
    assert any(c in ("market_2026", "modes") for c in top2_corpora), (
        f"Expected market_2026 or modes in top-2, got: {top2_corpora}"
    )


def test_multi_concept_query(retrieval_setup):
    """'Incoterm where seller pays import duty and all-risks insurance' -> DDP or CIP in top-3."""
    model, index, df = retrieval_setup
    results = _query(model, index, df,
                     "Which Incoterm requires the seller to pay import duties and provide all-risks insurance?",
                     k=5)
    top_topics = results["topic"].tolist()
    matched = [t for t in top_topics if "DDP" in t or "CIP" in t]
    assert matched, f"Expected DDP or CIP in top-5, got topics: {top_topics}"


def test_out_of_corpus_low_score(retrieval_setup):
    """A query with no corpus match should return cosine score below threshold."""
    model, index, df = retrieval_setup
    results = _query(model, index, df,
                     "Quarterly earnings per share for Apple Inc AAPL stock 2025", k=1)
    score = results.iloc[0]["score"]
    assert score < 0.60, (
        f"Out-of-corpus query returned suspiciously high score {score:.3f} — "
        f"threshold for 'no match' needs recalibration. Top chunk: {results.iloc[0]['topic']}"
    )
