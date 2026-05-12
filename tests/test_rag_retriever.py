"""Tests for src.rag.retriever."""
import pytest

from src.rag.retriever import retrieve


@pytest.fixture(scope="module", autouse=True)
def _require_sentence_transformers():
    pytest.importorskip("sentence_transformers")


def test_retrieve_returns_list():
    results = retrieve("What does DDP mean?")
    assert isinstance(results, list)


def test_retrieve_top_k_respected():
    results = retrieve("What does DDP mean?", top_k=2)
    assert len(results) <= 2


def test_retrieve_threshold_filters():
    results = retrieve("asdfghjkl random noise", top_k=3, threshold=0.45)
    assert results == []


def test_retrieve_score_ordering():
    results = retrieve("What are current ocean freight rates?", top_k=3)
    scores = [row["score"] for row in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_source_metadata_present():
    results = retrieve("Which HS chapter covers pharmaceutical products?", top_k=3)
    assert results, "Expected at least one retrieval result"
    row = results[0]
    assert {"text", "source", "category", "score"} <= set(row.keys())
    assert isinstance(row["source"], str) and row["source"]
    assert isinstance(row["category"], str) and row["category"]
