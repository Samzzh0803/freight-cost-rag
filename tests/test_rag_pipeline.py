"""Tests for src.rag.pipeline."""
import numpy as np

from src.rag import pipeline


def test_rag_query(monkeypatch):
    monkeypatch.setattr(pipeline, "retrieve", lambda query: [{"text": "DDP means Delivered Duty Paid.", "source": "Incoterms 2020", "category": "incoterms", "score": 0.9}])
    monkeypatch.setattr(pipeline, "generate_answer", lambda query, context_chunks: {"answer": "DDP means Delivered Duty Paid.", "sources": ["Incoterms 2020"], "model_used": "fake-model"})
    out = pipeline.rag_query("What does DDP mean?")
    assert out["answer"] == "DDP means Delivered Duty Paid."
    assert len(out["context_chunks"]) == 1


def test_rag_augmented_prediction(monkeypatch):
    context = [{"text": "Ocean freight rates are softer than air on major corridors.", "source": "Market 2026", "category": "market_2026", "score": 0.77}]
    monkeypatch.setattr(pipeline, "retrieve", lambda query, year_hint=None: context)
    monkeypatch.setattr(pipeline, "available_market_context_years", lambda: (2026,))

    out = pipeline.rag_augmented_prediction(
        prediction_usd=9200,
        shap_values=np.array([0.3, -0.2, 0.15]),
        feature_names=["log_weight", "mode_encoded", "unctad_rate_usd_kg"],
        feature_values={"log_weight": np.log1p(800.0), "mode_encoded": 1, "unctad_rate_usd_kg": 2.4},
        shipment_mode="Ocean",
        destination_country="South Africa",
        scenario_year=2026,
    )
    assert out["prediction_usd"] == 9200.0
    assert out["market_context"] == context
    assert out["sources"] == ["Market 2026"]
    assert out["context_year_used"] == 2026
    assert out["explanation"]
