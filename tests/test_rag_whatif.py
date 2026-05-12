"""Tests for src.rag.whatif."""
from __future__ import annotations

import numpy as np

from src.rag import whatif


def _shipment():
    return {
        "Weight_kg": 850.0,
        "mode": "Air",
        "inco_group": "ExWorks",
        "product_group": "ARV",
        "year": 2014,
        "origin_country": "India",
        "dest_country": "Kenya",
    }


def test_parse_mode_change():
    out = whatif.parse_whatif_intent("What if we switch this shipment from air to ocean?")
    assert out["is_what_if"] is True
    assert out["rag_only"] is False
    assert out["modification_type"] == "mode_change"
    assert out["modifications"]["mode"] == "Ocean"


def test_parse_route_change():
    out = whatif.parse_whatif_intent("Ship it to Nigeria instead")
    assert out["is_what_if"] is True
    assert out["modification_type"] == "route_change"
    assert out["modifications"]["dest_country"] == "Nigeria"


def test_parse_weight_change():
    out = whatif.parse_whatif_intent("Increase the weight to 1,250 kg")
    assert out["is_what_if"] is True
    assert out["modification_type"] == "weight_change"
    assert out["modifications"]["Weight_kg"] == 1250.0


def test_parse_plain_rag_question():
    out = whatif.parse_whatif_intent("What does DDP mean?")
    assert out["is_what_if"] is False
    assert out["rag_only"] is True
    assert out["modifications"] == {}


def test_apply_modifications_updates_copy():
    base = _shipment()
    updated = whatif.apply_modifications(base, {"mode": "Ocean", "Weight_kg": 1200.0})
    assert updated["mode"] == "Ocean"
    assert updated["Weight_kg"] == 1200.0
    assert base["mode"] == "Air"


def test_whatif_chat_routes_to_rag(monkeypatch):
    monkeypatch.setattr(whatif, "rag_query", lambda message: {"answer": "DDP means Delivered Duty Paid.", "sources": ["Incoterms 2020"], "context_chunks": []})
    out = whatif.whatif_chat("What does DDP mean?", _shipment())
    assert out["response_type"] == "rag_only"
    assert out["answer"] == "DDP means Delivered Duty Paid."


def test_whatif_chat_reruns_prediction(monkeypatch):
    calls = []

    def fake_predict(shipment_context):
        calls.append(shipment_context.copy())
        if shipment_context["mode"] == "Air":
            return {
                "prediction_usd": 1000.0,
                "shap_values": np.array([0.2, -0.1]),
                "feature_names": ["log_weight", "mode_encoded"],
                "feature_values": {"log_weight": 6.5, "mode_encoded": 0},
                "shipment_context": shipment_context.copy(),
            }
        return {
            "prediction_usd": 700.0,
            "shap_values": np.array([0.1, -0.2]),
            "feature_names": ["log_weight", "mode_encoded"],
            "feature_values": {"log_weight": 6.5, "mode_encoded": 1},
            "shipment_context": shipment_context.copy(),
        }

    monkeypatch.setattr(whatif, "predict_shipment", fake_predict)
    monkeypatch.setattr(
        whatif,
        "rag_augmented_prediction",
        lambda **kwargs: {
            "prediction_usd": kwargs["prediction_usd"],
            "explanation": "Ocean mode lowers the estimate.",
            "market_context": [{"source": "Market 2026", "text": "Ocean is cheaper.", "category": "market_2026", "score": 0.8}],
            "sources": ["Market 2026"],
            "scenario_year_requested": kwargs.get("scenario_year"),
            "context_year_used": 2026,
            "year_fallback_message": None,
            "available_context_years": [2026],
        },
    )

    out = whatif.whatif_chat("What if we switch this shipment to ocean?", _shipment())
    assert out["response_type"] == "what_if"
    assert out["modified_shipment"]["mode"] == "Ocean"
    assert out["original_prediction_usd"] == 1000.0
    assert out["modified_prediction_usd"] == 700.0
    assert out["delta_usd"] == -300.0
    assert out["direction"] == "decrease"
    assert out["sources"] == ["Market 2026"]
    assert len(calls) == 2


def test_predict_shipment_real_model_smoke():
    out = whatif.predict_shipment(_shipment())
    assert out["prediction_usd"] > 0
    assert out["feature_names"] == [
        "log_weight",
        "weight_bucket",
        "mode_encoded",
        "inco_encoded",
        "product_group_enc",
        "year",
        "route_label",
        "unctad_rate_usd_kg",
    ]
    assert len(out["shap_values"]) == len(out["feature_names"])
