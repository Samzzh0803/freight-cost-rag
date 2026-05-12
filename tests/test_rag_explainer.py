"""Tests for src.rag.explainer."""
import numpy as np

from src.rag.explainer import explain_prediction


def _sample_inputs():
    feature_names = ["log_weight", "mode_encoded", "unctad_rate_usd_kg", "year", "route_label"]
    shap_values = np.array([0.42, -0.11, 0.25, 0.05, -0.02])
    feature_values = {
        "log_weight": np.log1p(1200.0),
        "mode_encoded": 0,
        "unctad_rate_usd_kg": 4.75,
        "year": 2014,
        "route_label": 19,
    }
    return shap_values, feature_names, feature_values


def test_explain_prediction_contains_dollar_amount():
    shap_values, feature_names, feature_values = _sample_inputs()
    text = explain_prediction(shap_values, feature_names, feature_values, prediction_usd=18500)
    assert "$18,500" in text


def test_explain_unctad_framing():
    shap_values, feature_names, feature_values = _sample_inputs()
    text = explain_prediction(shap_values, feature_names, feature_values, prediction_usd=18500)
    lowered = text.lower()
    assert "destination-tier" in lowered
    assert "tariff" not in lowered
    assert "freight rate forecast" not in lowered


def test_explain_top3_features_mentioned():
    shap_values, feature_names, feature_values = _sample_inputs()
    text = explain_prediction(shap_values, feature_names, feature_values, prediction_usd=18500)
    assert "log_weight" in text
    assert "unctad_rate_usd_kg" in text
    assert "mode_encoded" in text


def test_explain_with_context_chunks():
    shap_values, feature_names, feature_values = _sample_inputs()
    context = [{"text": "# Air Freight\nAir freight markets in 2026 remain tight on key Africa corridors.", "source": "IATA Q1 2026"}]
    text = explain_prediction(shap_values, feature_names, feature_values, prediction_usd=18500, context_chunks=context)
    assert "Market context:" in text
    assert "IATA Q1 2026" in text


def test_explain_without_context_chunks():
    shap_values, feature_names, feature_values = _sample_inputs()
    text = explain_prediction(shap_values, feature_names, feature_values, prediction_usd=18500, context_chunks=None)
    assert "Market context:" not in text
