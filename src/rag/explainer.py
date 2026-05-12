"""Deterministic prediction explainer using SHAP attributions and retrieved context."""
from __future__ import annotations

import math
import re

import numpy as np

_MODE_NAME_MAP = {
    0: "Air",
    1: "Ocean",
    2: "Truck",
    3: "Other",
}


def _ordinal(rank: int) -> str:
    if 10 <= rank % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank % 10, "th")
    return f"{rank}{suffix}"


def _estimate_dollar_impact(prediction_usd: float, shap_value: float) -> float:
    return abs(float(prediction_usd) * (math.exp(abs(float(shap_value))) - 1.0))


def _direction_word(shap_value: float) -> str:
    return "increasing" if shap_value >= 0 else "decreasing"


def _format_value(value) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "unknown"
        return f"{float(value):,.2f}"
    return str(value)


def _clean_context_text(text: str) -> str:
    cleaned = re.sub(r"^#{1,6}\s+", "", text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _summarize_context(context_chunks: list[dict]) -> str:
    snippets: list[str] = []
    for chunk in context_chunks[:2]:
        cleaned = _clean_context_text(str(chunk.get("text", "")))
        if not cleaned:
            continue
        sentence_match = re.match(r"(.{40,220}?[.!?])(?:\s|$)", cleaned)
        snippet = sentence_match.group(1).strip() if sentence_match else cleaned[:220].rstrip()
        source = str(chunk.get("source", "")).strip()
        if source:
            snippets.append(f"{snippet} ({source})")
        else:
            snippets.append(snippet)
    return " ".join(snippets[:2]).strip()


def _feature_sentence(feature_name: str, feature_value, shap_value: float, rank: int, prediction_usd: float) -> str:
    direction = _direction_word(shap_value)
    impact = _estimate_dollar_impact(prediction_usd, shap_value)
    rank_text = _ordinal(rank)

    if feature_name == "log_weight":
        try:
            weight_text = f"{math.expm1(float(feature_value)):,.0f}"
        except (TypeError, ValueError, OverflowError):
            weight_text = _format_value(feature_value)
        return (
            f"Shipment weight ({weight_text} kg) is the {rank_text} driver, "
            f"{direction} the estimate by about ${impact:,.0f}."
        )

    if feature_name == "mode_encoded":
        try:
            mode_name = _MODE_NAME_MAP.get(int(feature_value), "Other")
        except (TypeError, ValueError):
            mode_name = str(feature_value)
        return (
            f"Shipment mode ({mode_name}) is the {rank_text} driver, "
            f"{direction} the estimate by about ${impact:,.0f}."
        )

    if feature_name == "unctad_rate_usd_kg":
        return (
            f"The destination-tier indicator (${_format_value(feature_value)}/kg reference rate) "
            f"suggests {'higher' if shap_value >= 0 else 'lower'} logistics costs for this corridor, "
            f"{direction} the estimate by about ${impact:,.0f}."
        )

    if feature_name == "year":
        return (
            f"The shipment year ({_format_value(feature_value)}) reflects historical cost trends, "
            f"{direction} the estimate by about ${impact:,.0f} as the {rank_text} driver."
        )

    return (
        f"{feature_name.replace('_', ' ').capitalize()} ({_format_value(feature_value)}) is the "
        f"{rank_text} driver, {direction} the estimate by about ${impact:,.0f}."
    )


def explain_prediction(
    shap_values: np.ndarray,
    feature_names: list,
    feature_values: dict,
    prediction_usd: float,
    context_chunks: list[dict] = None,
) -> str:
    """Build a deterministic natural-language explanation from SHAP and retrieved context."""
    shap_array = np.asarray(shap_values, dtype=float).reshape(-1)
    if len(shap_array) != len(feature_names):
        raise ValueError("Length mismatch between shap_values and feature_names")

    top_indices = np.argsort(np.abs(shap_array))[::-1][:3]
    driver_names = [str(feature_names[idx]) for idx in top_indices]
    intro = (
        f"The model predicts ${prediction_usd:,.0f} for this shipment. "
        f"The main cost drivers are: {', '.join(driver_names)}."
    )

    body = [
        _feature_sentence(
            feature_name=feature_names[idx],
            feature_value=feature_values.get(feature_names[idx]),
            shap_value=float(shap_array[idx]),
            rank=rank,
            prediction_usd=float(prediction_usd),
        )
        for rank, idx in enumerate(top_indices, start=1)
    ]

    explanation = " ".join([intro, *body]).strip()
    if context_chunks:
        context_summary = _summarize_context(context_chunks)
        if context_summary:
            explanation = f"{explanation} Market context: {context_summary}"
    return explanation
