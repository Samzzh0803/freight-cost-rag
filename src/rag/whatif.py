"""Week 9 what-if chat orchestration for shipment modifications and RAG routing."""
from __future__ import annotations

import copy
import json
import math
import os
import pickle
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.load_unctad import load_lookup
from src.data.load_usaid import load_clean, time_split
from src.features.features import (
    build_features,
    build_product_group_map,
    build_route_map,
    feature_names,
    get_weight_bin_edges,
)
from src.rag.generator import call_llm
from src.rag.pipeline import rag_augmented_prediction, rag_query

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_PATH = _REPO_ROOT / "models" / "usaid_model.pkl"
_MODE_ALIASES = {
    "air": "Air",
    "air freight": "Air",
    "air charter": "Air",
    "ocean": "Ocean",
    "sea": "Ocean",
    "truck": "Truck",
    "road": "Truck",
    "ground": "Truck",
    "other": "Other",
}
_STOPWORDS = {
    "the", "a", "an", "to", "for", "instead", "please", "shipment", "route", "mode",
    "change", "switch", "move", "send", "deliver", "weight", "kg", "kgs", "kilogram",
    "kilograms", "from", "via", "use", "set", "make", "it", "this", "that",
}


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_mode_phrase(value: str) -> bool:
    lowered = value.lower()
    return any(re.fullmatch(rf"{re.escape(alias)}", lowered) for alias in _MODE_ALIASES)


def _normalize_mode(value: str | None) -> str | None:
    if value is None:
        return None
    return _MODE_ALIASES.get(_normalize_spaces(value).lower())


def _normalize_country_token(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"[^A-Za-z,\s]", " ", value)
    cleaned = _normalize_spaces(cleaned)
    if not cleaned:
        return None
    words = [w for w in cleaned.split(" ") if w.lower() not in _STOPWORDS]
    if not words:
        return None
    normalized = " ".join(word.capitalize() if word.islower() else word for word in words)
    if _looks_like_mode_phrase(normalized):
        return None
    return normalized


def _extract_mode_change(message: str) -> str | None:
    lowered = message.lower()
    for alias, canonical in sorted(_MODE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return canonical
    return None


def _extract_weight_change(message: str) -> float | None:
    match = re.search(r"(\d[\d,]*\.?\d*)\s*(kg|kgs|kilogram|kilograms)\b", message.lower())
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def _extract_country_after_keyword(message: str, keyword: str) -> str | None:
    pattern = rf"\b{keyword}\b\s+([A-Za-z][A-Za-z,\s\-]{1,60})"
    match = re.search(pattern, message, flags=re.IGNORECASE)
    if not match:
        return None
    fragment = match.group(1)
    stop_pattern = r"\b(?:via|using|with|and|but|instead|please|switch|change)\b"
    if keyword.lower() == "from":
        stop_pattern = r"\b(?:to|via|using|with|and|but|instead|please|switch|change)\b"
    fragment = re.split(stop_pattern, fragment, maxsplit=1, flags=re.IGNORECASE)[0]
    return _normalize_country_token(fragment)


def _extract_destination_change(message: str) -> str | None:
    match = re.search(
        r"\b(?:destination|dest|to)\b\s+([A-Za-z][A-Za-z,\s\-]{1,60}?)(?:\b(?:instead|via|using|with|and|but|please|switch|change)\b|$)",
        message,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return _normalize_country_token(match.group(1))


def _extract_origin_change(message: str) -> str | None:
    match = re.search(
        r"\bfrom\b\s+([A-Za-z][A-Za-z,\s\-]{1,60}?)(?:\b(?:to|via|using|with|and|but|please|switch|change)\b|$)",
        message,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return _normalize_country_token(match.group(1))


def parse_whatif_intent(message: str) -> dict:
    """Rule-based parser for Week 9 routing and simple shipment modifications."""
    text = _normalize_spaces(message)
    lowered = text.lower()

    mode_value = _extract_mode_change(text)
    weight_value = _extract_weight_change(text)
    origin_value = _extract_origin_change(text)
    dest_value = _extract_destination_change(text)

    route_change_hint = any(phrase in lowered for phrase in ["instead", "reroute", "route to", "destination", "ship to", "send to", "from "])
    mode_change_hint = any(phrase in lowered for phrase in ["switch", "change mode", "via ", "use ", "move to", "air", "ocean", "sea", "truck", "road", "ground"])
    weight_change_hint = any(phrase in lowered for phrase in ["weight", "heavier", "lighter", "kg", "kilogram"])

    modifications: dict[str, Any] = {}
    modification_type = "other"

    if mode_value is not None:
        modifications["mode"] = mode_value
        modification_type = "mode_change"

    if origin_value is not None or dest_value is not None:
        if origin_value is not None:
            modifications["origin_country"] = origin_value
        if dest_value is not None:
            modifications["dest_country"] = dest_value
        modification_type = "route_change"

    if weight_value is not None:
        modifications["Weight_kg"] = weight_value
        if modification_type == "other":
            modification_type = "weight_change"

    trigger_words = ["what if", "switch", "change", "instead", "reroute", "increase", "decrease", "reduce", "raise", "set", "make"]
    is_what_if = bool(modifications) or (
        any([route_change_hint, mode_change_hint, weight_change_hint])
        and any(token in lowered for token in trigger_words)
    )
    rag_only = not bool(modifications) and not is_what_if

    return {
        "message": text,
        "modification_type": modification_type,
        "modifications": modifications,
        "is_what_if": bool(is_what_if and modifications),
        "rag_only": bool(rag_only),
    }


def apply_modifications(shipment_context: dict, modifications: dict) -> dict:
    updated = copy.deepcopy(shipment_context)
    updated.update(modifications)
    return updated


@lru_cache(maxsize=1)
def _load_model_bundle() -> dict:
    with _MODEL_PATH.open("rb") as fh:
        return pickle.load(fh)


@lru_cache(maxsize=1)
def _load_inference_assets() -> dict:
    df = load_clean()
    train_df, _ = time_split(df)
    return {
        "route_map": build_route_map(train_df),
        "weight_bin_edges": get_weight_bin_edges(train_df),
        "product_group_map": build_product_group_map(train_df),
        "unctad_lookup": load_lookup(),
    }


def _shipment_to_frame(shipment_context: dict) -> pd.DataFrame:
    row = {
        "Weight_kg": shipment_context.get("Weight_kg"),
        "mode": shipment_context.get("mode"),
        "inco_group": shipment_context.get("inco_group"),
        "product_group": shipment_context.get("product_group"),
        "year": shipment_context.get("year"),
        "origin_country": shipment_context.get("origin_country"),
        "dest_country": shipment_context.get("dest_country"),
    }
    return pd.DataFrame([row])


def predict_shipment(shipment_context: dict) -> dict:
    """Predict freight cost for a single shipment-like input row."""
    bundle = _load_model_bundle()
    assets = _load_inference_assets()

    shipment_df = _shipment_to_frame(shipment_context)
    X = build_features(
        shipment_df,
        unctad_lookup=assets["unctad_lookup"],
        route_map=assets["route_map"],
        weight_bin_edges=assets["weight_bin_edges"],
        product_group_map=assets["product_group_map"],
    )
    ordered_cols = bundle.get("feature_names") or feature_names()
    X = X[ordered_cols]

    model = bundle["model"]
    pred_log = float(model.predict(X)[0])
    prediction_usd = math.expm1(pred_log)

    booster = model.booster_
    contrib = booster.predict(X, pred_contrib=True)
    shap_values = np.asarray(contrib)[0][:-1]

    return {
        "prediction_usd": float(prediction_usd),
        "shap_values": shap_values,
        "feature_names": ordered_cols,
        "feature_values": X.iloc[0].to_dict(),
        "shipment_context": copy.deepcopy(shipment_context),
    }


def _build_delta_summary(original_prediction: float, updated_prediction: float) -> dict:
    delta = float(updated_prediction - original_prediction)
    pct = None if original_prediction == 0 else float(delta / original_prediction)
    return {
        "delta_usd": delta,
        "delta_pct": pct,
        "direction": "increase" if delta > 0 else "decrease" if delta < 0 else "no_change",
    }


_INTENT_SYSTEM = (
    "You are a freight logistics assistant. Classify the user's message and respond with "
    "valid JSON only — no markdown, no extra text."
)


def _llm_parse_intent(message: str, shipment_context: dict) -> dict:
    """LLM-based intent classification. Returns parsed dict with intent + modifications."""
    origin = shipment_context.get("origin_country", "unknown")
    dest = shipment_context.get("dest_country", "unknown")
    mode = shipment_context.get("mode", "unknown")
    weight = shipment_context.get("Weight_kg", "unknown")
    product = shipment_context.get("product_group", "unknown")

    user_prompt = (
        f'Current shipment: {origin} → {dest} | {mode} | {weight} kg | {product}\n'
        f'User message: "{message}"\n\n'
        'Respond with this JSON structure:\n'
        '{\n'
        '  "intent": "what_if" | "feasibility" | "informational",\n'
        '  "modifications": {"mode": null, "Weight_kg": null, "origin_country": null, "dest_country": null},\n'
        '  "feasibility_issue": true | false,\n'
        '  "feasibility_note": "string or null"\n'
        '}\n\n'
        'Rules:\n'
        f'- "what_if": user wants to change one or more shipment parameters\n'
        f'- "feasibility": user is asking whether a route or mode is physically possible\n'
        f'- "informational": user wants to understand trade terms, costs, or logistics concepts\n'
        f'- Set feasibility_issue=true if {mode} from {origin} to {dest} is physically impossible '
        f'(e.g. truck between continents separated by ocean)\n'
        f'- In modifications, include only fields that are changing; use null for unchanged fields\n'
        f'- For mode use exactly one of: "Air", "Ocean", "Truck", "Other"\n'
        f'- Weight_kg must be a number if provided\n'
    )

    raw = call_llm(_INTENT_SYSTEM, user_prompt)
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        raise ValueError(f"LLM did not return valid JSON for intent parsing: {raw[:300]}")
    return json.loads(match.group(0))


def whatif_chat(message: str, shipment_context: dict) -> dict:
    """
    Route a chat message to what-if prediction, feasibility answer, or informational RAG.
    Requires GROQ_API_KEY to be set.
    """
    if not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Add it to your .env file to enable the What-If Chat."
        )

    llm_intent = _llm_parse_intent(message, shipment_context)
    intent_type = llm_intent.get("intent", "informational")
    feasibility_issue = bool(llm_intent.get("feasibility_issue", False))
    feasibility_note = llm_intent.get("feasibility_note") or ""
    raw_mods = llm_intent.get("modifications") or {}
    modifications = {k: v for k, v in raw_mods.items() if v is not None}

    # what_if with no extractable modifications → treat as informational
    if intent_type == "what_if" and not modifications:
        intent_type = "informational"

    if intent_type == "feasibility":
        # Use the LLM's own note as the primary answer — RAG won't know geography
        rag_result = rag_query(message)
        rag_answer = rag_result.get("answer", "")
        answer = feasibility_note or "The route or mode may have physical or logistical constraints."
        usable_rag = rag_answer and "I don't have enough information" not in rag_answer
        if usable_rag:
            answer = f"{answer}\n\n**Market context:**\n{rag_answer}"
        return {
            "response_type": "rag_only",
            "answer": answer,
            "sources": rag_result.get("sources", []),
            "context_chunks": rag_result.get("context_chunks", []),
        }

    if intent_type == "informational":
        rag_result = rag_query(message)
        answer = rag_result.get("answer", "")
        if feasibility_issue and feasibility_note:
            answer = f"**Route Feasibility Note:** {feasibility_note}\n\n{answer}"
        return {
            "response_type": "rag_only",
            "answer": answer,
            "sources": rag_result.get("sources", []),
            "context_chunks": rag_result.get("context_chunks", []),
        }

    updated_shipment = apply_modifications(shipment_context, modifications)
    baseline = predict_shipment(shipment_context)
    modified = predict_shipment(updated_shipment)
    delta = _build_delta_summary(baseline["prediction_usd"], modified["prediction_usd"])

    augmented = rag_augmented_prediction(
        prediction_usd=modified["prediction_usd"],
        shap_values=modified["shap_values"],
        feature_names=modified["feature_names"],
        feature_values=modified["feature_values"],
        shipment_mode=str(updated_shipment.get("mode", "")),
        destination_country=str(updated_shipment.get("dest_country", "")),
        scenario_year=updated_shipment.get("year"),
        shipment_context=updated_shipment,
    )

    explanation = augmented["explanation"]
    if feasibility_issue and feasibility_note:
        explanation = f"**Route Feasibility Note:** {feasibility_note}\n\n{explanation}"

    return {
        "response_type": "what_if",
        "original_shipment": copy.deepcopy(shipment_context),
        "modified_shipment": updated_shipment,
        "original_prediction_usd": baseline["prediction_usd"],
        "modified_prediction_usd": modified["prediction_usd"],
        **delta,
        "explanation": explanation,
        "sources": augmented["sources"],
        "market_context": augmented["market_context"],
        "feature_values": modified["feature_values"],
        "scenario_year_requested": augmented["scenario_year_requested"],
        "context_year_used": augmented["context_year_used"],
        "year_fallback_message": augmented["year_fallback_message"],
        "available_context_years": augmented["available_context_years"],
    }
