"""High-level orchestration for retrieval-backed Q&A and prediction explanation."""
from __future__ import annotations

import math
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.rag.generator import call_llm, generate_answer
from src.rag.retriever import retrieve

_EXPLANATION_SYSTEM = (
    "You are a senior freight cost analyst. Given a shipment, its model prediction, "
    "the top SHAP feature drivers, and retrieved market context, write a clear 3–4 sentence "
    "explanation of why this shipment costs the predicted amount. Be concrete: reference "
    "dollar amounts, the transport mode, the corridor, and the dominant cost drivers. "
    "Do not invent figures that are not in the data. Do not repeat the prompt back."
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _dedupe_sources(context_chunks: list[dict]) -> list[str]:
    sources: list[str] = []
    for chunk in context_chunks:
        source = str(chunk.get("source", "")).strip()
        if source and source not in sources:
            sources.append(source)
    return sources


def rag_query(query: str) -> dict:
    """Single entry point for chat-style RAG queries."""
    context_chunks = retrieve(query)
    result = generate_answer(query, context_chunks)
    return {**result, "context_chunks": context_chunks}


@lru_cache(maxsize=1)
def available_market_context_years() -> tuple[int, ...]:
    years: set[int] = set()
    corpus_root = _REPO_ROOT / "data" / "corpus"
    if corpus_root.exists():
        for path in corpus_root.iterdir():
            if not path.is_dir():
                continue
            match = re.fullmatch(r"market_(\d{4})", path.name)
            if match:
                years.add(int(match.group(1)))
    return tuple(sorted(years))


def _resolve_context_year(scenario_year: int | None) -> dict:
    available_years = available_market_context_years()
    if scenario_year is None:
        used_year = available_years[-1] if available_years else None
        return {
            "scenario_year_requested": None,
            "context_year_used": used_year,
            "year_fallback_message": None,
            "available_context_years": list(available_years),
        }

    requested_year = int(scenario_year)
    if requested_year in available_years:
        return {
            "scenario_year_requested": requested_year,
            "context_year_used": requested_year,
            "year_fallback_message": None,
            "available_context_years": list(available_years),
        }

    used_year = available_years[-1] if available_years else None
    fallback_message = None
    if used_year is not None:
        fallback_message = (
            f"No year-specific market corpus is available for scenario year {requested_year}. "
            f"Using {used_year} market context instead."
        )
    return {
        "scenario_year_requested": requested_year,
        "context_year_used": used_year,
        "year_fallback_message": fallback_message,
        "available_context_years": list(available_years),
    }


def _build_llm_explanation_prompt(
    prediction_usd: float,
    shap_values: np.ndarray,
    feature_names: list,
    feature_values: dict,
    shipment_context: dict,
    context_chunks: list[dict],
) -> str:
    shap_arr = np.asarray(shap_values, dtype=float)
    top_idx = np.argsort(np.abs(shap_arr))[::-1][:5]
    driver_lines = []
    for i in top_idx:
        name = feature_names[i]
        val = float(shap_arr[i])
        impact = abs(float(prediction_usd) * (math.exp(abs(val)) - 1.0))
        direction = "increases" if val >= 0 else "decreases"
        raw_fv = feature_values.get(name)
        fv_str = f"{float(raw_fv):.4g}" if isinstance(raw_fv, (int, float)) else str(raw_fv)
        driver_lines.append(f"  - {name} = {fv_str}: {direction} cost by ~${impact:,.0f}")

    ctx_summary = ""
    for chunk in context_chunks[:3]:
        snippet = chunk.get("text", "")[:250].replace("\n", " ")
        ctx_summary += f"\n[{chunk.get('source', 'Market data')}] {snippet}"

    origin = shipment_context.get("origin_country", "unknown")
    dest = shipment_context.get("dest_country", "unknown")
    mode = shipment_context.get("mode", "unknown")
    weight = shipment_context.get("Weight_kg", "unknown")
    product = shipment_context.get("product_group", "unknown")
    inco = shipment_context.get("inco_group", "unknown")

    return (
        f"Shipment: {origin} → {dest} | {mode} | {weight} kg | {product} | INCO: {inco}\n"
        f"Model prediction: ${float(prediction_usd):,.0f}\n\n"
        f"Top cost drivers (SHAP):\n" + "\n".join(driver_lines) + "\n\n"
        f"Retrieved market context:{ctx_summary}\n\n"
        "Explain in 3–4 sentences why this shipment costs the predicted amount."
    )


def rag_augmented_prediction(
    prediction_usd: float,
    shap_values: np.ndarray,
    feature_names: list,
    feature_values: dict,
    shipment_mode: str,
    destination_country: str,
    scenario_year: int | None = None,
    shipment_context: dict | None = None,
) -> dict:
    """Attach retrieved market context and LLM explanation to a precomputed prediction."""
    if not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Add it to your .env file to enable AI explanations."
        )
    year_meta = _resolve_context_year(scenario_year)
    auto_query = f"{shipment_mode} freight {destination_country} cost"
    market_context = retrieve(auto_query, year_hint=year_meta["context_year_used"])
    ctx = shipment_context or {
        "mode": shipment_mode,
        "dest_country": destination_country,
        "origin_country": "",
        "Weight_kg": feature_values.get("log_weight", ""),
        "product_group": "",
        "inco_group": "",
    }
    explanation_prompt = _build_llm_explanation_prompt(
        prediction_usd=prediction_usd,
        shap_values=shap_values,
        feature_names=feature_names,
        feature_values=feature_values,
        shipment_context=ctx,
        context_chunks=market_context,
    )
    explanation = call_llm(_EXPLANATION_SYSTEM, explanation_prompt)
    return {
        "prediction_usd": float(prediction_usd),
        "explanation": explanation,
        "market_context": market_context,
        "sources": _dedupe_sources(market_context),
        **year_meta,
    }
