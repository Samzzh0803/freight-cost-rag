"""
SHAP + LLM explanation pipeline.

TODO:
- shap.TreeExplainer on the trained LightGBM model
- For a single prediction: compute SHAP values, extract top-5 feature contributions
- Retrieve relevant trade doc chunks via src.rag.retriever
- Assemble LLM prompt: shipment inputs + SHAP table + UNCTAD benchmark + retrieved chunks
- Call LLM via src.config.get_llm_client()
- Return structured dict: {summary, shap_drivers, citations, unctad_note}

Prompt template is in project_plan_v4.tex §3 Week 9.
Cache with functools.lru_cache keyed on input hash to avoid redundant LLM calls.
"""

import shap
from src.config import get_llm_client


def explain_prediction(model, inputs: dict, unctad_lookup: dict) -> dict:
    """
    inputs: dict of raw shipment fields (origin, destination, mode, weight, commodity, inco_term)
    Returns dict with keys: summary, shap_drivers (list), citations (list), unctad_note (str)
    """
    # TODO: implement explanation pipeline
    raise NotImplementedError
