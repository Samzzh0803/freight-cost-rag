"""RAG package exports with lazy loading to avoid optional dependency import errors."""

__all__ = [
    "load_index",
    "retrieve",
    "generate_answer",
    "explain_prediction",
    "rag_query",
    "rag_augmented_prediction",
    "parse_whatif_intent",
    "apply_modifications",
    "predict_shipment",
    "whatif_chat",
]


def __getattr__(name: str):
    if name in {"load_index", "retrieve"}:
        from src.rag.retriever import load_index, retrieve

        return {"load_index": load_index, "retrieve": retrieve}[name]

    if name == "generate_answer":
        from src.rag.generator import generate_answer

        return generate_answer

    if name == "explain_prediction":
        from src.rag.explainer import explain_prediction

        return explain_prediction

    if name in {"rag_query", "rag_augmented_prediction"}:
        from src.rag.pipeline import rag_augmented_prediction, rag_query

        return {"rag_query": rag_query, "rag_augmented_prediction": rag_augmented_prediction}[name]

    if name in {"parse_whatif_intent", "apply_modifications", "predict_shipment", "whatif_chat"}:
        from src.rag.whatif import apply_modifications, parse_whatif_intent, predict_shipment, whatif_chat

        return {
            "parse_whatif_intent": parse_whatif_intent,
            "apply_modifications": apply_modifications,
            "predict_shipment": predict_shipment,
            "whatif_chat": whatif_chat,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
