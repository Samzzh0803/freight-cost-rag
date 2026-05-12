"""LLM-backed answer generation for RAG chat queries."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

DEFAULT_SYSTEM_PROMPT = (
    "You are a freight logistics expert. Answer using ONLY the provided context. "
    "If the context doesn't contain the answer, say 'I don't have enough information to answer that.' "
    "Always cite which source document your answer comes from."
)
GROQ_MODEL = "llama-3.1-8b-instant"
OPENAI_MODEL = "gpt-4o-mini"

logger = logging.getLogger(__name__)
_CACHE_PAYLOADS: dict[str, tuple[str, str, str]] = {}


def _get_provider_name() -> str:
    return os.getenv("LLM_PROVIDER", "groq").strip().lower()


def _serialize_context(context_chunks: list[dict]) -> str:
    serializable = [
        {
            "text": chunk.get("text", ""),
            "source": chunk.get("source", ""),
            "category": chunk.get("category", ""),
            "score": round(float(chunk.get("score", 0.0)), 6),
        }
        for chunk in context_chunks
    ]
    return json.dumps(serializable, sort_keys=True, ensure_ascii=True)


def _build_cache_key(query: str, context_payload: str, system_prompt: str) -> str:
    digest = hashlib.sha256()
    digest.update(query.encode("utf-8"))
    digest.update(context_payload.encode("utf-8"))
    digest.update(system_prompt.encode("utf-8"))
    return digest.hexdigest()


def _build_user_prompt(query: str, context_chunks: list[dict]) -> str:
    context_blocks = []
    for idx, chunk in enumerate(context_chunks, start=1):
        context_blocks.append(
            f"[Source {idx}: {chunk.get('source', 'Unknown source')} | "
            f"Category: {chunk.get('category', 'unknown')} | "
            f"Score: {float(chunk.get('score', 0.0)):.3f}]\n{chunk.get('text', '')}"
        )
    joined_context = "\n\n".join(context_blocks)
    return f"Question: {query}\n\nContext:\n{joined_context}"


def _get_sources(context_chunks: list[dict]) -> list[str]:
    sources: list[str] = []
    for chunk in context_chunks:
        source = str(chunk.get("source", "")).strip()
        if source and source not in sources:
            sources.append(source)
    return sources


def _get_groq_client():
    from groq import Groq

    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def _get_openai_client():
    from openai import OpenAI

    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _call_chat_completion(provider: str, system_prompt: str, user_prompt: str) -> dict:
    if provider == "groq":
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        return {"answer": content.strip(), "model_used": GROQ_MODEL}

    if provider == "openai":
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        return {"answer": content.strip(), "model_used": OPENAI_MODEL}

    raise ValueError(f"Unknown LLM provider: {provider}")


def _is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True
    return "429" in str(exc)


@lru_cache(maxsize=50)
def _generate_answer_cached(cache_key: str) -> dict:
    query, context_payload, system_prompt = _CACHE_PAYLOADS[cache_key]
    context_chunks = json.loads(context_payload)
    user_prompt = _build_user_prompt(query, context_chunks)
    sources = _get_sources(context_chunks)
    provider = _get_provider_name()

    try:
        result = _call_chat_completion(provider, system_prompt, user_prompt)
        return {**result, "sources": sources}
    except Exception as exc:
        if provider == "groq" and _is_rate_limit_error(exc) and os.getenv("OPENAI_API_KEY"):
            logger.warning("Groq rate limit hit; falling back to OpenAI.")
            try:
                result = _call_chat_completion("openai", system_prompt, user_prompt)
                return {**result, "sources": sources}
            except Exception as fallback_exc:
                logger.warning("OpenAI fallback failed after Groq 429: %s", fallback_exc)
                return {
                    "answer": f"Error generating answer: {fallback_exc}",
                    "sources": sources,
                    "model_used": "error",
                }
        logger.warning("LLM generation failed: %s", exc)
        return {
            "answer": f"Error generating answer: {exc}",
            "sources": sources,
            "model_used": "error",
        }


def generate_answer(query: str, context_chunks: list[dict], system_prompt: str = None) -> dict:
    """Generate an answer grounded only in the provided retrieved context."""
    if not context_chunks:
        return {
            "answer": "I don't have enough information to answer that.",
            "sources": [],
            "model_used": "no_context",
        }

    prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    context_payload = _serialize_context(context_chunks)
    cache_key = _build_cache_key(query, context_payload, prompt)
    _CACHE_PAYLOADS[cache_key] = (query, context_payload, prompt)
    return _generate_answer_cached(cache_key)


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Direct LLM call for non-RAG use cases (explanation, intent parsing).
    Raises RuntimeError if no API key is configured."""
    if not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Add it to your .env file to enable AI features."
        )
    provider = _get_provider_name()
    result = _call_chat_completion(provider, system_prompt, user_prompt)
    return result["answer"]
