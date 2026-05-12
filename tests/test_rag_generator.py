"""Tests for src.rag.generator."""
from src.rag import generator


def setup_function():
    generator._generate_answer_cached.cache_clear()
    generator._CACHE_PAYLOADS.clear()


def test_generate_answer_returns_dict_keys(monkeypatch):
    def fake_call(provider, system_prompt, user_prompt):
        return {"answer": "DDP means Delivered Duty Paid.", "model_used": "fake-model"}

    monkeypatch.setattr(generator, "_call_chat_completion", fake_call)
    out = generator.generate_answer(
        "What does DDP mean?",
        [{"text": "DDP means Delivered Duty Paid.", "source": "Incoterms 2020", "category": "incoterms", "score": 0.91}],
    )
    assert {"answer", "sources", "model_used"} <= set(out.keys())


def test_generate_answer_empty_context():
    out = generator.generate_answer("What is the capital of France?", [])
    assert out["answer"] == "I don't have enough information to answer that."
    assert out["sources"] == []
    assert out["model_used"] == "no_context"


def test_cache_hit(monkeypatch):
    calls = {"count": 0}

    def fake_call(provider, system_prompt, user_prompt):
        calls["count"] += 1
        return {"answer": "Cached answer.", "model_used": "fake-model"}

    monkeypatch.setattr(generator, "_call_chat_completion", fake_call)
    context = [{"text": "Ocean freight rates remain elevated.", "source": "Market 2026", "category": "market_2026", "score": 0.88}]
    first = generator.generate_answer("What are current ocean freight rates?", context)
    second = generator.generate_answer("What are current ocean freight rates?", context)
    assert first == second
    assert calls["count"] == 1
