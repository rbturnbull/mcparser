import pytest

from mcparser.agents.team import ModelUsage
from mcparser.pricing import estimate, price_for


def test_prices():
    assert price_for("anthropic:claude-opus-5-5").output == 20
    assert price_for("claude-sonnet-5-5").cache_read == 0.10
    assert price_for("openai:gpt-9") is None


def test_estimate():
    # 1M input of which 600k cache reads and 100k cache writes, plus 10k output, on Opus 5.5
    assert estimate("anthropic:claude-opus-5-5", 1_000_000, 600_000, 100_000, 10_000) == pytest.approx(
        0.3 * 4 + 0.6 * 0.2 + 0.1 * 5 + 0.01 * 20
    )
    assert estimate("local:llama", 10, 0, 0, 10) is None


def test_estimate_other_vendors_uses_cache_rates():
    plain = estimate("openai:gpt-5.6", 1_000_000, 0, 0, 0)
    cached = estimate("openai:gpt-5.6", 1_000_000, 900_000, 0, 0)
    assert plain and cached and cached < plain / 2  # cache reads are much cheaper than fresh input
    assert estimate("google-gla:gemini-2.5-pro", 1000, 500, 0, 10) > 0
    assert estimate("gateway:gpt-5.6", 1000, 0, 0, 10) > 0  # unknown provider prefix: found by model name


def test_model_usage_accumulates():
    from pydantic_ai.usage import RunUsage

    usage = ModelUsage()
    usage.add(RunUsage(requests=2, input_tokens=1_000_000, cache_read_tokens=1_000_000, output_tokens=0), "claude-haiku-5-5")
    assert (usage.tasks, usage.requests, usage.cost) == (1, 2, pytest.approx(0.01))
