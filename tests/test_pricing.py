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


def test_model_usage_accumulates():
    from pydantic_ai.usage import RunUsage

    usage = ModelUsage()
    usage.add(RunUsage(requests=2, input_tokens=1_000_000, cache_read_tokens=1_000_000, output_tokens=0), "claude-haiku-5-5")
    assert (usage.tasks, usage.requests, usage.cost) == (1, 2, pytest.approx(0.01))
