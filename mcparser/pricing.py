"""Estimated cost of a run from token counts and list prices.

Prices are US dollars per million tokens (Anthropic list prices, October 2026).
Cache writes use the 5-minute rate, which is what ``cache=True`` uses. Haiku's
price is for prompts up to 100k tokens, far above the size of an agent request.
Models not in the table get no estimate.
"""

from __future__ import annotations

from pydantic import BaseModel


class Price(BaseModel):
    input: float
    output: float
    cache_write: float
    cache_read: float


PRICES: dict[str, Price] = {
    "claude-fable-5-1": Price(input=10, output=50, cache_write=12.5, cache_read=0.25),
    "claude-opus-5-5": Price(input=4, output=20, cache_write=5, cache_read=0.20),
    "claude-sonnet-5-5": Price(input=2, output=10, cache_write=2.5, cache_read=0.10),
    "claude-haiku-5-5": Price(input=0.10, output=0.50, cache_write=0.125, cache_read=0.01),
}


def price_for(model: str) -> Price | None:
    """The price of ``model``, given as ``provider:name`` or just the name."""
    return PRICES.get(model.rsplit(":", 1)[-1])


def estimate(model: str, input_tokens: int, cache_read_tokens: int, cache_write_tokens: int, output_tokens: int) -> float | None:
    """Dollars for this usage. ``input_tokens`` includes the cached and cache-written tokens."""
    price = price_for(model)
    if price is None:
        return None
    uncached = input_tokens - cache_read_tokens - cache_write_tokens
    return (
        uncached * price.input
        + cache_read_tokens * price.cache_read
        + cache_write_tokens * price.cache_write
        + output_tokens * price.output
    ) / 1_000_000
