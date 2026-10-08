"""Price list used to turn token counts into dollars.

Prices are USD per 1 million tokens, taken from the company LiteLLM proxy's public model hub
(https://llm.keyvalue.systems/public/model_hub) and the providers' pricing pages on 2026-10-01.
They are here to make the cost demos concrete. Check the provider's page before you quote them.

`cached` is the price of an input token that was read from the prompt cache.
`cache_write` is the price of writing to the cache (Anthropic charges extra; OpenAI, Gemini and
DeepSeek cache automatically and charge nothing extra to write).
"""

PRICES = {
    # model id on the proxy              input   output  cached  cache_write
    "openai/gpt-4.1-nano":            dict(input=0.10, output=0.40, cached=0.025, cache_write=0.10),
    "openai/gpt-4o-mini":             dict(input=0.15, output=0.60, cached=0.075, cache_write=0.15),
    "openai/gpt-5.4-nano":            dict(input=0.20, output=1.25, cached=0.02, cache_write=0.20),
    "openai/gpt-5.4-mini":            dict(input=0.75, output=4.50, cached=0.075, cache_write=0.75),
    "openai/gpt-5.5":                 dict(input=5.00, output=30.00, cached=0.50, cache_write=5.00),
    "anthropic/claude-haiku-4-5":     dict(input=1.00, output=5.00, cached=0.10, cache_write=1.25),
    "anthropic/claude-sonnet-4-6":    dict(input=3.00, output=15.00, cached=0.30, cache_write=3.75),
    "gemini/gemini-2.5-flash":        dict(input=0.30, output=2.50, cached=0.03, cache_write=0.30),
    "deepseek/deepseek-chat":         dict(input=0.28, output=0.42, cached=0.028, cache_write=0.28),
}

# Rough GPU rental prices for the self-hosting comparison (USD per hour, on-demand cloud).
GPU_HOURLY = {
    "1x L4 (24 GB)  - fits 7-8B models": 0.80,
    "1x A100 (80 GB) - fits ~30B models, or 70B quantised": 2.50,
    "1x H100 (80 GB) - fastest single GPU": 4.00,
    "4x H100        - 70B at full precision": 16.00,
}


def price_of(model: str) -> dict:
    """Prices for a model, or zeros (with a flag) if we don't know it."""
    if model in PRICES:
        return PRICES[model]
    return dict(input=0.0, output=0.0, cached=0.0, cache_write=0.0, unknown=True)


def cost(model: str, input_tokens: int, output_tokens: int,
         cached_tokens: int = 0, cache_write_tokens: int = 0) -> float:
    """Dollar cost of one request.

    `input_tokens` is the total prompt size as the API reports it. Cached and cache-write
    tokens are part of that total, so they are taken out and charged at their own rate.
    """
    p = price_of(model)
    fresh = max(input_tokens - cached_tokens - cache_write_tokens, 0)
    return (
        fresh * p["input"]
        + cached_tokens * p["cached"]
        + cache_write_tokens * p["cache_write"]
        + output_tokens * p["output"]
    ) / 1_000_000
