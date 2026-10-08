# Part 3 · Cost, caching, rate limits, async

## Concepts

**Token economics.** `cost = input tokens × input price + output tokens × output price` (prices per million tokens).
Output costs 2 to 8× more than input. In a typical RAG chatbot the user's question is ~1% of the tokens; the system prompt,
retrieved documents and history are over 90%. Reasoning models also bill their hidden "thinking" tokens as output.

**Prompt caching.** If many requests start with the same long text, the provider reuses the processed start of the prompt.

| Provider | How | Cached input price |
|---|---|---|
| OpenAI | Automatic, prompts ≥ 1,024 tokens | 10% (GPT-5.x) to 25% (GPT-4.1) |
| Anthropic | Explicit `cache_control` on a content block | 10% to read, 125% to write; lives 5 min (or 1 h) |
| Gemini | Automatic, or explicit cached content | roughly 10 to 25% |
| DeepSeek | Automatic | 10% |

The one rule: **static first, dynamic last**. Anything that changes (time, user name, retrieved chunks) goes after the part you want cached.

**Response caching** stores whole answers for identical requests (our LiteLLM proxy does this). **Semantic caching**
reuses answers for questions that mean the same thing, using embeddings.

**Rate limits.** RPM (requests/min), TPM (tokens/min), sometimes per day or a budget. Over the limit you get HTTP 429.
Retry 429, 5xx and timeouts with **exponential backoff and jitter**; never retry 400 or 401. The OpenAI and Anthropic SDKs
already do this (`max_retries`), `tenacity` does it for your own code, and gateways like LiteLLM add fallback models.

**Async and batching.** LLM calls are mostly waiting. `asyncio.gather` runs them concurrently; an `asyncio.Semaphore`
caps how many are in flight so you stay under rate limits. Putting several items in one prompt sends the instructions
once. Provider **Batch APIs** take a file of requests, return results within 24 hours, and cost ~50% less.

## The scripts

| Script | What happens | What we saw when testing |
|---|---|---|
| `01_token_economics.py` | Tokenising English, Hindi, Malayalam, JSON, code; anatomy of a request; cost at scale; levers | Malayalam took 1.9 characters per token vs 3.5 for English. One request: $0.0006 on gpt-4.1-nano, $0.032 on gpt-5.5. |
| `02_prompt_caching.py` | 3 questions over the 3,600-token handbook on OpenAI and Claude; the timestamp mistake; response caching | From call 2 on: 3,328 tokens cached on OpenAI (cost 75% lower), 4,066 on Claude (~90% lower). Timestamp first: 0 cached. A repeated identical request came back in 0.05 s. |
| `03_rate_limits_retries.py` | Real rate-limit headers; a fake API that allows 3 requests per 2 s | No retry: 3/10 succeeded. Backoff with jitter: 10/10. |
| `04_async_batching.py` | 12 tickets sequential vs gather vs Semaphore(4) vs one prompt | Async was 2 to 5× faster at the same cost. One batched prompt cost about the same with 1 request instead of 12. |

Latency on a shared proxy is noisy. If a timing looks odd, run it again.

## Exercises

1. In `02_prompt_caching.py`, move the `[handbook copy …]` tag from the start of the system prompt to the end. Does caching still work? Why?
2. In `04_async_batching.py`, try `Semaphore(1)`, `(2)` and `(8)`. What happens to the time?
3. In `03_rate_limits_retries.py`, remove the jitter (use `wait = min(cap, base * 2 ** attempt)`). Run 5 copies at once in different terminals. What's the risk?
4. Using `01_token_economics.py`, estimate the monthly bill for a feature you work on.
