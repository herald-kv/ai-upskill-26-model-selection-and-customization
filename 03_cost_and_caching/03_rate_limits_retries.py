"""03 · Rate limits and retries: what to do when the API says "slow down".

Every provider limits how much you can send:
  RPM  requests per minute        TPM  tokens per minute        RPD  requests per day
Limits are per model and per account (or per key, behind a gateway like our LiteLLM proxy).
Go over and you get HTTP 429 Too Many Requests. Servers also fail now and then (500, 503, timeouts).

The fix is a retry with EXPONENTIAL BACKOFF and JITTER:
  wait 1s, then 2s, then 4s, then 8s... plus a random bit, so 100 clients don't all retry at the same instant.

Part A  read the limits: the rate-limit headers on a real response
Part B  a flaky fake API: no retry vs naive retry vs backoff with jitter
Part C  you rarely write this yourself: SDK retries, and fallback models

Run:  python 03_cost_and_caching/03_rate_limits_retries.py        (--auto to skip the pauses)
"""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.common import MODELS, NO_RESPONSE_CACHE, bad, client, heading, note, ok, part, say, table


class RateLimited(Exception):
    """Stands in for openai.RateLimitError (HTTP 429)."""


class FlakyAPI:
    """A fake API that allows `limit` requests per `window` seconds, like a real rate limiter."""

    def __init__(self, limit=3, window=2.0):
        self.limit, self.window, self.calls = limit, window, []

    def call(self, i):
        now = time.monotonic()
        self.calls = [t for t in self.calls if now - t < self.window]
        if len(self.calls) >= self.limit:
            raise RateLimited("429 Too Many Requests")
        self.calls.append(now)
        return f"answer {i}"


def no_retry(api, n):
    done = 0
    for i in range(n):
        try:
            api.call(i)
            done += 1
        except RateLimited:
            pass
    return done, 0


def backoff(api, n, max_retries=6, base=0.25, cap=4.0):
    done = retries = 0
    for i in range(n):
        for attempt in range(max_retries + 1):
            try:
                api.call(i)
                done += 1
                break
            except RateLimited:
                if attempt == max_retries:
                    break
                # exponential: 0.25, 0.5, 1, 2, 4 ... capped; "full jitter": a random wait up to that
                wait = random.uniform(0, min(cap, base * 2 ** attempt))
                retries += 1
                time.sleep(wait)
    return done, retries


def headers_demo():
    model = MODELS["small"]
    raw = client().chat.completions.with_raw_response.create(
        model=model, messages=[{"role": "user", "content": "Say OK"}], max_tokens=5, extra_body=NO_RESPONSE_CACHE
    )
    h = raw.headers
    rows = [[k, h[k]] for k in sorted(h) if "ratelimit" in k.lower() and not k.lower().startswith("llm_provider")]
    if h.get("x-litellm-response-cost"):
        rows.append(["x-litellm-response-cost", f"${float(h['x-litellm-response-cost']):.8f}  (the proxy's own cost calculation)"])
    if rows:
        table(["header", "value"], rows)
    else:
        note("This provider didn't send rate-limit headers.")
    note("\nlimit = your allowance; remaining = what's left in this window; reset = when it refills.")
    note("Good clients slow down BEFORE they hit 0 remaining, and honour the Retry-After header on a 429.")


def main():
    heading("03.3 · Rate limits and retries")

    part("A", "the rate-limit headers on a real response")
    headers_demo()

    part("B", "a fake API that allows 3 requests per 2 seconds; we send 10")
    n = 10
    start = time.monotonic()
    done, _ = no_retry(FlakyAPI(), n)
    t1 = time.monotonic() - start
    start = time.monotonic()
    done_b, retries = backoff(FlakyAPI(), n)
    t2 = time.monotonic() - start
    table(["strategy", "succeeded", "retries", "time"], [
        ["no retry", f"{done}/{n}", 0, f"{t1:.1f}s"],
        ["backoff + jitter", f"{done_b}/{n}", retries, f"{t2:.1f}s"],
    ], "lrrr")
    bad(f"Without retries, {n - done} of {n} requests just failed. In production that's {n - done} angry users.")
    ok("With backoff, everything succeeds: slower, but nothing is lost.")
    note("Retry only what can succeed later: 429, 500, 502, 503, 504 and timeouts. Never retry 400 (bad request) or 401 (bad key).")
    note("(lab/common.py does retry 401, only because our shared proxy now and then picks a bad upstream key.)")

    part("C", "you rarely write this yourself")
    print('''
    # 1. The OpenAI and Anthropic SDKs already retry 429/5xx with backoff (2 retries by default):
    client = OpenAI(max_retries=5, timeout=30)

    # 2. For your own code, the `tenacity` library:
    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(6),
           retry=retry_if_exception_type((RateLimitError, APITimeoutError)))
    def ask(prompt): ...

    # 3. Fallback models: if one provider is down or rate-limited, try another.
    #    LiteLLM (our proxy) does this server-side: look for x-litellm-attempted-fallbacks
    #    and x-litellm-attempted-retries in the response headers.
''')
    say("Other ways to stay under the limits:")
    say("  • Limit concurrency (a semaphore, see 04_async_batching.py)    • Send fewer, bigger requests (batching)")
    say("  • Cache answers to repeated questions                          • Ask the provider for a higher tier")


if __name__ == "__main__":
    main()
