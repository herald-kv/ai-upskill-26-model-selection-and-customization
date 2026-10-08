"""Shared plumbing: the API client, the model names, measuring and printing.

Nothing in here is part of the lesson. Read it if you are curious.

`chat()` is `client.chat.completions.create()` that also measures how long the call took,
reads the token counts (including cached tokens), works out the dollar cost and prints it all.
Every model goes through the company LiteLLM proxy, which speaks the OpenAI API for every
provider. That is why one function can call GPT, Claude, Gemini and DeepSeek.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from lab.pricing import cost

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# override=True so a stale OPENAI_API_KEY in your shell can't win over the .env file
load_dotenv(ROOT / ".env", override=True)

# The models used in the demos, by role. Change them in .env.
MODELS = {
    "small": os.getenv("MODEL_SMALL", "openai/gpt-4.1-nano"),
    "mid": os.getenv("MODEL_MID", "openai/gpt-5.4-mini"),
    "frontier": os.getenv("MODEL_FRONTIER", "anthropic/claude-sonnet-4-6"),
    "google": os.getenv("MODEL_GOOGLE", "gemini/gemini-2.5-flash"),
    "open": os.getenv("MODEL_OPEN", "deepseek/deepseek-chat"),
}

AUTO = "--auto" in sys.argv          # run straight through without pausing
QUIET = "--hide-prompts" in sys.argv  # don't print the prompts


# --- the client --------------------------------------------------------------------

def _base_url() -> str:
    base = (os.getenv("LITELLM_BASE_URL") or "").strip().rstrip("/")
    if not base:
        sys.exit("LITELLM_BASE_URL is not set. Copy .env.example to .env and fill it in.")
    return base if base.endswith("/v1") else base + "/v1"


def _api_key() -> str:
    key = (os.getenv("LITELLM_API_KEY") or "").strip()
    if not key:
        sys.exit("LITELLM_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return key


_client = None
_async_client = None


def _ssl_context():
    """A plain certifi SSL context. The SDK's default (truststore) can crash the interpreter
    when dozens of async connections open at the same moment, as the bake-off does."""
    import ssl

    import certifi

    return ssl.create_default_context(cafile=certifi.where())


def client():
    """OpenAI SDK client pointed at the LiteLLM proxy."""
    global _client
    if _client is None:
        from openai import DefaultHttpxClient, OpenAI

        _client = OpenAI(base_url=_base_url(), api_key=_api_key(), timeout=120, max_retries=0,
                         http_client=DefaultHttpxClient(verify=_ssl_context()))
    return _client


def async_client():
    """Async version, for the async/batching demo. One per event loop, because each
    asyncio.run() starts a new loop and a client can't be shared across loops."""
    import asyncio

    global _async_client
    loop = asyncio.get_running_loop()
    if _async_client is None or _async_client[0] is not loop:
        from openai import AsyncOpenAI, DefaultAsyncHttpxClient

        _async_client = (loop, AsyncOpenAI(base_url=_base_url(), api_key=_api_key(), timeout=120, max_retries=0,
                                           http_client=DefaultAsyncHttpxClient(verify=_ssl_context())))
    return _async_client[1]


# --- one measured call -----------------------------------------------------------------

@dataclass
class Result:
    model: str
    text: str
    seconds: float
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    cost: float = 0.0
    raw: object = field(default=None, repr=False)


def read_usage(model: str, usage) -> dict:
    """Pull token counts out of the usage block. Providers report caching differently;
    LiteLLM maps them all into the OpenAI shape, with Anthropic's extra fields alongside."""
    if usage is None:
        return dict(input_tokens=0, output_tokens=0, cached_tokens=0, cache_write_tokens=0, reasoning_tokens=0)
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) or getattr(usage, "cache_read_input_tokens", None) or 0
    written = getattr(usage, "cache_creation_input_tokens", None) or 0
    out_details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(out_details, "reasoning_tokens", None) or 0
    return dict(
        input_tokens=usage.prompt_tokens or 0,
        output_tokens=usage.completion_tokens or 0,
        cached_tokens=cached,
        cache_write_tokens=written,
        reasoning_tokens=reasoning,
    )


def _to_result(model: str, response, seconds: float) -> Result:
    message = response.choices[0].message
    counts = read_usage(model, response.usage)
    return Result(
        model=model,
        text=(message.content or "").strip(),
        seconds=seconds,
        cost=cost(model, counts["input_tokens"], counts["output_tokens"],
                  counts["cached_tokens"], counts["cache_write_tokens"]),
        raw=response,
        **counts,
    )


# The shared proxy occasionally routes a request to a bad upstream key (a 401 or 5xx that
# works on the next try). Demos retry twice so a hiccup doesn't derail the session.
# 03_cost_and_caching/03_rate_limits_retries.py shows how to write retries properly.
RETRIES = 2

# The proxy also caches whole responses: an identical request returns the stored answer in ~1 ms.
# That would make every timing in these demos meaningless, so it's off unless you ask for it
# (03_cost_and_caching/02_prompt_caching.py, Part E, shows it on purpose).
NO_RESPONSE_CACHE = {"cache": {"no-cache": True}}


def _with_defaults(kwargs: dict) -> dict:
    if kwargs.pop("response_cache", False):
        return kwargs
    extra = dict(kwargs.get("extra_body") or {})
    extra.update(NO_RESPONSE_CACHE)
    return {**kwargs, "extra_body": extra}


def _retryable(error: Exception) -> bool:
    from openai import APIConnectionError, APIStatusError

    if isinstance(error, APIConnectionError):
        return True
    return isinstance(error, APIStatusError) and error.status_code in (401, 429, 500, 502, 503, 504)


def chat(model: str, messages: list, *, show: bool = True, **kwargs) -> Result:
    """Send one chat request through the proxy, time it, and work out its cost."""
    if show:
        log_request(model, messages, {k: v for k, v in kwargs.items() if k != "response_cache"})
    kwargs = _with_defaults(kwargs)
    for attempt in range(RETRIES + 1):
        start = time.perf_counter()
        try:
            response = client().chat.completions.create(model=model, messages=messages, **kwargs)
            break
        except Exception as error:
            if attempt == RETRIES or not _retryable(error):
                raise
            time.sleep(1 + attempt)
    result = _to_result(model, response, time.perf_counter() - start)
    if show:
        log_result(result)
    return result


async def achat(model: str, messages: list, **kwargs) -> Result:
    """Async version of chat(). Prints nothing; the caller decides what to show."""
    import asyncio

    kwargs = _with_defaults(kwargs)
    for attempt in range(RETRIES + 1):
        start = time.perf_counter()
        try:
            response = await async_client().chat.completions.create(model=model, messages=messages, **kwargs)
            break
        except Exception as error:
            if attempt == RETRIES or not _retryable(error):
                raise
            await asyncio.sleep(1 + attempt)
    return _to_result(model, response, time.perf_counter() - start)


def ask(model: str, prompt: str, system: str | None = None, **kwargs) -> Result:
    """Shortcut for a single user message (and an optional system prompt)."""
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    return chat(model, messages, **kwargs)


# --- counting tokens without calling a model ------------------------------------------------

def count_tokens(text: str) -> int:
    """Token count with OpenAI's o200k tokenizer. Other providers' tokenizers differ by
    roughly 10-30%, so treat this as an estimate for non-OpenAI models."""
    try:
        import tiktoken

        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)  # rule of thumb: ~4 characters per token in English


# --- printing ------------------------------------------------------------------------

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
AMBER, GREEN, RED, BLUE, CYAN, MAGENTA = "\033[33m", "\033[32m", "\033[31m", "\033[34m", "\033[36m", "\033[35m"


def heading(text: str):
    print(f"\n{BOLD}{AMBER}{'=' * 78}\n {text}\n{'=' * 78}{RESET}")


def part(letter: str, text: str):
    pause()
    print(f"\n{BOLD}{BLUE}--- Part {letter}: {text} {'-' * max(0, 62 - len(text))}{RESET}")


def note(text: str):
    print(f"{DIM}{text}{RESET}")


def ok(text: str):
    print(f"{GREEN}✓ {text}{RESET}")


def bad(text: str):
    print(f"{RED}✗ {text}{RESET}")


def say(text: str):
    print(text)


def pause():
    if AUTO or not sys.stdin.isatty():
        return
    try:
        input(f"{DIM}[Enter to continue]{RESET}")
    except EOFError:
        pass


def short(text: str, limit: int = 400) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f"{DIM} … (+{len(text) - limit} chars){RESET}"


def log_request(model: str, messages: list, kwargs: dict):
    print(f"{CYAN}▶ model: {model}{RESET}")
    if QUIET:
        return
    for key, value in kwargs.items():
        print(f"{CYAN}▶ {key}: {short(value, 120)}{RESET}")
    for m in messages:
        content = m["content"]
        if isinstance(content, list):  # content blocks (images, cache_control)
            content = " + ".join(
                "[image]" if b.get("type") == "image_url" else short(b.get("text", ""), 200) for b in content
            )
        print(f"{CYAN}▶ {m['role']}: {RESET}{short(content, 300)}")


def log_result(r: Result):
    print(f"{MAGENTA}◀ {RESET}{short(r.text, 600)}")
    print(f"{DIM}  {fmt_usage(r)}{RESET}")


def fmt_usage(r: Result) -> str:
    extra = ""
    if r.cached_tokens:
        extra += f" (cached {r.cached_tokens})"
    if r.cache_write_tokens:
        extra += f" (written to cache {r.cache_write_tokens})"
    if r.reasoning_tokens:
        extra += f" (of which thinking {r.reasoning_tokens})"
    return (f"{r.seconds:.2f}s · in {r.input_tokens} · out {r.output_tokens}{extra} · "
            f"${r.cost:.6f}")


def table(headers: list[str], rows: list[list], align: str | None = None):
    """Plain text table. `align` is a string of l/r per column."""
    cells = [[str(h) for h in headers]] + [[str(c) for c in row] for row in rows]
    widths = [max(len(_plain(r[i])) for r in cells) for i in range(len(headers))]
    align = align or "l" * len(headers)

    def line(row):
        out = []
        for i, c in enumerate(row):
            pad = widths[i] - len(_plain(c))
            out.append(c + " " * pad if align[i] == "l" else " " * pad + c)
        return "  ".join(out)

    print(BOLD + line(cells[0]) + RESET)
    print(DIM + "  ".join("-" * w for w in widths) + RESET)
    for row in cells[1:]:
        print(line(row))


def _plain(s: str) -> str:
    import re

    return re.sub(r"\033\[[0-9;]*m", "", s)


def money(x: float) -> str:
    if x >= 100:
        return f"${x:,.0f}"
    if x >= 1:
        return f"${x:,.2f}"
    return f"${x:.4f}" if x >= 0.0001 else f"${x:.6f}"
