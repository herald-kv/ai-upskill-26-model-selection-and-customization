"""02 · Prompt caching: pay full price for the same prompt only once.

When many requests start with the same long text (a system prompt, a handbook, few-shot
examples, a big document), the provider can keep the processed version of that prefix and
reuse it. Cached input tokens cost about 10% (Anthropic, OpenAI GPT-5.x, DeepSeek) to 25%
(OpenAI GPT-4.1) of the normal price, and the response starts faster.

Analogy: a kitchen preps the ingredients once in the morning and cooks many dishes from them.

  OpenAI, DeepSeek, Gemini  automatic, for prompts over ~1,024 tokens. Nothing to change.
  Anthropic (Claude)       explicit: mark where the cacheable part ends with cache_control.
                           Writing the cache costs 1.25×; reading it costs 0.1×. Lasts 5 minutes (or 1 hour).

The one rule: caching matches the PREFIX exactly. Static content first, dynamic content last.

Part A  OpenAI: automatic caching, 3 questions over the same handbook
Part B  Claude: explicit cache_control, cache write then cache reads
Part C  the classic mistake: a timestamp at the top of the prompt
Part D  what it saves at scale
Part E  a different kind of cache: storing the whole RESPONSE (our proxy does this)

Run:  python 03_cost_and_caching/02_prompt_caching.py        (--auto to skip the pauses)
"""

import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.common import DATA, MODELS, bad, chat, heading, money, note, ok, part, say, table
from lab.pricing import cost

HANDBOOK = (DATA / "handbook.md").read_text()
SYSTEM = "You are BrightCart's support assistant. Answer in one or two sentences, using only the handbook.\n\n" + HANDBOOK
QUESTIONS = [
    "How long do UPI refunds take?",
    "Can I return a laptop after 20 days?",
    "What does BrightCart Plus cost?",
]
# A tag that's new each run, so the first call really is a cache miss even if you rerun quickly.
RUN = datetime.now().strftime("%H%M%S")


def row(i, r):
    return [i, r.input_tokens, r.cached_tokens or "", r.cache_write_tokens or "", f"{r.seconds:.2f}s", money(r.cost)]


HEADERS = ["call", "input tokens", "cached", "written to cache", "latency", "cost"]


def openai_auto():
    model = MODELS["mid"]
    system = f"[handbook copy {RUN}]\n" + SYSTEM  # same for all 3 calls in this run
    rows = []
    for i, q in enumerate(QUESTIONS, 1):
        r = chat(model, [{"role": "system", "content": system}, {"role": "user", "content": q}], max_tokens=300)
        rows.append(row(i, r))
        print()
    table(HEADERS, rows, "rrrrrr")
    return rows


def claude_explicit():
    model = MODELS["frontier"]
    # cache_control marks the END of the part to cache. Everything before it is cached.
    system = [{"type": "text", "text": f"[handbook copy {RUN}]\n" + SYSTEM, "cache_control": {"type": "ephemeral"}}]
    rows = []
    for i, q in enumerate(QUESTIONS, 1):
        r = chat(model, [{"role": "system", "content": system}, {"role": "user", "content": q}], max_tokens=300)
        rows.append(row(i, r))
        print()
    table(HEADERS, rows, "rrrrrr")
    note("Call 1 writes the cache (costs a bit more than normal). Calls 2 and 3 read it at a tenth of the price.")


def timestamp_mistake():
    model = MODELS["mid"]
    rows = []
    for i, q in enumerate(QUESTIONS[:2], 1):
        # The mistake: something that changes on every call, at the START of the prompt.
        system = f"Current time: {datetime.now().isoformat()}\n" + SYSTEM
        r = chat(model, [{"role": "system", "content": system}, {"role": "user", "content": q}],
                 max_tokens=300, show=False)
        rows.append(row(i, r))
        time.sleep(1)
    table(HEADERS, rows, "rrrrrr")
    if all(not r[2] for r in rows):
        bad("Nothing was cached: the first line was different, so the whole prefix was different.")
    say("Fix: put the timestamp (user name, retrieved documents, anything that changes) AFTER the static part,")
    say("for example in the user message. Static first, dynamic last.")


def at_scale():
    model = MODELS["mid"]
    prompt, out, calls = 3600, 60, 1_000_000
    no_cache = cost(model, prompt, out) * calls
    with_cache = cost(model, prompt, out, cached_tokens=3300) * calls
    table(["", f"1M calls on {model}"], [
        ["Without caching", money(no_cache)],
        ["With caching (~92% of the prompt cached)", money(with_cache)],
        ["Saved", f"{money(no_cache - with_cache)}  ({100 * (1 - with_cache / no_cache):.0f}%)"],
    ], "lr")
    print()
    say("Where caching pays off:")
    say("  • Long system prompts and few-shot examples       • Chat: the history grows, the start stays the same")
    say("  • Agents: tool definitions sent on every step     • Many questions about the same big document")
    note("Caches expire after a few minutes without use, so caching helps steady traffic more than rare requests.")


def response_cache():
    model = MODELS["small"]
    question = [{"role": "user", "content": f"In one sentence, what is BrightCart's return window for shoes? (run {RUN})"}]
    rows = []
    for i in (1, 2):
        r = chat(model, question, max_tokens=60, show=False, response_cache=True)
        rows.append([i, f"{r.seconds:.3f}s", r.text[:70]])
    table(["call", "latency", "answer"], rows, "rrl")
    say("\nThe second call never reached OpenAI: the LiteLLM proxy returned the stored answer.")
    table(["", "prompt caching", "response caching"], [
        ["Who does it", "the model provider", "you, or your gateway (LiteLLM, Redis)"],
        ["Reuses", "the processed prompt PREFIX", "the whole ANSWER"],
        ["Matches", "same start of prompt", "exactly the same request"],
        ["Answer", "freshly generated", "identical, stored copy"],
        ["Good for", "long shared context", "FAQs, repeated identical calls"],
    ])
    note("Semantic caching goes one step further: reuse the answer when a NEW question means the same (via embeddings, Layer 3).")


def main():
    heading("03.2 · Prompt caching")
    say(f"The system prompt is our {len(HANDBOOK):,}-character support handbook, about 3,600 tokens.")

    part("A", f"OpenAI ({MODELS['mid']}): automatic caching")
    rows = openai_auto()
    if any(r[2] for r in rows[1:]):
        ok("Calls 2 and 3 reused the cached handbook: cheaper, and usually faster. We changed nothing in the code.")
    else:
        note("No cache hit this time. Caching is best-effort; rerun and it usually appears.")

    part("B", f"Claude ({MODELS['frontier']}): explicit cache_control")
    claude_explicit()

    part("C", "the classic mistake")
    timestamp_mistake()

    part("D", "what it saves")
    at_scale()

    part("E", "response caching")
    response_cache()


if __name__ == "__main__":
    main()
