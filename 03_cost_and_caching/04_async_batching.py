"""04 · Async and batching: same work, less waiting, lower bills.

An LLM call spends almost all its time WAITING for the model. While one request waits,
your program could be sending others.

Part A  sequential: 12 tickets, one after another
Part B  async: all 12 at once with asyncio.gather
Part C  async, but polite: at most 4 at a time with a Semaphore (stays under rate limits)
Part D  batching: 12 tickets in ONE request, and the provider Batch APIs (50% off)

Run:  python 03_cost_and_caching/04_async_batching.py        (--auto to skip the pauses)
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.common import DATA, MODELS, achat, chat, heading, money, note, ok, part, say, table

MODEL = MODELS["small"]
SYSTEM = ("You route customer support tickets for an online shop. Reply with exactly one word: "
          "billing, account, delivery, technical or returns.")
TICKETS = json.loads((DATA / "support_tickets.json").read_text())


def messages(ticket):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ticket["text"]}]


def sequential():
    results = []
    for t in TICKETS:
        results.append(chat(MODEL, messages(t), max_tokens=5, show=False))
        print(".", end="", flush=True)
    print()
    return results


async def all_at_once():
    return await asyncio.gather(*(achat(MODEL, messages(t), max_tokens=5) for t in TICKETS))


async def limited(max_in_flight=4):
    gate = asyncio.Semaphore(max_in_flight)  # TRY: change to 1, 2, 8
    in_flight = 0
    peak = 0

    async def one(t):
        nonlocal in_flight, peak
        async with gate:  # waits here if 4 requests are already running
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                return await achat(MODEL, messages(t), max_tokens=5)
            finally:
                in_flight -= 1

    results = await asyncio.gather(*(one(t) for t in TICKETS))
    return results, peak


def one_big_request():
    listing = "\n".join(f"{t['id']}: {t['text']}" for t in TICKETS)
    prompt = ("Route each ticket. Reply with JSON only: an object mapping ticket id to one of "
              "billing, account, delivery, technical, returns.\n\n" + listing)
    r = chat(MODEL, [{"role": "user", "content": prompt}], max_tokens=400, show=False,
             response_format={"type": "json_object"})
    return r


def accuracy(labels):
    return sum(label.strip().lower().strip(".") == t["category"] for label, t in zip(labels, TICKETS))


def main():
    heading("03.4 · Async and batching")
    say(f"{len(TICKETS)} tickets to route with {MODEL}.")
    timings = []

    part("A", "sequential")
    start = time.perf_counter()
    seq = sequential()
    timings.append(["sequential", time.perf_counter() - start, len(TICKETS), sum(r.cost for r in seq), accuracy(r.text for r in seq)])

    part("B", "async: everything at once")
    start = time.perf_counter()
    res = asyncio.run(all_at_once())
    timings.append(["asyncio.gather (12 at once)", time.perf_counter() - start, len(TICKETS), sum(r.cost for r in res), accuracy(r.text for r in res)])

    part("C", "async with a Semaphore (max 4 at a time)")
    start = time.perf_counter()
    res, peak = asyncio.run(limited(4))
    timings.append([f"gather + Semaphore(4), peak {peak}", time.perf_counter() - start, len(TICKETS), sum(r.cost for r in res), accuracy(r.text for r in res)])

    part("D", "batching: all tickets in one request")
    start = time.perf_counter()
    big = one_big_request()
    try:
        labels = json.loads(big.text)
        correct = accuracy(labels.get(t["id"], "") for t in TICKETS)
    except json.JSONDecodeError:
        correct = 0
    timings.append(["one request, 12 tickets", time.perf_counter() - start, 1, big.cost, correct])

    print()
    table(["approach", "time", "requests", "cost", "correct"],
          [[a, f"{t:.2f}s", n, money(c), f"{ok_}/{len(TICKETS)}"] for a, t, n, c, ok_ in timings], "lrrrr")
    fastest_parallel = min(timings[1][1], timings[2][1])
    ok(f"Async was about {timings[0][1] / fastest_parallel:.0f}× faster than sequential, for the same cost.")
    note("Batching in one prompt sends the instructions once instead of 12 times. Fewer requests also means")
    note("fewer chances to hit the rate limit. The trade-off: one bad answer can spoil the whole batch.")

    say("\nProvider Batch APIs (for work that can wait, e.g. overnight reports, evals, back-filling data):")
    print('''
    # Upload a JSONL file with one request per line, get results within 24 hours, pay ~50% less.
    batch_file = client.files.create(file=open("requests.jsonl", "rb"), purpose="batch")
    client.batches.create(input_file_id=batch_file.id, endpoint="/v1/chat/completions",
                          completion_window="24h")
    # Anthropic: client.messages.batches.create(requests=[...])   Gemini: batch mode, also ~50% off
''')
    say("Which to use?")
    table(["need", "use"], [
        ["A user is waiting for the answer", "async + Semaphore, streaming"],
        ["Many small items, one user waiting", "batch several items into one prompt"],
        ["Thousands of items, results by tomorrow", "provider Batch API (≈50% cheaper)"],
    ])


if __name__ == "__main__":
    main()
