"""01 · Model bake-off: the same tasks on five models, side by side.

Leaderboards tell you which model is best on average. You need to know which model is good
enough for *your* task at the lowest cost and latency. The only way to know is to try them.

Part A  one ticket, five models: see the answers, the speed and the price
Part B  the scoreboard, and what it costs at 1 million requests a month

Run:  python 01_choosing_models/01_model_bakeoff.py        (--auto to skip the pauses)
"""

import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.common import (DATA, MODELS, achat, ask, bad, heading, money, note, ok, part, say,
                        table)

# TRY: swap a model for any other id on the proxy (python setup_check.py --list)
CONTESTANTS = [MODELS["small"], MODELS["mid"], MODELS["frontier"], MODELS["google"], MODELS["open"]]

CATEGORIES = ["billing", "account", "delivery", "technical", "returns"]
CLASSIFY_SYSTEM = (
    "You route customer support tickets for an online shop. "
    f"Reply with exactly one word from this list: {', '.join(CATEGORIES)}."
)

EMAIL = """Hi team, this is Priya Nair. My order BC-90412 was charged Rs 3,499 but I only
got the charger, not the phone case. Can you send the case or refund it? Thanks, Priya"""
EXTRACT_PROMPT = (
    "Extract these fields from the email as JSON with keys "
    '"name", "order_id", "amount_rs" (a number), "missing_item". Reply with JSON only.\n\n' + EMAIL
)
EXPECTED_JSON = {"name": "Priya Nair", "order_id": "BC-90412", "amount_rs": 3499, "missing_item": "phone case"}

REASONING = [
    # Counting letters is hard for LLMs because they see tokens, not letters (Layer 1).
    ("How many times does the letter r appear in the phrase 'refrigerator repair request'? "
     "Reply with just the number.", "7"),
    # A small policy calculation: proportional refund on a combo offer.
    ("A combo of three items with list prices Rs 1000, Rs 2000 and Rs 3000 was sold for Rs 4800. "
     "The customer returns only the Rs 2000 item. The refund is that item's share of the combo price, "
     "in proportion to list price. How many rupees is the refund? Reply with just the number.", "1600"),
]


def tickets():
    return json.loads((DATA / "support_tickets.json").read_text())


def parse_json(text: str):
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def last_number(text: str) -> str:
    """Some models ignore "reply with just the number" and explain first, so score the last number."""
    numbers = re.findall(r"\d[\d,]*", text)
    return numbers[-1].replace(",", "") if numbers else ""


def json_score(data) -> bool:
    if not isinstance(data, dict):
        return False
    try:
        return (
            data.get("name", "").lower() == "priya nair"
            and data.get("order_id", "").upper() == "BC-90412"
            and float(data.get("amount_rs")) == 3499
            and "case" in data.get("missing_item", "").lower()
        )
    except (TypeError, ValueError):
        return False


async def run_model(model: str) -> dict:
    """All the eval calls for one model, sent concurrently (you'll see how in Part 3)."""
    ts = tickets()
    calls = [
        achat(model, [{"role": "system", "content": CLASSIFY_SYSTEM}, {"role": "user", "content": t["text"]}],
              max_tokens=1000)
        for t in ts
    ]
    calls.append(achat(model, [{"role": "user", "content": EXTRACT_PROMPT}], max_tokens=1000))
    calls += [achat(model, [{"role": "user", "content": q}], max_tokens=2000) for q, _ in REASONING]
    results = await asyncio.gather(*calls, return_exceptions=True)

    errors = [r for r in results if isinstance(r, Exception)]
    good = [r for r in results if not isinstance(r, Exception)]
    class_results, json_result, reason_results = results[: len(ts)], results[len(ts)], results[len(ts) + 1:]

    def text(r):
        return "" if isinstance(r, Exception) else r.text.strip().lower().strip(".")

    correct = sum(text(r) == t["category"] for r, t in zip(class_results, ts))
    misses = [(t["id"], t["category"], text(r)) for r, t in zip(class_results, ts) if text(r) != t["category"]]
    reasoning_ok = sum(last_number(text(r)) == ans for r, (_, ans) in zip(reason_results, REASONING))
    reasoning_answers = [last_number(text(r)) or "?" for r in reason_results]
    return dict(
        model=model,
        classify=correct,
        misses=misses,
        json=json_score(parse_json(text(json_result))) if not isinstance(json_result, Exception) else False,
        reasoning=reasoning_ok,
        reasoning_answers=reasoning_answers,
        seconds=sum(r.seconds for r in good) / max(len(good), 1),
        cost=sum(r.cost for r in good),
        calls=len(good),
        errors=errors,
    )


async def run_all():
    return await asyncio.gather(*(run_model(m) for m in CONTESTANTS))


def main():
    heading("01 · Model bake-off: which model is good enough for this job?")
    say("We're building a support assistant for BrightCart, a (fictional) online shop.")
    say("Contestants:")
    for m in CONTESTANTS:
        say(f"  • {m}")

    part("A", "one ticket, five models")
    ticket = tickets()[5]  # the messy coupon/store credit one
    say(f"Ticket {ticket['id']} (correct category: {ticket['category']}):\n  \"{ticket['text']}\"\n")
    for model in CONTESTANTS:
        ask(model, ticket["text"], system=CLASSIFY_SYSTEM, max_tokens=1000)
        print()
    note("Same question, but very different speed and price.")
    note("Notice Gemini's 'thinking' tokens: reasoning models think before they answer, and you pay for it.")

    part("B", "the scoreboard")
    say("Sending 15 requests to each model, all at once…")
    scores = asyncio.run(run_all())
    for s in scores:
        if s["errors"]:
            bad(f"{s['model']}: {len(s['errors'])} calls failed, e.g. {str(s['errors'][0])[:120]}")
    rows = []
    for s in scores:
        per_request = s["cost"] / max(s["calls"], 1)
        rows.append([
            s["model"],
            f"{s['classify']}/12",
            "✓" if s["json"] else "✗",
            f"{s['reasoning']}/2",
            f"{s['seconds']:.2f}s",
            money(s["cost"]),
            money(per_request * 1_000_000),
        ])
    table(["model", "routing", "JSON", "reasoning", "avg latency", "cost (15 calls)", "at 1M req/month"],
          rows, "lllrrrr")
    print()
    good_router = min((s for s in scores if s["classify"] >= 11 and s["json"]), key=lambda s: s["cost"], default=None)
    all_correct = min((s for s in scores if s["classify"] == 12 and s["json"] and s["reasoning"] == 2),
                      key=lambda s: s["cost"], default=None)
    if good_router:
        ok(f"If the job is routing and extraction: {good_router['model']} is good enough and cheapest.")
    if all_correct:
        ok(f"If the job also needs reasoning: {all_correct['model']} is the cheapest that got everything right.")
    else:
        bad("No model got everything right. Time for a bigger model, or a better prompt.")
    note("Results change from run to run (Layer 1: non-determinism). Real evals use hundreds of cases.")
    say("\nTakeaways")
    say("  1. On simple tasks (routing, extraction), small models often match frontier ones for a fraction of the price.")
    say("  2. The differences show up on reasoning, messy inputs and edge cases. Test on YOUR hard cases.")
    say("  3. Price per request × requests per month is the number your manager will ask about.")
    note("\n# TRY: add a ticket that you think is ambiguous to data/support_tickets.json and rerun.")
    note("# TRY: replace a contestant with a bigger model, e.g. openai/gpt-5.5 or anthropic/claude-opus-4-6.")


if __name__ == "__main__":
    main()
