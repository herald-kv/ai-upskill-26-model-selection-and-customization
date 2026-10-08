"""04 · Capstone: a model router. Everything from today in one small program.

A cheap model reads every ticket first and decides how hard it is (Part 1: right model for the job).
Simple tickets are answered by the cheap model; complex ones go to the frontier model.
Both answer from the handbook, which is cached (Part 3: caching). Calls run concurrently (Part 3: async).
The style comes from the handbook's examples; a fine-tuned small model could replace them (Part 2).

Part A  triage: the small model labels each ticket simple or complex
Part B  answer: simple → small model, complex → frontier model
Part C  the bill: router vs "always use the frontier model"

Run:  python 04_capstone_model_router.py        (--auto to skip the pauses)
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lab.common import DATA, MODELS, achat, heading, money, note, ok, part, say, short, table

CHEAP, STRONG = MODELS["small"], MODELS["frontier"]
TICKETS = json.loads((DATA / "support_tickets.json").read_text())
HANDBOOK = (DATA / "handbook.md").read_text()

TRIAGE = """Decide how hard this support ticket is to answer well.
"simple": one clear question with a direct answer in a policy.
"complex": several issues at once, money disputes, security, business accounts, or an upset customer.
Reply with JSON only: {"complexity": "simple" | "complex", "reason": "<five words>"}"""

ANSWER_SYSTEM = ("You are BrightCart's support assistant. Follow the handbook's tone of voice and reply "
                 "structure, and only state policies that are in it. Keep replies under 80 words.\n\n" + HANDBOOK)


def answer_messages(model, ticket):
    if model.startswith("anthropic/"):
        system = [{"type": "text", "text": ANSWER_SYSTEM, "cache_control": {"type": "ephemeral"}}]
    else:
        system = ANSWER_SYSTEM  # OpenAI caches automatically
    return [{"role": "system", "content": system}, {"role": "user", "content": ticket["text"]}]


async def triage(ticket):
    r = await achat(CHEAP, [{"role": "system", "content": TRIAGE}, {"role": "user", "content": ticket["text"]}],
                    max_tokens=60, response_format={"type": "json_object"})
    try:
        label = json.loads(r.text).get("complexity", "complex")
    except json.JSONDecodeError:
        label = "complex"  # when unsure, use the strong model
    return label, r


async def answer_all(pairs, concurrency=4):
    gate = asyncio.Semaphore(concurrency)

    async def one(model, ticket):
        async with gate:
            return await achat(model, answer_messages(model, ticket), max_tokens=250)

    # Warm the cache with one call per model first, so the rest can read it.
    first = {}
    results = [None] * len(pairs)
    for i, (model, ticket) in enumerate(pairs):
        if model not in first:
            first[model] = i
            results[i] = await one(model, ticket)
    rest = [i for i in range(len(pairs)) if results[i] is None]
    for i, r in zip(rest, await asyncio.gather(*(one(*pairs[i]) for i in rest))):
        results[i] = r
    return results


async def run():
    triaged = await asyncio.gather(*(triage(t) for t in TICKETS))
    routed = [(CHEAP if label == "simple" else STRONG, t) for (label, _), t in zip(triaged, TICKETS)]
    answers = await answer_all(routed)
    baseline = await answer_all([(STRONG, t) for t in TICKETS])
    return triaged, routed, answers, baseline


def main():
    heading("04 · Capstone: a model router")
    say(f"Cheap model: {CHEAP}     Strong model: {STRONG}")
    say(f"{len(TICKETS)} tickets. Running triage, routed answers, and an 'always strong' baseline…")
    triaged, routed, answers, baseline = asyncio.run(run())

    part("A", "triage")
    rows = []
    for (label, r), t in zip(triaged, TICKETS):
        agree = "✓" if label == t["complexity"] else "✗"
        rows.append([t["id"], short(t["text"], 60), label, f"{agree} (we said {t['complexity']})"])
    table(["ticket", "text", "router says", "matches our label?"], rows)
    note("Triage by a small model costs a fraction of a cent. When it's unsure, it should send the ticket up, not down.")

    part("B", "answers")
    for (model, t), r in zip(routed, answers):
        who = "small" if model == CHEAP else "STRONG"
        say(f"\n[{t['id']} → {who}] {short(t['text'], 90)}")
        say(f"   {short(r.text, 300)}")
        note(f"   {r.seconds:.1f}s · {money(r.cost)}" + (f" · cached {r.cached_tokens}" if r.cached_tokens else ""))
    print()
    note("Read the small model's answers critically: did it invent a customer name, a reference number or a step")
    note("that isn't in the handbook? Did triage send a security or money problem to the small model?")
    note("Spotting this automatically, on every release, is what evaluation is for (Layer 6).")

    part("C", "the bill")
    triage_cost = sum(r.cost for _, r in triaged)
    routed_cost = triage_cost + sum(r.cost for r in answers)
    strong_cost = sum(r.cost for r in baseline)
    n_strong = sum(1 for m, _ in routed if m == STRONG)
    share_strong = n_strong / len(TICKETS)

    def warm_average(pairs, results, model):
        # In production the cache is almost always warm, so skip the one call that wrote it.
        costs = [r.cost for (m, _), r in zip(pairs, results) if m == model][1:]
        return sum(costs) / len(costs) if costs else 0.0

    baseline_pairs = [(STRONG, t) for t in TICKETS]
    per_router = (triage_cost / len(TICKETS)
                  + (1 - share_strong) * warm_average(routed, answers, CHEAP)
                  + share_strong * warm_average(routed, answers, STRONG))
    per_strong = warm_average(baseline_pairs, baseline, STRONG)
    table(["approach", f"this run ({len(TICKETS)} tickets)", "per 1M tickets (cache warm)"], [
        [f"Router ({len(TICKETS) - n_strong} small, {n_strong} strong, + triage)", money(routed_cost), money(per_router * 1e6)],
        ["Always the strong model", money(strong_cost), money(per_strong * 1e6)],
    ], "lrr")
    if per_router < per_strong:
        ok(f"At scale the router costs {100 * (1 - per_router / per_strong):.0f}% less, and hard tickets still get the strong model.")
    note("This run includes one cache write per model, which costs extra. At 1M tickets that's a rounding error.")
    print()
    say("What you'd add before production (Layer 6):")
    say("  • An eval: are the small model's answers to 'simple' tickets actually good enough?")
    say("  • Monitoring: how often does triage get it wrong, and what does each route cost per day?")
    note("\n# TRY: make the router stricter (edit TRIAGE) or swap CHEAP for MODELS['mid'] and compare the bill.")


if __name__ == "__main__":
    main()
