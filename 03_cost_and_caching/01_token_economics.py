"""01 · Token economics: where the money goes. No API calls.

You pay per token, in and out. Output tokens cost 2 to 8 times more than input tokens.
A prototype that costs a few cents a day can cost thousands a month in production.

Part A  what a token is, and why some text costs more than other text
Part B  the anatomy of one chatbot request: which part is expensive?
Part C  the same request at 10k, 100k and 1M requests a month, on each model
Part D  the levers you can pull, and roughly how much each saves

Run:  python 03_cost_and_caching/01_token_economics.py        (--auto to skip the pauses)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.common import DATA, DIM, RESET, count_tokens, heading, money, note, part, say, table
from lab.pricing import PRICES, cost

SAMPLES = [
    ("English", "Where is my refund for order 88231?"),
    ("Hindi", "मेरे ऑर्डर 88231 का रिफंड कहाँ है?"),
    ("Malayalam", "എന്റെ ഓർഡർ 88231-ന്റെ റീഫണ്ട് എവിടെ?"),
    ("JSON", '{"order_id": 88231, "status": "refund_pending", "amount_rs": 2349}'),
    ("Python", "def refund(order_id: int) -> bool:\n    return api.post(f'/refunds/{order_id}').ok"),
]


def show_tokens(text: str):
    try:
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        pieces = [enc.decode([t]) for t in enc.encode(text)]
        return " ".join(f"[{p}]" for p in pieces)
    except Exception:
        return "(install tiktoken to see the pieces)"


def main():
    heading("03.1 · Token economics")

    part("A", "what a token is")
    rows = []
    for name, text in SAMPLES:
        rows.append([name, len(text), count_tokens(text), f"{len(text) / count_tokens(text):.1f}"])
    table(["text", "characters", "tokens", "chars per token"], rows, "lrrr")
    say(f"\nEnglish, split into tokens:\n  {DIM}{show_tokens(SAMPLES[0][1])}{RESET}")
    note("\nRule of thumb: 1 token ≈ 4 English characters ≈ ¾ of a word.")
    note("Indian languages often take 1.5 to 3× more tokens for the same meaning, so the same answer costs more.")
    note("Each provider has its own tokenizer: the same text is a different number of tokens on Claude or Gemini.")

    part("B", "anatomy of one chatbot request")
    handbook = (DATA / "handbook.md").read_text()
    parts = [
        ("System prompt (rules, persona)", 600, "input"),
        ("Retrieved documents (RAG, 5 chunks)", 2500, "input"),
        ("Conversation history (6 turns)", 1200, "input"),
        ("The user's question", 40, "input"),
        ("The model's answer", 350, "output"),
    ]
    model = "openai/gpt-5.4-mini"
    p = PRICES[model]
    rows = []
    total = 0.0
    for name, tokens, kind in parts:
        c = tokens * p[kind] / 1e6
        total += c
        rows.append([name, tokens, kind, money(c)])
    table(["part", "tokens", "type", f"cost on {model}"], rows, "lrlr")
    say(f"\nOne request ≈ {money(total)}. The question is 1% of the tokens; the context is ~93%.")
    say(f"Output: 8% of the tokens but {350 * p['output'] / 1e6 / total:.0%} of the cost "
        f"(output is {p['output'] / p['input']:.0f}× the price of input on this model).")
    note(f"For scale: our whole support handbook is {count_tokens(handbook):,} tokens.")

    part("C", "the same request, at scale, on each model")
    tokens_in = sum(t for _, t, k in parts if k == "input")
    tokens_out = sum(t for _, t, k in parts if k == "output")
    rows = []
    for m in PRICES:
        one = cost(m, tokens_in, tokens_out)
        rows.append([m, money(one), money(one * 10_000), money(one * 100_000), money(one * 1_000_000)])
    table(["model", "1 request", "10k/month", "100k/month", "1M/month"], rows, "lrrrr")
    note("\nSame app, same traffic: the model choice alone changes the bill by about 100×.")

    part("D", "the levers")
    base = cost(model, tokens_in, tokens_out)
    levers = [
        ("Use a smaller model (gpt-5.4-mini → gpt-4.1-nano)", cost("openai/gpt-4.1-nano", tokens_in, tokens_out)),
        ("Retrieve 3 chunks instead of 5", cost(model, tokens_in - 1000, tokens_out)),
        ("Summarise old history (1200 → 300 tokens)", cost(model, tokens_in - 900, tokens_out)),
        ("Cap the answer length (350 → 150 tokens)", cost(model, tokens_in, 150)),
        ("Prompt caching (system prompt cached)", cost(model, tokens_in, tokens_out, cached_tokens=600)),
        ("Batch API for non-urgent work (50% off)", base * 0.5),
    ]
    rows = [[name, money(c * 1_000_000), f"-{100 * (1 - c / base):.0f}%"] for name, c in levers]
    table(["lever", "per 1M requests", "saving"], [["Starting point", money(base * 1_000_000), ""]] + rows, "lrr")
    note("\nThe levers stack. Always measure quality after each one: cheaper is only better if it still works.")


if __name__ == "__main__":
    main()
