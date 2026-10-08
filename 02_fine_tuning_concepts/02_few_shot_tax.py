"""02 · The few-shot tax: when does fine-tuning pay for itself?

Few-shot prompting teaches a model a style by showing examples in the prompt. It works, but you
send (and pay for) those examples on every single call. Fine-tuning moves the examples into the
model's weights once, so the prompt shrinks.

Part A  zero-shot: the model's default style
Part B  few-shot: 8 examples of BrightCart's voice in the prompt. Better, but count the tokens.
Part C  the maths: few-shot vs few-shot with caching vs a fine-tuned model, per month

Run:  python 02_fine_tuning_concepts/02_few_shot_tax.py        (--auto to skip the pauses)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.common import DATA, MODELS, chat, heading, money, note, part, say, table

MODEL = MODELS["small"]  # gpt-4.1-nano: cheap, and OpenAI can fine-tune it

SYSTEM = "You are BrightCart's support assistant. Reply to the customer."
QUESTION = "[Customer: Priya] I returned my shoes a week ago but haven't got my money back. Paid by card."

# OpenAI fine-tuning prices for gpt-4.1-nano, USD per 1M tokens (October 2026, illustrative).
FT = dict(training=1.50, input=0.20, output=0.80)   # a fine-tuned model costs ~2x the base model per token
BASE = dict(input=0.10, output=0.40, cached=0.025)


def examples(n: int) -> list[dict]:
    rows = [json.loads(line) for line in (DATA / "finetune_sample.jsonl").read_text().splitlines()]
    messages = []
    for row in rows[:n]:
        messages += [m for m in row["messages"] if m["role"] != "system"]
    return messages


def main():
    heading("02 · The few-shot tax")
    say(f"Model: {MODEL}. Task: reply in BrightCart's voice (brief, first name, answer first, reference number).")

    part("A", "zero-shot: no examples")
    zero = chat(MODEL, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}], max_tokens=300)
    note("Polite, but generic: long, maybe a sign-off, no reference number. Not our voice.")

    part("B", "few-shot: 8 examples in the prompt")
    shots = examples(8)
    few = chat(MODEL, [{"role": "system", "content": SYSTEM}, *shots, {"role": "user", "content": QUESTION}],
               max_tokens=300)
    note("Now it sounds like us. The examples did the teaching.")
    note("But check the facts: any refund time or process it mentions came from the examples or was made up.")
    note("Examples teach STYLE. Facts must come from RAG or the prompt (the handbook says cards take 5 to 7 working days).")
    extra = few.input_tokens - zero.input_tokens
    say(f"\nThe examples added {extra} input tokens. You pay for them on EVERY call.")

    part("C", "the maths: when does fine-tuning pay off?")
    # TRY: change the volume. At low volume, few-shot wins easily.
    calls_per_month = 2_000_000
    prompt_zero, out = zero.input_tokens, few.output_tokens
    prompt_few = few.input_tokens

    def monthly(prompt_tokens, price_in, price_out, cached_tokens=0, price_cached=0.0):
        fresh = prompt_tokens - cached_tokens
        return calls_per_month * (fresh * price_in + cached_tokens * price_cached + out * price_out) / 1e6

    few_cost = monthly(prompt_few, BASE["input"], BASE["output"])
    few_cached = monthly(prompt_few, BASE["input"], BASE["output"], cached_tokens=extra, price_cached=BASE["cached"])
    ft_cost = monthly(prompt_zero, FT["input"], FT["output"])
    # Training: 2,000 examples × ~120 tokens × 3 epochs
    train_tokens = 2000 * 120 * 3
    train_cost = train_tokens * FT["training"] / 1e6

    say(f"{calls_per_month:,} calls a month, {out} output tokens each:\n")
    table(
        ["approach", "prompt tokens", "per month", "one-off"],
        [
            ["Few-shot (8 examples)", prompt_few, money(few_cost), "-"],
            ["Few-shot + prompt caching", f"{prompt_few} ({extra} cached)", money(few_cached), "-"],
            ["Fine-tuned model, no examples", prompt_zero, money(ft_cost), money(train_cost) + " training"],
        ],
        "lrrr",
    )
    print()
    say("Reading the table:")
    say("  • Fine-tuned models cost more per token, but the prompt is much shorter.")
    say("  • Prompt caching (Part 3) shrinks the few-shot tax a lot. Try it before fine-tuning.")
    say("    (OpenAI only caches prompts of 1,024+ tokens. These 8 short examples are under that;")
    say("    a real prompt with more examples and rules would qualify.)")
    say("  • Training itself is cheap. Collecting and cleaning thousands of good examples is not.")
    say("  • The money is rarely the whole story. Fine-tuning also means collecting data, evaluating,")
    say("    and retraining when the base model is upgraded. It's a project, not a setting.")
    say("  • Fine-tuning is worth it when it improves QUALITY you couldn't reach with prompting,")
    say("    or when a small fine-tuned model can replace a big expensive one.")
    note("\n# TRY: set calls_per_month = 50_000 and rerun Part C.")


if __name__ == "__main__":
    main()
