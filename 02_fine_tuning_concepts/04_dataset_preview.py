"""04 · What a fine-tuning dataset looks like. No training, no API calls.

Fine-tuning data is a file of example conversations: the input, and the exact output you
wish the model had given. Quality matters far more than quantity.

Part A  the format: one JSON object per line (JSONL)
Part B  validate it: the checks every provider runs, and some they don't
Part C  size and cost: tokens, epochs, and what training would cost
Part D  what the API call would look like

Run:  python 02_fine_tuning_concepts/04_dataset_preview.py        (--auto to skip the pauses)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.common import DATA, bad, count_tokens, heading, money, note, ok, part, say, table

FILE = DATA / "finetune_sample.jsonl"
TRAINING_PRICE = 1.50  # USD per 1M training tokens, gpt-4.1-nano (illustrative, October 2026)

# Examples of the mistakes that ruin fine-tuning runs.
BROKEN = [
    {"messages": [{"role": "user", "content": "Where is my order?"}]},                                   # no answer
    {"messages": [{"role": "user", "content": "Refund?"}, {"role": "assistant", "content": ""}]},        # empty answer
    {"messages": [{"role": "customer", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]},   # bad role
    {"messages": [{"role": "user", "content": "Can I return my phone?"},
                  {"role": "assistant", "content": "Sure, you have 30 days!!! 😀"}]},                     # wrong policy AND wrong voice
]


def check(example: dict) -> list[str]:
    problems = []
    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        return ["no 'messages' list"]
    for m in messages:
        if m.get("role") not in ("system", "user", "assistant"):
            problems.append(f"unknown role '{m.get('role')}'")
        if not str(m.get("content", "")).strip():
            problems.append(f"empty {m.get('role')} message")
    if messages[-1].get("role") != "assistant":
        problems.append("last message must be the assistant's answer")
    answer = messages[-1].get("content", "")
    # Checks no provider runs for you: does the answer match YOUR policy and voice?
    if "30 days" in answer and "return" in json.dumps(messages).lower():
        problems.append("policy: phones have a 10-day return window, not 30 (the model will learn the mistake)")
    if "!!" in answer or "😀" in answer:
        problems.append("voice: brand guide says one exclamation mark at most and no emoji")
    return problems


def main():
    heading("04 · A fine-tuning dataset")
    rows = [json.loads(line) for line in FILE.read_text().splitlines() if line.strip()]

    part("A", "the format")
    say(f"{FILE.relative_to(DATA.parent)}: {len(rows)} examples, one JSON object per line. The first one:\n")
    print(json.dumps(rows[0], indent=2, ensure_ascii=False))
    note("\nEach example: (optional) system prompt → user message → the ideal assistant reply.")
    note("The model learns to produce replies like the assistant messages. Every one of them is a lesson.")

    part("B", "validate before you train")
    for i, row in enumerate(rows, 1):
        problems = check(row)
        if problems:
            bad(f"example {i}: {'; '.join(problems)}")
    ok(f"All {len(rows)} examples in the file pass.")
    say("\nNow some broken ones, the kind you find in real exported chat logs:")
    for i, row in enumerate(BROKEN, 1):
        for p in check(row):
            bad(f"broken {i}: {p}")
    note("\nGarbage in, garbage out, but permanently: a fine-tuned model can't 'unsee' bad examples.")

    part("C", "how big, and how much?")
    tokens = [sum(count_tokens(m["content"]) for m in row["messages"]) for row in rows]
    avg = sum(tokens) / len(tokens)
    say(f"Average example: {avg:.0f} tokens. This file: {sum(tokens):,} tokens.\n")
    epochs = 3
    table_rows = []
    for n in (len(rows), 100, 1_000, 10_000):
        trained = n * avg * epochs
        table_rows.append([f"{n:,}", f"{trained:,.0f}", money(trained * TRAINING_PRICE / 1e6)])
    table(["examples", f"tokens trained ({epochs} epochs)", "training cost"], table_rows, "rrr")
    print()
    say("How many examples do you need?")
    say("  • 10 to 50: the minimum most providers accept; enough to see whether it's working.")
    say("  • 100 to 1,000: typical for style and format.")
    say("  • 1,000 to 10,000+: narrow skills such as classification into many labels.")
    say("Always keep a separate test set (say 10 to 20%) the model never trains on, and compare before/after.")

    part("D", "what the API call looks like (not run)")
    print('''
    from openai import OpenAI
    client = OpenAI()

    file = client.files.create(file=open("finetune_sample.jsonl", "rb"), purpose="fine-tune")
    job = client.fine_tuning.jobs.create(model="gpt-4.1-nano-2025-04-14", training_file=file.id)
    # ...minutes to hours later...
    client.fine_tuning.jobs.retrieve(job.id).fine_tuned_model
    # -> "ft:gpt-4.1-nano-2025-04-14:brightcart::abc123"  ← use this like any other model id
''')
    note("Our LiteLLM proxy key can only call models, so we don't run this today.")
    note("Open-source route: the same JSONL works with Hugging Face TRL / Unsloth on a single GPU (LoRA/QLoRA).")


if __name__ == "__main__":
    main()
