"""01 · Prompt, RAG or fine-tune? A quiz for the room. No API calls.

The ladder, cheapest and fastest first:
  1. Prompting   = giving instructions          changes: what you ask
  2. RAG         = handing over the manual      changes: what the model can look up
  3. Fine-tuning = sending someone on a course  changes: the model's weights, so how it behaves by default

Rule of thumb: RAG for KNOWLEDGE (facts, documents, anything that changes).
               Fine-tuning for BEHAVIOUR (style, format, a narrow skill), at scale.
               Prompting first, always.

Part A  the quiz: for each scenario, the room shouts P, R, F or "R+F"
Part B  the cheat sheet

Run:  python 02_fine_tuning_concepts/01_prompt_rag_or_finetune.py     (--auto prints the answers straight away)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.common import AUTO, BOLD, GREEN, RESET, heading, part, say, table

SCENARIOS = [
    ("A team of 10 wants meeting notes summarised into action items.",
     "P", "Any decent model does this from a clear instruction. Nothing to look up, nothing to train."),
    ("A bot must answer questions about our 2,000-page internal wiki, which changes every week.",
     "R", "This is knowledge, and it changes. RAG fetches today's version; a fine-tuned model would be out of date by next week."),
    ("Users ask about a product we launched yesterday.",
     "R", "Fine-tuning is a poor way to teach new facts: it's slow, and the model may still hallucinate details. Put the facts in the context."),
    ("Extract invoice fields into JSON for our accounting system.",
     "P", "Prompting plus structured outputs (Layer 2) handles this. Fine-tune only if you've hit a ceiling on accuracy."),
    ("Every reply must follow our brand voice and a fixed 4-part structure. We have 5,000 great example replies, "
     "and a 3,000-token few-shot prompt is sent 2 million times a month.",
     "F", "Behaviour and format, at scale, with lots of good examples. Fine-tuning bakes the examples into the weights, "
          "so you stop paying for them on every call (see 02_few_shot_tax.py)."),
    ("Classify radiology reports into 40 internal codes. Prompting reaches 78% accuracy, we have 20,000 labelled "
     "reports, and it must run on a small model inside the hospital.",
     "F", "A narrow, repetitive task with lots of labelled data and a small on-prem model: the classic case for LoRA "
          "fine-tuning an open-weight model."),
    ("A support bot must quote current policies AND sound exactly like our best agents, cheaply, at high volume.",
     "R+F", "Knowledge comes from RAG (policies change), behaviour comes from fine-tuning (tone, structure). They combine well."),
    ("A startup is building a prototype and has no example data yet.",
     "P", "No data means no fine-tuning. Prompt first, log real conversations, and those logs become your training data later."),
]

LABELS = {"P": "Prompting", "R": "RAG", "F": "Fine-tuning", "R+F": "RAG + fine-tuning"}


def main():
    heading("01 · Prompt, RAG or fine-tune?")
    say("For each scenario, pick one: P = prompting, R = RAG, F = fine-tuning, R+F = both.")

    part("A", "the quiz")
    score = 0
    for i, (scenario, answer, why) in enumerate(SCENARIOS, 1):
        say(f"\n{BOLD}{i}. {scenario}{RESET}")
        if not AUTO and sys.stdin.isatty():
            guess = input("   Your answer (P / R / F / R+F): ").strip().upper()
            score += guess == answer
        say(f"   {GREEN}→ {LABELS[answer]}.{RESET} {why}")
    if not AUTO and sys.stdin.isatty():
        say(f"\nYou got {score}/{len(SCENARIOS)}.")

    part("B", "the cheat sheet")
    table(
        ["", "Prompting", "RAG", "Fine-tuning"],
        [
            ["Analogy", "Give instructions", "Hand over the manual", "Send on a training course"],
            ["Changes", "The input", "The input (adds documents)", "The model's weights"],
            ["Good for", "Almost everything, first", "Facts, documents, freshness", "Style, format, narrow skills"],
            ["Bad for", "Huge rule sets", "Changing behaviour", "Adding or updating facts"],
            ["Needs", "A clear prompt", "Documents + a pipeline", "Hundreds+ of good examples"],
            ["Time to try", "Minutes", "Days", "Days to weeks"],
            ["Cost", "Per call", "Per call + vector DB", "Training + hosting/inference"],
            ["Update", "Edit the prompt", "Re-index the documents", "Retrain"],
        ],
    )
    say("\nBefore fine-tuning, ask:")
    say("  1. Have I really pushed prompting (clear instructions, examples, structured outputs)?")
    say("  2. Is the problem missing KNOWLEDGE? Then it's RAG, not fine-tuning.")
    say("  3. Do I have hundreds to thousands of high-quality examples, and an eval to prove it helped?")
    say("  4. Will I retrain when the base model is upgraded?")


if __name__ == "__main__":
    main()
