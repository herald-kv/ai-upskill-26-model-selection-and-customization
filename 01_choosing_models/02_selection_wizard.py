"""02 · Model selection wizard: turn requirements into a shortlist. No API calls.

Choosing a model is mostly answering five questions in the right order:
  1. Can the data leave our network?        → hosted API, private cloud, or self-hosted
  2. What goes in and out? (text, images, audio, code)  → which model families qualify
  3. How hard is the task?                   → small, mid or frontier tier
  4. How fast must it answer?                → latency budget
  5. How many requests, and what budget?     → cost per month, and whether self-hosting pays

Part A  three example projects, run through the wizard
Part B  your project: answer the questions yourself
Part C  hosted vs self-hosted: the "taxi vs owning a car" maths

Run:  python 01_choosing_models/02_selection_wizard.py        (--auto runs Part A and C only)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.common import AUTO, BOLD, DIM, GREEN, RESET, heading, money, note, part, say, table
from lab.pricing import GPU_HOURLY, PRICES

# Example model families per tier, October 2026. Names change fast; the tiers don't.
TIERS = {
    "small": "GPT-4.1 nano / GPT-5.4 nano, Claude Haiku 4.5, Gemini Flash-Lite, Llama 3.1 8B, Qwen3 8B, Mistral Small",
    "mid": "GPT-5.4 mini, Gemini 2.5/3 Flash, DeepSeek V3/V4, Llama 4 Maverick, Qwen3 32B, Mistral Medium",
    "frontier": "GPT-5.5, Claude Sonnet / Opus, Gemini 3 Pro, DeepSeek V4 Pro, Qwen3 Max",
}
OPEN_WEIGHT = "Llama 4, Qwen3 / Qwen3-VL, Mistral / Ministral, DeepSeek V3/V4, Gemma 4, gpt-oss"
MODALITY = {
    "text": "any model",
    "images": "vision models: GPT-5.x, Claude, Gemini; open: Qwen3-VL, Llama 4, Gemma 4",
    "audio": "speech-to-text first (Whisper, gpt-4o-transcribe, Gemini audio), then any text model",
    "code": "coding-tuned models: GPT-5.x Codex, Claude Sonnet/Opus; open: Qwen3-Coder, Devstral, Codestral",
    "long documents": "long context models: Gemini (1M), GPT-4.1/5.5 (1M), Claude Sonnet 4.6 (1M); open: Llama 4 Scout, Qwen3; or use RAG (Layer 3)",
}

QUESTIONS = [
    ("privacy", "Can the data be sent to an external API?",
     {"yes": "Yes, normal business data", "region": "Only to a provider in our region/cloud account (e.g. Azure, Bedrock, Vertex)",
      "no": "No, it must stay on our own servers"}),
    ("modality", "What does the model need to read?",
     {"text": "Text", "images": "Images / screenshots / scans", "audio": "Audio / calls",
      "code": "Code", "long documents": "Very long documents"}),
    ("difficulty", "How hard is the task?",
     {"easy": "Classify, extract, summarise, rewrite", "medium": "Answer from documents, write emails, simple tools",
      "hard": "Multi-step reasoning, agents, complex code, high stakes"}),
    ("latency", "How fast must it answer?",
     {"realtime": "Under 1 second (autocomplete, voice)", "interactive": "A few seconds (chat)",
      "batch": "Minutes or hours are fine (overnight reports)"}),
    ("volume", "How many requests a month?",
     {"low": "Under 100k", "medium": "100k to 10 million", "high": "Over 10 million"}),
]

EXAMPLES = [
    ("BrightCart ticket router", dict(privacy="yes", modality="text", difficulty="easy", latency="interactive", volume="high")),
    ("Hospital discharge-summary assistant", dict(privacy="no", modality="long documents", difficulty="medium", latency="interactive", volume="low")),
    ("Internal coding agent for 200 developers", dict(privacy="region", modality="code", difficulty="hard", latency="interactive", volume="medium")),
]


def recommend(a: dict) -> list[str]:
    """The decision tree. Every line of advice comes with its reason."""
    advice = []

    # 1. Privacy decides WHERE the model runs, before anything else.
    if a["privacy"] == "no":
        advice.append(f"Self-host an open-weight model ({OPEN_WEIGHT}). Why: the data can't leave your servers.")
    elif a["privacy"] == "region":
        advice.append("Use a frontier model through your own cloud account (Azure OpenAI, AWS Bedrock, Google Vertex) "
                      "in your region, with no training on your data. Why: data stays in your contract and region.")
    else:
        advice.append("Use a hosted API. Why: no servers to run, newest models, pay per token.")

    # 2. Modality filters WHICH models qualify.
    advice.append(f"Needs {a['modality']}: {MODALITY[a['modality']]}.")

    # 3 + 4. Difficulty and latency pick the TIER.
    tier = {"easy": "small", "medium": "mid", "hard": "frontier"}[a["difficulty"]]
    if a["latency"] == "realtime" and tier == "frontier":
        tier = "mid"
        advice.append("Real-time latency rules out the slowest frontier models; use a mid model and keep prompts short.")
    if a["latency"] == "realtime" and tier == "mid":
        advice.append("For sub-second answers, look at small models, streaming, and providers built for speed (Groq, Cerebras).")
    advice.append(f"Start with the {tier} tier: {TIERS[tier]}. Why: {a['difficulty']} task. "
                  "Then try one tier down; if it passes your eval, use it.")

    # 5. Volume and latency decide HOW you call it.
    if a["latency"] == "batch":
        advice.append("Use the provider's Batch API: about 50% cheaper, results within 24 hours.")
    if a["volume"] == "high":
        advice.append("High volume: price per token dominates. Use prompt caching, the smallest model that passes, "
                      "and consider routing easy requests to a small model (see the capstone).")
        if a["privacy"] != "no":
            advice.append("At this volume, compare against self-hosting an open-weight model (Part C).")
    if a["difficulty"] == "hard" and a["privacy"] == "no":
        advice.append("Hard task + self-hosted: you'll need the biggest open models (DeepSeek V4, Qwen3 235B, Llama 4) "
                      "and multi-GPU servers. Budget for it, or revisit whether a private cloud deployment is allowed.")
    return advice


def show(name: str, answers: dict):
    say(f"\n{BOLD}{name}{RESET}")
    say(DIM + "  " + ", ".join(f"{k}={v}" for k, v in answers.items()) + RESET)
    for line in recommend(answers):
        say(f"  {GREEN}→{RESET} {line}")


def ask_user() -> dict:
    answers = {}
    for key, question, options in QUESTIONS:
        say(f"\n{BOLD}{question}{RESET}")
        keys = list(options)
        for i, k in enumerate(keys, 1):
            say(f"  {i}. {options[k]}")
        while True:
            choice = input("  choose a number: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(keys):
                answers[key] = keys[int(choice) - 1]
                break
    return answers


def self_hosting_maths():
    # TRY: change the traffic and the answer flips
    requests_per_month = 3_000_000
    tokens_in, tokens_out = 800, 200  # per request
    say(f"Traffic: {requests_per_month:,} requests/month × ({tokens_in} in + {tokens_out} out) tokens\n")

    rows = []
    for model in ["openai/gpt-4.1-nano", "deepseek/deepseek-chat", "openai/gpt-5.4-mini", "anthropic/claude-sonnet-4-6"]:
        p = PRICES[model]
        monthly = requests_per_month * (tokens_in * p["input"] + tokens_out * p["output"]) / 1_000_000
        rows.append([f"Taxi: hosted API, {model}", money(monthly), "none"])
    for gpu, hourly in GPU_HOURLY.items():
        rows.append([f"Own car: {gpu}", money(hourly * 730), "you run it"])
    table(["option", "per month", "ops work"], rows, "lrl")
    print()
    say("What the GPU price leaves out:")
    say("  • Engineers to deploy, monitor, patch and scale it (often the biggest cost).")
    say("  • Paying for the GPU 24/7, even at 3 a.m. when nobody uses it.")
    say("  • Open models you can run on one GPU are usually weaker than frontier APIs.")
    say("When self-hosting wins: strict data rules, very high steady volume, a fine-tuned open model, "
        "or a need to control the model version forever.")
    note("GPU prices are rough on-demand cloud rates; reserved and spot instances are cheaper.")


def main():
    heading("02 · Model selection wizard")
    say("Order matters: privacy → modality → difficulty → latency → volume.")
    say("Each answer narrows the field. Benchmarks come last, and your own eval decides.")

    part("A", "three example projects")
    for name, answers in EXAMPLES:
        show(name, answers)

    if not AUTO:
        part("B", "your project")
        show("Your project", ask_user())

    part("C", "hosted vs self-hosted: taxi vs owning a car")
    self_hosting_maths()


if __name__ == "__main__":
    main()
