"""03 · LoRA intuition: why you don't need to retrain the whole model. No API calls, no GPU.

A model is mostly big matrices of numbers (weights). Full fine-tuning updates all of them.
LoRA (Low-Rank Adaptation) freezes the original weights and learns a small "patch" instead:

        W_new = W + B @ A          W: d × d (frozen)    B: d × r    A: r × d    r is tiny (4–64)

Analogy: instead of rewriting the textbook, you add sticky notes. Same book, new behaviour,
and you can peel the notes off or swap in a different set.

Part A  a toy 6×6 "model" and a rank-1 patch you can see
Part B  the same idea at real scale: how many numbers you actually train
Part C  GPU memory: full fine-tuning vs LoRA vs QLoRA
Part D  one base model, many adapters

Run:  python 02_fine_tuning_concepts/03_lora_intuition.py        (--auto to skip the pauses)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.common import heading, note, part, say, table

np.set_printoptions(precision=2, suppress=True, linewidth=120)


def toy():
    rng = np.random.default_rng(0)
    d, r = 6, 1
    W = rng.normal(0, 1, (d, d))
    say(f"W, the frozen weights ({d}×{d} = {d * d} numbers):")
    print(W)

    # LoRA starts with B = 0, so on day one the model behaves exactly like the original.
    B = np.zeros((d, r))
    A = rng.normal(0, 1, (r, d))
    say(f"\nB ({d}×{r}) starts at zero, A ({r}×{d}) is random. So B @ A = 0 and nothing changes at first:")
    print(B @ A)

    # Pretend training nudged B. (Real training finds these values with gradient descent.)
    B = rng.normal(0, 0.5, (d, r))
    delta = B @ A
    say(f"\nAfter 'training', B @ A is a full {d}×{d} change, built from only {d * r + r * d} numbers:")
    print(delta)
    x = rng.normal(0, 1, d)
    say("\nSame input, original model vs patched model:")
    say(f"  W @ x       = {W @ x}")
    say(f"  (W + BA) @ x = {(W + delta) @ x}")
    note(f"Full fine-tuning would train {d * d} numbers here. LoRA trains {d * r + r * d}.")


def scale():
    # A 7B-class model: hidden size 4096, 32 layers, 4 attention matrices per layer (q, k, v, o).
    # Real models also have bigger MLP matrices; LoRA is often applied to those too.
    hidden, layers, mats = 4096, 32, 4
    total = 6_700_000_000
    rows = [["Full fine-tuning", f"{total:,}", "100%"]]
    for r in (8, 16, 64):
        trainable = layers * mats * (2 * hidden * r)
        rows.append([f"LoRA r={r} on q,k,v,o", f"{trainable:,}", f"{100 * trainable / total:.2f}%"])
    table(["approach (7B model)", "numbers trained", "of the model"], rows, "lrr")
    note("\nr (the rank) is the size of the patch. Bigger r = more capacity to change, more to train.")
    note("Typical values are 8 to 64. Results are often close to full fine-tuning for style and format tasks.")


def memory():
    # Rough rules of thumb for training memory (activations and batch size add more on top):
    #   full fine-tuning in mixed precision with Adam ≈ 16 bytes per parameter
    #   LoRA: frozen weights in 16-bit (2 bytes/param) + small adapter + optimizer states
    #   QLoRA: frozen weights quantised to 4-bit (0.5 bytes/param) + adapter
    rows = []
    for name, params in (("7B", 7e9), ("13B", 13e9), ("70B", 70e9)):
        full = params * 16 / 1e9
        lora = params * 2 / 1e9 * 1.2
        qlora = params * 0.5 / 1e9 * 1.4
        rows.append([name, f"~{full:,.0f} GB", f"~{lora:,.0f} GB", f"~{qlora:,.0f} GB"])
    table(["model", "full fine-tune", "LoRA (16-bit base)", "QLoRA (4-bit base)"], rows, "lrrr")
    say("\nWhat that means in hardware:")
    say("  • 7B full fine-tune: several 80 GB GPUs.   7B QLoRA: one gaming GPU or a free Colab T4.")
    say("  • 70B full fine-tune: a cluster.           70B QLoRA: one 80 GB GPU.")
    note("These are rough estimates; real numbers depend on sequence length, batch size and tricks like gradient checkpointing.")


def adapters():
    r, hidden, layers, mats = 16, 4096, 32, 4
    adapter_mb = layers * mats * 2 * hidden * r * 2 / 1e6  # 16-bit numbers
    say(f"A LoRA adapter for a 7B model at r={r} is about {adapter_mb:.0f} MB. The base model is about 14 GB.")
    say("So one server can hold one base model and swap small adapters per request:")
    say("    base model (14 GB) + adapter 'legal-tone' (34 MB)")
    say("                       + adapter 'support-voice' (34 MB)")
    say("                       + adapter 'customer-A-format' (34 MB)")
    say("Serving tools such as vLLM and LoRAX do exactly this (multi-LoRA serving).")
    print()
    say("The PEFT family (Parameter-Efficient Fine-Tuning), at awareness level:")
    table(
        ["method", "idea in one line"],
        [
            ["LoRA", "learn a small low-rank patch B @ A next to frozen weights"],
            ["QLoRA", "LoRA on a 4-bit quantised base model: same idea, a quarter of the memory"],
            ["Adapters", "insert small new layers between frozen ones"],
            ["Prompt / prefix tuning", "learn a few 'virtual tokens' put in front of every input"],
        ],
    )
    say("\nWhere you'd actually do it:")
    say("  • Hosted: OpenAI, Google Vertex, AWS Bedrock, Together, Fireworks fine-tuning APIs (upload JSONL, get a model id)")
    say("  • Open source: Hugging Face PEFT + TRL, Unsloth, Axolotl, LLaMA-Factory")
    note("You don't need to implement any of this. Know it exists, what it costs, and when to reach for it.")


def main():
    heading("03 · LoRA intuition")
    part("A", "a toy model and a rank-1 patch")
    toy()
    part("B", "real scale: how much do we actually train?")
    scale()
    part("C", "GPU memory needed to train")
    memory()
    part("D", "one base model, many adapters")
    adapters()


if __name__ == "__main__":
    main()
