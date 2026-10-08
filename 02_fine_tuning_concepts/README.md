# Part 2 · Fine-tuning, the concepts

Awareness level: know what it is, when it's worth it and roughly how it works. Nobody trains a model in this session.

## Concepts

**The customisation ladder.** Try them in this order:

| | Prompting | RAG | Fine-tuning |
|---|---|---|---|
| Analogy | Give instructions | Hand over the manual | Send on a training course |
| Changes | What you send | What you send (adds documents) | The model's weights |
| For | Almost everything, first | Knowledge: facts, documents, freshness | Behaviour: style, format, narrow skills |
| Bad for | Huge rule sets | Changing tone or format | Adding or updating facts |
| Needs | A clear prompt | Documents and a pipeline | Hundreds to thousands of good examples |

**RAG is for what the model knows. Fine-tuning is for how it behaves.**

**What fine-tuning changes.** A model is built in three stages: pre-training (the provider, on trillions of tokens),
instruction tuning (the provider, so it follows instructions) and *your* fine-tune (hundreds to thousands of your
examples). Your fine-tune nudges the weights so the model behaves your way by default, without long prompts.

**PEFT (Parameter-Efficient Fine-Tuning).** Full fine-tuning updates every weight: a 7B model needs ~112 GB of GPU memory to train.
PEFT methods freeze the model and train a small add-on instead.

- **LoRA** learns a small low-rank patch: `W_new = W + B @ A`, where B and A are thin matrices. ~0.1 to 1% of the parameters.
- **QLoRA** does the same on a base model compressed to 4 bits: a 7B fine-tune fits on one consumer GPU.
- **Adapters** are small files (~34 MB for 7B) you can swap per task or customer on the same base model.

**Where it's done.** Hosted APIs (OpenAI, Vertex, Bedrock, Together, Fireworks): upload JSONL, get a model id.
Open source: Hugging Face TRL + PEFT, Unsloth, Axolotl.

## The scripts

| Script | What happens | What we saw when testing |
|---|---|---|
| `01_prompt_rag_or_finetune.py` | 8 scenarios; you answer P, R, F or R+F, then see why | The new-product scenario fools most people: it's RAG, not fine-tuning. |
| `02_few_shot_tax.py` | Zero-shot vs 8 examples; monthly cost of few-shot vs caching vs fine-tuned | 8 examples added 591 tokens per call. At 2M calls/month: $182 few-shot, $94 with caching, $128 fine-tuned. Caching can beat fine-tuning. |
| `03_lora_intuition.py` | A visible 6×6 LoRA patch; parameter and memory tables | LoRA r=16 on a 7B model trains 0.25% of the parameters. QLoRA on 7B needs ~5 GB. |
| `04_dataset_preview.py` | Validates `data/finetune_sample.jsonl`; catches broken examples; estimates cost | Training 1,000 examples costs about $0.33. The data is the expensive part. |

## Exercises

1. In `02_few_shot_tax.py`, set `calls_per_month = 50_000`. Which approach wins now?
2. In `03_lora_intuition.py`, change `r` in `toy()` from 1 to 3. How many numbers are trained? When would r = d make sense?
3. Write three more examples for `data/finetune_sample.jsonl` in BrightCart's voice. Add one with a deliberate mistake and see if `04_dataset_preview.py` catches it. What can't a script catch?
