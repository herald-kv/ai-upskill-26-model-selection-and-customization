# Layer 5 · Model Selection & Customization

Hands-on scripts and slides for the AI Upskill session on **choosing a model, customising it, and keeping the bill under control**.
Every example is about BrightCart, a fictional online shop that wants an AI support assistant.

| Part | The question | You'll learn |
|---|---|---|
| **1 · Choosing the right model** | Which model? | Frontier vs open-weight, model tiers, hosted vs self-hosted, picking by modality, testing on your own data |
| **2 · Fine-tuning (concepts)** | Can we make it ours? | Prompting vs RAG vs fine-tuning, what fine-tuning changes, PEFT / LoRA / QLoRA at awareness level |
| **3 · Cost & caching** | What will it cost? | Token economics, prompt caching, rate limits and retries, async, batching |

**Mental model for the session**

```
Choosing a model     = hiring: an intern, an experienced hire or a senior consultant. Hire the cheapest who can do the job.
Hosted vs self-host  = taking a taxi vs owning a car
Prompt / RAG / tune  = giving instructions / handing over the manual / sending someone on a training course
LoRA                 = sticky notes on a textbook: the book stays the same, the notes change the behaviour
Prompt caching       = prepping ingredients once in the morning, cooking many dishes from them
```

## Setup (10 minutes, before the session)

```bash
cd ai-upskill-26-model-selection-and-customization
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                 # then paste the LiteLLM key the presenter shares
python setup_check.py                # five green ticks = ready
```

All models go through the company LiteLLM proxy (`https://llm.keyvalue.systems`). It speaks the OpenAI API for every
provider, so one script can call GPT, Claude, Gemini and DeepSeek. To try other models, change the `MODEL_*` lines in
`.env`; `python setup_check.py --list` shows what the proxy offers.

## The scripts

Run everything from the repo root. Each script is split into parts (A, B, C…) and waits for Enter between them.

| Flag | What it does |
|---|---|
| `--auto` | runs straight through without pausing |
| `--hide-prompts` | prints only the responses |

| # | Script | What it shows | API calls? |
|---|---|---|---|
| | `setup_check.py` | Packages, key, and a ping to each model | yes |
| 1.1 | `01_choosing_models/01_model_bakeoff.py` | The same 15 tasks on five models: accuracy, latency, cost | yes |
| 1.2 | `01_choosing_models/02_selection_wizard.py` | Requirements → model shortlist; hosted vs self-hosted maths | no |
| 1.3 | `01_choosing_models/03_multimodal_choice.py` | A receipt image: a text-only model silently fails, vision models read it | yes |
| 2.1 | `02_fine_tuning_concepts/01_prompt_rag_or_finetune.py` | Quiz: 8 scenarios, prompt or RAG or fine-tune? | no |
| 2.2 | `02_fine_tuning_concepts/02_few_shot_tax.py` | Few-shot examples cost tokens on every call; when fine-tuning pays off | yes |
| 2.3 | `02_fine_tuning_concepts/03_lora_intuition.py` | LoRA with NumPy: W + B·A, parameter and memory counts | no |
| 2.4 | `02_fine_tuning_concepts/04_dataset_preview.py` | A fine-tuning JSONL file: format, validation, cost | no |
| 3.1 | `03_cost_and_caching/01_token_economics.py` | What a token costs, where the money goes, the cost levers | no |
| 3.2 | `03_cost_and_caching/02_prompt_caching.py` | OpenAI automatic caching, Claude `cache_control`, the timestamp mistake, response caching | yes |
| 3.3 | `03_cost_and_caching/03_rate_limits_retries.py` | Rate-limit headers, 429s, exponential backoff with jitter | 1 call |
| 3.4 | `03_cost_and_caching/04_async_batching.py` | Sequential vs `asyncio.gather` vs Semaphore vs one batched prompt | yes |
| 4 | `04_capstone_model_router.py` | A cheap model triages, a strong model takes the hard tickets; the bill vs "always strong" | yes |

How to read the output: cyan `▶` is what was sent, magenta `◀` is what came back, grey lines are token counts, latency
and cost, green `✓` and red `✗` mark what each part is showing. Lines marked `# TRY:` in the code are places to experiment.

The whole run costs a few cents.

## Repo layout

```
lab/            shared plumbing: proxy client, measured chat() call, prices, printing (not part of the lesson)
data/           support tickets, the BrightCart handbook (~3,600 tokens), a sample fine-tuning dataset
01_choosing_models/  02_fine_tuning_concepts/  03_cost_and_caching/   the scripts, each folder with a README
04_capstone_model_router.py
slides/
```

Prices in `lab/pricing.py` come from the proxy's model hub and provider pricing pages on 1 October 2026. They're there
to make the maths concrete; check the provider's page before quoting them. BrightCart, its policies and its customers are fictional.
