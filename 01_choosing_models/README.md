# Part 1 · Choosing the right model

## Concepts

**Closed vs open-weight.** Closed models (OpenAI GPT, Anthropic Claude, Google Gemini) are only available through an API.
Open-weight models (Meta Llama, Mistral, Qwen, DeepSeek, Google Gemma, OpenAI gpt-oss) publish their weights, so you can
run them anywhere, including on your own servers. "Open-weight" is not always "open source": the licence decides what
you're allowed to do.

**Tiers.** Every provider sells a small, a mid and a frontier model. Prices differ by 30 to 100×.

| Tier | Like hiring… | Good at | Examples (Oct 2026) |
|---|---|---|---|
| Small | an intern | classify, extract, short rewrites | GPT-4.1 nano, GPT-5.4 nano, Claude Haiku, Gemini Flash-Lite, Llama 8B |
| Mid | an experienced hire | most everyday work, RAG answers | GPT-5.4 mini, Gemini Flash, DeepSeek V3/V4, Llama 4 Maverick |
| Frontier | a senior consultant | hard reasoning, agents, complex code | GPT-5.5, Claude Sonnet / Opus, Gemini Pro |

**What to compare:** quality on *your* task, cost (input and output), latency, context window, modalities, features
(tools, structured outputs, caching, batch), data and privacy terms, rate limits and licence.

**Benchmarks vs your own eval.** Leaderboards give you a shortlist. A set of 20 to 50 of your real, hard examples picks the winner.

**Hosted vs self-hosted (taxi vs your own car).** Hosted APIs: no ops, best models, pay per token. Self-hosted: data never
leaves, fixed GPU cost, full control, but you run it. In between: frontier models through your own cloud account
(Azure OpenAI, AWS Bedrock, Google Vertex) in your region.

**Picking by modality.** Vision for images and scans; speech-to-text (Whisper) then a text model for audio; coding-tuned
models for code; long-context models (or RAG) for long documents.

## The scripts

| Script | What happens | What we saw when testing |
|---|---|---|
| `01_model_bakeoff.py` | 12 tickets, 1 extraction, 2 reasoning questions on 5 models | All models routed 11 or 12 of 12 correctly. Only Claude Sonnet got both reasoning questions. gpt-4.1-nano would cost ~$8 per million requests vs ~$516 for Sonnet. |
| `02_selection_wizard.py` | Five questions → recommendation with reasons; taxi vs car maths | At 3M requests/month the cheapest API ($480) beats a single GPU ($584) before counting engineering time. |
| `03_multimodal_choice.py` | A generated receipt sent to a text-only model and two vision models | deepseek-chat received only 25 tokens (the image was dropped) and made up a total. Vision models read Rs 2,646 correctly. |

## Exercises

1. Add two tickets you think are ambiguous to `data/support_tickets.json`. Do the models disagree?
2. Replace a contestant in `01_model_bakeoff.py` with `openai/gpt-5.5` or `anthropic/claude-haiku-4-5`. Is the extra cost worth it?
3. Run the wizard (without `--auto`) for a project on your team. Do you agree with its answer?
4. In `02_selection_wizard.py`, change `requests_per_month` until self-hosting looks attractive. What else would you need to be true?
