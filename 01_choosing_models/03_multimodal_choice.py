"""03 · Choosing a model for images: check the capability before you pick.

Not every model can see. Worse, a text-only model behind an OpenAI-compatible gateway may
not raise an error: the image is quietly dropped and the model answers anyway.

Part A  make a receipt image (generated with Pillow, no files needed)
Part B  send it to a text-only model and to two vision models
Part C  look the capability up instead of guessing: the proxy's model hub

Run:  python 01_choosing_models/03_multimodal_choice.py        (--auto to skip the pauses)
"""

import base64
import io
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.common import DATA, MODELS, bad, chat, heading, note, ok, part, say, table

# TRY: change the items; the correct total is computed from them
ITEMS = [("Wireless mouse", 799), ("USB-C cable x2", 598), ("Laptop sleeve", 1249)]
TOTAL = sum(price for _, price in ITEMS)

TEXT_ONLY = MODELS["open"]                       # deepseek-chat: text only
VISION = [MODELS["small"], MODELS["google"]]     # both accept images


def make_receipt() -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (460, 330), "white")
    d = ImageDraw.Draw(img)
    try:
        font, big = ImageFont.load_default(size=20), ImageFont.load_default(size=26)
    except TypeError:  # Pillow < 10.1
        font = big = ImageFont.load_default()
    d.text((20, 15), "BrightCart Store #214", font=big, fill="black")
    d.text((20, 55), "01 Oct 2026   Bill no. 55871", font=font, fill="gray")
    y = 100
    for name, price in ITEMS:
        d.text((20, y), name, font=font, fill="black")
        d.text((330, y), f"Rs {price:,}", font=font, fill="black")
        y += 35
    d.line((20, y + 5, 440, y + 5), fill="black", width=2)
    d.text((20, y + 20), "TOTAL", font=big, fill="black")
    d.text((310, y + 20), f"Rs {TOTAL:,}", font=big, fill="black")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def image_message(png: bytes, question: str) -> list:
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    return [{"role": "user", "content": [
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": {"url": url}},
    ]}]


def main():
    heading("03 · Choosing a model for images")

    part("A", "a receipt image")
    png = make_receipt()
    out = DATA / "generated" / "receipt.png"
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(png)
    say(f"Saved {out.relative_to(DATA.parent)} (open it to see what the models see). Correct total: Rs {TOTAL:,}")

    question = "What is the total on this receipt, and how many line items are there? Answer in one sentence."
    part("B", "text-only model vs vision models")
    say(f"Text-only model: {TEXT_ONLY}")
    r = chat(TEXT_ONLY, image_message(png, question), max_tokens=200)
    bad(f"Only {r.input_tokens} input tokens: the image never reached the model, and no error was raised.")
    if str(TOTAL) not in r.text.replace(",", ""):
        bad("It answered anyway, with a made-up total. A silent hallucination.")
    print()
    for model in VISION:
        r = chat(model, image_message(png, question), max_tokens=1000)
        if f"{TOTAL:,}" in r.text or str(TOTAL) in r.text:
            ok(f"{model} read the total: {r.input_tokens} input tokens, almost all of them the image.")
        else:
            bad(f"{model} did not find Rs {TOTAL:,}")
        print()
    note("Images are converted to tokens too. Bigger images cost more. Downscale before sending.")

    part("C", "look it up instead of guessing")
    try:
        with urllib.request.urlopen("https://llm.keyvalue.systems/public/model_hub", timeout=30) as resp:
            hub = {m["model_group"]: m for m in json.load(resp)}
        rows = []
        for model in [MODELS["small"], MODELS["mid"], MODELS["frontier"], MODELS["google"], MODELS["open"]]:
            m = hub.get(model, {})
            rows.append([model,
                         "✓" if m.get("supports_vision") else "✗",
                         "✓" if m.get("supports_function_calling") else "✗",
                         "✓" if m.get("supports_reasoning") else "✗",
                         f"{int(m['max_input_tokens']):,}" if m.get("max_input_tokens") else "?"])
        table(["model", "vision", "tools", "reasoning", "context window"], rows, "lcccr")
    except Exception as e:
        note(f"Could not reach the model hub ({e}). Check the provider's docs instead.")
    print()
    say("Choosing by modality:")
    say("  • Images, screenshots, scans → a vision model (GPT-5.x, Claude, Gemini; open: Qwen3-VL, Llama 4, Gemma 4)")
    say("  • Audio → speech-to-text first (Whisper / gpt-4o-transcribe), then a text model. Gemini also takes audio directly.")
    say("  • Code → coding-tuned models (Codex, Claude Sonnet/Opus; open: Qwen3-Coder, Devstral)")
    say("  • Generate images → a different kind of model entirely (gpt-image, Imagen, Flux)")
    note("\n# TRY: send the receipt to MODELS['frontier'] and compare the answer and the token count.")


if __name__ == "__main__":
    main()
