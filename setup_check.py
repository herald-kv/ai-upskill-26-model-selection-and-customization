"""Check your setup before the session: packages, .env, and a ping to each demo model.

Run:  python setup_check.py           check everything
      python setup_check.py --list    also list the chat models the proxy offers
"""

import importlib.util
import sys

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"

PACKAGES = ["openai", "dotenv", "pydantic", "PIL", "tiktoken", "numpy", "certifi"]


def main():
    print("Packages")
    missing = [p for p in PACKAGES if importlib.util.find_spec(p) is None]
    for p in PACKAGES:
        print(f"  {RED + '✗' if p in missing else GREEN + '✓'} {p}{RESET}")
    if missing:
        sys.exit("\nRun: pip install -r requirements.txt")

    from lab.common import MODELS, chat, table  # needs the packages above

    print("\nModels (one tiny request each)")
    rows = []
    failed = False
    for role, model in MODELS.items():
        try:
            r = chat(model, [{"role": "user", "content": "Reply with the word OK."}], max_tokens=200, show=False)
            rows.append([role, model, f"{GREEN}✓{RESET}", f"{r.seconds:.1f}s", r.text[:20]])
        except Exception as e:
            failed = True
            rows.append([role, model, f"{RED}✗{RESET}", "", str(e)[:70]])
    table(["role", "model", "ok", "latency", "reply"], rows)

    if "--list" in sys.argv:
        from lab.common import client

        ids = sorted({m.id for m in client().models.list()})
        chat_like = [i for i in ids if not any(x in i for x in ("image", "dall", "tts", "whisper", "embed", "-x-", "*"))]
        print(f"\nThe proxy lists {len(ids)} model ids ({len(chat_like)} chat-like). Not all of them have healthy deployments.")
        for i in chat_like:
            print(" ", i)

    if failed:
        sys.exit(f"\n{RED}Some models failed. Check LITELLM_API_KEY in .env, or pick other ids (python setup_check.py --list).{RESET}")
    print(f"\n{GREEN}All set.{RESET}")


if __name__ == "__main__":
    main()
