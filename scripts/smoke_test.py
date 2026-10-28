"""One cheap call per backend. Never prints keys.

Usage: .venv/bin/python scripts/smoke_test.py
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PROMPT = "Reply with exactly the word OK."
SECRETS = [v for k, v in os.environ.items() if k.endswith(("_KEY", "_SECRET")) and v]


def redact(text: str) -> str:
    for s in SECRETS:
        text = text.replace(s, "***")
    return text


def run(name, fn):
    t0 = time.time()
    try:
        out = fn()
        if not out or str(out).startswith("None"):
            raise RuntimeError(f"empty output: {out!r}")
        print(f"[PASS] {name} ({time.time() - t0:.1f}s): {redact(str(out))[:200]!r}")
        return True
    except Exception as e:
        print(f"[FAIL] {name} ({time.time() - t0:.1f}s): {type(e).__name__}: {redact(str(e))[:300]}")
        return False


def client(base_url, key_env):
    key = os.environ.get(key_env)
    if not key:
        raise RuntimeError(f"{key_env} missing in .env")
    return OpenAI(base_url=base_url, api_key=key, timeout=60)


def meta_models():
    ids = [m.id for m in client("https://api.meta.ai/v1", "MODEL_API_KEY").models.list()]
    return ", ".join(ids)


def meta_responses():
    r = client("https://api.meta.ai/v1", "MODEL_API_KEY").responses.create(
        model="muse-spark-1.3-contributor", input=PROMPT, max_output_tokens=1024
    )
    return f"{r.output_text} (status={r.status}, usage={r.usage})"


def meta_chat():
    r = client("https://api.meta.ai/v1", "MODEL_API_KEY").chat.completions.create(
        model="muse-spark-1.3-contributor",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=1024,
    )
    return f"{r.choices[0].message.content} (finish={r.choices[0].finish_reason})"


def chat(base_url, key_env, model):
    r = client(base_url, key_env).chat.completions.create(
        model=model, messages=[{"role": "user", "content": PROMPT}], max_tokens=1024
    )
    return f"{r.choices[0].message.content} (finish={r.choices[0].finish_reason})"


def fireworks_muse_models():
    c = client("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY")
    return ", ".join(m.id for m in c.models.list() if "muse" in m.id.lower()) or "(none)"


def modal_token():
    if not (Path.home() / ".modal.toml").exists():
        raise RuntimeError("~/.modal.toml missing; run `modal setup`")
    return "token file present"


if __name__ == "__main__":
    results = [
        run("meta /models", meta_models),
        run("meta /responses", meta_responses),
        run("meta /chat/completions", meta_chat),
        run("baseten chat", lambda: chat("https://inference.baseten.co/v1", "BASETEN_API_KEY",
                                         "meta-models/muse-spark-open")),
        run("fireworks chat", lambda: chat("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY",
                                           "accounts/fireworks/models/muse-spark-open")),
        run("fireworks muse models", fireworks_muse_models),
        run("modal token", modal_token),
    ]
    sys.exit(0 if all(results) else 1)
