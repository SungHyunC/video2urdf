"""Model endpoint settings. Every model sits behind one OpenAI-compatible base URL.

Switch between the Meta Model API and local open weights with `V2U_PROFILE` in `.env`:
    meta   https://api.meta.ai/v1, key in MODEL_API_KEY, Muse Spark
    local  vLLM on the A100 reached through `ssh -L 8000:localhost:8000 <host>`, Muse Glimmer
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

PROFILES = {
    "meta": {"base_url": "https://api.meta.ai/v1", "model": "muse-spark-1.3", "key_env": "MODEL_API_KEY"},
    # vLLM ignores the key unless started with --api-key.
    "local": {"base_url": "http://localhost:8000/v1", "model": "muse-glimmer", "key_env": None},
}


@dataclass(frozen=True)
class Settings:
    profile: str
    base_url: str
    model: str
    api_key: str


def load_settings(profile: str | None = None) -> Settings:
    load_dotenv()
    name = profile or os.getenv("V2U_PROFILE", "meta")
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")
    p = PROFILES[name]
    key = os.getenv("V2U_API_KEY") or (os.getenv(p["key_env"]) if p["key_env"] else None) or "EMPTY"
    if p["key_env"] and key == "EMPTY":
        raise RuntimeError(f"profile {name!r} needs {p['key_env']} in .env (see .env.example)")
    return Settings(
        profile=name,
        base_url=os.getenv("V2U_BASE_URL", p["base_url"]),
        model=os.getenv("V2U_MODEL", p["model"]),
        api_key=key,
    )
