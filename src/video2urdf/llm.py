"""Thin OpenAI-SDK client for whichever profile is configured (Meta Model API or local open weights).

Muse Glimmer must not run greedy, so this module never sets temperature; the servers apply the model defaults.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from openai import OpenAI

from .config import Settings, load_settings


def make_client(settings: Settings | None = None) -> tuple[OpenAI, Settings]:
    settings = settings or load_settings()
    return OpenAI(base_url=settings.base_url, api_key=settings.api_key), settings


def image_part(path: str | Path) -> dict:
    """Chat Completions content part with the image inlined as a data URL."""
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def chat(messages: list[dict], settings: Settings | None = None, **kwargs):
    client, settings = make_client(settings)
    return client.chat.completions.create(model=settings.model, messages=messages, **kwargs)
