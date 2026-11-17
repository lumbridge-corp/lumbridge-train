"""A teacher is any OpenAI-compatible chat endpoint: a local vLLM, a hosted API, your own model."""

from __future__ import annotations

import json
import os
import re
import urllib.request

from .config import Teacher

_JSON = re.compile(r"\{.*\}", re.S)


def chat(t: Teacher, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 400) -> str:
    body = {"model": t.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    body.update(t.extra_body)
    headers = {"content-type": "application/json", **t.headers}
    if t.api_key_env:
        key = os.environ.get(t.api_key_env)
        if not key:
            raise RuntimeError(f"teacher {t.name}: ${t.api_key_env} is not set")
        headers["authorization"] = f"Bearer {key}"
    req = urllib.request.Request(
        t.base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), headers=headers
    )
    with urllib.request.urlopen(req, timeout=t.timeout_s) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"].get("content") or ""


def parse_json(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    for candidate in (text, *(_JSON.findall(text)[:1])):
        try:
            v = json.loads(candidate)
            if isinstance(v, dict):
                return v
        except ValueError:
            pass
    return None
