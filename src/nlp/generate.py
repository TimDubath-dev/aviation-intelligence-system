"""Thin wrappers around OpenAI and Anthropic for the explainer LLM."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def call_openai(system: str, user: str, model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


def call_anthropic(system: str, user: str, model: str = "claude-haiku-4-5-20251001") -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=model,
        system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=400,
    )
    return resp.content[0].text.strip()


def generate(system: str, user: str, provider: str = "openai", **kw) -> str:
    if provider == "openai":
        return call_openai(system, user, **kw)
    if provider == "anthropic":
        return call_anthropic(system, user, **kw)
    raise ValueError(provider)
