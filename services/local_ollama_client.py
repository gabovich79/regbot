"""Constrained request builders for local Ollama inference."""

from __future__ import annotations

from typing import Any


def build_constrained_chat_payload(*, model: str, prompt: str) -> dict[str, Any]:
    """Create a deterministic, bounded local synthesis request."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {
            "num_ctx": 8192,
            "num_predict": 400,
            "temperature": 0.0,
        },
    }
