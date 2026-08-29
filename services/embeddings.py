"""Embedding helpers for retrieval (OpenAI-compatible)."""

from __future__ import annotations

import os

from openai import AsyncOpenAI

from config import EMBEDDING_MODEL, OPENAI_API_KEY

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _client


async def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    client = get_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        vectors.extend(item.embedding for item in response.data)
    return vectors
