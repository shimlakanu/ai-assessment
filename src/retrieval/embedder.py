"""Embeddings via nomic-embed-text-v1.5 on Fireworks (Step 2)."""

from __future__ import annotations

import os

from openai import OpenAI

_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
_MODEL = "nomic-ai/nomic-embed-text-v1.5"
_BATCH_SIZE = 32

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=_FIREWORKS_BASE_URL,
            api_key=os.environ["FIREWORKS_API_KEY"],
        )
    return _client


def embed(text: str) -> list[float]:
    """Embed a single string. Returns a float vector."""
    return embed_batch([text])[0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings in batches. Returns list of float vectors."""
    results: list[list[float]] = []
    client = _get_client()
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        response = client.embeddings.create(model=_MODEL, input=batch)
        results.extend(item.embedding for item in response.data)
    return results
