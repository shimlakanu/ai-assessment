"""Semantic chunking with Claude context enrichment (Step 1)."""

from __future__ import annotations

import os
import re

import tiktoken
from anthropic import Anthropic

_ENC = tiktoken.get_encoding("cl100k_base")
_TARGET_MIN = 256
_TARGET_MAX = 512
_OVERLAP_RATIO = 0.12   # 12% overlap at chunk boundaries

_CONTEXT_PROMPT = (
    "Document: {full_text}\n\n"
    "Chunk: {chunk_text}\n\n"
    "In 1-2 sentences, describe what this chunk is about in the context of "
    "the full document. Be specific."
)


# ── Public ────────────────────────────────────────────────────────────────────

def chunk_and_enrich(
    doc_output: dict,
) -> list[dict]:
    """Chunk docProcessing output regions and enrich each chunk with Claude context.

    Returns a list of chunk dicts:
        chunk_id, doc_id, chunk_text, contextualized_text,
        region_indices, bboxes, page
    """
    doc_id = doc_output["doc_id"]
    full_text = doc_output["full_text"]
    regions = [r for r in doc_output["regions"] if r["text"] and r["text"] != "ILLEGIBLE"]

    raw_chunks = _group_regions(regions)
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    context_window = full_text[:2000]

    enriched: list[dict] = []
    for i, raw in enumerate(raw_chunks):
        context = _get_context(client, context_window, raw["chunk_text"])
        enriched.append({
            "chunk_id": f"{doc_id}_chunk_{i:04d}",
            "doc_id": doc_id,
            "chunk_text": raw["chunk_text"],
            "contextualized_text": f"{context}\n\n{raw['chunk_text']}",
            "region_indices": raw["region_indices"],
            "bboxes": raw["bboxes"],
            "page": raw["page"],
        })

    return enriched


# ── Internal ──────────────────────────────────────────────────────────────────

def _token_count(text: str) -> int:
    return len(_ENC.encode(text))


def _split_at_sentence_boundary(text: str, max_tokens: int) -> tuple[str, str]:
    """Split text so the first part is <= max_tokens, ending at a sentence boundary."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    head: list[str] = []
    tokens = 0
    for s in sentences:
        t = _token_count(s)
        if tokens + t > max_tokens and head:
            rest_start = text.find(s, sum(len(h) for h in head))
            return text[:rest_start].strip(), text[rest_start:].strip()
        head.append(s)
        tokens += t
    return text.strip(), ""


def _overlap_prefix(text: str, ratio: float) -> str:
    """Return the last `ratio` fraction of tokens from text as an overlap prefix."""
    tokens = _ENC.encode(text)
    n = max(1, int(len(tokens) * ratio))
    return _ENC.decode(tokens[-n:])


def _group_regions(regions: list[dict]) -> list[dict]:
    """Greedily group regions into chunks respecting token bounds and sentence integrity."""
    chunks: list[dict] = []
    current_texts: list[str] = []
    current_indices: list[int] = []
    current_bboxes: list = []
    current_page: int = regions[0]["page"] if regions else 1
    current_tokens = 0
    overlap_prefix = ""

    def flush(overlap: str) -> None:
        nonlocal current_texts, current_indices, current_bboxes, current_page, current_tokens, overlap_prefix
        if not current_texts:
            return
        chunk_text = "\n".join(current_texts)
        chunks.append({
            "chunk_text": chunk_text,
            "region_indices": list(current_indices),
            "bboxes": list(current_bboxes),
            "page": current_page,
        })
        overlap_prefix = _overlap_prefix(chunk_text, _OVERLAP_RATIO)
        current_texts = [overlap] if overlap else []
        current_indices = []
        current_bboxes = []
        current_tokens = _token_count(overlap) if overlap else 0

    for region in regions:
        text = region["text"].strip()
        if not text:
            continue

        t = _token_count(text)

        # Flush before adding if this single region alone would exceed max
        if t > _TARGET_MAX:
            flush(overlap_prefix)
            # Split the oversized region at sentence boundary
            remaining = text
            while remaining:
                head, remaining = _split_at_sentence_boundary(remaining, _TARGET_MAX)
                chunks.append({
                    "chunk_text": head,
                    "region_indices": [region["index"]],
                    "bboxes": [region.get("bbox")],
                    "page": region.get("page", 1),
                })
            overlap_prefix = _overlap_prefix(chunks[-1]["chunk_text"], _OVERLAP_RATIO)
            continue

        if current_tokens + t > _TARGET_MAX and current_tokens >= _TARGET_MIN:
            flush(overlap_prefix)

        if not current_texts and overlap_prefix:
            current_texts.append(overlap_prefix)
            current_tokens += _token_count(overlap_prefix)
            overlap_prefix = ""

        current_texts.append(text)
        current_indices.append(region["index"])
        bbox = region.get("bbox") or region.get("bboxes")
        current_bboxes.append(bbox)
        current_page = region.get("page", current_page)
        current_tokens += t

    flush(overlap_prefix)
    return chunks


def _get_context(client: Anthropic, context_window: str, chunk_text: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": _CONTEXT_PROMPT.format(
                full_text=context_window,
                chunk_text=chunk_text[:800],
            ),
        }],
    )
    return response.content[0].text.strip()
