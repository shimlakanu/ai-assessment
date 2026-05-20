"""Evidence package formatting for grounded generation (Step 5)."""

from __future__ import annotations

_GENERATION_SYSTEM = (
    "You are a precise document analyst. "
    "Answer using ONLY the evidence provided. "
    "For every claim you make, cite the chunk with [chunk_N]. "
    "If the evidence does not support a claim, write UNSUPPORTED — do not invent."
)


def format_evidence_package(chunks: list[dict]) -> str:
    """Format retrieved chunks into a source-annotated evidence block."""
    lines: list[str] = []
    for i, chunk in enumerate(chunks):
        region_str = ", ".join(str(r) for r in chunk.get("region_indices", []))
        page = chunk.get("page", "?")
        score = chunk.get("rerank_score", 0.0)
        lines.append(
            f"[chunk_{i}] {chunk['chunk_text']}\n"
            f"sources: regions {{{region_str}}} | page {page} | score {score:.3f}"
        )
    return "\n\n".join(lines)


def build_generation_prompt(question: str, evidence_package: str) -> str:
    """Build the full generation prompt that enforces grounded-only output."""
    return (
        f"{_GENERATION_SYSTEM}\n\n"
        f"Evidence:\n{evidence_package}\n\n"
        f"Question: {question}\n\n"
        "Answer (cite every claim with [chunk_N]; write UNSUPPORTED for any "
        "claim not supported by the evidence above):"
    )
