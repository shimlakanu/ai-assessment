"""Evidence sufficiency gate (Step 4).

Generation must not proceed unless avg(rerank_scores) >= 0.4.
This is the contract that makes the system trustworthy.
"""

from __future__ import annotations

_SUFFICIENCY_THRESHOLD = 0.4


def check_sufficiency(results: list[dict]) -> dict:
    """Evaluate whether retrieved evidence is strong enough for generation.

    Returns a dict with:
        status: "sufficient" | "insufficient_evidence"
        sufficiency_score: float
        message: str (only present when insufficient)
        chunks: the top_k results (always present for inspection)
    """
    if not results:
        return {
            "status": "insufficient_evidence",
            "sufficiency_score": 0.0,
            "message": (
                "No evidence retrieved. "
                "Do not generate. Flag for human review."
            ),
            "chunks": [],
        }

    avg_score = sum(r["rerank_score"] for r in results) / len(results)

    if avg_score < _SUFFICIENCY_THRESHOLD:
        return {
            "status": "insufficient_evidence",
            "sufficiency_score": avg_score,
            "message": (
                "Retrieved evidence too weak to support grounded generation. "
                "Do not generate. Flag for human review."
            ),
            "chunks": results,
        }

    return {
        "status": "sufficient",
        "sufficiency_score": avg_score,
        "chunks": results,
    }
