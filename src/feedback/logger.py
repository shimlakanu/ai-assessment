"""Append per-document OCR difficulty signals to feedback_log.json."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

FEEDBACK_LOG = Path("data/feedback_log.json")


def log_feedback(doc_id: str, region_index: dict[int, dict]) -> None:
    """Write one entry per uncertain region to feedback_log.json.

    Only regions that were flagged, recovered, or marked needs_review are logged
    (i.e. anything that wasn't a clean >=0.85 pass).
    """
    uncertain = [
        (i, r) for i, r in region_index.items()
        if r.get("flagged") or r.get("recovered") or r.get("needs_review")
    ]
    if not uncertain:
        return

    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if FEEDBACK_LOG.exists():
        try:
            existing = json.loads(FEEDBACK_LOG.read_text())
        except json.JSONDecodeError:
            existing = []

    now = datetime.now(timezone.utc).isoformat()
    for idx, region in uncertain:
        conf = region.get("confidence", 0.0)
        existing.append({
            "timestamp": now,
            "doc_id": doc_id,
            "issue": _classify_issue(conf, region),
            "region_index": idx,
            "confidence": conf,
            "resolution": _resolution(region),
        })

    FEEDBACK_LOG.write_text(json.dumps(existing, indent=2))


def _classify_issue(confidence: float, region: dict) -> str:
    if region.get("needs_review"):
        return "illegible_unresolvable"
    if confidence < 0.3:
        return "very_low_confidence_scan_quality"
    if confidence < 0.5:
        return "low_confidence_font_or_noise"
    return "moderate_confidence_layout_or_ambiguity"


def _resolution(region: dict) -> str:
    if region.get("needs_review"):
        return "ILLEGIBLE"
    if region.get("recovered"):
        return "vision_recovery"
    if region.get("needs_recovery"):
        return "vision_recovery_critical"
    return "flagged_acceptable"
