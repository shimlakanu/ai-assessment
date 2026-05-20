"""Citation index construction and per-field evidence highlighting (Step 6)."""

from __future__ import annotations

from PIL import Image, ImageDraw

from src.retrieval.indexer import IndexManager

_HIGHLIGHT_COLOR = (255, 50, 50)   # red
_HIGHLIGHT_WIDTH = 2


def build_citation_index(
    draft_fields: dict[str, str],
    retrieved_chunks: list[dict],
    doc_id: str,
    index_manager: IndexManager,
) -> dict[str, dict]:
    """Map each draft field to the chunks that support it.

    draft_fields: {field_name: generated_value}
    retrieved_chunks: the top_k results from retrieve() that produced this draft

    Each entry in the returned index:
        {
            "draft_field": str,
            "value": str,
            "supported_by": [chunk_id, ...],
            "region_indices": [int, ...],
            "bboxes": [[...], ...],
            "rerank_scores": [float, ...],
            "page": int | None,
        }
    """
    citation_index: dict[str, dict] = {}

    for field_name, value in draft_fields.items():
        # Identify which chunks cite this field (chunks whose chunk_id appears in value)
        supporting = [
            c for c in retrieved_chunks
            if _chunk_cited_in(c["chunk_id"], value)
        ]
        # Fallback: if no explicit citation found, attribute all retrieved chunks
        if not supporting:
            supporting = retrieved_chunks

        all_region_indices: list[int] = []
        all_bboxes: list = []
        all_scores: list[float] = []
        page = None

        for chunk in supporting:
            all_region_indices.extend(chunk.get("region_indices") or [])
            all_bboxes.extend(b for b in (chunk.get("bboxes") or []) if b is not None)
            all_scores.append(chunk.get("rerank_score", 0.0))
            if page is None:
                page = chunk.get("page")

        citation_index[field_name] = {
            "draft_field": field_name,
            "value": value,
            "supported_by": [c["chunk_id"] for c in supporting],
            "region_indices": all_region_indices,
            "bboxes": all_bboxes,
            "rerank_scores": all_scores,
            "page": page,
        }

    index_manager.store_citations(doc_id, citation_index)
    return citation_index


def highlight_evidence(
    image: Image.Image,
    citation_index: dict[str, dict],
    field_name: str,
) -> Image.Image:
    """Draw bounding polygons for cited regions of field_name on a copy of image.

    Only annotates regions cited by field_name — never the full page.
    Returns the annotated image.
    """
    record = citation_index.get(field_name)
    if not record or not record.get("bboxes"):
        return image.copy()

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for bbox in record["bboxes"]:
        if not bbox:
            continue
        # bbox is [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] polygon
        poly = [tuple(map(int, pt)) for pt in bbox]
        draw.polygon(poly, outline=_HIGHLIGHT_COLOR, width=_HIGHLIGHT_WIDTH)

    return annotated


# ── Internal ──────────────────────────────────────────────────────────────────

def _chunk_cited_in(chunk_id: str, text: str) -> bool:
    """Check if chunk_id appears as a citation marker in the generated text."""
    # Accepts patterns like [chunk_0002] or [chunk_2] normalised
    bare = chunk_id.split("_chunk_")[-1].lstrip("0") or "0"
    return (
        f"[{chunk_id}]" in text
        or f"[chunk_{bare}]" in text
        or chunk_id in text
    )
