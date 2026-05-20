"""Full document processing pipeline (Steps 1–6)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.feedback.logger import log_feedback
from src.ingestion.pdf_loader import extract_native_text_pages, load_pdf
from src.ingestion.store import store_pdf
from src.ocr.assembler import assemble
from src.ocr.criticality import assess_criticality
from src.ocr.extractor import extract
from src.ocr.preprocessor import preprocess
from src.ocr.recovery import recover_regions


def process(source_pdf: str | Path) -> dict:
    """Run the full pipeline on a PDF and return the structured output dict.

    Steps:
      1. Store PDF → doc_id + confirmed path
      2. Per-page: PyMuPDF native text OR rasterise → preprocess → PaddleOCR
      3. Confidence routing (assemble): high / mid / low tiers
         + immediate vision recovery for conf < 0.5
      4. Criticality assessment on mid-tier flagged regions
         + vision recovery for needs_recovery=True regions
      5. Feedback logging
      6. Final JSON output
    """
    # ── Step 1: Store ───────────────────────────────────────────────────────
    doc_id, saved_path = store_pdf(source_pdf)

    # ── Step 2: Extract ─────────────────────────────────────────────────────
    native_pages = extract_native_text_pages(saved_path)
    raster_images: list[Image.Image] | None = None   # lazy; only if needed

    all_regions: list[dict] = []
    page_images: dict[int, Image.Image] = {}  # page_num → original PIL image

    for page_num, native in enumerate(native_pages, start=1):
        if native is not None:
            # Native PDF text found — use directly
            all_regions.extend(native)
        else:
            # Sparse / scanned — rasterise, preprocess, OCR
            if raster_images is None:
                raster_images = load_pdf(saved_path)
            image = raster_images[page_num - 1]
            page_images[page_num] = image          # keep original for cropping
            preprocessed = preprocess(image)
            for region in extract(preprocessed):
                all_regions.append({"page": page_num, **region})

    # ── Step 3: Confidence routing ──────────────────────────────────────────
    flat_text, region_index = assemble(all_regions)

    # Immediately recover conf < 0.5 regions via Claude vision
    region_index = recover_regions(region_index, page_images, recovery_key="needs_vision_recovery")

    # Rebuild flat_text so criticality has accurate context
    flat_text = _rebuild_flat_text(region_index)

    # ── Step 4: Critical region recovery ────────────────────────────────────
    # LLM decides which of the mid-tier (⚠ flagged) regions are semantically
    # critical.  Only those get sent to Claude vision — not all flagged ones.
    region_index = assess_criticality(flat_text, region_index)
    region_index = recover_regions(region_index, page_images, recovery_key="needs_recovery")

    # ── Step 5: Feedback logging ─────────────────────────────────────────────
    log_feedback(doc_id, region_index)

    # ── Step 6: Final output ─────────────────────────────────────────────────
    return _build_output(doc_id, region_index)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _rebuild_flat_text(region_index: dict[int, dict]) -> str:
    """Reconstruct flat_text after vision recovery has patched low-conf regions."""
    lines = []
    for i, r in sorted(region_index.items()):
        text = r["text"]
        if r.get("needs_review"):
            lines.append(f"[{i}] ILLEGIBLE ⚠ needs_review")
        elif r.get("flagged"):
            conf = r.get("confidence", 0.0)
            lines.append(f"[{i}] ⚠ {text} (conf={conf:.2f})")
        else:
            lines.append(f"[{i}] {text}")
    return "\n".join(lines)


def _build_output(doc_id: str, region_index: dict[int, dict]) -> dict:
    """Assemble the final structured output document."""
    sorted_regions = []
    needs_review: list[int] = []

    for i, r in sorted(region_index.items()):
        entry = {
            "index": i,
            "text": r["text"],
            "confidence": r.get("confidence", 0.0),
            "bbox": r.get("bbox"),
            "page": r.get("page"),
            "flagged": r.get("flagged", False),
            "recovered": r.get("recovered", False),
        }
        sorted_regions.append(entry)
        if r.get("needs_review"):
            needs_review.append(i)

    full_text = "\n".join(
        r["text"] for r in sorted_regions if r["text"] != "ILLEGIBLE"
    )

    return {
        "doc_id": doc_id,
        "full_text": full_text,
        "regions": sorted_regions,
        "needs_review": needs_review,
    }
