from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
from pdf2image import convert_from_path

_SPARSE_CHAR_THRESHOLD = 50


def load_pdf(pdf_path: str | Path) -> list[Image.Image]:
    """Rasterize all pages to PIL Images (used as OCR fallback)."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {pdf_path.suffix}")
    return convert_from_path(pdf_path)


def extract_native_text_pages(pdf_path: str | Path) -> list[list[dict] | None]:
    """Return per-page text regions from PyMuPDF.

    Each entry is a list of region dicts (text, confidence=1.0, bbox) if the
    page has readable text, or None if the page is empty/sparse (i.e. needs OCR).
    """
    doc = fitz.open(str(pdf_path))
    pages: list[list[dict] | None] = []

    for page_num, page in enumerate(doc, start=1):
        raw_text = page.get_text().strip()
        if len(raw_text) < _SPARSE_CHAR_THRESHOLD:
            pages.append(None)
            continue

        regions: list[dict] = []
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, _block_no, block_type = block
            if block_type != 0:  # skip image blocks
                continue
            text = text.strip()
            if not text:
                continue
            regions.append({
                "page": page_num,
                "text": text,
                "confidence": 1.0,
                "bbox": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                "source": "native",
            })

        pages.append(regions if regions else None)

    doc.close()
    return pages
