from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path


def load_pdf(pdf_path: str | Path) -> list[Image.Image]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {pdf_path.suffix}")
    return convert_from_path(pdf_path)
