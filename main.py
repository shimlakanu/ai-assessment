import sys

from src.ingestion.pdf_loader import load_pdf
from src.ocr.extractor import extract_pages

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_pdf>")
        sys.exit(1)

    pages = load_pdf(sys.argv[1])
    regions = extract_pages(pages)
    for r in regions:
        print(r)
