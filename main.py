import sys

from src.ingestion.pdf_loader import load_pdf
from src.ocr.assembler import assemble
from src.ocr.criticality import assess_criticality
from src.ocr.extractor import extract_pages

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_pdf>")
        sys.exit(1)

    pages = load_pdf(sys.argv[1])
    regions = extract_pages(pages)
    flat_text, region_index = assemble(regions)
    region_index = assess_criticality(flat_text, region_index)
    print(flat_text)
