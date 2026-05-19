from src.ingestion.pdf_loader import load_pdf
from src.ocr.extractor import extract

PDF_PATH = "data/sample_docs/sample1.pdf"

pages = load_pdf(PDF_PATH)
print(f"Pages loaded: {len(pages)}")

for i, page in enumerate(pages):
    print(f"\n--- Page {i + 1} ---")
    regions = extract(page)
    for region in regions:
        print(region)
