"""Smoke test: verify assess_criticality correctly classifies flagged regions."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ocr.criticality import assess_criticality

flat_text = (
    "[0] Invoice No. 1042\n"
    "[1] Total Amount Due: $4,200.00\n"
    "[2] Thank you for your business."
)

region_index = {
    0: {"text": "Invoice No. 1042",           "confidence": 0.95, "page": 1, "bbox": []},
    1: {"text": "Total Amount Due: $4,200.00", "confidence": 0.95, "page": 1, "bbox": []},
    2: {"text": "Thank you for your business.","confidence": 0.95, "page": 1, "bbox": []},
    3: {"text": "Inv0ice N0. 1O42",            "confidence": 0.55, "page": 1, "bbox": [], "flagged": True},
    4: {"text": "T0tal Amovnt Dve: $4,2OO.OO", "confidence": 0.48, "page": 1, "bbox": [], "flagged": True},
    5: {"text": "Thenk yov for yovr busines.", "confidence": 0.61, "page": 1, "bbox": [], "flagged": True},
}

result = assess_criticality(flat_text, region_index)

print("\n--- criticality smoke test results ---")
for i, region in result.items():
    if region.get("flagged"):
        status = "needs_recovery" if region.get("needs_recovery") else "flagged/acceptable"
        print(f"  [{i}] {status:20s}  conf={region['confidence']:.2f}  \"{region['text']}\"")
