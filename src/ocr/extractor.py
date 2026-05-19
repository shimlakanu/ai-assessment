import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

_ocr = None


def _get_ocr() -> PaddleOCR:
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(use_textline_orientation=True, lang="en")
    return _ocr


def extract(image: Image.Image) -> list[dict]:
    img_array = np.array(image)
    results = _get_ocr().ocr(img_array)

    regions = []
    for page_result in (results or []):
        texts = page_result.get("rec_texts", [])
        scores = page_result.get("rec_scores", [])
        polys = page_result.get("rec_polys", [])
        for text, confidence, bbox in zip(texts, scores, polys):
            regions.append({
                "bbox": bbox.tolist() if hasattr(bbox, "tolist") else bbox,
                "text": text,
                "confidence": confidence,
            })
    return regions


def extract_pages(images: list[Image.Image]) -> list[dict]:
    regions = []
    for page_num, image in enumerate(images, start=1):
        for region in extract(image):
            regions.append({"page": page_num, **region})
    return regions
