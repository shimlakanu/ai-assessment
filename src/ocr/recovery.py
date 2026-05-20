"""Vision recovery for low-confidence and critical OCR regions.

Uses Qwen2.5-VL 32B on Fireworks — optimised for document understanding
and visual grounding, available via the existing FIREWORKS_API_KEY.
"""

from __future__ import annotations

import base64
import os
from io import BytesIO

from openai import OpenAI
from PIL import Image

_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
_MODEL = "accounts/fireworks/models/qwen2p5-vl-32b-instruct"
_PROMPT = (
    'OCR guess: "{guess}"\n'
    "What does this image actually say? "
    'Reply with ONLY the corrected text, or exactly "ILLEGIBLE" if unreadable.'
)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=_FIREWORKS_BASE_URL,
            api_key=os.environ["FIREWORKS_API_KEY"],
        )
    return _client


def recover_regions(
    region_index: dict[int, dict],
    page_images: dict[int, Image.Image],
    recovery_key: str,
) -> dict[int, dict]:
    """Send regions marked with recovery_key=True to Qwen2.5-VL 32B for recovery.

    For each region:
    - Crops the bounding box from the original page image.
    - Asks the model to correct the OCR guess.
    - If the model returns "ILLEGIBLE", marks needs_review=True, recovered=False.
    - Otherwise, replaces text with the recovered value, recovered=True.

    Returns an updated copy of region_index.
    """
    to_recover = {i: r for i, r in region_index.items() if r.get(recovery_key)}
    if not to_recover:
        return region_index

    client = _get_client()
    result = dict(region_index)

    for idx, region in to_recover.items():
        page_num = region.get("page", 1)
        image = page_images.get(page_num)
        crop_b64 = _crop_to_b64(region, image)

        response = client.chat.completions.create(
            model=_MODEL,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{crop_b64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": _PROMPT.format(guess=region["text"]),
                    },
                ],
            }],
        )

        recovered_text = response.choices[0].message.content.strip()
        if recovered_text.upper() == "ILLEGIBLE":
            result[idx] = {
                **region,
                "text": "ILLEGIBLE",
                "flagged": True,
                "recovered": False,
                "needs_review": True,
            }
        else:
            result[idx] = {
                **region,
                "text": recovered_text,
                "flagged": True,
                "recovered": True,
            }

    return result


def _crop_to_b64(region: dict, image: Image.Image | None) -> str:
    """Crop the region bbox from the page image and return as base64 PNG."""
    if image is None:
        buf = BytesIO()
        Image.new("RGB", (100, 30), color=(255, 255, 255)).save(buf, format="PNG")
        return base64.standard_b64encode(buf.getvalue()).decode()

    bbox = region.get("bbox")
    if bbox:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        pad = 8
        x1 = max(0, int(min(xs)) - pad)
        y1 = max(0, int(min(ys)) - pad)
        x2 = min(image.width,  int(max(xs)) + pad)
        y2 = min(image.height, int(max(ys)) + pad)
        cropped = image.crop((x1, y1, x2, y2))
    else:
        cropped = image

    buf = BytesIO()
    cropped.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()
