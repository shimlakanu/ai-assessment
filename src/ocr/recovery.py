"""Claude vision recovery for low-confidence and critical OCR regions."""

from __future__ import annotations

import base64
import os
from io import BytesIO

import anthropic
from PIL import Image

_MODEL = "claude-opus-4-5"
_PROMPT = (
    'OCR guess: "{guess}"\n'
    "What does this image actually say? "
    'Reply with ONLY the corrected text, or exactly "ILLEGIBLE" if unreadable.'
)


def recover_regions(
    region_index: dict[int, dict],
    page_images: dict[int, Image.Image],
    recovery_key: str,
) -> dict[int, dict]:
    """Send regions marked with recovery_key=True to Claude vision.

    For each region:
    - Crops the bounding box from the original page image.
    - Asks Claude to correct the OCR guess.
    - If Claude returns "ILLEGIBLE", marks needs_review=True, recovered=False.
    - Otherwise, replaces text with the recovered value, recovered=True.

    Returns an updated copy of region_index.
    """
    to_recover = {i: r for i, r in region_index.items() if r.get(recovery_key)}
    if not to_recover:
        return region_index

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        for idx in to_recover:
            region_index[idx] = {
                **region_index[idx],
                "text": "ILLEGIBLE",
                "recovered": False,
                "needs_review": True,
                "recovery_skipped": "ANTHROPIC_API_KEY not set",
            }
        return region_index

    client = anthropic.Anthropic(api_key=api_key)
    result = dict(region_index)

    for idx, region in to_recover.items():
        page_num = region.get("page", 1)
        image = page_images.get(page_num)
        crop_b64 = _crop_to_b64(region, image)

        response = client.messages.create(
            model=_MODEL,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": crop_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": _PROMPT.format(guess=region["text"]),
                    },
                ],
            }],
        )

        recovered_text = response.content[0].text.strip()
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
