import json
import os
import re

from openai import OpenAI

_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
_MODEL = "accounts/fireworks/models/deepseek-v4-pro"

_SYSTEM = (
    "You are a document quality analyst. "
    "You will be given a document and a list of low-confidence OCR regions. "
    "Respond only with valid JSON."
)

_USER_TMPL = """\
Document context:
{flat_text}

Flagged regions (low OCR confidence):
{flagged_list}

For each flagged region, determine whether the uncertainty could change the
meaning, validity, or actionability of the output.

If yes  -> needs_recovery: true
If no   -> needs_recovery: false

Respond with a JSON object mapping each region id to its needs_recovery value:
{{"0": true, "3": false, ...}}
"""


def _extract_json(text: str) -> dict:
    """Extract the last JSON object from a response that may contain reasoning text."""
    matches = re.findall(r"\{[^{}]+\}", text, re.DOTALL)
    if not matches:
        raise ValueError(f"No JSON object found in response:\n{text}")
    return json.loads(matches[-1])


def assess_criticality(
    flat_text: str,
    region_index: dict[int, dict],
) -> dict[int, dict]:
    flagged = {i: r for i, r in region_index.items() if r.get("flagged")}
    if not flagged:
        return region_index

    flagged_list = "\n".join(f"[{i}] {r['text']}" for i, r in flagged.items())
    prompt = _USER_TMPL.format(flat_text=flat_text, flagged_list=flagged_list)

    client = OpenAI(base_url=_FIREWORKS_BASE_URL, api_key=os.environ["FIREWORKS_API_KEY"])
    response = client.chat.completions.create(
        model=_MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    raw = response.choices[0].message.content
    decisions = _extract_json(raw)

    result = dict(region_index)
    for str_id, needs_recovery in decisions.items():
        idx = int(str_id)
        if idx in result and needs_recovery:
            result[idx] = {**result[idx], "needs_recovery": True}

    return result
