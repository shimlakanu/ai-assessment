_HIGH = 0.85   # include as-is
_LOW = 0.50    # include with ⚠ flag; below this → vision recovery


def assemble(regions: list[dict]) -> tuple[str, dict[int, dict]]:
    """Route regions by confidence into three tiers.

    >= 0.85  → included in flat_text, no flag
    0.5–0.85 → included in flat_text, flagged=True (LLM criticality check next)
    < 0.5    → NOT included yet, flagged=True + needs_vision_recovery=True
    """
    region_index = {i: dict(region) for i, region in enumerate(regions)}

    lines = []
    for i, region in region_index.items():
        conf = region["confidence"]
        if conf >= _HIGH:
            lines.append(f"[{i}] {region['text']}")
        elif conf >= _LOW:
            region["flagged"] = True
            lines.append(f"[{i}] ⚠ {region['text']} (conf={conf:.2f})")
        else:
            region["flagged"] = True
            region["needs_vision_recovery"] = True

    flat_text = "\n".join(lines)
    return flat_text, region_index
