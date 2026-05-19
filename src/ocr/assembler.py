CONFIDENCE_THRESHOLD = 0.70


def assemble(regions: list[dict]) -> tuple[str, dict[int, dict]]:
    region_index = {i: dict(region) for i, region in enumerate(regions)}

    lines = []
    for i, region in region_index.items():
        if region["confidence"] >= CONFIDENCE_THRESHOLD:
            lines.append(f"[{i}] {region['text']}")
        else:
            region["flagged"] = True

    flat_text = "\n".join(lines)
    return flat_text, region_index
