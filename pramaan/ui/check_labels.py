"""Plain-language labels for verifier checks shown in the public interface."""

from __future__ import annotations

import re


_CHECK_LABELS = {
    "candidate_parse": "Readable pattern",
    "candidate_truncated": "Complete response",
    "schema_valid": "Pattern structure",
    "complexity_bound": "Primitive limit",
    "palette_limit": "Colour limit",
    "periodic_raster": "Tile continuity",
    "layer_registration": "Layer alignment",
    "exclusive_ink_collision": "Separate print layers",
    "minimum_gap": "Minimum spacing",
    "coverage_bound": "Pattern coverage",
    "dot_clearance": "Dot spacing",
    "translation_period": "Repeat period",
    "cell_quantization": "Cell alignment",
    "kolam_mirrors_valid": "Kolam crossings",
    "kolam_loop_count": "Kolam stroke count",
    "kolam_mirror_symmetry": "Kolam symmetry",
    "jaali_single_solid": "Connected lattice",
    "jaali_holes_interior": "Interior openings",
    "jaali_open_fraction": "Open area",
    "jaali_min_web": "Lattice thickness",
}


def check_label(check_id: object) -> str:
    """Turn a verifier id into readable copy without exposing implementation names."""
    value = check_id if isinstance(check_id, str) else ""
    if value in _CHECK_LABELS:
        return _CHECK_LABELS[value]
    if value.startswith("symmetry_"):
        mode = value.removeprefix("symmetry_").upper()
        return "Symmetry" if mode == "NONE" else f"Symmetry ({mode})"
    cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", value).strip()
    return cleaned.capitalize() if cleaned else "Geometric rule"


def check_labels(check_ids: object, *, limit: int = 8) -> str:
    if not isinstance(check_ids, list):
        return ""
    return ", ".join(check_label(item) for item in check_ids[:limit] if isinstance(item, str))
