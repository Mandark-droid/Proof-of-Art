"""Plain labels for verifier checks on v2 pages.

One mapping: the v1 labels in ``pramaan.ui.check_labels``, overridden by the friendlier wording the
v2 designs use (spec 3.1).
"""
from __future__ import annotations

import re

from pramaan.ui.check_labels import check_label as _v1_check_label

_DESIGN_LABELS = {
    "candidate_parse": "Unreadable proposal",
    "candidate_truncated": "Cut off at the token limit",
    "schema_valid": "Readable pattern",
    "complexity_bound": "Within the design size limit",
    "palette_limit": "Within the colour limit",
    "periodic_raster": "Repeat tiles match",
    "layer_registration": "Layers line up",
    "exclusive_ink_collision": "Ink layers do not overlap",
    "minimum_gap": "Motifs keep their distance",
    "coverage_bound": "Ink coverage is within range",
    "dot_clearance": "Dots keep their distance",
}

# Two or three words for a broken rule, used after "broke" ("0.725 · broke symmetry").
_SHORT = {
    "palette_limit": "colour limit", "complexity_bound": "size limit", "periodic_raster": "repeat tiles",
    "layer_registration": "layer alignment", "exclusive_ink_collision": "ink overlap", "minimum_gap": "spacing",
    "coverage_bound": "ink coverage", "dot_clearance": "dot spacing", "translation_period": "repeat period",
    "cell_quantization": "cell alignment", "kolam_mirrors_valid": "kolam crossings", "kolam_loop_count": "kolam loop count",
    "kolam_mirror_symmetry": "kolam symmetry", "jaali_single_solid": "connected lattice",
    "jaali_holes_interior": "interior openings", "jaali_open_fraction": "lattice open area", "jaali_min_web": "lattice thickness",
}


def check_label(check_id: str) -> str:
    """The visitor label for a check id."""
    if isinstance(check_id, str) and check_id.startswith("symmetry_"):
        return "Symmetric"
    return _DESIGN_LABELS.get(check_id) or _v1_check_label(check_id)


def check_row_label(check: dict) -> str:
    """The label for a check row; the colour limit names its number ("At most 4 colours")."""
    if check.get("id") == "palette_limit":
        match = re.search(r"\blimit\s+(\d+)\b", str(check.get("message", "")))
        if match:
            return f"At most {match.group(1)} colours"
    return check_label(check.get("id", ""))


def short_rule(check_id: str) -> str:
    """Two or three lower-case words naming a rule."""
    if isinstance(check_id, str) and check_id.startswith("symmetry_"):
        return "symmetry"
    return _SHORT.get(check_id) or check_label(check_id).lower()
