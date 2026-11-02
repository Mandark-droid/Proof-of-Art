"""Canonicalize the hand-authored demo and refresh its verifier certificate."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pramaan.certificate import make_certificate
from pramaan.checks.core import verify_pattern
from pramaan.dsl import Pattern
from pramaan.export import render_png

DESIGN = ROOT / "examples" / "ajrakh_starfield.pattern.json"
CERT = ROOT / "examples" / "ajrakh_starfield.certificate.json"
PREVIEW = ROOT / "assets" / "ajrakh_starfield_preview.png"


def orbit(point):
    result = set()
    for reflected in (False, True):
        x, y = point
        if reflected:
            x = 63 - x
        for _ in range(4):
            result.add((x, y))
            x, y = 63 - y, x
    return sorted(result, key=lambda p: (p[1], p[0]))


def main():
    data = json.loads(DESIGN.read_text(encoding="utf-8"))
    for layer in data["layers"]:
        expanded = {}
        for motif in layer["motifs"]:
            for center in orbit(motif["center"]):
                copy = dict(motif)
                copy["center"] = list(center)
                expanded[(motif["kind"], center, motif.get("radius"), motif.get("stroke"))] = copy
        layer["motifs"] = list(expanded.values())
    pattern = Pattern.model_validate_json(json.dumps(data))
    DESIGN.write_text(json.dumps(pattern.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    checks, masks, grid = verify_pattern(pattern)
    certificate = make_certificate(pattern, checks, masks, grid)
    CERT.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    render_png(grid, pattern.palette, scale=5, repeats=3).save(PREVIEW)
    if not certificate["all_hard_constraints_pass"]:
        raise SystemExit("Built-in example failed its independent checks")
    print(f"Wrote {DESIGN.relative_to(ROOT)}, {CERT.relative_to(ROOT)}, and {PREVIEW.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
