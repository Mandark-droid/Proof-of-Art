from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pramaan.traditions import ALL_STATES, GRAMMARS, Tradition, load_registry
from pramaan.tasks import canonical_jsonl_bytes

DEFAULT_OUTPUT = Path(__file__).parents[1] / "benchmarks" / "suites"
SYMMETRY = {
    "periodic_raster": "D2",
    "dot_field": "D2",
    "stripe_band": "D2",
    "counted_grid": "D2",
    "rotational_field": "C4",
    "kolam_loop": "D2",
    "jaali_lattice": "D2",
}
DENSITIES = ("sparse", "balanced", "close-spaced", "dense but legible")


def validate_constraint_profile(grammar: str, profile: dict[str, Any]) -> None:
    """Validate through ConstraintsV2 when available; keep an interim stub for A2."""
    try:
        from pramaan.dsl import ConstraintsV2
    except (ImportError, AttributeError):
        # TODO: remove once the public v0.2 interface is available.
        allowed = {
            "symmetry",
            "min_coverage",
            "max_coverage",
            "min_clearance",
            "translation_period",
            "cell_size",
            "required_loops",
            "frame",
            "min_web",
            "min_open_fraction",
            "max_open_fraction",
        }
        if set(profile) - allowed:
            raise ValueError(f"unknown constraint fields: {set(profile) - allowed}")
        if profile.get("symmetry", SYMMETRY[grammar]) not in {"None", "D2", "D4", "C2", "C4"}:
            raise ValueError("invalid symmetry")
        if "cell_size" in profile and profile["cell_size"] not in {2, 4, 8}:
            raise ValueError("invalid cell size")
        if "translation_period" in profile:
            period = profile["translation_period"]
            if period.get("axis") not in {"x", "y", "xy"} or period.get("period", 0) < 1:
                raise ValueError("invalid translation period")
        if "min_clearance" in profile and profile["min_clearance"] < 0:
            raise ValueError("invalid minimum clearance")
        return
    if "frame" in profile:
        if grammar != "jaali_lattice":
            raise ValueError("frame is only valid for jaali_lattice")
        frame = profile["frame"]
        if type(frame) is not int or not 0 <= frame <= 12:
            raise ValueError("jaali frame must be an integer in [0, 12]")
    ConstraintsV2.model_validate({key: value for key, value in profile.items() if key != "frame"})


def _primary_for_state(registry: dict[str, Tradition], state: str) -> Tradition:
    candidates = [
        item
        for item in registry.values()
        if item.primary_state == state and item.priority == "primary"
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected one primary tradition for {state}, found {len(candidates)}")
    return candidates[0]


def _hard_alternate(registry: dict[str, Tradition], state: str, primary: Tradition) -> Tradition:
    candidates = [
        item
        for item in registry.values()
        if item.primary_state == state
        and item.priority == "alternate"
        and item.confidence == "high"
        and item.grammar != primary.grammar
    ]
    return candidates[0] if candidates else primary


def _suite_palette(tradition: Tradition) -> list[str]:
    """Keep host-owned palettes within PatternV2's six-colour schema bound."""
    palette = tradition.palette_presets["traditional"]
    if not palette:
        raise ValueError(f"traditional palette for {tradition.id} is empty")
    # Select only colours that already have sourced derivation in the registry.
    return palette[:6]


def _task_profile(grammar: str, profile: dict[str, Any], brief: str) -> dict[str, Any]:
    """Fill task-owned Jaali openness bounds before locking a suite row.

    Briefs with an explicit 40-50% interval keep that exact requirement. Other
    Jaali briefs use the documented broad default [0.25, 0.75], which leaves
    room for varied lattices while making the verifier's openness check defined.
    """
    result = dict(profile)
    if grammar == "jaali_lattice":
        normalized = brief.lower().replace("–", "-").replace("—", "-")
        if "40-50% open area" in normalized:
            result["min_open_fraction"] = 0.4
            result["max_open_fraction"] = 0.5
        else:
            result.setdefault("min_open_fraction", 0.25)
            result.setdefault("max_open_fraction", 0.75)
    return result


def _eval_row(state: str, tradition: Tradition, level: str, brief: str, number: int) -> dict[str, Any]:
    profile = _task_profile(tradition.grammar, dict(tradition.default_profile), brief)
    profile.setdefault("symmetry", SYMMETRY[tradition.grammar])
    validate_constraint_profile(tradition.grammar, profile)
    return {
        "id": f"{tradition.id}-{level}-{number}",
        "suite_id": "eval-v2.1",
        "state": state,
        "tradition_id": tradition.id,
        "grammar": tradition.grammar,
        "level": level,
        "symmetry": profile["symmetry"],
        "constraint_profile": profile,
        "palette": _suite_palette(tradition),
        "brief": brief,
    }


def _train_brief(tradition: Tradition, variant: int) -> str:
    blocked = ("sacred", "deity", "goddess", "god", "conch", "chakra", "lotus", "swastik", "ancestor", "ghotul", "memorial", "pakhangba", "ritual", "identity mark")
    motifs = [m for m in tradition.motif_vocabulary if not any(word in m.lower() for word in blocked)]
    motifs = motifs or ["abstract repeated geometric forms"]
    motif = motifs[(variant - 1) % len(motifs)].strip().rstrip(".;")
    colors = _suite_palette(tradition)
    offset = variant % len(colors)
    rotated = colors[offset:] + colors[:offset]
    palette = ", ".join(rotated[: min(4, len(rotated))])
    density = DENSITIES[(variant - 1) % len(DENSITIES)]
    brief = (
        f"Training v1 variant {variant}: Create an original geometric tile inspired by {tradition.name}. "
        f"Use abstract motif guidance from {motif}; distribute it with {density} spacing and a clear center. "
        "Avoid named sacred or identity-specific symbols. "
        f"Use this palette order: {palette}. Treat the palette as approximate color guidance."
    )
    fixed_text_length = len(brief) - len(motif)
    motif = motif[:max(0, 500 - fixed_text_length)]
    brief = (
        f"Training v1 variant {variant}: Create an original geometric tile inspired by {tradition.name}. "
        f"Use abstract motif guidance from {motif}; distribute it with {density} spacing and a clear center. "
        "Avoid named sacred or identity-specific symbols. "
        f"Use this palette order: {palette}. Treat the palette as approximate color guidance."
    )
    if len(brief) > 500:
        raise ValueError(f"training brief for {tradition.id} exceeds the orchestrator limit")
    return brief


def build_suites(output_dir: Path | str = DEFAULT_OUTPUT, *, train_suite_id: str = "train-v1.2") -> dict[str, str]:
    output_dir = Path(output_dir)
    eval_path = output_dir / "eval-v2.1.jsonl"
    train_path = output_dir / f"{train_suite_id}.jsonl"
    if eval_path.exists() or train_path.exists():
        raise FileExistsError(f"refusing to overwrite existing suite file in {output_dir}")

    registry = load_registry()
    eval_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    for state_index, state in enumerate(ALL_STATES):
        primary = _primary_for_state(registry, state)
        eval_rows.append(_eval_row(state, primary, "easy", primary.draft_briefs["easy"], state_index))
        hard_tradition = _hard_alternate(registry, state, primary)
        eval_rows.append(_eval_row(state, hard_tradition, "hard", hard_tradition.draft_briefs["hard"], state_index))
        for variant in range(1, 5):
            brief = _train_brief(primary, variant)
            profile = _task_profile(primary.grammar, dict(primary.default_profile), brief)
            profile.setdefault("symmetry", SYMMETRY[primary.grammar])
            validate_constraint_profile(primary.grammar, profile)
            train_rows.append({
                "id": f"{primary.id}-train-{variant}",
                "suite_id": train_suite_id,
                "state": state,
                "tradition_id": primary.id,
                "grammar": primary.grammar,
                "level": "train",
                "symmetry": profile["symmetry"],
                "constraint_profile": profile,
                "palette": _suite_palette(primary),
                "brief": brief,
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"eval-v2.1": eval_rows, train_suite_id: train_rows}
    digests: dict[str, str] = {}
    for name, rows in outputs.items():
        path = output_dir / f"{name}.jsonl"
        content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
        path.write_text(content, encoding="utf-8")
        # The runner hashes the bytes on disk. On Windows, text-mode writes
        # translate newlines, so hash the written file rather than pre-write text.
        digests[name] = hashlib.sha256(canonical_jsonl_bytes(path.read_bytes())).hexdigest()
    return digests


if __name__ == "__main__":
    for suite_id, digest in build_suites().items():
        print(f"{suite_id}: sha256={digest}")
