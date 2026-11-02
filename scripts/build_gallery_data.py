"""Build compact, replay-verified model-gallery assets from benchmark trajectories."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

from pramaan import VALIDATOR_VERSION
from pramaan.certificate import make_certificate
from pramaan.checks.core import all_pass, verify_pattern
from pramaan.dsl import ConstraintsV2, parse_pattern
from pramaan.stats import wilson_interval
from pramaan.tasks import Task, canonical_jsonl_bytes, load_suite
from pramaan.traditions import ALL_STATES, load_registry, traditions_for_state

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "benchmarks" / "suites" / "eval-v2.1.jsonl"
DEFAULT_TRAJECTORY = ROOT / "benchmarks" / "data" / "eval-v2.1-trajectories.jsonl"
DEFAULT_OUTPUT = ROOT / "assets" / "gallery"
GRAMMAR_LABELS = {
    "periodic_raster": "G1", "dot_field": "G2", "stripe_band": "G3",
    "counted_grid": "G4", "rotational_field": "G5", "kolam_loop": "G6",
    "jaali_lattice": "G7",
}
STATE_GRID = {
    "Jammu and Kashmir": [3, 0], "Ladakh": [4, 0], "Punjab": [3, 1], "Himachal Pradesh": [4, 1],
    "Chandigarh": [2, 2], "Haryana": [3, 2], "Uttarakhand": [4, 2], "Sikkim": [6, 2], "Arunachal Pradesh": [9, 2],
    "Rajasthan": [2, 3], "Delhi": [4, 3], "Uttar Pradesh": [5, 3], "Bihar": [6, 3], "Assam": [8, 3], "Nagaland": [9, 3],
    "Gujarat": [1, 4], "Madhya Pradesh": [3, 4], "Chhattisgarh": [4, 4], "Jharkhand": [6, 4], "West Bengal": [7, 4],
    "Meghalaya": [8, 4], "Manipur": [9, 4], "Dadra and Nagar Haveli and Daman and Diu": [1, 5], "Maharashtra": [3, 5],
    "Odisha": [5, 5], "Tripura": [8, 5], "Mizoram": [9, 5], "Goa": [2, 6], "Telangana": [4, 6], "Andhra Pradesh": [5, 6],
    "Karnataka": [3, 7], "Lakshadweep": [1, 8], "Kerala": [3, 8], "Tamil Nadu": [4, 8], "Puducherry": [5, 8],
    "Andaman and Nicobar Islands": [8, 8],
}
WITNESS_KEYS = {
    "original": "orig", "transformed": "xform", "pixel": "pixel", "other_pixel": "other",
    "point": "point", "first": "first", "second": "second", "center": "center",
}
MARKER_COLOURS = {
    "orig": "#FF2D95", "xform": "#00E5FF", "pixel": "#FFE600", "other": "#FF8C00",
    "point": "#BC6FF1", "first": "#50C878", "second": "#F05D5E", "center": "#00A6A6",
}


def _read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return rows


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _witness_points(report: dict[str, Any]) -> list[list[Any]]:
    points: list[list[Any]] = []
    for check in report.get("checks", []):
        witness = check.get("counterexample")
        if not isinstance(witness, dict):
            continue
        for key, role in WITNESS_KEYS.items():
            point = witness.get(key)
            if (isinstance(point, (list, tuple)) and len(point) == 2
                    and all(isinstance(axis, int) and not isinstance(axis, bool) for axis in point)):
                item = [int(point[0]), int(point[1]), role]
                if item not in points:
                    points.append(item)
    return points


def _indexed_png(grid, palette: list[str], witnesses: list[list[Any]]) -> str:
    image = Image.fromarray(grid.astype("uint8"), mode="P")
    rgb_palette = [tuple(bytes.fromhex(colour.lstrip("#"))) for colour in palette]
    for role in MARKER_COLOURS:
        rgb_palette.append(tuple(bytes.fromhex(MARKER_COLOURS[role].lstrip("#"))))
    flat = [channel for colour in rgb_palette for channel in colour]
    image.putpalette(flat + [0] * (768 - len(flat)))
    draw = ImageDraw.Draw(image)
    role_index = {role: len(palette) + index for index, role in enumerate(MARKER_COLOURS)}
    for x, y, role in witnesses:
        if not (0 <= x < 64 and 0 <= y < 64):
            continue
        color = role_index[role]
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=color, width=1)
        draw.point((x, y), fill=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _replay_candidate(candidate: Any) -> tuple[dict | None, dict | None, list[dict], Any]:
    if not isinstance(candidate, dict):
        return None, None, [], None
    pattern = parse_pattern(json.dumps(candidate, ensure_ascii=False))
    checks, masks, grid = verify_pattern(pattern)
    certificate = make_certificate(pattern, checks, masks, grid)
    return pattern.model_dump(mode="json"), certificate, checks, grid


def _certificate_sha(certificate: dict | None) -> str | None:
    if certificate is None:
        return None
    packed = json.dumps(certificate, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return _sha256(packed)


def _attempt_payload(call: dict[str, Any], expected_index: int) -> dict[str, Any]:
    stored_report = call.get("verifier_report") if isinstance(call.get("verifier_report"), dict) else {}
    replayed_candidate, certificate, checks, grid = _replay_candidate(call.get("candidate"))
    if replayed_candidate is not None:
        stored_checks = stored_report.get("checks")
        if stored_checks != checks:
            raise ValueError(f"call {call.get('call_index')} stored checks differ from independent replay")
        accepted = all_pass(checks)
        if bool(call.get("accepted")) != accepted:
            raise ValueError(f"call {call.get('call_index')} acceptance differs from independent replay")
    else:
        checks = stored_report.get("checks", [])
        accepted = False
    witnesses = _witness_points({"checks": checks})
    palette = (call.get("candidate") or {}).get("palette", [])
    tile = _indexed_png(grid, palette, witnesses) if grid is not None else None
    index = call.get("candidate_index") or call.get("call_index") or expected_index
    return {
        "index": int(index),
        "passed": bool(accepted),
        "failing_checks": [str(check.get("id")) for check in checks if check.get("status") == "FAIL"],
        "witness": witnesses,
        "tile_png_b64": tile,
    }


def _certificate_for_accepted(calls: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    for call in reversed(calls):
        if not call.get("accepted") or not isinstance(call.get("candidate"), dict):
            continue
        _candidate, certificate, _checks, _grid = _replay_candidate(call["candidate"])
        return _certificate_sha(certificate), certificate.get("mask_sha256") if certificate else None
    return None, None


def _near_bound(call: dict[str, Any]) -> bool:
    if not call.get("accepted") or not isinstance(call.get("candidate"), dict):
        return False
    candidate, certificate, checks, grid = _replay_candidate(call["candidate"])
    if grid is None:
        return False
    background = int(candidate.get("background_index", 0))
    coverage = float((grid != background).mean())
    constraints = candidate.get("constraints") or {}
    bounds = [constraints.get("min_coverage"), constraints.get("max_coverage")]
    return any(isinstance(bound, (int, float)) and abs(coverage - float(bound)) <= 0.01 for bound in bounds)


def _build_attempts(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for row in rows:
        calls = row.get("calls") if isinstance(row.get("calls"), list) else []
        attempts = [_attempt_payload(call, index) for index, call in enumerate(calls, 1)]
        passed_at = next((item["index"] for item in attempts if item["passed"]), None)
        cert_sha, mask_sha = _certificate_for_accepted(calls)
        if row.get("accepted_mask_sha256") and mask_sha != row.get("accepted_mask_sha256"):
            raise ValueError(f"accepted mask digest differs from replay for {row.get('prompt_id')}")
        rewards = [call.get("reward") for call in calls if isinstance(call.get("reward"), (int, float))]
        tasks[row["prompt_id"]] = {
            "status": "completed", "state": row["state"], "level": row.get("level"),
            "grammar": row.get("grammar"), "tradition_id": row.get("tradition_id"), "brief": row["brief"],
            "passed_at": passed_at, "reward": max(rewards, default=0.0),
            "cert_sha256": cert_sha, "mask_sha256": mask_sha,
            "attempts": attempts,
        }
    return tasks


def _arm_summary(key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if not total:
        raise ValueError(f"arm {key} has no rows")
    pass1 = sum(row.get("validity_at_1") is True for row in rows)
    pass3 = sum(row.get("validity_at_3") is True for row in rows)
    schemas = sum(row.get("valid_schema_at_1") is True for row in rows)
    initial_failures = sum(row.get("validity_at_1") is not True for row in rows)
    rescued = sum(row.get("repair_rescued") is True for row in rows)
    rewards = []
    tokens = []
    costs = []
    latencies = []
    calls = []
    near_bounds = 0
    normalizations = {"strip_code_fence": 0, "strip_think_block": 0}
    normalization_calls = 0
    by_grammar_counts: dict[str, list[int]] = {label: [0, 0] for label in GRAMMAR_LABELS.values()}
    for row in rows:
        row_calls = row.get("calls") if isinstance(row.get("calls"), list) else []
        calls.extend(row_calls)
        rewards.append(max([call.get("reward", 0.0) for call in row_calls
                            if isinstance(call.get("reward", 0.0), (int, float))] or [0.0]))
        usage = (row.get("telemetry") or {}).get("token_usage") or {}
        if isinstance(usage.get("total_tokens"), (int, float)):
            tokens.append(float(usage["total_tokens"]))
        cost = (row.get("telemetry") or {}).get("cost") or {}
        if isinstance(cost.get("estimated_amount"), (int, float)):
            costs.append(float(cost["estimated_amount"]))
        latency = (row.get("telemetry") or {}).get("latency_ms") or {}
        latency_ms = latency.get("provider_reported_total") or latency.get("request_elapsed_total")
        if isinstance(latency_ms, (int, float)):
            latencies.append(float(latency_ms))
        for call in row_calls:
            normalization_calls += 1
            metadata = call.get("metadata") if isinstance(call.get("metadata"), dict) else {}
            normalized = metadata.get("normalizations")
            if normalized is None:
                normalized = call.get("normalizations", [])
            if isinstance(normalized, list):
                actions = {name for name in normalized
                           if isinstance(name, str) and name in normalizations}
                for name in actions:
                    normalizations[name] += 1
            elif isinstance(normalized, dict):
                # Accept previously published count maps while new trajectories store action lists.
                for name in normalizations:
                    value = normalized.get(name, 0)
                    if type(value) in (int, float) and value >= 0:
                        normalizations[name] += int(value)
            near_bounds += int(_near_bound(call))
        grammar = GRAMMAR_LABELS.get(row.get("grammar"))
        if grammar:
            by_grammar_counts[grammar][0] += 1
            by_grammar_counts[grammar][1] += int(row.get("validity_at_3") is True)
    intervals1 = wilson_interval(pass1, total)
    intervals3 = wilson_interval(pass3, total)
    accepted_count = pass3
    model = rows[0]
    return {
        "key": key,
        "label": "Beam" if key == "beam-hf" else key,
        "provider": model.get("provider", "unknown"),
        "run_kind": "community" if "community" in str(model.get("source", "")).lower() else "official",
        "request_params": model.get("request_params", {}),
        "n_tasks": total,
        "completed_prompt_ids": sorted(str(row["prompt_id"]) for row in rows),
        "pass_at_1": pass1 / total,
        "pass_at_3": pass3 / total,
        "ci_pass_at_1": list(intervals1),
        "ci_pass_at_3": list(intervals3),
        "rescue_rate": rescued / initial_failures if initial_failures else 0.0,
        "mean_reward": statistics.fmean(rewards),
        "valid_schema_at_1": schemas / total,
        "tokens_per_task": statistics.fmean(tokens) if tokens else None,
        "est_cost_per_verified": sum(costs) / accepted_count if costs and accepted_count else None,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "length_rate": (sum((call.get("metadata") or {}).get("finish_reason") == "length" for call in calls) / len(calls)
                        if calls else 0.0),
        "near_bound_rate": near_bounds / pass3 if pass3 else 0.0,
        "normalizations": normalizations,
        "normalization_calls": normalization_calls,
        "normalization_rates": {name: count / normalization_calls if normalization_calls else None
                               for name, count in normalizations.items()},
        "by_grammar": {grammar: {"n": values[0], "passed_at_3": values[1]}
                       for grammar, values in by_grammar_counts.items()},
    }


def _baseline_candidate(task: Task, *, centered_dot: bool) -> dict[str, Any]:
    grammar = task.grammar
    colors = list(task.palette)
    ink_index = 1 if len(colors) > 1 else 0
    constraints = {key: value for key, value in task.constraint_profile.items()
                   if key in ConstraintsV2.model_fields}
    constraints["symmetry"] = task.symmetry
    constraints.setdefault("max_colours", min(4, len(colors)))
    candidate: dict[str, Any] = {
        "schema_version": "0.2", "grammar": grammar, "tradition_id": task.tradition_id,
        "title": "Single centred dot baseline" if centered_dot else "Empty tile baseline",
        "tile": {"width": 64, "height": 64, "wrap_x": True, "wrap_y": True},
        "palette": colors, "background_index": 0,
        "layers": [{"id": "baseline-ink", "colour_index": ink_index, "anchor": [0, 0],
                    "motifs": ([{"kind": "dot", "center": [32, 32], "radius": 1}] if centered_dot else [])}],
        "constraints": constraints,
    }
    if grammar == "kolam_loop":
        candidate["tile"] = {"width": 64, "height": 64, "wrap_x": False, "wrap_y": False}
        candidate["kolam"] = {"rows": 1, "cols": 1, "spacing": 6, "mirrors": [], "stroke": 1, "dot_radius": 0}
    elif grammar == "jaali_lattice":
        frame = int(task.constraint_profile.get("frame", 0))
        candidate["tile"] = {"width": 64, "height": 64, "wrap_x": frame == 0, "wrap_y": frame == 0}
        candidate["jaali"] = {"frame": frame, "open_colour_index": 0, "holes": []}
    return candidate


def _baseline_rates(tasks: list[Task]) -> list[dict[str, Any]]:
    result = []
    for key, centered_dot in (("empty-tile", False), ("single-centred-dot", True)):
        passed = 0
        for task in tasks:
            pattern = parse_pattern(json.dumps(_baseline_candidate(task, centered_dot=centered_dot)))
            checks, masks, grid = verify_pattern(pattern)
            make_certificate(pattern, checks, masks, grid)
            passed += int(all_pass(checks))
        result.append({"key": key, "n_tasks": len(tasks), "pass_at_1": passed / len(tasks), "passed": passed})
    return result


def _state_index(claims_path: Path) -> list[dict[str, Any]]:
    claims_doc = json.loads(claims_path.read_text(encoding="utf-8"))
    claim_map = {claim["id"]: claim for claim in claims_doc.get("claims", [])}
    ut_names = {"Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
                "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"}
    output = []
    for state in ALL_STATES:
        traditions = traditions_for_state(state)
        if not traditions:
            raise ValueError(f"no primary tradition for {state}")
        item = traditions[0]
        lens = item.festival_lens
        diwali_tie = None
        if lens is not None and lens.diwali_status != "none" and lens.claim_ids:
            claim = claim_map[lens.claim_ids[0]]
            source = claim["sources"][0]
            diwali_tie = {"claim": claim["text"], "source_url": source["url"]}
        output.append({
            "state": state, "state_type": "union_territory" if state in ut_names else "state",
            "grid": STATE_GRID[state], "tradition_id": item.id, "tradition": item.name,
            "grammar": item.grammar, "tier": item.tier, "caution": item.cautions,
            "sources": [{"title": source.title, "url": source.url} for source in item.sources],
            "diwali_tie": diwali_tie,
        })
    if len(STATE_GRID) != len(ALL_STATES) or len({tuple(value) for value in STATE_GRID.values()}) != 36:
        raise ValueError("the India quilt requires 36 unique state cells")
    return output


def _validate_gallery_summary(arms: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    summary_models = summary.get("by_model")
    if not isinstance(summary_models, dict):
        raise ValueError("summary has no by_model mapping")
    gallery_models = {arm.get("key"): arm for arm in arms}
    if len(gallery_models) != 10 or set(gallery_models) != set(summary_models):
        raise ValueError(
            "gallery and summary model keys differ "
            f"(gallery={sorted(gallery_models)}, summary={sorted(summary_models)})"
        )
    for key, arm in gallery_models.items():
        overall = summary_models[key].get("overall")
        if not isinstance(overall, dict):
            raise ValueError(f"summary for {key} has no overall counts")
        checks = {
            "n": arm["n_tasks"],
            "scored_n": arm["n_tasks"],
            "passed_at_1": round(arm["pass_at_1"] * arm["n_tasks"]),
            "passed_at_3": round(arm["pass_at_3"] * arm["n_tasks"]),
        }
        for field, gallery_count in checks.items():
            summary_count = overall.get(field)
            if summary_count != gallery_count:
                raise ValueError(
                    f"{key} {field} mismatch: gallery={gallery_count}, summary={summary_count}"
                )


def build_gallery_data(*, output_dir: Path = DEFAULT_OUTPUT, suite_path: Path = DEFAULT_SUITE,
                       trajectory_paths: Iterable[Path] | None = None,
                       summary_path: Path | None = None,
                       claims_path: Path | None = None,
                       generated_at: str | None = None) -> Path:
    """Replay benchmark candidates and emit one small index plus one payload per model."""
    trajectory_paths = tuple(trajectory_paths or (DEFAULT_TRAJECTORY,))
    claims_path = claims_path or ROOT / "pramaan" / "data" / "claims.json"
    suite_bytes = suite_path.read_bytes()
    tasks, suite_sha = load_suite("eval-v2.1" if suite_path == DEFAULT_SUITE else _suite_name(suite_path))
    rows = _read_jsonl(trajectory_paths)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path else None
    expected_ids = {task.id for task in tasks}
    task_by_id = {task.id: task for task in tasks}
    allowed_suite_digests = {suite_sha}
    if summary is not None:
        source_digests = summary.get("source_suite_digests")
        if not isinstance(source_digests, dict):
            raise ValueError("summary has no source_suite_digests mapping")
        allowed_suite_digests.update(value for value in source_digests.values()
                                     if isinstance(value, str) and len(value) == 64)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row_digest = row.get("prompt_suite_sha256")
        if row.get("suite_id") != tasks[0].suite_id or (
            row_digest != suite_sha and (summary is None or row_digest not in allowed_suite_digests)
        ):
            raise ValueError(f"trajectory row has mismatched suite identity: {row.get('prompt_id')}")
        task = task_by_id.get(row.get("prompt_id"))
        if task is None:
            raise ValueError(f"trajectory row has prompt id outside the suite: {row.get('prompt_id')}")
        key = str(row.get("model_key") or "")
        if not key:
            raise ValueError(f"trajectory row has no model_key: {row.get('prompt_id')}")
        if summary is not None and (
            row.get("data_role") != "benchmark_evaluation" or row.get("training_eligible") is not False
        ):
            raise ValueError(f"non-evaluation trajectory row in strict gallery input: {row.get('prompt_id')}")
        if summary is not None:
            for field in ("state", "level", "grammar", "tradition_id", "brief"):
                if row.get(field) != getattr(task, field):
                    raise ValueError(f"{key} {row.get('prompt_id')} {field} differs from frozen suite")
        grouped[key].append(row)
    arms = []
    arm_payloads = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, arm_rows in sorted(grouped.items()):
        ids = [row.get("prompt_id") for row in arm_rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"arm {key} has duplicate prompt rows")
        if not set(ids) <= expected_ids:
            raise ValueError(f"arm {key} includes prompt ids outside the frozen suite")
        arm_tasks = _build_attempts(arm_rows)
        for task in tasks:
            arm_tasks.setdefault(task.id, {
                "status": "not_run", "state": task.state, "level": task.level,
                "grammar": task.grammar, "tradition_id": task.tradition_id,
                "brief": task.brief, "passed_at": None, "reward": 0.0,
                "cert_sha256": None, "mask_sha256": None, "attempts": [],
            })
        model = arm_rows[0]
        payload = {
            "key": key,
            "data_role": model.get("data_role", "benchmark_evaluation"),
            "training_eligible": model.get("training_eligible", False),
            "tasks": arm_tasks,
        }
        arm_payloads.append(payload)
        arms.append(_arm_summary(key, arm_rows))
    if summary is not None:
        _validate_gallery_summary(arms, summary)
    for payload in arm_payloads:
        arm_path = output_dir / f"{payload['key']}.json"
        arm_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    baselines = _baseline_rates(tasks)
    completed_sets = [set(arm["completed_prompt_ids"]) for arm in arms]
    common_prompt_ids = sorted(set.intersection(*completed_sets)) if completed_sets else []
    index = {
        "suite_id": tasks[0].suite_id, "suite_sha256": _sha256(canonical_jsonl_bytes(suite_bytes)),
        "validator_version": VALIDATOR_VERSION,
        "generated_at": generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "states": _state_index(claims_path), "arms": arms,
        "common_prompt_ids": common_prompt_ids, "baselines": baselines,
    }
    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return index_path


def _suite_name(path: Path) -> str:
    name = path.name.removesuffix(".jsonl")
    if name not in {"eval-v2", "eval-v2.1", "train-v1"}:
        raise ValueError(f"unsupported suite filename: {path.name}")
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--trajectory", type=Path, action="append", dest="trajectories")
    parser.add_argument("--summary", type=Path,
                        help="summary JSON to validate the model set and per-model counts against")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--claims", type=Path, default=ROOT / "pramaan" / "data" / "claims.json")
    args = parser.parse_args()
    path = build_gallery_data(output_dir=args.output, suite_path=args.suite,
                              trajectory_paths=args.trajectories, summary_path=args.summary,
                              claims_path=args.claims)
    index = json.loads(path.read_text(encoding="utf-8"))
    print(f"PASS gallery: {len(index['states'])} states, {len(index['arms'])} arms, {len(index['baselines'])} baselines")
    for arm in index["arms"]:
        size = (path.parent / f"{arm['key']}.json").stat().st_size
        print(f"  {arm['key']}: {arm['n_tasks']} tasks, pass@1 {arm['pass_at_1']:.3f}, pass@3 {arm['pass_at_3']:.3f}, {size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
