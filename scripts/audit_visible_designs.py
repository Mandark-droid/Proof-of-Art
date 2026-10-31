"""Read-only audit of accepted patterns whose rendered ink is nearly invisible."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from pramaan.checks.core import all_pass, verify_pattern
from pramaan.dsl import parse_pattern
from pramaan.visibility import MIN_VISIBLE_SHARE

ROOT = Path(__file__).resolve().parents[1]
PIXELS_PER_TILE = 64 * 64
MIN_VISIBLE_PIXELS = math.ceil(PIXELS_PER_TILE * MIN_VISIBLE_SHARE)


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected an object")
                yield row


def _candidate(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _render_stats(value: Any) -> dict[str, Any] | None:
    candidate = _candidate(value)
    if candidate is None:
        return None
    try:
        pattern = parse_pattern(json.dumps(candidate, ensure_ascii=False))
        checks, _masks, grid = verify_pattern(pattern)
    except Exception:
        return None
    if grid is None or not hasattr(grid, "size"):
        return None
    background = candidate.get("background_index", 0)
    if type(background) is not int:
        return None
    pixels = int(grid.size)
    visible_pixels = int((grid != background).sum())
    return {
        "accepted": all_pass(checks),
        "pixels": pixels,
        "visible_pixels": visible_pixels,
        "visible_share": visible_pixels / pixels if pixels else 0.0,
        "meets_threshold": bool(pixels and visible_pixels >= math.ceil(pixels * MIN_VISIBLE_SHARE)),
    }


def audit_benchmark(trajectory_path: Path, summary_path: Path) -> tuple[list[dict[str, Any]], dict[str, dict]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_models = summary.get("by_model")
    if not isinstance(expected_models, dict):
        raise ValueError("summary JSON has no by_model mapping")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _jsonl(trajectory_path):
        key = str(row.get("model_key") or "")
        if not key:
            raise ValueError(f"trajectory row {row.get('prompt_id')} has no model_key")
        grouped[key].append(row)
    if set(grouped) != set(expected_models):
        raise ValueError(f"model keys differ: trajectories={sorted(grouped)}, summary={sorted(expected_models)}")

    results: list[dict[str, Any]] = []
    indexed_rows: dict[str, dict] = {}
    for key in sorted(grouped):
        rows = grouped[key]
        seen_ids: set[str] = set()
        pass1 = pass1_visible = pass3 = pass3_visible = solid = under2 = 0
        for row in rows:
            prompt_id = str(row.get("prompt_id") or "")
            if not prompt_id or prompt_id in seen_ids:
                raise ValueError(f"{key} has a missing or duplicate prompt_id: {prompt_id!r}")
            seen_ids.add(prompt_id)
            indexed_rows[f"{key}:{prompt_id}"] = row
            accepted: list[tuple[int, dict[str, Any]]] = []
            for call in row.get("calls", []):
                if not isinstance(call, dict):
                    continue
                candidate_index = call.get("candidate_index")
                stats = _render_stats(call.get("candidate"))
                if stats is None or type(candidate_index) is not int:
                    continue
                if stats["accepted"]:
                    accepted.append((candidate_index, stats))
            accepted.sort(key=lambda item: item[0])
            first = next((stats for index, stats in accepted if index == 1), None)
            within_three = next((stats for index, stats in accepted if 1 <= index <= 3), None)
            replay_first, replay_three = first is not None, within_three is not None
            if replay_first != (row.get("validity_at_1") is True):
                raise ValueError(f"{key}:{prompt_id} replay disagrees with validity_at_1")
            if replay_three != (row.get("validity_at_3") is True):
                raise ValueError(f"{key}:{prompt_id} replay disagrees with validity_at_3")
            if first:
                pass1 += 1
                pass1_visible += int(first["meets_threshold"])
            if within_three:
                pass3 += 1
                pass3_visible += int(within_three["meets_threshold"])
                solid += int(within_three["visible_pixels"] == 0)
                under2 += int(0 < within_three["visible_share"] < MIN_VISIBLE_SHARE)

        overall = expected_models[key].get("overall", {})
        checks = {
            "n": len(rows), "scored_n": len(rows),
            "passed_at_1": pass1, "passed_at_3": pass3,
        }
        for field, observed in checks.items():
            if overall.get(field) != observed:
                raise ValueError(f"{key} {field} differs from summary: replay={observed}, summary={overall.get(field)}")
        results.append({
            "model": key, "tasks": len(rows),
            "pass1_total": pass1, "pass1_visible": pass1_visible,
            "pass1_low": pass1 - pass1_visible,
            "pass3_total": pass3, "pass3_visible": pass3_visible,
            "pass3_low": pass3 - pass3_visible,
            "pass3_solid": solid, "pass3_under2": under2,
        })
    return results, indexed_rows


def audit_training_config(path: Path, name: str, fields: tuple[str, ...]) -> dict[str, int | str]:
    rows = list(_jsonl(path))
    candidates = accepted = low = solid = invalid = 0
    for row in rows:
        for field in fields:
            if field not in row:
                continue
            if field == "calls" and isinstance(row[field], list):
                values = [call.get("candidate") or call.get("raw_response")
                          for call in row[field] if isinstance(call, dict)]
            else:
                values = row[field] if isinstance(row[field], list) else [row[field]]
            for value in values:
                candidates += 1
                stats = _render_stats(value)
                if stats is None:
                    invalid += 1
                elif stats["accepted"]:
                    accepted += 1
                    low += int(not stats["meets_threshold"])
                    solid += int(stats["visible_pixels"] == 0)
    return {"config": name, "rows": len(rows), "assistant_candidates": candidates,
            "verifier_accepted": accepted, "accepted_under_2pct": low,
            "accepted_solid": solid, "unparsed_or_unrenderable": invalid}


def audit_references(path: Path) -> dict[str, int]:
    rows = list(_jsonl(path))
    accepted = low = solid = 0
    for row in rows:
        stats = _render_stats(row.get("candidate_pattern"))
        if stats and stats["accepted"]:
            accepted += 1
            low += int(not stats["meets_threshold"])
            solid += int(stats["visible_pixels"] == 0)
    return {"rows": len(rows), "accepted": accepted, "accepted_under_2pct": low,
            "accepted_solid": solid}


def audit_gallery(gallery_dir: Path, model_keys: set[str],
                  trajectory_rows: dict[str, dict]) -> dict[str, int]:
    index = json.loads((gallery_dir / "index.json").read_text(encoding="utf-8"))
    arms = {arm["key"] for arm in index.get("arms", []) if isinstance(arm, dict) and "key" in arm}
    if arms != model_keys:
        raise ValueError(f"gallery model keys differ from audited data: {sorted(arms)}")
    tasks = nonempty_roles = accepted_low = accepted_solid = accepted_under2 = 0
    for key in sorted(arms):
        document = json.loads((gallery_dir / f"{key}.json").read_text(encoding="utf-8"))
        model_tasks = document.get("tasks", {})
        if len(model_tasks) != 72:
            raise ValueError(f"gallery {key} has {len(model_tasks)} tasks, expected 72")
        if document.get("data_role") != "benchmark_evaluation" or document.get("training_eligible") is not False:
            raise ValueError(f"gallery {key} is not marked evaluation-only")
        tasks += len(model_tasks)
        nonempty_roles += int(any(task.get("attempts") for task in model_tasks.values()))
        for prompt_id, gallery_task in model_tasks.items():
            source = trajectory_rows.get(f"{key}:{prompt_id}")
            if source is None:
                raise ValueError(f"gallery task is missing from trajectories: {key}:{prompt_id}")
            gallery_passes = {(item.get("index"), item.get("passed"))
                              for item in gallery_task.get("attempts", [])}
            for call in source.get("calls", []):
                if not isinstance(call, dict):
                    continue
                candidate_index = call.get("candidate_index")
                if type(candidate_index) is not int or not 1 <= candidate_index <= 3:
                    continue
                stats = _render_stats(call.get("candidate"))
                if not stats or not stats["accepted"]:
                    continue
                if (candidate_index, True) not in gallery_passes:
                    raise ValueError(f"gallery is missing accepted attempt {key}:{prompt_id}:{candidate_index}")
                if not stats["meets_threshold"]:
                    accepted_low += 1
                    accepted_solid += int(stats["visible_pixels"] == 0)
                    accepted_under2 += int(0 < stats["visible_share"] < MIN_VISIBLE_SHARE)
    return {"models": len(arms), "tasks": tasks, "models_with_attempts": nonempty_roles,
            "accepted_under_2pct": accepted_low, "accepted_solid": accepted_solid,
            "accepted_nonzero_under_2pct": accepted_under2}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, default=ROOT / ".pramaan-exports/merged-publication/eval-v2.1-trajectories.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / "benchmarks/data/eval-v2.1-summary.json")
    parser.add_argument("--train-trajectories", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--preference", type=Path, required=True)
    parser.add_argument("--references", type=Path, default=ROOT / "benchmarks/data/eval-v2.1-references.jsonl")
    parser.add_argument("--gallery-dir", type=Path, default=ROOT / "assets/gallery")
    parser.add_argument("--home-prompt-id", default="gj_bandhani-easy-6")
    args = parser.parse_args()

    models, trajectory_rows = audit_benchmark(args.trajectories, args.summary)
    print("MODEL | First try visible / accepted | Within three visible / accepted | <2% within three (solid / nonzero)")
    for item in models:
        print(f"{item['model']} | {item['pass1_visible']}/{item['pass1_total']} | "
              f"{item['pass3_visible']}/{item['pass3_total']} | "
              f"{item['pass3_solid']} / {item['pass3_under2']}")

    training = [
        audit_training_config(args.train_trajectories, "train trajectories", ("calls",)),
        audit_training_config(args.sft, "SFT", ("completion",)),
        audit_training_config(args.repair, "repair", ("completion",)),
        audit_training_config(args.preference, "preference", ("chosen", "rejected")),
    ]
    print("\nDATASET | rows | assistant candidates | verifier accepted | accepted <2% | solid | unparsed/unrenderable")
    for item in training:
        print(f"{item['config']} | {item['rows']} | {item['assistant_candidates']} | "
              f"{item['verifier_accepted']} | {item['accepted_under_2pct']} | "
              f"{item['accepted_solid']} | {item['unparsed_or_unrenderable']}")

    home_key = f"beam-hf:{args.home_prompt_id}"
    home = trajectory_rows.get(home_key)
    if home is None:
        raise ValueError(f"Home proof replay row not found: {home_key}")
    home_stats = []
    for call in home.get("calls", []):
        stats = _render_stats(call.get("candidate")) if isinstance(call, dict) else None
        if stats and stats["accepted"]:
            home_stats.append((call.get("candidate_index"), stats))
    home_stats.sort(key=lambda pair: pair[0] if type(pair[0]) is int else 999)
    if not home_stats:
        raise ValueError("Home Proof Replay has no independently accepted candidate")
    home_pick = home_stats[0][1]
    reference_stats = audit_references(args.references)
    gallery_stats = audit_gallery(args.gallery_dir, {item["model"] for item in models}, trajectory_rows)
    print("\nEXAMPLES")
    print(f"Home Proof Replay {args.home_prompt_id}: visible={home_pick['visible_pixels']}/{home_pick['pixels']} "
          f"({home_pick['visible_share']:.2%}); meets_2pct={home_pick['meets_threshold']}")
    print(f"Gallery: {gallery_stats['models']} models, {gallery_stats['tasks']} tasks; "
          f"{gallery_stats['accepted_under_2pct']} low-visibility PASS@3 designs are present as saved attempts "
          f"({gallery_stats['accepted_solid']} solid, {gallery_stats['accepted_nonzero_under_2pct']} nonzero under 2%).")
    print(f"References: {reference_stats['accepted']}/{reference_stats['rows']} accepted, "
          f"{reference_stats['accepted_under_2pct']} accepted under 2% visible, "
          f"{reference_stats['accepted_solid']} solid.")

    print("\nREAD-ONLY audit complete; no score, suite, validator, or dataset file was written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
