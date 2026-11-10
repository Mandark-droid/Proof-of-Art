"""Write the v2 site data: deterministic, offline after the first download, no model calls.

    python -m pramaan_v2.build.site_data            # rebuild pramaan_v2/site_data/
    python -m pramaan_v2.build.site_data --check    # fail if the committed files are stale
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from pramaan.checks.core import verify_pattern
from pramaan.dsl import parse_pattern
from pramaan.tasks import load_suite

from ..constants import GRAMMAR_LABELS, GRAMMAR_ORDER, STATE_CODES, STATE_ORDER
from .kolam_paths import kolam_data
from .metrics import failing_first_try, grammar_brief_counts, model_metrics
from .replay import Run, Try, clean_rgba, low_ink, pixels_sha256, replay_all
from .sources import Published, check_ids, load_published
from .traditions_data import traditions_data

PACKAGE = Path(__file__).resolve().parents[1]
OUT = PACKAGE / "site_data"
ROOT = PACKAGE.parent
REFERENCES = ROOT / "benchmarks" / "data" / "eval-v2.1-references.jsonl"
SHORT_NAMES = PACKAGE / "short_names.json"
CELL = 64
EVIDENCE = (79, 209, 197, 255)
REWARD_VERSION = "pramaan-reward-v1"
UNION_TERRITORIES = frozenset(STATE_ORDER[28:])


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def tile_png(rgba: np.ndarray) -> bytes:
    return _png_bytes(Image.fromarray(rgba, "RGBA").quantize(colors=256, method=Image.Quantize.FASTOCTREE,
                                                             dither=Image.Dither.NONE))


def mask_png(mask: np.ndarray) -> bytes:
    out = np.zeros(mask.shape + (4,), dtype=np.uint8)
    out[mask] = EVIDENCE
    return _png_bytes(Image.fromarray(out, "RGBA"))


def _short_names() -> dict[str, str]:
    return json.loads(SHORT_NAMES.read_text(encoding="utf-8"))


def _references() -> dict[str, np.ndarray]:
    tiles = {}
    for line in REFERENCES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        candidate = row.get("candidate_pattern")
        if not row.get("all_hard_constraints_pass") or not isinstance(candidate, dict):
            continue
        pattern = parse_pattern(json.dumps(candidate, ensure_ascii=False))
        _checks, _masks, grid = verify_pattern(pattern)
        palette = np.array([tuple(bytes.fromhex(c.lstrip("#"))) for c in candidate["palette"]], dtype=np.uint8)
        rgb = palette[np.asarray(grid)]
        tiles[row["task_id"]] = np.concatenate([rgb, np.full((64, 64, 1), 255, np.uint8)], axis=2)
    return tiles


def _try_json(item: Try) -> dict:
    entry = {"n": item.n, "outcome": item.outcome, "reward": item.reward,
             "rules_held": list(item.rules_held) if item.rules_held else None,
             "tokens": item.tokens, "latency_ms": item.latency_ms, "checks": item.checks,
             "pair": item.pair, "measured": item.measured, "proposal": item.proposal,
             "mismatch_px": int(item.mismatch.sum()) if item.mismatch is not None else None}
    if item.grid is not None:
        entry["tile_png"] = _b64(tile_png(clean_rgba(item)))
    if item.mismatch is not None:
        entry["mask_png"] = _b64(mask_png(item.mismatch))
    if item.outcome == "passed":
        entry["pixels_sha256"] = pixels_sha256(item)
    return entry


def _write(out: Path, name: str, data: bytes, hashes: dict[str, str]) -> None:
    path = out / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    hashes[name] = hashlib.sha256(data).hexdigest()


def build(out: Path, published: Published) -> dict[str, str]:
    """Write every site data file under ``out``; return {relative path: sha256}."""
    out = Path(out)
    check_ids(published.rows)
    runs, stats = replay_all(published.rows)
    tasks, suite_sha = load_suite("eval-v2.1")
    metrics = model_metrics(runs, published.summary, tasks)
    order = [m.key for m in metrics]
    labels = {m.key: m.label for m in metrics}
    short = _short_names()
    registry_names = {s["id"]: s["name"] for state in traditions_data()["states"] for s in state["traditions"]}
    tradition_name = lambda tid: short.get(tid) or registry_names.get(tid, tid)  # noqa: E731
    by_state = {}
    for task in tasks:
        slot = by_state.setdefault(task.state, {})
        if task.level in slot:
            raise ValueError(f"briefs {slot[task.level].id} and {task.id} have the same state and level "
                             f"({task.state}, {task.level}); the wall has one cell for each")
        slot[task.level] = task
    hashes: dict[str, str] = {}

    # Wall
    rows = 2 * len(order)
    sprite = Image.new("RGBA", (36 * CELL, rows * CELL), (0, 0, 0, 0))
    wall_runs = []
    for mi, model in enumerate(order):
        for si, state in enumerate(STATE_ORDER):
            for li, level in enumerate(("easy", "hard")):
                task = by_state[state][level]
                run: Run | None = runs.get((model, task.id))
                if run is None:
                    continue
                row = mi * 2 + li
                if run.shown is not None:
                    sprite.paste(Image.fromarray(clean_rgba(run.shown), "RGBA"), (si * CELL, row * CELL))
                wall_runs.append({"model": model, "task": task.id, "state": STATE_CODES[state], "level": level,
                                  "col": si, "row": row, "status": run.status,
                                  "low_ink": run.status in "123" and low_ink(run.shown),
                                  "tradition": tradition_name(task.tradition_id)})
    _write(out, "wall.png", _png_bytes(sprite), hashes)
    _write(out, "wall.json", _json_bytes({"sprite": "wall.png", "cell": CELL, "cols": 36, "rows": rows,
                                          "runs": wall_runs}), hashes)

    # Meta
    statuses = [run.status for run in runs.values()]
    evidence = sum(1 for run in runs.values() for t in run.tries if t.outcome != "passed" and t.pair)
    validator = next((row.get("validator_version") for row in published.rows if row.get("validator_version")), None)
    meta = {
        "suite_id": "eval-v2.1", "suite_sha256": suite_sha, "validator": validator, "reward": REWARD_VERSION,
        "dataset": published.dataset, "revision": published.revision,
        "models": [{"key": m.key, "label": m.label, "provider": m.provider, "request_params": m.request_params,
                    "rank": i + 1 if m.complete else None, "complete": m.complete}
                   for i, m in enumerate(metrics)],
        "states": [{"name": s, "code": STATE_CODES[s], "type": "union_territory" if s in UNION_TERRITORIES else "state",
                    "index": i, "easy": by_state[s]["easy"].id, "hard": by_state[s]["hard"].id}
                   for i, s in enumerate(STATE_ORDER)],
        "grammars": [{"id": g, "label": GRAMMAR_LABELS[g], "briefs": n} for g, n in grammar_brief_counts(tasks).items()],
        "totals": {"models": len(order), "briefs": len(tasks), "runs": len(runs),
                   "proposals": stats.proposals, "rendered": stats.rendered, "pixel_evidence": evidence,
                   "passed": sum(s in "123" for s in statuses), "broke": statuses.count("x"),
                   "never": statuses.count("-"), "low_ink": sum(r["low_ink"] for r in wall_runs)},
    }
    _write(out, "meta.json", _json_bytes(meta), hashes)

    # Leaderboard
    groups = [{"grammar": g, "tasks": [by_state[s][lvl].id for s in STATE_ORDER for lvl in ("easy", "hard")
                                       if by_state[s][lvl].grammar == g]} for g in GRAMMAR_ORDER]
    _write(out, "leaderboard.json", _json_bytes({
        "models": [asdict(m) for m in metrics], "failing_first_try": failing_first_try(runs),
        "grammar_briefs": grammar_brief_counts(tasks), "matrix": {"groups": groups},
    }), hashes)

    # Replays, one file per brief
    references = _references()
    for si, state in enumerate(STATE_ORDER):
        for li, level in enumerate(("easy", "hard")):
            task = by_state[state][level]
            model_runs = [runs[(m, task.id)] for m in order if (m, task.id) in runs]
            reference = references.get(task.id)
            _write(out, f"replays/{task.id}.json", _json_bytes({
                "task": task.id, "number": si * 2 + li + 1, "state": state, "state_code": STATE_CODES[state],
                "level": level, "tradition_id": task.tradition_id, "tradition": tradition_name(task.tradition_id),
                "grammar": task.grammar, "symmetry": task.symmetry, "brief": task.brief,
                "reference": {"tile_png": _b64(tile_png(reference))} if reference is not None else None,
                "summary": {"first_try": sum(r.status == "1" for r in model_runs),
                            "repaired": sum(r.status in ("2", "3") for r in model_runs),
                            "never": sum(r.status in ("x", "-") for r in model_runs)},
                "models": [{"key": r.model_key, "label": labels[r.model_key], "status": r.status,
                            "total_tokens": r.total_tokens, "model_time_ms": r.model_time_ms,
                            "check_time_ms": r.check_time_ms, "est_cost_usd": r.est_cost_usd,
                            "tries": [_try_json(t) for t in r.tries]} for r in model_runs],
            }), hashes)

    _write(out, "kolam.json", _json_bytes(kolam_data(runs, tasks)), hashes)
    _write(out, "traditions.json", _json_bytes(traditions_data()), hashes)
    return hashes


def _committed_hashes(out: Path) -> dict[str, str]:
    return {path.relative_to(out).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(out.rglob("*")) if path.is_file() and path.name != ".gitattributes"}


def check(out: Path, published: Published) -> list[str]:
    """Rebuild into a temporary directory; return the paths that differ from ``out``."""
    with tempfile.TemporaryDirectory() as temp:
        fresh = build(Path(temp), published)
    committed = _committed_hashes(Path(out))
    return sorted(name for name in set(fresh) | set(committed) if fresh.get(name) != committed.get(name))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--compact", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    published = load_published(args.compact, args.summary)
    if args.check:
        stale = check(args.out, published)
        if stale:
            print("site data is stale; rebuild with python -m pramaan_v2.build.site_data:\n  " + "\n  ".join(stale))
            return 1
        print(f"site data is current ({published.revision})")
        return 0
    hashes = build(args.out, published)
    print(f"wrote {len(hashes)} files to {args.out} from {published.dataset}@{published.revision[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
