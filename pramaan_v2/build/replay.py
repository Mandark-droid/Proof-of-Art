"""Replay every saved try through the shared checker and classify each run.

The stored reports are trusted only after the checker reproduces them: any
difference is a hard failure (``ReplayMismatch``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import numpy as np

from pramaan.checks.core import all_pass, verify_pattern
from pramaan.dsl import parse_pattern
from pramaan.render import mask_digest
from pramaan.reward import reward_for_report

Outcome = Literal["passed", "broke", "unreadable"]
FORMAT_CHECKS = frozenset({"candidate_parse", "candidate_truncated"})
LOW_INK_SHARE = 0.02


class ReplayMismatch(RuntimeError):
    """A recomputed check list differs from the stored report."""


@dataclass
class Try:
    n: int
    outcome: Outcome
    grid: np.ndarray | None
    palette: list[str] | None
    background_index: int | None
    symmetry: str | None
    checks: list[dict]
    reward: float
    rules_held: tuple[int, int] | None
    pair: dict | None
    mismatch: np.ndarray | None
    measured: str | None
    tokens: int | None
    latency_ms: int | None
    proposal: dict | None
    kolam: dict | None
    masks: dict | None = None


@dataclass
class Run:
    model_key: str
    task_id: str
    tries: list[Try]
    status: str
    shown: Try | None
    total_tokens: int | None
    model_time_ms: int | None
    check_time_ms: int | None
    est_cost_usd: float | None


@dataclass(frozen=True)
class ReplayStats:
    proposals: int
    rendered: int
    mismatches: int


def _int_or_none(value) -> int | None:
    return value if type(value) is int else None


def _group(symmetry: str):
    def rotations(g):
        return [np.rot90(g, k) for k in (1, 2, 3)]
    flips = lambda g: [g[:, ::-1], g[::-1, :], g[::-1, ::-1]]  # noqa: E731
    return {
        "D2": flips,
        "C2": lambda g: [g[::-1, ::-1]],
        "R180": lambda g: [g[::-1, ::-1]],
        "C4": rotations,
        "D4": lambda g: rotations(g) + [g[:, ::-1], g[::-1, :], g.T, np.rot90(g, 2).T],
    }.get(symmetry)


def symmetry_mismatch(grid: np.ndarray, symmetry: str) -> np.ndarray | None:
    """Pixels whose colour differs from their image under any transform of the group."""
    transforms = _group(symmetry)
    if transforms is None:
        return None
    mask = np.zeros(grid.shape, dtype=bool)
    for image in transforms(grid):
        mask |= grid != image
    return mask


def _pair(checks: list[dict]) -> dict | None:
    for check in checks:
        if check.get("status") != "FAIL":
            continue
        example = check.get("counterexample")
        if not isinstance(example, dict):
            continue
        if "original" in example and "transformed" in example:
            return {"check": check["id"], "transform": example.get("transform"),
                    "original": example["original"], "transformed": example["transformed"]}
        if "pixel" in example:
            return {"check": check["id"], "pixel": example["pixel"]}
    return None


def _rules_held(checks: list[dict]) -> tuple[int, int] | None:
    hard = [check for check in checks if check.get("id") != "schema_valid"]
    if not hard or any(check.get("id") in FORMAT_CHECKS for check in checks):
        return None
    return sum(check.get("status") == "PASS" for check in hard), len(hard)


def _replay_try(attempt: dict) -> tuple[Try, bool]:
    report = attempt.get("report") or {}
    stored = report.get("checks") or []
    metadata = attempt.get("metadata") or {}
    usage = metadata.get("usage") or {}
    candidate = attempt.get("candidate")
    failed_ids = {check.get("id") for check in stored if check.get("status") == "FAIL"}
    base = dict(n=attempt["index"], checks=stored, reward=reward_for_report(report),
                tokens=_int_or_none(usage.get("total_tokens")),
                latency_ms=_int_or_none(metadata.get("provider_elapsed_ms")),
                proposal=candidate if isinstance(candidate, dict) else None)
    if not isinstance(candidate, dict) or failed_ids & FORMAT_CHECKS:
        failing = next((check for check in stored if check.get("status") == "FAIL"), None)
        return Try(outcome="unreadable", grid=None, palette=None, background_index=None, symmetry=None,
                   rules_held=None, pair=None, mismatch=None,
                   measured=failing.get("message") if failing else None, kolam=None, **base), False
    pattern = parse_pattern(json.dumps(candidate, ensure_ascii=False))
    checks, masks, grid = verify_pattern(pattern)
    if checks != [check for check in stored if check.get("id") != "candidate_truncated"]:
        raise ReplayMismatch(f"try {attempt['index']}: recomputed checks differ from the stored report")
    passed = all_pass(checks)
    symmetry = pattern.constraints.symmetry
    mismatch = None
    if not passed and any(check["status"] == "FAIL" and check["id"].startswith("symmetry_") for check in checks):
        mismatch = symmetry_mismatch(np.asarray(grid), symmetry)
    pair = None if passed else _pair(checks)
    failing = next((check for check in checks if check["status"] == "FAIL"), None)
    return Try(outcome="passed" if passed else "broke", grid=np.asarray(grid, dtype=np.uint8),
               palette=list(candidate.get("palette") or []),
               background_index=candidate.get("background_index", 0), symmetry=symmetry,
               rules_held=_rules_held(checks), pair=pair, mismatch=mismatch,
               measured=None if passed or failing is None else failing.get("message"),
               kolam=candidate.get("kolam"), masks=masks, **base), True


def _status(tries: list[Try]) -> tuple[str, Try | None]:
    for item in tries:
        if item.outcome == "passed":
            return str(item.n), item
    rendered = [item for item in tries if item.grid is not None]
    return ("x", rendered[-1]) if rendered else ("-", None)


def replay_all(rows: list[dict]) -> tuple[dict[tuple[str, str], Run], ReplayStats]:
    runs: dict[tuple[str, str], Run] = {}
    proposals = rendered = 0
    for row in rows:
        tries = []
        for attempt in row["attempts"]:
            try:
                item, did_render = _replay_try(attempt)
            except ReplayMismatch as exc:
                raise ReplayMismatch(f"{row['model_key']} / {row['prompt_id']} {exc}") from None
            proposals += 1
            rendered += did_render
            tries.append(item)
        status, shown = _status(tries)
        latency = row.get("latency_ms") or {}
        cost = row.get("cost") or {}
        runs[(row["model_key"], row["prompt_id"])] = Run(
            model_key=row["model_key"], task_id=row["prompt_id"], tries=tries, status=status, shown=shown,
            total_tokens=_int_or_none(row.get("total_tokens")),
            model_time_ms=_int_or_none(latency.get("request_elapsed_total")),
            check_time_ms=_int_or_none(latency.get("verification_total")),
            est_cost_usd=cost.get("estimated_amount") if isinstance(cost.get("estimated_amount"), (int, float)) else None,
        )
    return runs, ReplayStats(proposals=proposals, rendered=rendered, mismatches=0)


def clean_rgba(item: Try) -> np.ndarray:
    """The tile as drawn: ``palette[grid]`` with full alpha, no markings."""
    colours = np.array([tuple(bytes.fromhex(value.lstrip("#"))) for value in item.palette], dtype=np.uint8)
    rgb = colours[item.grid]
    alpha = np.full(item.grid.shape + (1,), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=2)


def low_ink(item: Try | None) -> bool:
    """Fewer than 2% of pixels differ in colour from the background colour."""
    if item is None or item.grid is None:
        return False
    palette = [value.lower() for value in item.palette]
    background = palette[item.background_index or 0]
    differs = np.array([value != background for value in palette])
    return float(differs[item.grid].mean()) < LOW_INK_SHARE


def pixels_sha256(item: Try) -> str:
    """The certificate's pixel digest for a rendered try."""
    return mask_digest(item.masks, item.grid)
