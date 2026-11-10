"""Per-model leaderboard numbers derived from replayed runs and the published summary."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from pramaan.ui.findings import wilson_interval

from ..constants import GRAMMAR_ORDER
from .replay import FORMAT_CHECKS, Run, low_ink

_ANCHOR_LOCATION = re.compile(r"layers\.\d+\.anchor(\.\d+)?")


@dataclass
class ModelMetrics:
    key: str
    label: str
    provider: str | None
    request_params: dict
    provenance: str
    n: int
    within3: int
    first: dict[str, int]
    readable: int
    passed_of_readable: tuple[int, int]
    repaired: tuple[int, int]
    wilson95: tuple[float, float] | None
    tokens_per_brief: float | None
    latency_s: float | None
    spend_usd: float | None
    spend_cap_usd: float | None
    by_grammar: dict[str, int]
    low_ink: int
    anchor_only: int
    truncated: int
    complete: bool
    on_frontier: bool = False
    radar: dict[str, float] = field(default_factory=dict)


def _failed_ids(checks: list[dict]) -> list[str]:
    return [check["id"] for check in checks if check.get("status") == "FAIL"]


def is_anchor_only(checks: list[dict]) -> bool:
    """A parse rejection whose every validation error sits under ``layers[n].anchor``."""
    parse = next((c for c in checks if c.get("id") == "candidate_parse" and c.get("status") == "FAIL"), None)
    if parse is None:
        return False
    lines = str(parse.get("message") or "").splitlines()[1:]
    locations = [line.strip() for line in lines if line and not line[0].isspace()]
    return bool(locations) and all(_ANCHOR_LOCATION.fullmatch(location) for location in locations)


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def radar_values(m: ModelMetrics) -> dict[str, float]:
    """Spec 9.7: six spokes in 0..1, outward is always better.

    The economy spokes use the values as displayed (tokens to 0.1k, latency to
    0.1 s, spend to the cent) so the chart agrees with the table beside it.
    """
    missed = m.repaired[1]
    tokens_k = round(m.tokens_per_brief / 1000, 1) if m.tokens_per_brief else None
    latency = round(m.latency_s, 1) if m.latency_s else None
    spend = round(m.spend_usd, 2) if m.spend_usd else None
    return {
        "passed": _clamp(m.within3 / m.n) if m.n else 0.0,
        "readable": _clamp(m.readable / m.n) if m.n else 0.0,
        "repaired": _clamp(m.repaired[0] / missed) if missed else 0.0,
        "tokens": _clamp(math.log2(16 / tokens_k) / 3) if tokens_k else 0.0,
        "speed": _clamp(math.log2(64 / latency) / 4) if latency else 0.0,
        "spend": _clamp((math.log10(32) - math.log10(spend)) / (math.log10(32) - math.log10(0.1)))
        if spend else 0.0,
    }


def _model(key: str, runs: list[Run], document: dict, task_grammar: dict[str, str]) -> ModelMetrics:
    overall = document.get("overall") or {}
    profile = document.get("request_profile") or {}
    first = Counter()
    readable = anchor = truncated = low = 0
    repaired_num = repaired_den = 0
    grammar = Counter()
    for run in runs:
        # A run with no saved attempts is never readable (status "-") and has no first try to count.
        try1 = run.tries[0] if run.tries else None
        if try1 is not None:
            first[try1.outcome] += 1
            if try1.outcome != "unreadable":
                readable += 1
            anchor += is_anchor_only(try1.checks)
            truncated += "candidate_truncated" in _failed_ids(try1.checks)
        if run.status != "1":
            repaired_den += 1
            repaired_num += run.status in ("2", "3")
        if run.status in ("1", "2", "3"):
            grammar[task_grammar[run.task_id]] += 1
            low += low_ink(run.shown)
    n = len(runs)
    within = sum(grammar.values())
    usd = (overall.get("estimated_cost_totals_by_currency") or {}).get("USD")
    latency_ms = overall.get("mean_request_elapsed_ms_per_prompt")
    return ModelMetrics(
        key=key, label=str(document.get("label") or key), provider=profile.get("provider"),
        request_params=dict(profile.get("request_params") or {}), provenance="canonical", n=n,
        within3=within, first={"passed": first["passed"], "broke": first["broke"], "unreadable": first["unreadable"]},
        readable=readable, passed_of_readable=(first["passed"], readable),
        repaired=(repaired_num, repaired_den), wilson95=wilson_interval(within, n),
        tokens_per_brief=overall.get("mean_tokens_when_available"),
        latency_s=latency_ms / 1000 if isinstance(latency_ms, (int, float)) else None,
        spend_usd=usd if isinstance(usd, (int, float)) else None,
        spend_cap_usd=overall.get("spend_cap_usd"),
        by_grammar={g: grammar[g] for g in GRAMMAR_ORDER},
        low_ink=low, anchor_only=anchor, truncated=truncated, complete=False,
    )


def model_metrics(runs: dict[tuple[str, str], Run], summary: dict, tasks) -> list[ModelMetrics]:
    """Every model's numbers, in leaderboard order (complete runs ranked first)."""
    task_grammar = {task.id: task.grammar for task in tasks}
    by_model: dict[str, list[Run]] = {}
    for (model_key, _task), run in runs.items():
        by_model.setdefault(model_key, []).append(run)
    documents = summary.get("by_model") or {}
    metrics = [_model(key, model_runs, documents.get(key) or {}, task_grammar)
               for key, model_runs in by_model.items()]
    for m in metrics:
        m.complete = m.n == len(task_grammar)
    metrics.sort(key=lambda m: (not m.complete, -m.within3 / max(m.n, 1), -m.first["passed"] / max(m.n, 1),
                                m.label.casefold()))
    best = -1
    for m in sorted((m for m in metrics if m.spend_usd is not None), key=lambda m: m.spend_usd):
        if m.within3 > best:
            m.on_frontier = True
            best = m.within3
    for m in metrics:
        m.radar = radar_values(m)
    return metrics


def failing_first_try(runs: dict[tuple[str, str], Run]) -> list[dict]:
    """Failing checks over every first proposal, most frequent first."""
    counts = Counter(check_id for run in runs.values() if run.tries for check_id in _failed_ids(run.tries[0].checks))
    return [{"check": check_id, "count": count, "kind": "format" if check_id in FORMAT_CHECKS else "geometry"}
            for check_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def grammar_brief_counts(tasks) -> dict[str, int]:
    counts = Counter(task.grammar for task in tasks)
    return {g: counts[g] for g in GRAMMAR_ORDER}
