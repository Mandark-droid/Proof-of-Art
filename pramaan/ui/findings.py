"""Small, evidence-linked headline metrics from the canonical eval summary."""

from __future__ import annotations

import math
from html import escape as html_escape

from .visibility import accepted_visibility_counts


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float] | None:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if type(successes) is not int or type(total) is not int or total <= 0 or not 0 <= successes <= total:
        return None
    p = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    centre = (p + z2 / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z2 / (4 * total)) / total) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _repairs_after_rule_failure(document: object) -> int | None:
    tasks = document.get("tasks") if isinstance(document, dict) else None
    if not isinstance(tasks, dict):
        return None
    total = 0
    for task in tasks.values():
        if not isinstance(task, dict) or type(task.get("passed_at")) is not int or task["passed_at"] <= 1:
            continue
        attempts = task.get("attempts")
        first = attempts[0] if isinstance(attempts, list) and attempts else None
        failing = first.get("failing_checks") if isinstance(first, dict) else None
        if (isinstance(failing, list) and failing and first.get("passed") is False
                and "candidate_parse" not in failing):
            total += 1
    return total


def _jaali_pass_counts(models: dict) -> tuple[int, int] | None:
    passed = total = 0
    for document in models.values():
        grammars = document.get("by_grammar") if isinstance(document, dict) else None
        jaali = grammars.get("jaali_lattice") if isinstance(grammars, dict) else None
        if not isinstance(jaali, dict) or type(jaali.get("n")) is not int or type(jaali.get("accepted_designs")) is not int:
            return None
        passed += jaali["accepted_designs"]
        total += jaali["n"]
    return (passed, total) if total else None


def _request_latency_ms(document: object) -> float | None:
    overall = document.get("overall") if isinstance(document, dict) else None
    value = overall.get("mean_request_elapsed_ms_per_prompt") if isinstance(overall, dict) else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0 else None


def render_findings_strip(summary: dict | None, *, gallery_documents: dict[str, dict] | None = None,
                          task_backgrounds: dict[str, str] | None = None) -> str:
    """Render saved-data findings about repair sources, grammar, latency and visibility."""
    summary = summary if isinstance(summary, dict) else {}
    models = summary.get("by_model") if isinstance(summary.get("by_model"), dict) else {}
    gallery_documents = gallery_documents if isinstance(gallery_documents, dict) else {}
    beam = models.get("beam-hf", {})
    opus = models.get("claude-opus-5-5", {})
    beam_overall = beam.get("overall", {}) if isinstance(beam, dict) else {}
    opus_overall = opus.get("overall", {}) if isinstance(opus, dict) else {}
    beam_repairs = _repairs_after_rule_failure(gallery_documents.get("beam-hf"))
    opus_repairs = _repairs_after_rule_failure(gallery_documents.get("claude-opus-5-5"))
    beam_total = beam_overall.get("repair_rescued") if isinstance(beam_overall, dict) else None
    opus_total = opus_overall.get("repair_rescued") if isinstance(opus_overall, dict) else None
    repair_text = (
        f"{beam_repairs} of Beam’s {beam_total} came from a readable design that broke a rule; "
        f"{opus_repairs} of Opus’s {opus_total} did."
        if all(type(value) is int for value in (beam_repairs, beam_total, opus_repairs, opus_total))
        else "Repair starting points are not available from the saved attempt records."
    )
    jaali = _jaali_pass_counts(models)
    jaali_text = (f"Jaali had {jaali[0]} accepted results out of {jaali[1]} model-task results."
                  if jaali else "Jaali results are not available in the saved summary.")
    latencies = [(key, _request_latency_ms(document)) for key, document in models.items()]
    valid_latencies = [(key, value) for key, value in latencies if value is not None]
    beam_latency = _request_latency_ms(beam)
    next_fastest = min((value for key, value in valid_latencies if key != "beam-hf"), default=None)
    next_key = next((key for key, value in valid_latencies if key != "beam-hf" and value == next_fastest), None)
    next_label = models.get(next_key, {}).get("label", next_key) if next_key else None
    latency_text = (
        f"Beam averaged {beam_latency / 1000:.1f} s per request; the next fastest was "
        f"{next_fastest / 1000:.1f} s ({html_escape(str(next_label))}). Request settings differ by provider."
        if beam_latency is not None and next_fastest is not None else
        "Request latency is not available for all models in the saved summary. Request settings differ by provider."
    )
    visibility = {key: accepted_visibility_counts(document, task_backgrounds)
                  for key, document in gallery_documents.items()}
    known_visibility = [value for value in visibility.values() if isinstance(value, tuple)]
    if known_visibility and len(known_visibility) == len(models):
        low_total = sum(value[0] for value in known_visibility)
        accepted_total = sum(value[1] for value in known_visibility)
        adjusted = []
        for key, document in models.items():
            counts = visibility.get(key)
            if not isinstance(counts, tuple):
                continue
            adjusted.append((counts[1] - counts[0], str(document.get("label", key))))
        adjusted.sort(key=lambda row: (-row[0], row[1].casefold()))
        order = ", ".join(f"{html_escape(label)} {count}" for count, label in adjusted[:4])
        visibility_text = (f"{low_total} of {accepted_total} accepted designs show little or no visible ink. "
                           f"Without them the top order would be {order}. Published ranks stay as run.")
    else:
        visibility_text = "Near-invisible totals are not available from the saved gallery records. Published ranks stay as run."
    cards = [
        f'<article data-testid="finding-repair-origins"><strong>Where repairs started</strong><span>{repair_text}</span></article>',
        f'<article data-testid="finding-jaali"><strong>Hardest grammar</strong><span>{jaali_text}</span></article>',
        f'<article data-testid="finding-request-time"><strong>Request time</strong><span>{latency_text}</span></article>',
        f'<article data-testid="finding-visibility-order"><strong>Visibility changes the raw top order</strong><span>{visibility_text}</span></article>',
    ]
    return ('<section class="pramaan-findings" data-testid="findings-strip" aria-label="Benchmark findings">'
            '<h2>What the checker found</h2><div class="pramaan-findings-grid">' + "".join(cards) +
            '</div><p class="pramaan-findings-note">These findings are computed from the saved summary and replay-audited gallery. Published scores and ranks remain as run.</p></section>')
