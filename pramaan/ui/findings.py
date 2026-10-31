"""Small, evidence-linked headline metrics from the canonical eval summary."""

from __future__ import annotations

import math


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


def _metric(successes: object, total: object) -> tuple[int, int] | None:
    if type(successes) is not int or type(total) is not int or total <= 0 or not 0 <= successes <= total:
        return None
    return successes, total


def render_findings_strip(summary: dict | None) -> str:
    """Render the three canonical findings and truthful status for each benchmark arm.

    ``summary`` is loaded from ``benchmarks/data/eval-v2.1-summary.json`` by the
    caller when rendering; metrics are not copied into UI constants.
    """
    summary = summary if isinstance(summary, dict) else {}
    beam = summary.get("by_model", {}).get("beam-hf", {}) if isinstance(summary.get("by_model"), dict) else {}
    overall = beam.get("overall", {}) if isinstance(beam, dict) and isinstance(beam.get("overall"), dict) else {}
    first_try = _metric(overall.get("passed_at_1"), overall.get("n"))
    if first_try is None:
        first_try = _metric(round(overall.get("validity_at_1", -1) * overall.get("n", 0)), overall.get("n"))
    repair = _metric(overall.get("repair_rescued"), overall.get("repair_requested"))
    final = _metric(overall.get("accepted_designs"), overall.get("n"))
    if final is None:
        final = _metric(overall.get("passed_at_3"), overall.get("n"))

    def rate_text(value: tuple[int, int] | None, label: str) -> str:
        if value is None:
            return f'<strong>{label}</strong><span>Not available in the published summary</span>'
        successes, total = value
        interval = wilson_interval(successes, total)
        pct = successes / total
        ci_text = (f' · 95% Wilson {interval[0]:.1%}–{interval[1]:.1%}' if interval else "")
        return (f'<strong>{label}</strong><span>{successes}/{total} · {pct:.1%}{ci_text}</span>')

    # The baseline is the initial answer; the repaired result allows up to three
    # proposals after checker feedback. The label explicitly describes protocol.
    cards = [
        '<article data-testid="finding-first-try">' + rate_text(first_try, "Beam on the first proposal") +
        '<small>Without repair feedback</small></article>',
        '<article data-testid="finding-repair-gain">' + rate_text(repair, "Repairs accepted") +
        '<small>Of initial failures sent for checker feedback</small></article>',
        '<article data-testid="finding-proofs">' + rate_text(final, "Replay-verified results") +
        '<small>Pass@3 after checker feedback; task count is the denominator</small></article>',
    ]
    return ('<section class="pramaan-findings" data-testid="findings-strip" aria-label="Benchmark findings">'
            '<h2>What the checker found</h2><div class="pramaan-findings-grid">' + "".join(cards) +
            '</div><p class="pramaan-findings-note">Scores count geometric verifier outcomes, not cultural authenticity. '
            'Confidence intervals use the Wilson score method.</p></section>')
