"""Accessible inline-SVG comparison charts for shared benchmark arms."""

from __future__ import annotations

import html
import math
import base64
import re
from pathlib import Path
from numbers import Real


_COLORS = ("#287d78", "#a84436", "#536272", "#916f2b", "#5964a5", "#47709a")

_ARM_KEY = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,100}\Z")
_BASE64_PNG = re.compile(r"[A-Za-z0-9+/]+={0,2}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REPLAYABLE_ATTEMPT_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,150}\Z")


def render_arena(
    index: dict | None,
    arm_documents: dict[str, dict],
    *,
    suite_id: str,
    suite_sha256: str,
    prompt_id: str,
    replay_prompt_id: str,
) -> str:
    """Show all available model attempts for one frozen, identical benchmark brief."""
    if (not isinstance(index, dict) or index.get("suite_id") != suite_id
            or index.get("suite_sha256") != suite_sha256
            or not _REPLAYABLE_ATTEMPT_ID.fullmatch(prompt_id)):
        return '<section class="pramaan-arena" data-testid="arena"><p role="status">The same-brief Arena data is unavailable.</p></section>'

    arms = index.get("arms") if isinstance(index.get("arms"), list) else []
    selected = []
    for arm in arms:
        if not isinstance(arm, dict):
            continue
        key = arm.get("key")
        if not isinstance(key, str) or not _ARM_KEY.fullmatch(key):
            continue
        document = arm_documents.get(key)
        tasks = document.get("tasks") if isinstance(document, dict) else None
        task = tasks.get(prompt_id) if isinstance(tasks, dict) else None
        selected.append((key, str(arm.get("label") or key), task))
    available = [(key, label, task) for key, label, task in selected if isinstance(task, dict)]
    if not available:
        return ('<section class="pramaan-arena" data-testid="arena" aria-label="Same-brief model Arena">'
                '<h3>Arena</h3><p role="status">No model has published attempts for the selected brief yet.</p></section>')

    briefs = {task.get("brief") for _key, _label, task in available
              if isinstance(task.get("brief"), str)}
    missing_brief = any(not isinstance(task.get("brief"), str) for _key, _label, task in available)
    if missing_brief or len(briefs) != 1:
        return ('<section class="pramaan-arena" data-testid="arena" aria-label="Same-brief model Arena">'
                '<h3>Arena</h3><p class="pramaan-muted" role="alert">Saved briefs differ or are missing for this task; '
                'the attempts are not compared.</p></section>')

    brief = html.escape(next(iter(briefs)))
    cards = []
    for key, label, task in selected:
        safe_key, safe_label = html.escape(key, quote=True), html.escape(label)
        if not isinstance(task, dict):
            cards.append(f'<article class="pramaan-arena-arm" data-testid="arena-arm-{safe_key}" '
                         'data-arm-state="not-run"><h4>' + safe_label + '</h4>'
                         '<p class="pramaan-arena-empty">Not run on this brief yet</p></article>')
            continue
        attempts = task.get("attempts") if isinstance(task.get("attempts"), list) else []
        attempt_rows = []
        for attempt in attempts[:3]:
            if not isinstance(attempt, dict):
                continue
            index_num = attempt.get("index")
            if type(index_num) is not int or not 1 <= index_num <= 3:
                continue
            passed = attempt.get("passed") is True
            verdict = "Pass" if passed else "Fail"
            checks = attempt.get("failing_checks") if isinstance(attempt.get("failing_checks"), list) else []
            check_text = ", ".join(str(value) for value in checks[:8])
            tile = attempt.get("tile_png_b64")
            preview = ""
            if isinstance(tile, str) and 0 < len(tile) <= 50000 and _BASE64_PNG.fullmatch(tile):
                try:
                    decoded = base64.b64decode(tile, validate=True)
                except (ValueError, base64.binascii.Error):
                    decoded = b""
                if decoded.startswith(b"\x89PNG\r\n\x1a\n"):
                    preview = (f'<img class="pramaan-arena-tile" src="data:image/png;base64,{tile}" '
                               f'alt="{verdict} attempt {index_num} tile for {safe_label}" loading="lazy" decoding="async">')
            attempt_rows.append(
                f'<li data-testid="arena-attempt-{safe_key}-{index_num}" data-attempt-state="{"pass" if passed else "fail"}">'
                f'<span class="pramaan-arena-verdict" aria-label="{verdict} attempt {index_num}">'
                f'{"✓" if passed else "×"} {verdict}</span><strong>Attempt {index_num}</strong>'
                + preview + (f'<small>Checks: {html.escape(check_text)}</small>' if check_text and not passed else "")
                + '</li>')
        passed_at = task.get("passed_at")
        status = (f"Verified on attempt {passed_at}" if type(passed_at) is int
                  else "Not verified · no passing attempt" if attempts
                  else "Not run on this brief yet")
        replay = (f'<a class="pramaan-ui-button secondary" href="#proof-replay" '
                  f'data-testid="arena-replay-{safe_key}">Replay this brief</a>'
                  if prompt_id == replay_prompt_id else "")
        cards.append(
            f'<article class="pramaan-arena-arm" data-testid="arena-arm-{safe_key}" '
            f'data-arm-state="{"verified" if type(passed_at) is int else "attempted" if attempts else "not-run"}">'
            f'<h4>{safe_label}</h4><p class="pramaan-arena-status">{html.escape(status)}</p>'
            + ('<ol class="pramaan-arena-attempts">' + "".join(attempt_rows) + '</ol>'
               if attempt_rows else '<p class="pramaan-arena-empty">No recorded attempts</p>')
            + replay + '</article>')

    return ('<section class="pramaan-arena" data-testid="arena" aria-label="Same-brief model Arena">'
            '<p class="pramaan-ui-eyebrow">Same brief · same checker</p><h3>Arena</h3>'
            f'<p class="pramaan-same-brief-prompt">{brief}</p>'
            '<p>Each attempt is replay evidence from the saved benchmark record. A check or cross reports the geometric verifier outcome.</p>'
            '<div class="pramaan-arena-grid">' + "".join(cards) + '</div></section>')


def render_same_brief_wall(
    index: dict | None,
    state_name: str,
    *,
    root: Path,
    suite_id: str,
    suite_sha256: str,
    gallery_json,
    model_key: str = "all",
) -> str:
    """Render replay-audited task rows grouped only by identical task ID and brief."""
    if (not isinstance(index, dict) or index.get("suite_id") != suite_id
            or index.get("suite_sha256") != suite_sha256):
        return '<p class="pramaan-muted">The replay-audited model gallery is not available in this Space build.</p>'

    arms = index.get("arms") if isinstance(index.get("arms"), list) else []
    selected = [arm for arm in arms if isinstance(arm, dict)
                and (model_key in (None, "all") or arm.get("key") == model_key)]
    by_prompt: dict[str, list[tuple[dict, dict]]] = {}
    for arm in selected:
        key = arm.get("key")
        if not isinstance(key, str) or not _ARM_KEY.fullmatch(key):
            continue
        document = gallery_json(root / "assets" / "gallery" / f"{key}.json")
        tasks = document.get("tasks") if isinstance(document, dict) else None
        if not isinstance(tasks, dict):
            continue
        for prompt_id, task in tasks.items():
            if isinstance(prompt_id, str) and isinstance(task, dict) and task.get("state") == state_name:
                by_prompt.setdefault(prompt_id, []).append((arm, task))

    if not by_prompt:
        return (f'<p class="pramaan-muted">No model task records are available for '
                f'{html.escape(str(state_name))}.</p>')

    groups = []
    for prompt_id, rows in sorted(by_prompt.items()):
        brief_values = [task.get("brief") for _arm, task in rows]
        briefs = {brief for brief in brief_values if isinstance(brief, str)}
        if len(rows) > 1 and (len(briefs) != 1 or any(not isinstance(brief, str) for brief in brief_values)):
            groups.append(
                '<article class="pramaan-same-brief"><h4>' + html.escape(prompt_id) + '</h4>'
                '<p class="pramaan-muted" role="alert">Saved briefs differ across model records for this task; '
                'these outputs are not compared.</p></article>'
            )
            continue

        brief = next(iter(briefs), "")
        cards = []
        for arm, task in sorted(rows, key=lambda item: str(
                item[0].get("label", item[0].get("key", ""))).casefold()):
            attempts = task.get("attempts") if isinstance(task.get("attempts"), list) else []
            passed_at = task.get("passed_at") if type(task.get("passed_at")) is int else None
            passed_attempt = attempts[passed_at - 1] if passed_at is not None and 1 <= passed_at <= len(attempts) else {}
            accepted = isinstance(passed_attempt, dict) and passed_attempt.get("passed") is True
            attempt = passed_attempt if accepted else (attempts[-1] if attempts else {})
            attempt = attempt if isinstance(attempt, dict) else {}
            tile = attempt.get("tile_png_b64")
            image_html = ""
            if isinstance(tile, str) and 0 < len(tile) <= 50000 and _BASE64_PNG.fullmatch(tile):
                try:
                    decoded = base64.b64decode(tile, validate=True)
                except (ValueError, base64.binascii.Error):
                    decoded = b""
                if decoded.startswith(b"\x89PNG\r\n\x1a\n"):
                    label = html.escape(str(arm.get("label", arm.get("key", "model"))), quote=True)
                    safe_prompt = html.escape(prompt_id, quote=True)
                    verification = "Verified" if accepted else "Unverified"
                    image_html = (f'<img class="pramaan-model-tile" src="data:image/png;base64,{tile}" '
                                  'loading="lazy" decoding="async" '
                                  f'alt="{verification} tile for {label} · {safe_prompt}">')

            label = html.escape(str(arm.get("label", arm.get("key", "model"))))
            status = f"Verified · passed on attempt {passed_at}" if accepted else (
                "Not run" if task.get("status") == "not_run" else "Not verified · no passing attempt")
            checks = attempt.get("failing_checks") if isinstance(attempt.get("failing_checks"), list) else []
            failures = (" · failing checks: " + html.escape(", ".join(str(item) for item in checks[:6]))
                        if checks and not accepted else "")
            evidence = []
            for field, name in (("cert_sha256", "certificate"), ("mask_sha256", "mask")):
                digest = task.get(field)
                if isinstance(digest, str) and _SHA256.fullmatch(digest):
                    evidence.append(f"{name} SHA-256 {digest[:16]}…")
            cards.append(
                '<article class="pramaan-model-tile-card"><h5>' + label + '</h5>'
                + (image_html or '<div class="pramaan-model-tile-placeholder">No tile preview</div>')
                + f'<p class="pramaan-model-tile-status">{html.escape(status)}{failures}</p>'
                + (f'<small>{html.escape(" · ".join(evidence))}</small>' if evidence else "")
                + '</article>'
            )

        groups.append(
            '<article class="pramaan-same-brief"><h4>' + html.escape(prompt_id) + '</h4>'
            + (f'<p class="pramaan-same-brief-prompt">{html.escape(brief)}</p>' if brief else "")
            + '<div class="pramaan-model-tile-grid">' + "".join(cards) + '</div></article>'
        )
    return '<div class="pramaan-same-brief-wall" aria-label="Same-brief model comparison">' + "".join(groups) + '</div>'


def _valid_count(value) -> bool:
    return type(value) is int and value >= 0


def _pass_rate(row: dict, key: str) -> float | None:
    n, successes = row.get("n"), row.get(key)
    if not _valid_count(n) or n == 0 or not _valid_count(successes) or successes > n:
        return None
    return successes / n


def _axes(aria_label: str, title: str, y_label: str) -> str:
    return (
        f'<svg viewBox="0 0 640 260" role="img" aria-label="{aria_label}" focusable="false">'
        f'<title>{title}</title>'
        '<g fill="none" stroke="#d8d0c5" stroke-width="1">'
        '<line x1="92" y1="32" x2="92" y2="210"/><line x1="92" y1="210" x2="570" y2="210"/>'
        '<line x1="92" y1="121" x2="570" y2="121" stroke-dasharray="4 5"/>'
        '<line x1="92" y1="32" x2="570" y2="32" stroke-dasharray="4 5"/>'
        '</g><g fill="#596775" font-size="12" font-family="system-ui,sans-serif">'
        '<text x="52" y="214">0%</text><text x="43" y="125">50%</text><text x="43" y="36">100%</text>'
        f'<text transform="translate(18 145) rotate(-90)">{y_label}</text>'
        '</g>'
    )


def _repair_arc(rows: list[dict]) -> str:
    chart = [_axes("Repair arc, pass at one to pass at three", "Repair arc", "Pass rate")]
    for index, row in enumerate(rows):
        first = _pass_rate(row, "passed1")
        final = _pass_rate(row, "passed3")
        if first is None or final is None:
            continue
        color = _COLORS[index % len(_COLORS)]
        y1, y3 = 210 - first * 178, 210 - final * 178
        label = html.escape(str(row.get("label") or row.get("key") or "Model"), quote=True)
        chart.append(
            f'<line x1="145" y1="{y1:.2f}" x2="485" y2="{y3:.2f}" stroke="{color}" stroke-width="3"/>'
            f'<circle cx="145" cy="{y1:.2f}" r="6" fill="{color}"><title>{label}: pass@1 {first:.1%}</title></circle>'
            f'<circle cx="485" cy="{y3:.2f}" r="6" fill="{color}"><title>{label}: pass@3 {final:.1%}</title></circle>'
            f'<text x="505" y="{y3 + 4:.2f}" fill="{color}" font-size="12">{label}</text>'
        )
    chart.append(
        '<g fill="#26384b" font-size="13" font-family="system-ui,sans-serif">'
        '<text x="120" y="238">Pass@1</text><text x="458" y="238">Pass@3</text></g></svg>'
    )
    return "".join(chart)


def _cost_quality(rows: list[dict]) -> str | None:
    points = []
    for row in rows:
        cost = row.get("est_cost_per_verified")
        rate = _pass_rate(row, "passed3")
        if isinstance(cost, Real) and not isinstance(cost, bool) and math.isfinite(float(cost)) and cost >= 0 and rate is not None:
            points.append((row, float(cost), rate))
    if len(points) < 2:
        return None
    maximum_cost = max(cost for _row, cost, _rate in points)
    if maximum_cost <= 0:
        maximum_cost = 1.0
    chart = [_axes("Cost per verified result versus pass at three", "Cost versus pass@3", "Pass@3")]
    for index, (row, cost, rate) in enumerate(points):
        x, y = 92 + cost / maximum_cost * 478, 210 - rate * 178
        color = _COLORS[index % len(_COLORS)]
        label = html.escape(str(row.get("label") or row.get("key") or "Model"), quote=True)
        chart.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="{color}"><title>{label}: '
            f'USD {cost:.4f} per verified result; pass@3 {rate:.1%}</title></circle>'
            f'<text x="{min(x + 10, 500):.2f}" y="{y + 4:.2f}" fill="{color}" font-size="12">{label}</text>'
        )
    chart.append(
        f'<g fill="#26384b" font-size="13" font-family="system-ui,sans-serif">'
        f'<text x="92" y="238">USD 0</text><text x="500" y="238">USD {maximum_cost:.4g}</text>'
        '<text x="284" y="255">Estimated cost per verified result</text></g></svg>'
    )
    return "".join(chart)


def render_comparison_charts(rows: list[dict]) -> str:
    """Render repair and cost charts using only complete, common-suite arms."""
    eligible = [row for row in rows if isinstance(row, dict) and row.get("complete") is True
                and _pass_rate(row, "passed1") is not None and _pass_rate(row, "passed3") is not None]
    wrapper = ['<section class="pramaan-comparison-charts" aria-label="Model comparison charts">'
               '<h4>Shared-suite comparisons</h4>']
    if len(eligible) < 2:
        wrapper.append('<p class="pramaan-muted">At least two complete models on the same questions are needed '
                       'for the repair-arc comparison. No cost-quality comparison is shown until two complete '
                       'models have cost data.</p></section>')
        return "".join(wrapper)

    wrapper.append('<div class="pramaan-comparison-chart-grid">'
                   '<article><h5>Repair arc · pass@1 to pass@3</h5>' + _repair_arc(eligible) + '</article>')
    cost_chart = _cost_quality(eligible)
    if cost_chart:
        wrapper.append('<article><h5>Estimated cost vs pass@3</h5>' + cost_chart + '</article>')
    else:
        wrapper.append('<article><h5>Estimated cost vs pass@3</h5><p class="pramaan-muted">'
                       'No cost-quality comparison is shown until two complete models have cost data.</p></article>')
    wrapper.append('</div></section>')
    return "".join(wrapper)
