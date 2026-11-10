"""Accessible leaderboard renderers for canonical and community results."""

from __future__ import annotations

import html
import math
import re
from pathlib import Path

from html import escape as html_escape

from .findings import wilson_interval
from .visibility import accepted_pattern_is_low_visibility, accepted_visibility_counts
from .check_labels import check_label


_MODEL_LABELS = {
    "beam-hf": "Beam",
    "claude-haiku-5-5": "Claude Haiku 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-opus-5-5": "Claude Opus 5.5",
    "claude-fable-5-1": "Claude Fable 5.1",
    "claude-fable-5-2": "Claude Fable 5.2",
    "deepseek-v4.1-flash": "DeepSeek V4 Flash",
    "gpt-6.1-sol": "GPT-6.1 Sol",
    "gpt-6.1-terra": "GPT-6.1 Terra",
    "gpt-6-luna": "GPT-6 Luna",
    "gpt-6-astra": "GPT-6 Astra",
}


def model_display_name(key: str, label: object = None) -> str:
    """Use the leaderboard's canonical name whenever an artifact stores its raw key."""
    if isinstance(label, str) and label and label != key:
        return label
    return _MODEL_LABELS.get(key, label if isinstance(label, str) and label else key)


def _summary_rate(overall: dict, rate_key: str, count_key: str, n: int) -> int:
    count = overall.get(count_key)
    if type(count) is int and 0 <= count <= n:
        return count
    rate = overall.get(rate_key)
    if isinstance(rate, (int, float)) and not isinstance(rate, bool) and math.isfinite(rate):
        return max(0, min(n, round(rate * n)))
    return 0


def _usd_cost(overall: dict) -> str:
    reported = overall.get("reported_cost_totals_by_currency")
    estimated = overall.get("estimated_cost_totals_by_currency")
    values = []
    for label, costs in (("reported", reported), ("estimated", estimated)):
        amount = costs.get("USD") if isinstance(costs, dict) else None
        if isinstance(amount, (int, float)) and not isinstance(amount, bool) and math.isfinite(amount):
            values.append(f"{label} USD {amount:.2f}")
    return " · ".join(values) if values else "not reported"


def _usd_cost_number(overall: dict) -> float | None:
    for field in ("reported_cost_totals_by_currency", "estimated_cost_totals_by_currency"):
        costs = overall.get(field)
        amount = costs.get("USD") if isinstance(costs, dict) else None
        if isinstance(amount, (int, float)) and not isinstance(amount, bool) and math.isfinite(amount):
            return float(amount)
    return None


def _estimated_usd_cost_number(overall: dict) -> float | None:
    costs = overall.get("estimated_cost_totals_by_currency")
    amount = costs.get("USD") if isinstance(costs, dict) else None
    if isinstance(amount, (int, float)) and not isinstance(amount, bool) and math.isfinite(amount) and amount > 0:
        return float(amount)
    return None


def _first_try_breakdown(document: object) -> tuple[int, int, int, int]:
    """Count saved first attempts as pass, readable rule failure, or unreadable."""
    tasks = document.get("tasks") if isinstance(document, dict) else None
    if not isinstance(tasks, dict):
        return 0, 0, 0, 0
    passed = broke_rule = unreadable = observed = 0
    for task in tasks.values():
        attempts = task.get("attempts") if isinstance(task, dict) else None
        first = attempts[0] if isinstance(attempts, list) and attempts and isinstance(attempts[0], dict) else None
        if first is None:
            continue
        failing = first.get("failing_checks")
        if first.get("passed") is True:
            passed += 1
        elif first.get("passed") is False and isinstance(failing, list) and failing:
            if "candidate_parse" in failing:
                unreadable += 1
            else:
                broke_rule += 1
        else:
            continue
        observed += 1
    return passed, broke_rule, unreadable, observed


def render_model_report(summary: dict | None, *, expected_tasks: int = 72,
                        gallery_documents: dict[str, dict] | None = None,
                        task_backgrounds: dict[str, str] | None = None,
                        suite_tasks: list[dict] | None = None) -> str:
    """Render the canonical model chart and single-row-per-model report from a summary."""
    summary = summary if isinstance(summary, dict) else {}
    published_expected = summary.get("expected_prompt_count")
    if type(published_expected) is int and published_expected > 0:
        expected_tasks = published_expected
    source = summary.get("by_model")
    models = source if isinstance(source, dict) else {}
    records = []
    for key, document in models.items():
        if not isinstance(key, str) or not isinstance(document, dict):
            continue
        overall = document.get("overall") if isinstance(document.get("overall"), dict) else {}
        study = document.get("study") if isinstance(document.get("study"), dict) else {}
        n = overall.get("n")
        if type(n) is not int or n < 0:
            n = 0
        n = min(n, expected_tasks)
        first = _summary_rate(overall, "validity_at_1", "passed_at_1", n)
        within = _summary_rate(overall, "validity_at_3", "accepted_designs", n)
        label = str(document.get("label") or _MODEL_LABELS.get(key, key))
        cap_reached = any(value is True for value in (
            document.get("cap_reached"), study.get("cap_reached"), study.get("budget_cap_reached"),
            study.get("stopped_at_cap"), overall.get("cap_reached"), overall.get("stopped_at_cap"),
        ))
        stop_reason = study.get("stop_reason")
        cap_reached = cap_reached or (isinstance(stop_reason, str) and stop_reason in {"budget_cap", "spend_cap", "usd_cap"})
        cost_ledger = document.get("cost_ledger") if isinstance(document.get("cost_ledger"), dict) else {}
        cap_usd = cost_ledger.get("cap_usd")
        if not isinstance(cap_usd, (int, float)) or isinstance(cap_usd, bool) or not math.isfinite(cap_usd) or cap_usd <= 0:
            cap_usd = 10.0 if cap_reached else None
        records.append({"key": key, "label": label, "n": n, "first": first,
                        "within": within, "cost": _usd_cost(overall), "cost_number": _usd_cost_number(overall),
                        "latency_ms": overall.get("mean_request_elapsed_ms_per_prompt"),
                        "cap": cap_reached, "cap_usd": cap_usd, "overall": overall})

    # Rank only arms with complete coverage. A high score on a smaller subset
    # must not displace models evaluated on the full frozen suite.
    ranked = sorted((row for row in records if row["n"] == expected_tasks),
                    key=lambda row: (-(row["within"] / row["n"]), -(row["first"] / row["n"]), row["label"].casefold()))
    rank_by_key = {}
    previous = None
    current_rank = 0
    for position, row in enumerate(ranked, 1):
        score = (row["within"] / row["n"], row["first"] / row["n"])
        if score != previous:
            current_rank = position
            previous = score
        rank_by_key[row["key"]] = current_rank

    gallery_documents = gallery_documents if isinstance(gallery_documents, dict) else {}
    chart_rows = []
    for row in sorted(records, key=lambda item: (rank_by_key.get(item["key"], expected_tasks + 1), item["label"].casefold())):
        label = html_escape(row["label"], quote=True)
        if row["n"] <= 0:
            chart_rows.append(f'<li data-model-key="{html_escape(row["key"], quote=True)}"><strong>{label}</strong><span>No evaluated tasks yet</span></li>')
            continue
        rate = row["within"] / row["n"]
        passed, broke_rule, unreadable, observed = _first_try_breakdown(gallery_documents.get(row["key"]))
        if observed == row["n"] and passed + broke_rule + unreadable == row["n"]:
            first_try = (
                f'<div class="pramaan-model-bar-track pramaan-model-bar-stacked" role="img" '
                f'aria-label="First try: {passed} passed, {broke_rule} broke a rule, {unreadable} unreadable">'
                f'<span class="pramaan-model-bar-segment is-pass" style="width:{passed / row["n"] * 100:.3f}%"></span>'
                f'<span class="pramaan-model-bar-segment is-rule-fail" style="width:{broke_rule / row["n"] * 100:.3f}%"></span>'
                f'<span class="pramaan-model-bar-segment is-unreadable" style="width:{unreadable / row["n"] * 100:.3f}%"></span>'
                '</div>'
                f'<p class="pramaan-model-first-try">First try: {passed} passed · {broke_rule} broke a rule · '
                f'{unreadable} unreadable</p>'
            )
            data_counts = (f' data-first-pass="{passed}" data-first-rule-fail="{broke_rule}" '
                           f'data-first-unreadable="{unreadable}"')
        else:
            first_try = '<p class="pramaan-model-first-try" role="status">First-try breakdown unavailable from saved attempt records.</p>'
            data_counts = ""
        chart_rows.append(
            f'<li data-model-key="{html_escape(row["key"], quote=True)}" data-testid="model-chart-{html_escape(row["key"], quote=True)}"{data_counts}>'
            f'<div class="pramaan-model-bar-label"><strong>{label}</strong><span>Within three: {row["within"]}/{row["n"]} · {rate:.1%}</span></div>'
            + first_try + '</li>')

    visibility_counts = {
        key: accepted_visibility_counts(document, task_backgrounds)
        for key, document in gallery_documents.items()
    }
    table_rows = []
    for row in sorted(records, key=lambda item: (rank_by_key.get(item["key"], expected_tasks + 1), item["label"].casefold())):
        n = row["n"]
        first_pct = f"{row['first'] / n:.1%}" if n else "—"
        within_pct = f"{row['within'] / n:.1%}" if n else "—"
        rank = f"Tie · rank {rank_by_key[row['key']]}" if rank_by_key.get(row["key"]) is not None and sum(value == rank_by_key[row["key"]] for value in rank_by_key.values()) > 1 else (f"Rank {rank_by_key[row['key']]}" if row["key"] in rank_by_key else "Not ranked · incomplete coverage")
        cap_note = (f' <small title="Independent evaluation spending ceiling">USD {row["cap_usd"]:g} cap</small>'
                    if row["cap_usd"] is not None else "")
        p1_rate = row["first"] / n if n else None
        p3_rate = row["within"] / n if n else None
        p1_interval = wilson_interval(row["first"], n) if n else None
        p3_interval = wilson_interval(row["within"], n) if n else None
        def interval_html(interval):
            if not isinstance(interval, tuple) or len(interval) != 2:
                return ""
            return f'<small>95% Wilson {interval[0]:.1%}–{interval[1]:.1%}</small>'
        invisible = visibility_counts.get(row["key"])
        invisible_text = (f'{invisible[0]} / {invisible[1]} accepted'
                          if isinstance(invisible, tuple) else "not available")
        invisible_attrs = (f' data-near-invisible="{invisible[0]}" data-accepted-designs="{invisible[1]}"'
                           if isinstance(invisible, tuple) else "")
        latency = row.get("latency_ms")
        latency_text = (
            f"{latency / 1000:.1f}s per request"
            if isinstance(latency, (int, float)) and not isinstance(latency, bool)
            and math.isfinite(latency) and latency >= 0 else "not reported"
        )
        table_rows.append(
            f'<tr data-leaderboard-row data-source="official" data-testid="model-report-row-{html_escape(row["key"], quote=True)}" data-model-key="{html_escape(row["key"], quote=True)}" '
            f'data-score-pass1="{leaderboard_numeric_attr(p1_rate)}" data-score-pass3="{leaderboard_numeric_attr(p3_rate)}" '
            f'data-coverage="{n}" data-cost="{leaderboard_numeric_attr(row["cost_number"])}" '
            f'data-latency-ms="{leaderboard_numeric_attr(row["latency_ms"])}"{invisible_attrs}>'
            f'<td>{html_escape(rank)}</td><th scope="row">{html_escape(row["label"])}</th>'
            f'<td>{row["first"]} / {n} <small>{first_pct}</small>{interval_html(p1_interval)}</td>'
            f'<td>{row["within"]} / {n} <small>{within_pct}</small>{interval_html(p3_interval)}</td>'
            f'<td>{html_escape(invisible_text)}</td>'
            f'<td>{n} of {expected_tasks} evaluated{cap_note}</td><td>{html_escape(row["cost"])}</td>'
            f'<td>{html_escape(latency_text)}</td></tr>')

    body = ("".join(table_rows) if table_rows else '<tr><td colspan="8">No model results have been published.</td></tr>')
    chart = ("".join(chart_rows) if chart_rows else '<li role="status">No model results have been published.</li>')
    return (
        '<section class="pramaan-model-report" data-testid="model-report" aria-label="Canonical model results">'
        '<h3>Model results</h3><p>Deterministic geometric scores. Provider errors are excluded from evaluated counts and scores.</p>'
        + '<section id="leaderboard-composition" class="pramaan-model-chart" data-testid="model-report-chart" tabindex="0" aria-label="First-try outcomes and within-three results by model">'
        '<h4>First try, then within three proposals</h4><p>Each stacked bar divides first attempts into passed, broke a rule, and unreadable; within-three counts appear beside it.</p>'
        '<ul class="pramaan-model-composition-legend" aria-label="First-try outcome legend">'
        '<li><span class="is-pass" aria-hidden="true"></span>Passed</li>'
        '<li><span class="is-rule-fail" aria-hidden="true"></span>Broke a rule</li>'
        '<li><span class="is-unreadable" aria-hidden="true"></span>Unreadable</li></ul>'
        '<ol>' + chart + '</ol><p class="pramaan-model-chart-alt">Text alternative: first-try counts and within-three results are written beside every bar; the table repeats the rates and costs.</p></section>'
        + _render_results_matrix(records, gallery_documents, suite_tasks or [], task_backgrounds or {})
        + _render_first_try_failures_chart(records, gallery_documents)
        + _render_cost_frontier_chart(records, expected_tasks)
        + '<section id="leaderboard-full-results" class="pramaan-model-full-results"><h4>Full results</h4>'
        + '<div class="pramaan-leaderboard-scroll"><table class="pramaan-model-table" data-testid="model-report-table" data-leaderboard-table>'
        '<caption>One row per model; missing provider answers are excluded.</caption>'
        '<thead><tr><th scope="col">Rank</th><th scope="col">Model</th><th scope="col">First try</th><th scope="col">Within three tries</th><th scope="col">Near-invisible</th><th scope="col">Evaluated</th><th scope="col">Cost</th><th scope="col">Mean request time</th></tr></thead>'
        '<tbody>' + body + '</tbody></table></div>'
        + _visibility_footnote(visibility_counts)
        + '<p class="pramaan-model-footnote">One unsatisfiable task remains in every model’s denominator.</p></section>'
        '<dialog class="pramaan-chart-detail-dialog" data-testid="chart-detail-dialog" '
        'aria-labelledby="pramaan-chart-detail-title">'
        '<h3 id="pramaan-chart-detail-title" data-chart-detail-title>Model result details</h3>'
        '<p data-chart-detail-body></p>'
        '<div class="pramaan-ui-actions"><a class="pramaan-ui-button secondary" data-chart-detail-arena hidden>Open this brief in the Arena</a>'
        '<button type="button" class="pramaan-ui-button secondary" data-chart-detail-close>Close</button></div></dialog>'
        '</section>')


def _render_cost_frontier_chart(records: list[dict], expected_tasks: int) -> str:
    points = []
    for row in records:
        cost = _estimated_usd_cost_number(row.get("overall", {}))
        if row["n"] != expected_tasks or cost is None or row["n"] <= 0:
            continue
        points.append({**row, "estimated_cost": cost, "rate": row["within"] / row["n"]})
    if not points:
        return ('<section id="leaderboard-cost" class="pramaan-model-cost" data-testid="model-cost-chart" tabindex="0" '
                'aria-label="Estimated cost against pass rate"><h4>Cost against within three</h4>'
                '<p role="status">A complete saved cost comparison is not available.</p></section>')

    # Logarithmic cost coordinates keep low-cost models readable beside the higher-cost runs.
    x0, x1, y0, y1 = 78.0, 700.0, 350.0, 58.0
    costs = [point["estimated_cost"] for point in points]
    min_log = math.log10(min(costs))
    max_log = math.log10(max(costs))
    if max_log == min_log:
        max_log += 1
    def xy(point):
        x = x0 + (math.log10(point["estimated_cost"]) - min_log) / (max_log - min_log) * (x1 - x0)
        y = y0 - point["rate"] * (y0 - y1)
        return x, y

    frontier = []
    best_rate = -1.0
    for point in sorted(points, key=lambda item: (item["estimated_cost"], -item["rate"], item["label"].casefold())):
        if point["rate"] > best_rate:
            frontier.append(point)
            best_rate = point["rate"]
    frontier_keys = {point["key"] for point in frontier}
    path = " ".join(("M" if index == 0 else "L") + f"{xy(point)[0]:.2f},{xy(point)[1]:.2f}"
                     for index, point in enumerate(frontier))
    svg = [
        '<svg class="pramaan-cost-plot" viewBox="0 0 760 430" role="img" '
        'aria-label="Estimated evaluation cost in US dollars on a logarithmic horizontal axis; within-three pass rate vertically. Points are labeled by model; a dashed line marks the Pareto frontier.">',
        '<line x1="78" y1="350" x2="700" y2="350" class="axis"/><line x1="78" y1="58" x2="78" y2="350" class="axis"/>',
    ]
    tick_values = [value for value in (0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
                   if min(costs) <= value <= max(costs)]
    for value in tick_values:
        x = x0 + (math.log10(value) - min_log) / (max_log - min_log) * (x1 - x0)
        svg.append(f'<line x1="{x:.2f}" y1="350" x2="{x:.2f}" y2="355" class="axis"/>')
        svg.append(f'<text x="{x:.2f}" y="374" text-anchor="middle" class="tick" data-cost-tick="{value:g}">${value:g}</text>')
    for pct in (0, 25, 50, 75, 100):
        y = y0 - pct / 100 * (y0 - y1)
        svg.append(f'<line x1="73" y1="{y:.2f}" x2="78" y2="{y:.2f}" class="axis"/>')
        svg.append(f'<text x="66" y="{y + 4:.2f}" text-anchor="end" class="tick">{pct}%</text>')
    svg.append(f'<path d="{path}" class="frontier" aria-label="Pareto frontier"/>')
    label_boxes: list[tuple[float, float, float, float]] = []
    label_positions = {}
    label_points = sorted(points, key=lambda item: (xy(item)[1], xy(item)[0], item["label"].casefold()))
    vertical_offsets = (0, -14, 14, -28, 28, -42, 42, -56, 56, -70, 70, -84, 84, -98, 98)
    for point in label_points:
        x, y = xy(point)
        label = html_escape(point["label"], quote=True)
        width = max(28.0, len(point["label"]) * 6.4)
        preferred_sides = (1, -1) if x < (x0 + x1) / 2 else (-1, 1)
        chosen = None
        for side in preferred_sides:
            anchor = "start" if side > 0 else "end"
            label_x = x + side * 10
            left, right = (label_x, label_x + width) if side > 0 else (label_x - width, label_x)
            for offset in vertical_offsets:
                label_y = y + offset
                top, bottom = label_y - 10, label_y + 3
                if left < 48 or right > 752 or top < 45 or bottom > 384:
                    continue
                if any(left < other_right + 5 and right + 5 > other_left
                       and top < other_bottom + 3 and bottom + 3 > other_top
                       for other_left, other_top, other_right, other_bottom in label_boxes):
                    continue
                chosen = (label_x, label_y, anchor, left, top, right, bottom, side)
                break
            if chosen:
                break
        if chosen is None:
            raise ValueError(f"Could not place a non-overlapping cost label for {point['label']}")
        label_x, label_y, anchor, left, top, right, bottom, side = chosen
        label_boxes.append((left, top, right, bottom))
        label_positions[point["key"]] = (x, y, label_x, label_y, anchor, left, top, right, bottom, side)

    for point in sorted(points, key=lambda item: item["label"].casefold()):
        x, y, label_x, label_y, anchor, left, top, right, bottom, side = label_positions[point["key"]]
        label = html_escape(point["label"], quote=True)
        key = html_escape(point["key"], quote=True)
        cost = point["estimated_cost"]
        rate_pct = point["rate"] * 100
        frontier_attr = "true" if point["key"] in frontier_keys else "false"
        svg.append(
            f'<g class="point{" is-frontier" if point["key"] in frontier_keys else ""}" tabindex="0" role="img" '
            f'data-model-key="{key}" data-testid="model-cost-point-{key}" '
            f'data-touch-detail="cost" data-detail-model="{label}" '
            f'data-estimated-cost="{cost:.8f}" data-within-three="{point["within"]}" '
            f'data-denominator="{point["n"]}" data-frontier="{frontier_attr}">'
            f'<title>{label}: estimated USD {cost:.2f}; {point["within"]}/{point["n"]} within three proposals</title>'
            f'<path class="label-leader" d="M{x:.2f},{y:.2f} L{(left if side > 0 else right):.2f},{max(top + 3, min(bottom - 3, y)):.2f}" aria-hidden="true"/>'
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6"/>'
            f'<text x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="{anchor}" class="point-label" '
            f'data-label-box="{left:.2f},{top:.2f},{right:.2f},{bottom:.2f}">{label}</text></g>'
        )
    svg.extend([
        '<text x="389" y="410" text-anchor="middle" class="axis-label">Estimated evaluation cost (USD, log scale)</text>',
        '<text x="18" y="205" transform="rotate(-90 18 205)" text-anchor="middle" class="axis-label">Within three</text>',
        '</svg>',
    ])
    frontier_count = len(frontier)
    return (
        '<section id="leaderboard-cost" class="pramaan-model-cost" data-testid="model-cost-chart" tabindex="0" '
        'aria-label="Estimated cost against within-three pass rate">'
        '<h4>Cost against within three</h4>'
        '<p>Dashed line shows the Pareto frontier: lower estimated evaluation cost and a higher within-three rate. '
        'Costs are estimates, request settings differ by provider, and the horizontal axis is logarithmic.</p>'
        + "".join(svg)
        + f'<p class="pramaan-model-cost-alt">Text alternative: {len(points)} complete models plotted from saved estimates; '
        f'{frontier_count} are on the displayed cost / pass frontier. Every point has its model name, estimated USD cost, '
        'within-three count, and denominator in its focus tooltip and data record.</p></section>'
    )


def _render_results_matrix(records: list[dict], gallery_documents: dict,
                           suite_tasks: list[dict], task_backgrounds: dict[str, str]) -> str:
    tasks = [task for task in suite_tasks if isinstance(task, dict) and isinstance(task.get("id"), str)]
    tasks.sort(key=lambda task: (str(task.get("grammar", "")), str(task.get("id", ""))))
    if not tasks:
        return ('<section id="leaderboard-results-matrix" class="pramaan-results-matrix" '
                'data-testid="model-results-matrix" tabindex="0"><h4>Results matrix</h4>'
                '<p role="status">The frozen task list is unavailable in this build.</p></section>')
    grammar_groups: list[tuple[str, list[dict]]] = []
    for task in tasks:
        grammar = str(task.get("grammar", "Unclassified"))
        if not grammar_groups or grammar_groups[-1][0] != grammar:
            grammar_groups.append((grammar, []))
        grammar_groups[-1][1].append(task)
    group_header = ''.join(
        f'<th class="grammar-group" scope="colgroup" colspan="{len(group)}">{html_escape(grammar.replace("_", " ").title())}</th>'
        for grammar, group in grammar_groups
    )


    task_header = ''.join(
        f'<th scope="col" title="{html_escape(str(task.get("state", "")) + " · " + str(task.get("brief", "")), quote=True)}">'
        f'{index + 1}</th>' for index, task in enumerate(tasks)
    )
    rows = []
    for row in sorted(records, key=lambda item: item["label"].casefold()):
        model_key = str(row["key"])
        model_docs = gallery_documents.get(model_key, {})
        model_tasks = model_docs.get("tasks") if isinstance(model_docs, dict) else {}
        model_tasks = model_tasks if isinstance(model_tasks, dict) else {}
        cells = []
        for index, task in enumerate(tasks):
            task_id = str(task["id"])
            saved = model_tasks.get(task_id)
            attempts = saved.get("attempts") if isinstance(saved, dict) else None
            attempts = [item for item in attempts if isinstance(item, dict)] if isinstance(attempts, list) else []
            first_pass = bool(attempts and attempts[0].get("passed") is True)
            passing = next((item for item in attempts if item.get("passed") is True), None)
            if first_pass:
                state, state_label = "first", "Passed first try"
                attempt_label = "attempt 1"
            elif passing:
                state, state_label = "repair", "Passed after repair"
                attempt_label = f"attempt {passing.get('index', 'unknown')}"
            elif attempts:
                state, state_label = "fail", "Did not pass"
                attempt_label = f"{len(attempts)} attempts; no passing result"
            else:
                state, state_label = "not-run", "Not run"
                attempt_label = "no attempt saved"
            near_invisible = bool(passing and accepted_pattern_is_low_visibility(
                passing, task_backgrounds.get(task_id)
            ))
            task_name = html_escape(task_id, quote=True)
            state_name = html_escape(str(task.get("state", "State")), quote=True)
            brief = html_escape(str(task.get("brief", "")), quote=True)
            model_label = html_escape(str(row["label"]), quote=True)
            accessible = html_escape(
                f'{row["label"]} · {state_label} · {task.get("state", "State")} · '
                f'{task_id} · {attempt_label}' + (" · near-invisible accepted design" if near_invisible else ""),
                quote=True,
            )
            pass_number = None
            if passing:
                candidate_index = passing.get("index")
                pass_number = candidate_index if type(candidate_index) is int and 1 <= candidate_index <= 3 else next(
                    (attempt_index for attempt_index, attempt in enumerate(attempts, 1) if attempt is passing),
                    None,
                )
                attempt_label = f"attempt {pass_number}" if pass_number is not None else "passing attempt"
            hatch = " is-near-invisible" if near_invisible else ""
            cells.append(
                f'<td><a class="pramaan-matrix-cell is-{state}{hatch}" '
                f'href="#/leaderboard?task={task_name}" data-testid="matrix-cell-{model_key}-{task_name}" '
                f'data-matrix-cell data-touch-detail="matrix" data-detail-model="{model_label}" '
                f'data-detail-state="{state_name}" data-detail-task="{task_name}" '
                f'data-detail-result="{html_escape(state_label, quote=True)}" '
                f'data-detail-attempt="{html_escape(attempt_label, quote=True)}" data-detail-brief="{brief}" '
                f'data-model-key="{html_escape(model_key, quote=True)}" '
                f'data-task-id="{task_name}" data-cell-state="{state}" '
                f'data-near-invisible="{str(near_invisible).lower()}" '
                f'title="{accessible} · {brief}" aria-label="{accessible}">'
                f'<span class="sr-only">{html_escape(state_label)}: {attempt_label}</span>'
                f'<span aria-hidden="true">{pass_number if pass_number is not None else "×" if state == "fail" else "—"}</span></a></td>'
            )
        rows.append(
            f'<tr data-testid="matrix-row-{html_escape(model_key, quote=True)}">'
            f'<th scope="row" class="matrix-model">{html_escape(str(row["label"]))}</th>'
            + ''.join(cells) + '</tr>'
        )
    return (
        '<section id="leaderboard-results-matrix" class="pramaan-results-matrix" '
        'data-testid="model-results-matrix" tabindex="0" aria-label="Saved model results across benchmark tasks">'
        '<h4>Results matrix</h4>'
        '<p>Tap, focus, or hover a cell for its model, task, state, and attempt; use its detail card to open that brief in the Arena. '
        'Hatching marks an accepted result with under 2% visible ink.</p>'
        '<div class="pramaan-results-matrix-scroll" role="region" tabindex="0" aria-label="Scrollable model by task results matrix">'
        '<table><thead><tr><th class="matrix-model matrix-corner" scope="col" rowspan="2">Model</th>'
        + group_header + '</tr><tr>' + task_header + '</tr></thead><tbody>' + ''.join(rows)
        + '</tbody></table></div>'
        '<ul class="pramaan-matrix-legend" aria-label="Results matrix legend">'
        '<li><span class="is-first" aria-hidden="true">1</span>Passed on attempt 1</li>'
        '<li><span class="is-repair" aria-hidden="true">2/3</span>Passed on attempt 2 or 3</li>'
        '<li><span class="is-fail" aria-hidden="true">×</span>Did not pass</li>'
        '<li><span class="is-not-run" aria-hidden="true">—</span>Not run</li>'
        '<li><span class="is-hatched" aria-hidden="true"></span>Accepted, under 2% visible ink</li></ul>'
        '<p class="pramaan-model-cost-alt">Text alternative: rows identify models; columns are grouped by grammar, and each linked cell names its task, state, result, and attempt.</p>'
        '</section>'
    )


def _render_first_try_failures_chart(records: list[dict], gallery_documents: dict) -> str:
    """Count saved first-attempt failing checks, separating unreadable outputs."""
    counts: dict[str, int] = {}
    total_observed = 0
    for row in records:
        document = gallery_documents.get(row["key"], {})
        tasks = document.get("tasks") if isinstance(document, dict) else {}
        for task in tasks.values() if isinstance(tasks, dict) else []:
            attempts = task.get("attempts") if isinstance(task, dict) else None
            first = attempts[0] if isinstance(attempts, list) and attempts and isinstance(attempts[0], dict) else None
            if first is None or first.get("passed") is not False:
                continue
            failing = first.get("failing_checks")
            if not isinstance(failing, list):
                continue
            total_observed += 1
            if "candidate_parse" in failing:
                counts["__unreadable__"] = counts.get("__unreadable__", 0) + 1
                continue
            for check_id in set(item for item in failing if isinstance(item, str)):
                counts[check_id] = counts.get(check_id, 0) + 1
    categories = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    maximum = max((value for _, value in categories), default=0)
    bars = []
    for check_id, count in categories:
        unreadable = check_id == "__unreadable__"
        label = "Unreadable output" if unreadable else check_label(check_id)
        css_kind = " is-unreadable" if unreadable else ""
        bars.append(
            f'<li data-testid="first-try-failure-{html_escape(check_id, quote=True)}" '
            f'data-check-id="{html_escape(check_id, quote=True)}" data-count="{count}">'
            f'<span class="pramaan-failure-label">{html_escape(label)}</span>'
            f'<span class="pramaan-failure-track" role="img" aria-label="{html_escape(label, quote=True)}: {count} first attempts">'
            f'<span class="pramaan-failure-bar{css_kind}" style="width:{(count / maximum * 100) if maximum else 0:.3f}%"></span></span>'
            f'<b>{count}</b></li>'
        )
    body = ''.join(bars) if bars else '<li role="status">No failed first attempts are present in saved records.</li>'
    return (
        '<section id="leaderboard-failing-checks" class="pramaan-first-failures" '
        'data-testid="first-try-failures-chart" tabindex="0" aria-label="Most common first-try failing checks">'
        '<h4>What fails on the first try?</h4>'
        '<p>Counts use saved first attempts across model-task pairs. A task contributes at most once to each check; '
        'unreadable outputs are counted separately.</p>'
        f'<ol>{body}</ol>'
        f'<p class="pramaan-model-cost-alt">Text alternative: {len(categories)} categories summarize {total_observed} '
        'failed first attempts; the count for each check is printed beside its bar.</p></section>'
    )


def render_models_section_nav(available_markup: str) -> str:
    """Link only to dashboard sections present in the supplied rendered markup."""
    sections = (
        ("leaderboard-composition", "Composition"),
        ("leaderboard-results-matrix", "Results matrix"),
        ("leaderboard-failing-checks", "Failing checks"),
        ("leaderboard-cost", "Cost"),
        ("leaderboard-full-results", "Full results"),
        ("model-arena", "Arena"),
        ("leaderboard-methodology", "Methodology"),
        ("leaderboard-community-results", "Community results"),
    )
    links = "".join(
        f'<a href="#{anchor}">{html_escape(label)}</a>'
        for anchor, label in sections
        if f'id="{anchor}"' in available_markup
    )
    return (f'<nav class="pramaan-leaderboard-nav is-sticky" data-testid="models-section-menu" '
            f'aria-label="Models dashboard sections">{links}</nav>')


def _visibility_footnote(visibility_counts: dict[str, tuple[int, int] | None]) -> str:
    known = [value for value in visibility_counts.values() if isinstance(value, tuple)]
    if not known or len(known) != len(visibility_counts):
        return '<p class="pramaan-model-footnote" data-testid="visibility-limitation">The checker measures drawn shapes, not visibility; saved accepted designs do not provide a complete visibility count in this build. <a href="#/rl">See the Manipur example in Build with it.</a></p>'
    low_visibility = sum(value[0] for value in known)
    accepted = sum(value[1] for value in known)
    return (f'<p class="pramaan-model-footnote" data-testid="visibility-limitation">'
            f'The checker measures drawn shapes, not visibility: {low_visibility} of {accepted} accepted designs '
            'show little or no visible ink. <a href="#/rl">See the Manipur example in Build with it.</a></p>')


def leaderboard_numeric_attr(value) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return ""
    return f"{value:.8f}"


def render_community_leaderboard(snapshot: dict) -> str:
    status = html_escape(str(snapshot.get("status", "Community results are unavailable.")))
    summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), dict) else {}
    common = summary.get("common_completed") if isinstance(summary.get("common_completed"), dict) else {}
    common_models = common.get("model_keys") if isinstance(common.get("model_keys"), list) else []
    common_count = common.get("prompt_count", 0)
    metrics = common.get("by_model") if isinstance(common.get("by_model"), dict) else {}
    selected = snapshot.get("selected") if isinstance(snapshot.get("selected"), dict) else {}
    all_models = summary.get("by_model") if isinstance(summary.get("by_model"), dict) else {}
    rows_html = []
    ranked = []
    for model_key, study in all_models.items():
        coverage = study.get("study", {})
        chosen = metrics.get(model_key, {})
        if model_key in common_models and isinstance(chosen.get("validity_at_3"), (int, float)):
            ranked.append((float(chosen["validity_at_3"]), float(chosen.get("validity_at_1") or 0), model_key))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    rank_by_model = {model: index + 1 for index, (_pass3, _pass1, model) in enumerate(ranked)}
    for model_key, study in all_models.items():
        coverage = study.get("study", {})
        observed = coverage.get("observed_prompt_count", 0)
        if not isinstance(observed, int) or isinstance(observed, bool) or observed < 0:
            observed = 0
        result = metrics.get(model_key, {})
        overall = study.get("overall", {})
        run = selected.get(model_key)
        provenance = ("Community · self-reported model ID" if run else
                      ("Canonical Beam baseline" if observed else "No canonical Beam rows in this build")
                      if model_key == "beam-hf" else "Canonical result")
        run_id = str(run.get("run_id", "")) if isinstance(run, dict) else ""
        pass1 = result.get("validity_at_1")
        pass3 = result.get("validity_at_3")
        pass1_text = f"{pass1:.1%}" if common_count and isinstance(pass1, (int, float)) else "—"
        pass3_text = f"{pass3:.1%}" if common_count and isinstance(pass3, (int, float)) else "—"
        rank_text = (f"Rank {rank_by_model[model_key]}" if model_key in rank_by_model and len(ranked) >= 2
                     else "Not ranked")
        token_mean = overall.get("mean_tokens_when_available")
        time_mean = overall.get("mean_elapsed_ms")
        request_mean = overall.get("mean_request_elapsed_ms_per_prompt")
        verifier_mean = overall.get("mean_verification_elapsed_ms_per_prompt")
        estimated_costs = overall.get("estimated_cost_totals_by_currency")
        estimated_usd = estimated_costs.get("USD") if isinstance(estimated_costs, dict) else None
        def cost_label(field):
            values = overall.get(field)
            if not isinstance(values, dict) or not values:
                return "not reported"
            return ", ".join(f"{currency} {amount:.4f}" for currency, amount in sorted(values.items())
                              if isinstance(amount, (int, float))) or "not reported"
        telemetry = (
            f"Tokens {token_mean:.0f} mean" if isinstance(token_mean, (int, float)) else "Tokens not reported"
        )
        telemetry += f" · wall {time_mean / 1000:.1f} s mean" if isinstance(time_mean, (int, float)) else " · wall time unavailable"
        telemetry += f" · request {request_mean:.0f} ms mean/prompt" if isinstance(request_mean, (int, float)) else " · request latency unavailable"
        telemetry += f" · verifier {verifier_mean:.1f} ms mean/prompt" if isinstance(verifier_mean, (int, float)) else " · verifier latency unavailable"
        telemetry += f" · reported {cost_label('reported_cost_totals_by_currency')} · estimated {cost_label('estimated_cost_totals_by_currency')}"
        coverage_text = f"{observed}/72"
        if common_count and model_key in common_models:
            coverage_text += f" · {common_count} shared"
        run_detail = f"<small>Run {html_escape(run_id)}</small>" if run_id else ""
        rows_html.append(
            f'<tr data-leaderboard-row data-source="{"community" if run else "official"}" '
            f'data-model-key="{html_escape(str(model_key), quote=True)}" '
            f'data-score-pass1="{leaderboard_numeric_attr(pass1)}" data-score-pass3="{leaderboard_numeric_attr(pass3)}" '
            f'data-coverage="{observed}" data-tokens="{leaderboard_numeric_attr(token_mean)}" '
            f'data-latency="{leaderboard_numeric_attr(request_mean)}" data-cost="{leaderboard_numeric_attr(estimated_usd)}">'
            f'<th scope="row">{html_escape(str(model_key))}<small>{html_escape(provenance)}</small>{run_detail}</th>'
            f'<td>{html_escape(rank_text)}</td><td>{coverage_text}</td>'
            f'<td>{pass1_text}</td><td>{pass3_text}</td>'
            f'<td>{html_escape(telemetry)}</td>'
            f'<td>{html_escape(cost_label("reported_cost_totals_by_currency"))}</td>'
            f'<td>{html_escape(cost_label("estimated_cost_totals_by_currency"))}</td></tr>'
        )
    if not rows_html:
        return ('<section class="pramaan-community-results" aria-label="Community benchmark runs">'
                '<h3>Community comparison</h3><p role="status">' + status + '</p></section>')
    note = (f"Compared on {common_count} identical completed questions. Quality scores are geometric verifier outcomes, not cultural-authenticity ratings."
            if common_count and len(common_models) >= 2 else
            "Coverage is shown for each selected run. A ranking appears only when at least two models share completed questions.")
    return ('<section class="pramaan-community-results" aria-label="Community benchmark runs">'
            '<h3>Latest consented community runs</h3><p>' + html_escape(note) + '</p>'
            + '<div class="pramaan-leaderboard-scroll"><table class="pramaan-leaderboard" data-leaderboard-table>'
            '<caption>Community and canonical results · verifier outcomes measure geometric constraints, not cultural authenticity</caption>'
            '<thead><tr><th scope="col">Model / provenance</th><th scope="col">Rank</th>'
            '<th scope="col">Coverage</th><th scope="col">Pass@1</th>'
            '<th scope="col">Pass@3 · shared tasks</th><th scope="col">Tokens and latency</th>'
            '<th scope="col">Provider-reported cost</th><th scope="col">Rate-card estimate</th>'
            '</tr></thead><tbody>' + "".join(rows_html) + '</tbody></table></div></section>')

def render_canonical_leaderboard(index: dict | None = None, *, root: Path, suite_id: str, suite_sha256: str, load_gallery_index, gallery_json, wilson_interval, comparison_charts) -> str:
    index = index if isinstance(index, dict) else load_gallery_index()
    if (not isinstance(index, dict) or index.get("suite_id") != suite_id
            or index.get("suite_sha256") != suite_sha256):
        return ('<section class="pramaan-canonical-leaderboard" aria-label="Canonical model benchmark">'
                '<h3>Canonical model results</h3><p role="status">The replay-audited gallery index for this suite is not available in this Space build.</p></section>')
    common_ids = index.get("common_prompt_ids")
    if not isinstance(common_ids, list):
        common_ids = []
    common_ids = list(dict.fromkeys(item for item in common_ids if isinstance(item, str) and item))
    if not common_ids:
        return ('<section class="pramaan-canonical-leaderboard" aria-label="Canonical model benchmark">'
                '<h3>Canonical model results</h3><p role="status">No shared completed questions are available for comparison yet.</p></section>')

    grammar_codes = {
        "periodic_raster": "G1", "dot_field": "G2", "stripe_band": "G3",
        "counted_grid": "G4", "rotational_field": "G5", "kolam_loop": "G6", "jaali_lattice": "G7",
    }
    records = []
    arms = index.get("arms") if isinstance(index.get("arms"), list) else []
    for arm in arms:
        if not isinstance(arm, dict):
            continue
        key = str(arm.get("key", ""))
        if not key or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,100}", key):
            continue
        arm_doc = gallery_json(root / "assets" / "gallery" / f"{key}.json")
        task_map = arm_doc.get("tasks") if isinstance(arm_doc, dict) else None
        if not isinstance(task_map, dict):
            continue
        arm_ids = set(item for item in arm.get("completed_prompt_ids", []) if isinstance(item, str))
        task_rows = []
        for prompt_id in common_ids:
            row = task_map.get(prompt_id)
            if (prompt_id in arm_ids and isinstance(row, dict)
                    and row.get("status") == "completed"):
                task_rows.append(row)
        n = len(task_rows)
        passed1 = sum(type(row.get("passed_at")) is int and row["passed_at"] == 1 for row in task_rows)
        passed3 = sum(type(row.get("passed_at")) is int and 1 <= row["passed_at"] <= 3
                      for row in task_rows)
        grammar_results = {}
        for row in task_rows:
            code = grammar_codes.get(row.get("grammar"))
            if code:
                tally = grammar_results.setdefault(code, [0, 0])
                tally[0] += 1
                tally[1] += int(type(row.get("passed_at")) is int and 1 <= row["passed_at"] <= 3)
        records.append({
            "key": key, "label": str(arm.get("label") or key),
            "provider": str(arm.get("provider") or ""), "n": n, "complete": n == len(common_ids),
            "passed1": passed1, "passed3": passed3, "grammar": grammar_results,
            "mean_reward": arm.get("mean_reward"), "tokens_per_task": arm.get("tokens_per_task"),
            "est_cost_per_verified": arm.get("est_cost_per_verified"),
            "median_latency_ms": arm.get("median_latency_ms"),
            "normalizations": arm.get("normalizations"),
            "normalization_calls": arm.get("normalization_calls"),
        })
    eligible = [row for row in records if row["complete"]]
    eligible.sort(key=lambda row: (-row["passed3"], -row["passed1"], row["label"].casefold()))
    rank_by_key = {}
    tie_by_key = {}
    if len(eligible) >= 2:
        for row in eligible:
            score = (row["passed3"], row["passed1"])
            tied = [candidate for candidate in eligible
                    if (candidate["passed3"], candidate["passed1"]) == score]
            rank_by_key[row["key"]] = min(i for i, candidate in enumerate(eligible, start=1)
                                             if (candidate["passed3"], candidate["passed1"]) == score)
            if len(tied) > 1:
                tie_by_key[row["key"]] = True

    rows_html = []
    for row in records:
        n = row["n"]
        p1 = row["passed1"] / n if n else 0.0
        p3 = row["passed3"] / n if n else 0.0
        ci1 = wilson_interval(row["passed1"], n)
        ci3 = wilson_interval(row["passed3"], n)
        if row["key"] in tie_by_key:
            rank = f"Tie · rank {rank_by_key[row['key']]}"
        elif row["key"] in rank_by_key:
            rank = f"Rank {rank_by_key[row['key']]}"
        elif row["complete"]:
            rank = "Not ranked · fewer than two complete arms"
        else:
            rank = "Not ranked · incomplete shared coverage"
        heat_cells = []
        for code in (f"G{number}" for number in range(1, 8)):
            count, successes = row["grammar"].get(code, [0, 0])
            heat_cells.append(
                f'<span class="pramaan-grammar-cell" title="{code}: {successes}/{count} passed within three proposals">'
                f'<b>{code}</b><small>{successes}/{count}</small></span>'
            )
        telemetry = []
        for label, value, suffix in (
            ("Mean reward", row["mean_reward"], ""), ("Mean tokens/task", row["tokens_per_task"], ""),
            ("Estimated USD/verified", row["est_cost_per_verified"], ""),
            ("Median latency", row["median_latency_ms"], " ms"),
        ):
            if isinstance(value, (int, float)):
                telemetry.append(f"{label} {value:.3f}{suffix}" if suffix == "" else f"{label} {value:.0f}{suffix}")
        normalization_data = row["normalizations"]
        if isinstance(normalization_data, dict):
            fence_count = normalization_data.get("strip_code_fence")
            think_count = normalization_data.get("strip_think_block")
            if (type(fence_count) is int and fence_count >= 0
                    and type(think_count) is int and think_count >= 0):
                normalization_calls = row["normalization_calls"]
                if type(normalization_calls) is int and normalization_calls > 0:
                    normalization_text = (
                        f"Fence {fence_count / normalization_calls:.1%} ({fence_count}/{normalization_calls}); "
                        f"think {think_count / normalization_calls:.1%} ({think_count}/{normalization_calls})"
                    )
                else:
                    normalization_text = f"Fence {fence_count}; think {think_count} actions (rate unavailable)"
            else:
                normalization_text = "not reported"
        else:
            normalization_text = "not reported"
        rows_html.append(
            '<tr data-leaderboard-row data-source="official" '
            f'data-model-key="{html_escape(row["key"], quote=True)}" '
            f'data-score-pass1="{leaderboard_numeric_attr(p1)}" data-score-pass3="{leaderboard_numeric_attr(p3)}" '
            f'data-coverage="{n}" data-tokens="{leaderboard_numeric_attr(row["tokens_per_task"])}" '
            f'data-latency="{leaderboard_numeric_attr(row["median_latency_ms"])}" '
            f'data-cost="{leaderboard_numeric_attr(row["est_cost_per_verified"])}">'
            '<th scope="row">' + html_escape(row["label"]) + '<small>' + html_escape(row["provider"]) + '</small></th>'
            + f'<td>{n}/{len(common_ids)}<small>{html_escape(rank)}</small></td>'
            + f'<td>{p1:.1%}<small>95% Wilson {ci1[0]:.1%}–{ci1[1]:.1%}</small></td>'
            + f'<td>{p3:.1%}<small>95% Wilson {ci3[0]:.1%}–{ci3[1]:.1%}</small></td>'
            + '<td><div class="pramaan-grammar-heat" aria-label="Pass@3 by grammar">' + "".join(heat_cells) + '</div></td>'
            + '<td>' + html_escape(normalization_text) + '</td>'
            + '<td>' + html_escape(" · ".join(telemetry) or "No telemetry available") + '</td></tr>'
        )
    baseline_rows = []
    baselines = index.get("baselines") if isinstance(index.get("baselines"), list) else []
    for baseline in baselines:
        if not isinstance(baseline, dict):
            continue
        name = html_escape(str(baseline.get("key", "Verifier baseline")))
        total = baseline.get("n_tasks", 0)
        passed = baseline.get("passed")
        rate = baseline.get("pass_at_1")
        passed_text = f"{passed} / {total}" if isinstance(passed, int) and isinstance(total, int) else "not available"
        rate_text = f"{rate:.1%}" if isinstance(rate, (int, float)) else "not available"
        baseline_rows.append(f"<li>{name}: {passed_text} tasks passed ({rate_text} pass@1, full suite)</li>")
    baseline_html = ("<h4>Verifier baselines · full suite, shown for calibration</h4><ul>"
                     + "".join(baseline_rows) + "</ul>" if baseline_rows else "")
    hero_html = _leaderboard_hero(records, common_ids, wilson_interval)
    body = ("<p>Ranks use shared completed prompts. Intervals are 95% Wilson; partial arms stay visible but unranked.</p>"
            + hero_html +
            '<div class="pramaan-leaderboard-scroll"><table class="pramaan-leaderboard" data-leaderboard-table><thead><tr>'
            '<th scope="col">Model</th><th scope="col">Coverage / rank</th><th scope="col">Pass@1</th>'
            '<th scope="col">Pass@3</th><th scope="col">G1–G7 pass@3</th><th scope="col">Output cleanup</th><th scope="col">Telemetry</th>'
            '</tr></thead><tbody>' + "".join(rows_html) + '</tbody></table></div>' + baseline_html
            + comparison_charts(records))
    return ('<section class="pramaan-canonical-leaderboard" aria-label="Canonical model benchmark">'
            f'<h3>Canonical model results · {html_escape(suite_id)}</h3>'
            f'<p>Shared questions: {len(common_ids)} · suite digest {html_escape(str(suite_sha256)[:12])}…</p>'
            + body + '</section>')


def _leaderboard_hero(records: list[dict], common_ids: list[str], wilson_interval) -> str:
    """Render the shared-completion pass@3 chart, keeping partial arms visible and unranked."""
    complete = [row for row in records if row.get("complete") is True and row.get("n", 0) > 0]
    complete.sort(key=lambda row: (-row["passed3"], -row["passed1"], row["label"].casefold()))
    rank_by_key: dict[str, int] = {}
    tied: set[str] = set()
    for row in complete:
        score = (row["passed3"], row["passed1"])
        group = [item for item in complete if (item["passed3"], item["passed1"]) == score]
        rank_by_key[row["key"]] = min(i for i, item in enumerate(complete, 1)
                                       if (item["passed3"], item["passed1"]) == score)
        if len(group) > 1:
            tied.update(item["key"] for item in group)

    items = []
    for row in complete:
        n = row["n"]
        successes = row["passed3"]
        rate = successes / n
        lo, hi = wilson_interval(successes, n)
        low_pct, high_pct = max(0, lo * 100), min(100, hi * 100)
        if len(complete) < 2:
            rank = "Baseline · no other complete arm"
        elif row["key"] in tied:
            rank = f"Tie · rank {rank_by_key[row['key']]}"
        else:
            rank = f"Rank {rank_by_key[row['key']]}"
        label = html_escape(row["label"], quote=True)
        items.append(
            f'<li data-testid="leaderboard-hero-{html_escape(row["key"], quote=True)}" '
            f'data-model-key="{html_escape(row["key"], quote=True)}" data-coverage="{n}/{len(common_ids)}">'
            f'<div class="pramaan-leaderboard-hero-label"><strong>{label}</strong>'
            f'<span>{rank} · {successes}/{n} on shared questions · Pass@3 {rate:.1%}</span></div>'
            '<div class="pramaan-leaderboard-hero-track" role="img" '
            f'aria-label="{label}: pass at three {successes} of {n}, {rate:.1%}; 95 percent Wilson interval {lo:.1%} to {hi:.1%}">'
            f'<span class="pramaan-leaderboard-hero-bar" style="width:{rate * 100:.3f}%"></span>'
            f'<span class="pramaan-leaderboard-hero-interval" style="left:{low_pct:.3f}%;width:{high_pct - low_pct:.3f}%"></span>'
            f'</div><small>95% Wilson interval {lo:.1%}–{hi:.1%}</small></li>'
        )
    for row in records:
        if row.get("complete") is True:
            continue
        if row.get("n", 0) <= 0:
            continue
        label = html_escape(row["label"], quote=True)
        items.append(
            f'<li class="is-partial" data-testid="leaderboard-hero-{html_escape(row["key"], quote=True)}" '
            f'data-model-key="{html_escape(row["key"], quote=True)}" data-coverage="{row["n"]}/{len(common_ids)}">'
            f'<div class="pramaan-leaderboard-hero-label"><strong>{label}</strong>'
            f'<span>Partial coverage {row["n"]}/{len(common_ids)}; not ranked until shared coverage is complete</span></div>'
            '</li>'
        )
    if not items:
        content = '<p role="status">No completed model results yet. Arms appear here as results land.</p>'
    else:
        content = '<ol class="pramaan-leaderboard-hero-list">' + ''.join(items) + '</ol>'
    return ('<section class="pramaan-leaderboard-hero" data-testid="leaderboard-hero" '
            'aria-label="Pass at three on shared completed questions">'
            '<h4>Pass@3 on shared questions</h4>'
            f'<p>Each bar compares the same {len(common_ids)} completed questions. Lines show 95% Wilson intervals. '
            'Partial arms remain visible but are not ranked.</p>' + content +
            '<p class="pramaan-leaderboard-hero-text-alt">Text alternative: the model, pass count, rate, coverage, '
            'rank or tie, and confidence interval are printed beside each bar.</p></section>')
