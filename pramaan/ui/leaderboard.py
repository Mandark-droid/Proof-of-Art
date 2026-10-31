"""Accessible leaderboard renderers for canonical and community results."""

from __future__ import annotations

import html
import math
import re
from pathlib import Path

from html import escape as html_escape


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
    body = ("<p>Ranks use only the intersection of completed prompt IDs. Confidence intervals are 95% Wilson intervals; partial arms stay visible without a cross-model rank.</p>"
            '<div class="pramaan-leaderboard-scroll"><table class="pramaan-leaderboard" data-leaderboard-table><thead><tr>'
            '<th scope="col">Model</th><th scope="col">Coverage / rank</th><th scope="col">Pass@1</th>'
            '<th scope="col">Pass@3</th><th scope="col">G1–G7 pass@3</th><th scope="col">Output cleanup</th><th scope="col">Telemetry</th>'
            '</tr></thead><tbody>' + "".join(rows_html) + '</tbody></table></div>' + baseline_html
            + comparison_charts(records))
    return ('<section class="pramaan-canonical-leaderboard" aria-label="Canonical model benchmark">'
            f'<h3>Canonical model results · {html_escape(suite_id)}</h3>'
            f'<p>Shared questions: {len(common_ids)} · suite digest {html_escape(str(suite_sha256)[:12])}…</p>'
            + body + '</section>')
