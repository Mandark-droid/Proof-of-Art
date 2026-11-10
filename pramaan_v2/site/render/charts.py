"""Server-side charts for the leaderboard: inline SVG and HTML, no chart library (spec section 9).

Markup and geometry are ported from ``Leaderboard.dc.html`` and the reference scripts
``radar.py`` and ``matrix.py``. Every value comes from ``leaderboard.json`` or ``wall.json``;
these functions only format and lay out. Every mark carries a ``title`` (HTML) or ``<title>``
(SVG) with the model, the measure and the raw count; ``40-leaderboard.js`` turns them into a
styled tooltip that also shows on keyboard focus.
"""
from __future__ import annotations

import math
from html import escape

from pramaan_v2.constants import GRAMMAR_LABELS

from .replays import run_url

MONO = "'JetBrains Mono', ui-monospace, Menlo, monospace"
SANS = "'Anek Latin', 'Segoe UI', sans-serif"
TURMERIC, BROKE, UNREADABLE, TRACK, HAIRLINE, BORDER = "#F4B63F", "#C8452B", "#5E554E", "#211B18", "#2E2621", "#4A3F38"
CHALK, MUTED, FAINT, GROUND, PANEL = "#F4EFE6", "#B8ADA1", "#948980", "#15110F", "#1D1815"
SERIES = (("#C08318", "circle"), ("#2FA596", "square"), ("#9A7BE0", "diamond"))
DASH = "–"  # shown where the published data has no value


def esc(value) -> str:
    return escape("" if value is None else str(value), quote=True)


# ---------------------------------------------------------------- number formats

def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def tokens_k(tokens: float | None) -> str:
    return DASH if tokens is None else f"{tokens / 1000:.1f}k"


def seconds(s: float | None) -> str:
    return DASH if s is None else f"{s:.1f} s"


def dollars(usd: float | None) -> str:
    return DASH if usd is None else f"${usd:.2f}"


def rank_text(rank: int | None, width: int = 0) -> str:
    """A rank, or a dash for a model that did not run every brief and so is not ranked."""
    return DASH if rank is None else f"{rank:0{width}d}"


def share(count: int, n: int) -> float:
    return count / n if n else 0.0


def priced(rows: list[dict]) -> list[dict]:
    """Models with a published spend; the others stay off the cost chart and its frontier."""
    return [m for m in rows if isinstance(m.get("spend_usd"), (int, float))]


def half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def wilson_label(interval) -> str:
    lo, hi = interval
    return f"{half_up(lo * 100)}–{half_up(hi * 100)}%"


def short_grammar(gid: str) -> str:
    """Spoke and matrix-header word: first word of the grammar label."""
    return GRAMMAR_LABELS.get(gid, gid).split()[0]


# Caption nouns and tooltip labels from radar.py (design wording).
_CAPTION_NOUN = {"jaali_lattice": "jaali", "rotational_field": "rotational", "kolam_loop": "kolam"}
_TIP_LABEL = {"rotational_field": "Rotational"}


def caption_noun(gid: str) -> str:
    return _CAPTION_NOUN.get(gid, GRAMMAR_LABELS.get(gid, gid).lower())


def tip_label(gid: str) -> str:
    return _TIP_LABEL.get(gid, GRAMMAR_LABELS.get(gid, gid))


# ---------------------------------------------------------------- 9.1 ranking

RANK_GRID = "28px minmax(110px, 190px) minmax(0, 1fr) 150px"


def ranking_row(m: dict, rank: int | None, *, hidden: bool) -> str:
    lo, hi = m["wilson95"]
    width = share(m["within3"], m["n"]) * 100
    label = esc(m["label"])
    bar_tip = f"{label} · passed within three: {m['within3']} of {m['n']}"
    int_tip = f"{label} · 95% Wilson interval: {pct(lo)} to {pct(hi)}"
    return (
        f'<div class="lb-rank-row" data-model="{esc(m["key"])}"{" hidden" if hidden else ""} '
        f'style="display: grid; grid-template-columns: {RANK_GRID}; gap: 14px; align-items: center; '
        f'min-height: 45px; border-top: 1px solid {HAIRLINE}">'
        f'<span style="font-family: {MONO}; font-size: 13px; color: {FAINT}">{rank_text(rank, 2)}</span>'
        f'<span class="lb-rank-name" style="font-weight: 600">{label}</span>'
        f'<span class="lb-rank-bars" style="display: flex; flex-direction: column; gap: 4px">'
        f'<span class="lb-tipmark" tabindex="0" title="{bar_tip}" style="display: block; height: 14px; background: {TRACK}">'
        f'<span style="display: block; width: {width:.2f}%; height: 100%; background: {TURMERIC}"></span></span>'
        f'<span style="position: relative; display: block; height: 6px">'
        f'<span class="lb-tipmark" title="{int_tip}" style="position: absolute; left: {lo * 100:.1f}%; '
        f'width: {(hi - lo) * 100:.1f}%; top: 0; height: 6px; box-sizing: border-box; border-left: 2px solid {MUTED}; '
        f'border-right: 2px solid {MUTED}"><span style="position: absolute; left: 0; right: 0; top: 2px; height: 2px; '
        f'background: {MUTED}"></span></span></span></span>'
        f'<span class="lb-rank-value" style="font-family: {MONO}; font-size: 13px; text-align: right; white-space: nowrap">'
        f'{m["within3"]} of {m["n"]} <span style="color: {FAINT}">{wilson_label(m["wilson95"])}</span></span></div>'
    )


def ranking_table(rows: list[dict], ranks: dict[str, int | None]) -> str:
    """The ranking's numbers as a real table (for screen readers; the bars carry the same values)."""
    body = "".join(
        f"<tr><th scope=\"row\">{esc(m['label'])}</th><td>{rank_text(ranks.get(m['key']))}</td><td>{m['within3']} of {m['n']}</td>"
        f"<td>{pct(m['wilson95'][0])} to {pct(m['wilson95'][1])}</td></tr>" for m in rows)
    return ('<div class="lb-sr"><table><caption>Briefs passed within three tries, with 95% Wilson intervals</caption>'
            '<thead><tr><th scope="col">Model</th><th scope="col">Rank</th><th scope="col">Within three</th>'
            f'<th scope="col">95% interval</th></tr></thead><tbody>{body}</tbody></table></div>')


# ---------------------------------------------------------------- 9.2 results table

def first_try_split(m: dict) -> list[float]:
    f = m["first"]
    parts = [round(share(f["passed"], m["n"]) * 100, 2), round(share(f["broke"], m["n"]) * 100, 2)]
    return parts + [round(100 - sum(parts), 2)]


def first_try_bar(m: dict) -> str:
    f = m["first"]
    label = esc(m["label"])
    widths = first_try_split(m)
    segs = (("passed", f["passed"], TURMERIC, "first try passed"), ("broke", f["broke"], BROKE, "first try broke a rule"),
            ("unreadable", f["unreadable"], UNREADABLE, "first try unreadable"))
    spans = "".join(
        f'<span class="lb-tipmark" title="{label} · {words}: {n} of {m["n"]}" '
        f'style="width: {w:g}%; background: {colour}"></span>'
        for (_, n, colour, words), w in zip(segs, widths))
    return (f'<span style="display: flex; height: 16px">{spans}</span>'
            f'<span style="display: block; margin-top: 6px; font-size: 12px; color: {MUTED}">'
            f'{f["passed"]} passed · {f["broke"]} broke a rule · {f["unreadable"]} unreadable</span>')


def results_row(m: dict, *, hidden: bool) -> str:
    td = "padding: 12px"
    return (
        f'<tr class="lb-res-row" data-model="{esc(m["key"])}"{" hidden" if hidden else ""} '
        f'style="border-top: 1px solid {HAIRLINE}">'
        f'<th scope="row" style="padding: 12px 12px 12px 0; text-align: left; font-family: {SANS}; font-size: 17px; '
        f'font-weight: 600">{esc(m["label"])}</th>'
        f'<td style="{td}"><span style="font-size: 22px; font-weight: 700; color: {TURMERIC}">{m["within3"]}</span> '
        f'<span style="color: {FAINT}">{pct(share(m["within3"], m["n"]))}</span></td>'
        f'<td style="{td}">{first_try_bar(m)}</td>'
        f'<td style="{td}">{m["passed_of_readable"][0]} of {m["passed_of_readable"][1]}</td>'
        f'<td style="{td}; text-align: right">{m["repaired"][0]} of {m["repaired"][1]}</td>'
        f'<td style="{td}; text-align: right">{tokens_k(m["tokens_per_brief"])}</td>'
        f'<td style="{td}; text-align: right">{seconds(m["latency_s"])}</td>'
        f'<td style="padding: 12px 0 12px 12px; text-align: right">{dollars(m["spend_usd"])}</td></tr>'
    )


def results_table(rows: list[dict], hidden: set[str], briefs: int) -> str:
    th = "padding: 10px 12px; font-weight: 500"
    head = (
        f'<tr style="font-size: 11px; letter-spacing: 0.08em; color: {FAINT}; text-align: left">'
        f'<th scope="col" style="padding: 10px 12px 10px 0; font-weight: 500; width: 180px">MODEL</th>'
        f'<th scope="col" style="{th}; width: 120px">WITHIN THREE</th>'
        f'<th scope="col" style="{th}">FIRST TRY, ALL {briefs} BRIEFS</th>'
        f'<th scope="col" style="{th}; width: 120px">PASSED, OF READABLE</th>'
        f'<th scope="col" style="{th}; width: 76px; text-align: right">REPAIRED</th>'
        f'<th scope="col" style="{th}; width: 70px; text-align: right">TOKENS</th>'
        f'<th scope="col" style="{th}; width: 76px; text-align: right">LATENCY</th>'
        f'<th scope="col" style="padding: 10px 0 10px 12px; font-weight: 500; width: 70px; text-align: right">SPEND</th></tr>'
    )
    body = "".join(results_row(m, hidden=m["key"] in hidden) for m in rows)
    return (
        '<div class="lb-scroll lb-sticky" style="overflow-x: auto">'
        f'<table style="width: 100%; min-width: 1120px; border-collapse: collapse; font-family: {MONO}; font-size: 14px">'
        f'<thead>{head}</thead><tbody data-lb-rows="results" style="border-bottom: 1px solid {HAIRLINE}">{body}'
        '</tbody></table></div>'
    )


# ---------------------------------------------------------------- 9.3 results matrix

CELL_W, CELL_GAP, GROUP_GAP = 30, 3, 14   # matrix.py units: 30 x 40 cells, 3 between, 14 between groups
STATUS_CLASS = {"1": "s1", "2": "s2", "3": "s3", "x": "sx", "-": "sn"}
STATUS_WORDS = {"1": "passed on try 1", "2": "passed on try 2", "3": "passed on try 3", "x": "broke a rule",
                "-": "never readable"}


def _group_units(n: int) -> int:
    return n * CELL_W + (n - 1) * CELL_GAP


def matrix_header_label(gid: str, n: int) -> str:
    label = GRAMMAR_LABELS.get(gid, gid)
    if n >= 6:
        return label.upper()
    word = label.split()[0].upper()
    return word if len(word) <= 6 else word[:5] + "."


def matrix(models: list[dict], groups: list[dict], runs: list[dict], state_names: dict[str, str]) -> str:
    cell = {(r["model"], r["task"]): r for r in runs}
    sizes = [len(g["tasks"]) for g in groups]
    total = sum(_group_units(n) for n in sizes) + GROUP_GAP * (len(groups) - 1)
    outer_gap = f"{GROUP_GAP / total * 100:.3f}%"

    def group_box(n: int, inner: str, extra: str = "") -> str:
        units = _group_units(n)
        return (f'<div style="flex: {units} 1 0; min-width: 0; display: grid; '
                f'grid-template-columns: repeat({n}, minmax(0, 1fr)); column-gap: {CELL_GAP / units * 100:.3f}%{extra}">'
                f'{inner}</div>')

    head = "".join(
        f'<div style="flex: {_group_units(n)} 1 0; min-width: 0; white-space: nowrap">'
        f'{esc(matrix_header_label(g["grammar"], n))}<br>{n}</div>' for g, n in zip(groups, sizes))
    rows = [
        '<div class="lb-mrow" aria-hidden="true" style="display: flex; gap: 12px; height: 34px">'
        '<span class="lb-mname" style="flex: 0 0 150px"></span>'
        f'<div style="flex: 1 1 0; min-width: 0; display: flex; column-gap: {outer_gap}; font-family: {MONO}; '
        f'font-size: 10px; line-height: 1.3; color: {FAINT}">{head}</div></div>'
    ]
    for m in models:
        key, label = m["key"], m["label"]
        groups_html = []
        for g in groups:
            cells = []
            for task in g["tasks"]:
                run = cell.get((key, task))
                if run is None:  # no run of this model on this brief: an empty, inert cell
                    cells.append('<span class="lb-c lb-gap" aria-hidden="true"></span>')
                    continue
                status = run["status"]
                classes = STATUS_CLASS[status] + (" ink" if run["low_ink"] else "")
                title = f"{label} · {state_names.get(run['state'], run['state'])} {run['level']} · {STATUS_WORDS[status]}"
                if run["low_ink"]:
                    title += " · low ink"
                cells.append(f'<a class="lb-c {classes}" href="{esc(run_url(task, key))}" title="{esc(title)}" '
                             f'aria-label="{esc(title)}"></a>')
            groups_html.append(group_box(len(g["tasks"]), "".join(cells)))
        rows.append(
            f'<div class="lb-mrow" data-model="{esc(key)}" style="display: flex; align-items: center; gap: 12px">'
            f'<span class="lb-mname" style="flex: 0 0 150px; font-size: 13px; line-height: 1.1">{esc(label)}</span>'
            f'<div style="flex: 1 1 0; min-width: 0; display: flex; column-gap: {outer_gap}">{"".join(groups_html)}'
            '</div></div>')
    return (
        '<div class="lb-scroll" style="overflow-x: auto">'
        f'<div class="lb-matrix" role="group" aria-label="Results matrix: {len(models)} models down, '
        f'{sum(sizes)} briefs across in {len(groups)} grammar groups" '
        'style="display: flex; flex-direction: column; gap: 2px; min-width: 980px">'
        + "".join(rows) + "</div></div>"
    )


def matrix_legend(low_ink: int) -> str:
    swatch = '<span style="width: 16px; height: 16px; background: {}"></span>'
    items = [(TURMERIC, "1 · first try"), ("#9F772D", "2 · second try"), ("#654C20", "3 · third try"),
             (BROKE, "× drew it, broke a rule"), ("#3A322D", "– never readable")]
    out = "".join(f'<span style="display: inline-flex; align-items: center; gap: 8px">{swatch.format(c)}{t}</span>'
                  for c, t in items)
    out += ('<span style="display: inline-flex; align-items: center; gap: 8px">'
            '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><rect width="16" height="16" '
            f'fill="{TURMERIC}"></rect><path d="M8 0h8v8z" fill="{GROUND}"></path></svg>'
            f'Notched corner · accepted with under 2% visible ink ({low_ink} cells)</span>')
    return (f'<div style="display: flex; flex-wrap: wrap; gap: 8px 22px; margin-top: 14px; font-size: 14px; '
            f'color: {MUTED}">{out}</div>')


# ---------------------------------------------------------------- 9.4 failing checks

FORMAT_CHECKS = {"candidate_parse", "candidate_truncated"}


def failing_rows(failing: list[dict], top: int = 11) -> tuple[list[dict], list[dict]]:
    ordered = sorted(failing, key=lambda r: -r["count"])
    return ordered[:top], ordered[top:]


def failing_bars(failing: list[dict], first_tries: int, label, other_label) -> str:
    head, rest = failing_rows(failing)
    peak = max(r["count"] for r in failing) if failing else 1
    rows = [(label(r["check"]), r["count"], r.get("kind") == "format" or r["check"] in FORMAT_CHECKS,
             f"{label(r['check'])} · first tries failing: {r['count']} of {first_tries}") for r in head]
    if rest:
        n = sum(r["count"] for r in rest)
        parts = ", ".join(f"{label(r['check'])} {r['count']}" for r in rest)
        rows.append((other_label(len(rest)), n, False, f"{other_label(len(rest))} · first tries failing: {n} ({parts})"))
    out = []
    for text, n, fmt, tip in rows:
        out.append(
            f'<div class="lb-fail-row" data-count="{n}" style="display: grid; grid-template-columns: 190px minmax(0, 1fr) 36px; '
            f'gap: 10px; align-items: center"><span>{esc(text)}</span>'
            f'<span class="lb-tipmark" tabindex="0" title="{esc(tip)}" style="height: 16px; background: {TRACK}">'
            f'<span style="display: block; width: {n / peak * 100:.1f}%; height: 100%; '
            f'background: {UNREADABLE if fmt else BROKE}"></span></span>'
            f'<span style="font-family: {MONO}; text-align: right">{n}</span></div>')
    table = "".join(f"<tr><th scope=\"row\">{esc(label(r['check']))}</th><td>{r['count']}</td></tr>"
                    for r in head + rest)
    return (f'<div style="display: flex; flex-direction: column; gap: 9px; font-size: 14px">{"".join(out)}</div>'
            f'<div class="lb-sr"><table><caption>Failing checks on the first try, out of {first_tries} first proposals'
            '</caption><thead><tr><th scope="col">Check</th><th scope="col">First tries</th>'
            f'</tr></thead><tbody>{table}</tbody></table></div>')


# ---------------------------------------------------------------- 9.5 cost scatter

COST_W, COST_H = 640, 380
X0, X1, Y0, Y1 = 56, 616, 16, 336
LOG_MIN, LOG_MAX = math.log10(0.1), math.log10(30)
COST_TICKS = ((0.1, "$0.10"), (0.3, "$0.30"), (1, "$1"), (3, "$3"), (10, "$10"), (30, "$30"))
LABEL_SIZE, CHAR_W = 12, 7.2   # JetBrains Mono advance is 0.6 em
PLOT_BOTTOM = 348              # labels stay above the x tick labels


def cost_x(spend: float) -> float:
    v = math.log10(max(spend, 1e-6))
    v = min(max(v, LOG_MIN), LOG_MAX)
    return round(X0 + (X1 - X0) * (v - LOG_MIN) / (LOG_MAX - LOG_MIN), 1)


def cost_y(passed: int, briefs: int) -> float:
    return round(Y1 - (Y1 - Y0) * share(passed, briefs), 1)


def cost_ticks(briefs: int) -> list[int]:
    """Briefs-passed grid lines: the brief count and its quarters, down to zero."""
    return sorted({round(briefs * k / 4) for k in range(5)}, reverse=True)


def frontier(rows: list[dict]) -> list[dict]:
    """Models flagged on the frontier, cheapest first (the builder computes the flag from spec 7.5)."""
    return sorted((m for m in priced(rows) if m.get("on_frontier")), key=lambda m: m["spend_usd"])


# Candidate label positions around a point: (dx, dy, anchor). Order is preference.
_CANDIDATES = (
    (10, 4, "start"), (-10, 4, "end"), (-10, -8, "end"), (10, -8, "start"), (11, 17, "start"), (-11, 17, "end"),
    (0, -12, "middle"), (0, 22, "middle"), (12, -20, "start"), (-12, -20, "end"), (12, 29, "start"),
    (-12, 29, "end"), (0, -25, "middle"), (0, 34, "middle"), (24, 4, "start"), (-24, 4, "end"),
    (-20, -12, "start"), (20, -12, "end"), (-20, 22, "start"), (20, 22, "end"),
)


def _box(x: float, y: float, anchor: str, width: float) -> tuple[float, float, float, float]:
    left = x if anchor == "start" else x - width if anchor == "end" else x - width / 2
    return (left, y - 10, left + width, y + 3)


def _overlap(a, b, pad=2.0) -> float:
    w = min(a[2], b[2]) + pad - max(a[0], b[0])
    h = min(a[3], b[3]) + pad - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _distance(box, cx, cy) -> float:
    nx, ny = min(max(cx, box[0]), box[2]), min(max(cy, box[1]), box[3])
    return math.hypot(cx - nx, cy - ny)


def _point_hit(box, cx, cy, r=8.0) -> float:
    nx, ny = min(max(cx, box[0]), box[2]), min(max(cy, box[1]), box[3])
    d = math.hypot(cx - nx, cy - ny)
    return 0.0 if d >= r else (r - d) * 40


def _segment_hits(box, segments) -> int:
    hits = 0
    for (x1, y1), (x2, y2) in segments:
        steps = max(2, int(math.hypot(x2 - x1, y2 - y1) / 2))
        for i in range(steps + 1):
            t = i / steps
            x, y = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
            if box[0] - 1 <= x <= box[2] + 1 and box[1] - 1 <= y <= box[3] + 1:
                hits += 1
    return hits


def place_cost_labels(rows: list[dict], briefs: int) -> list[dict]:
    """Direct labels for every point, placed so they overlap no label, no point and no frontier segment."""
    pts = [(m, cost_x(m["spend_usd"]), cost_y(m["within3"], briefs)) for m in priced(rows)]
    line = [(cost_x(m["spend_usd"]), cost_y(m["within3"], briefs)) for m in frontier(rows)]
    segments = list(zip(line, line[1:]))

    def cost(i: int, cand: int, chosen: dict[int, int]) -> float:
        m, cx, cy = pts[i]
        dx, dy, anchor = _CANDIDATES[cand]
        box = _box(cx + dx, cy + dy, anchor, len(m["label"]) * CHAR_W)
        score = cand * 0.5
        if box[0] < 0 or box[2] > COST_W or box[1] < 0 or box[3] > PLOT_BOTTOM:
            score += 1e6
        if box[0] < X0 - 4 and box[3] > Y1 - 4:
            score += 1e5
        own = _distance(box, cx, cy)
        for j, (_, ox, oy) in enumerate(pts):
            if j == i:
                continue
            score += _point_hit(box, ox, oy) * 1000
            if _distance(box, ox, oy) < own + 2:  # the label would read as another point's
                score += 5000
        score += _segment_hits(box, segments) * 200
        for j, c in chosen.items():
            if j == i:
                continue
            mj, ox, oy = pts[j]
            ddx, ddy, anc = _CANDIDATES[c]
            other = _box(ox + ddx, oy + ddy, anc, len(mj["label"]) * CHAR_W)
            score += _overlap(box, other) * 1000
        return score

    # Most crowded points first.
    def crowd(i):
        _, x, y = pts[i]
        return -sum(1 for _, ox, oy in pts if abs(ox - x) < 120 and abs(oy - y) < 30)

    order = sorted(range(len(pts)), key=crowd)
    chosen: dict[int, int] = {}
    for i in order:
        chosen[i] = min(range(len(_CANDIDATES)), key=lambda c: cost(i, c, chosen))
    for _ in range(6):
        changed = False
        for i in order:
            best = min(range(len(_CANDIDATES)), key=lambda c: cost(i, c, chosen))
            if best != chosen[i] and cost(i, best, chosen) < cost(i, chosen[i], chosen):
                chosen[i], changed = best, True
        if not changed:
            break
    out = []
    for i, (m, cx, cy) in enumerate(pts):
        dx, dy, anchor = _CANDIDATES[chosen[i]]
        x, y = cx + dx, cy + dy
        out.append({"key": m["key"], "x": round(x, 1), "y": round(y, 1), "anchor": anchor,
                    "box": _box(x, y, anchor, len(m["label"]) * CHAR_W)})
    return out


def cost_scatter(rows: list[dict], briefs: int) -> str:
    rows = priced(rows)
    front = frontier(rows)
    names = ", ".join(m["label"] for m in front[:-1]) + (" and " + front[-1]["label"] if len(front) > 1 else
                                                          (front[0]["label"] if front else ""))
    o = [f'<svg class="lb-cost" viewBox="0 0 {COST_W} {COST_H}" role="img" aria-label="Scatter chart of estimated run '
         f'cost against briefs passed within three tries. The cost frontier runs through {esc(names)}." '
         f'style="display: block; width: 100%; height: auto; font-family: {MONO}">']
    grid = "".join(f"M{X0} {cost_y(v, briefs):g}H{X1}" for v in cost_ticks(briefs))
    o.append(f'<path d="{grid}" stroke="{HAIRLINE}" stroke-width="1" fill="none"></path>')
    o.append("".join(f'<text x="48" y="{cost_y(v, briefs) + 4:g}" text-anchor="end" font-size="11" '
                     f'fill="{FAINT}">{v}</text>' for v in cost_ticks(briefs)))
    ticks = []
    for v, text in COST_TICKS:
        x = min(cost_x(v), COST_W - 36)
        ticks.append(f'<text x="{x:g}" y="360" text-anchor="middle" font-size="11" fill="{FAINT}">{text}</text>')
    o.append("".join(ticks))
    pts = " ".join("%.1f,%.1f" % (cost_x(m["spend_usd"]), cost_y(m["within3"], briefs)) for m in front)
    o.append(f'<polyline class="lb-frontier" points="{pts}" data-models="{esc(",".join(m["key"] for m in front))}" '
             f'stroke="{TURMERIC}" stroke-width="1.5" stroke-dasharray="5 5" fill="none"></polyline>')
    labels = {p["key"]: p for p in place_cost_labels(rows, briefs)}
    for m in rows:
        cx, cy = cost_x(m["spend_usd"]), cost_y(m["within3"], briefs)
        lab = labels[m["key"]]
        on = m.get("on_frontier")
        tip = (f"{m['label']} · briefs passed within three: {m['within3']} of {m['n']} · estimated spend "
               f"{dollars(m['spend_usd'])}" + (" · on the cost frontier" if on else ""))
        o.append(f'<g data-model="{esc(m["key"])}"><circle class="lb-tipmark" tabindex="0" cx="{cx:g}" cy="{cy:g}" r="7" '
                 f'fill="{TURMERIC if on else MUTED}" stroke="{GROUND}" stroke-width="2"><title>{esc(tip)}</title></circle>'
                 f'<text x="{lab["x"]:g}" y="{lab["y"]:g}"'
                 f'{"" if lab["anchor"] == "start" else " text-anchor=" + chr(34) + lab["anchor"] + chr(34)} '
                 f'font-size="{LABEL_SIZE}" fill="{CHALK}">{esc(m["label"])}</text></g>')
    o.append("</svg>")
    return "".join(o)


# ---------------------------------------------------------------- 9.6 grammar heatmap

def heat_alpha(count: int, n: int) -> float:
    """Fill alpha 0.9 x share, snapped out of the 0.47..0.66 band (spec 6.5 contrast rule)."""
    alpha = round(0.9 * count / n, 2) if n else 0.0
    if 0.47 < alpha < 0.66:
        alpha = 0.47 if alpha < 0.53 else 0.66
    return alpha


def heat_style(count: int, n: int) -> str:
    if count == 0:
        return f"background: {PANEL}; color: {FAINT}"
    alpha = heat_alpha(count, n)
    return f"background: rgba(244, 182, 63, {alpha:g})" + (f"; color: {GROUND}" if alpha >= 0.66 else "")


def heat_header(gid: str) -> str:
    return "Rotational" if gid == "rotational_field" else GRAMMAR_LABELS.get(gid, gid)


def grammar_heatmap(rows: list[dict], grammars: list[tuple[str, int]]) -> str:
    head = "".join(f'<th scope="col" style="padding: 6px 4px; font-weight: 500">{esc(heat_header(g).upper())}<br>{n}</th>'
                   for g, n in grammars)
    body = []
    for m in rows:
        cells = "".join(
            f'<td class="lb-tipmark" title="{esc(m["label"])} · {esc(GRAMMAR_LABELS.get(g, g).lower())}: '
            f'{m["by_grammar"].get(g, 0)} of {n}" style="{heat_style(m["by_grammar"].get(g, 0), n)}">'
            f'{m["by_grammar"].get(g, 0)}</td>' for g, n in grammars)
        body.append(f'<tr data-model="{esc(m["key"])}"><th scope="row" style="padding: 9px 8px 9px 0; text-align: left; '
                    f'font-family: {SANS}; font-size: 15px; font-weight: 600">{esc(m["label"])}</th>{cells}</tr>')
    return (
        '<div class="lb-scroll lb-sticky" style="flex: 2 1 640px; min-width: 0; overflow-x: auto">'
        f'<table style="width: 100%; min-width: 720px; border-collapse: separate; border-spacing: 2px; font-family: {MONO}; '
        'font-size: 14px; text-align: center"><thead>'
        f'<tr style="font-size: 11px; letter-spacing: 0.04em; color: {FAINT}">'
        '<th scope="col" style="padding: 6px 8px 6px 0; text-align: left; font-weight: 500">PASSED WITHIN THREE</th>'
        f'{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
    )


# ---------------------------------------------------------------- 9.7 radars

def _pt(cx, cy, r, i, n):
    a = -math.pi / 2 + 2 * math.pi * i / n
    return cx + r * math.cos(a), cy + r * math.sin(a)


def _poly(points) -> str:
    return " ".join("%.1f,%.1f" % p for p in points)


def _anchor(i, n) -> str:
    x = math.cos(-math.pi / 2 + 2 * math.pi * i / n)
    return "middle" if abs(x) < 0.15 else ("start" if x > 0 else "end")


def _rings(cx, cy, R, n) -> list[str]:
    o = []
    for f in (0.25, 0.5, 0.75, 1.0):
        o.append('<polygon points="%s" fill="none" stroke="%s" stroke-width="1"></polygon>' % (
            _poly([_pt(cx, cy, R * f, i, n) for i in range(n)]), BORDER if f == 1.0 else HAIRLINE))
    for i in range(n):
        x, y = _pt(cx, cy, R, i, n)
        o.append('<line x1="%d" y1="%d" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1"></line>' % (cx, cy, x, y, HAIRLINE))
    return o


def radar_caption(by_grammar: dict, grammars: list[tuple[str, int]]) -> str:
    zeros = [g for g, _ in grammars if by_grammar.get(g, 0) == 0]
    if len(zeros) >= 3:
        return f"No passes in {len(zeros)} of {len(grammars)} grammars."
    if len(zeros) == 2:
        return f"No passes in {caption_noun(zeros[0])} or {caption_noun(zeros[1])}."
    g, n = min(grammars, key=lambda gn: by_grammar.get(gn[0], 0) / gn[1])
    return f"Lowest: {caption_noun(g)}, {by_grammar.get(g, 0)} of {n}."


def small_radar(m: dict, grammars: list[tuple[str, int]]) -> str:
    W, H, cx, cy, R = 280, 198, 140, 104, 58
    n = len(grammars)
    vals = [m["by_grammar"].get(g, 0) / k for g, k in grammars]
    aria = m["label"] + ": " + ", ".join(f"{heat_header(g).lower()} {m['by_grammar'].get(g, 0)} of {k}" for g, k in grammars)
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(aria)}" style="display: block; width: 100%; height: auto">']
    o += _rings(cx, cy, R, n)
    o.append(f'<polygon points="{_poly([_pt(cx, cy, R * v, i, n) for i, v in enumerate(vals)])}" fill="{TURMERIC}" '
             f'fill-opacity="0.2" stroke="{TURMERIC}" stroke-width="2" stroke-linejoin="round"></polygon>')
    for i, (v, (g, k)) in enumerate(zip(vals, grammars)):
        x, y = _pt(cx, cy, R * v, i, n)
        o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{TURMERIC}" stroke="{PANEL}" stroke-width="1.5">'
                 f'<title>{esc(tip_label(g))}: {m["by_grammar"].get(g, 0)} of {k}</title></circle>')
    for i, (g, _) in enumerate(grammars):
        x, y = _pt(cx, cy, R + 10, i, n)
        dy = -3 if i == 0 else (11 if i in (3, 4) else 4)
        o.append(f'<text x="{x:.1f}" y="{y + dy:.1f}" text-anchor="{_anchor(i, n)}" font-size="13.5" fill="{MUTED}">'
                 f'{esc(short_grammar(g))}</text>')
    o.append("</svg>")
    return "".join(o)


def radar_card(m: dict, grammars: list[tuple[str, int]]) -> str:
    return (
        f'<article class="lb-radar-card" data-model="{esc(m["key"])}" style="min-width: 0; padding: 14px 0 12px; '
        f'background: {PANEL}; border: 1px solid {HAIRLINE}; border-radius: 6px">'
        '<div style="display: flex; align-items: baseline; justify-content: space-between; gap: 8px; padding: 0 14px">'
        f'<h4 style="margin: 0; font-size: 16px; font-weight: 600">{esc(m["label"])}</h4>'
        f'<span style="font-family: {MONO}; font-size: 12px; color: {MUTED}; white-space: nowrap">{m["within3"]} of {m["n"]}</span>'
        f'</div>{small_radar(m, grammars)}'
        f'<p style="margin: 2px 14px 0; font-size: 14px; color: {MUTED}">{esc(radar_caption(m["by_grammar"], grammars))}</p>'
        '</article>'
    )


SPOKES = (
    # key in leaderboard radar, spoke label, rim note, raw-values row label
    ("passed", "Passed within three", "rim 100%", "Passed within three"),
    ("readable", "Readable at try 1", "rim 100%", "Readable at try 1"),
    ("repaired", "Repaired after a miss", "rim 100%", "Repaired after a miss"),
    ("tokens", "Fewer tokens", "rim 2k per brief", "Tokens per brief"),
    ("speed", "Faster", "rim 4 s", "Mean latency"),
    ("spend", "Lower spend", "rim $0.10", "Run spend"),
)
def spoke_defs(briefs: int) -> tuple[tuple[str, str, str, str], ...]:
    return (
        ("Passed within three", "Briefs passed in at most three tries", f"0 of {briefs}", f"{briefs} of {briefs}"),
        ("Readable at try 1", "First proposals that parsed and rendered", "0%", "100%"),
        ("Repaired after a miss", "Of briefs missed on try 1, those passed by try 3", "0%", "100%"),
        ("Fewer tokens", "Tokens per brief, log scale", "16k", "2k"),
        ("Faster", "Mean latency, log scale", "64 s", "4 s"),
        ("Lower spend", "Total spend for the run, log scale", "$32", "$0.10"),
    )


def radar_values(m: dict) -> list[float]:
    """Spoke values from leaderboard.json, clamped to 0..1 so an outside model still draws."""
    return [min(1.0, max(0.0, float(m["radar"][k]))) for k, *_ in SPOKES]


def radar_display(m: dict) -> list[str]:
    rep = m["repaired"]
    return [f"{m['within3']} of {m['n']}", pct(share(m["readable"], m["n"])), f"{rep[0]} of {rep[1]}",
            tokens_k(m["tokens_per_brief"]), seconds(m["latency_s"]), dollars(m["spend_usd"])]


def marker(shape: str, x: float, y: float, colour: str, title: str) -> str:
    t = f"<title>{esc(title)}</title>"
    common = f'class="lb-tipmark" tabindex="0" fill="{colour}" stroke="{GROUND}" stroke-width="2"'
    if shape == "circle":
        return f'<circle {common} cx="{x:.1f}" cy="{y:.1f}" r="5">{t}</circle>'
    if shape == "square":
        return f'<rect {common} x="{x - 4.8:.1f}" y="{y - 4.8:.1f}" width="9.6" height="9.6">{t}</rect>'
    return (f'<rect {common} x="{x - 4.5:.1f}" y="{y - 4.5:.1f}" width="9" height="9" '
            f'transform="rotate(45 {x:.1f} {y:.1f})">{t}</rect>')


def swatch(slot: int) -> str:
    colour, shape = SERIES[slot]
    inner = {"circle": f'<circle cx="8" cy="8" r="5" fill="{colour}"></circle>',
             "square": f'<rect x="3.2" y="3.2" width="9.6" height="9.6" fill="{colour}"></rect>',
             "diamond": f'<rect x="3.5" y="3.5" width="9" height="9" transform="rotate(45 8 8)" fill="{colour}"></rect>'}[shape]
    return f'<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" style="flex: none">{inner}</svg>'


CMP = {"W": 660, "H": 470, "cx": 330, "cy": 240, "R": 158}


def compare_series(selected: list[dict]) -> str:
    cx, cy, R, n = CMP["cx"], CMP["cy"], CMP["R"], len(SPOKES)
    polys, marks = [], []
    for slot, m in enumerate(selected):
        colour, shape = SERIES[slot]
        vals = radar_values(m)
        pts = [_pt(cx, cy, R * v, i, n) for i, v in enumerate(vals)]
        polys.append(f'<polygon class="lb-series" data-model="{esc(m["key"])}" data-values="'
                     + ",".join(f"{round(v, 3):g}" for v in vals)
                     + f'" points="{_poly(pts)}" fill="{colour}" fill-opacity="0.1" stroke="{colour}" '
                       'stroke-width="2.25" stroke-linejoin="round"></polygon>')
        for (x, y), (_, spoke, _, _), shown in zip(pts, SPOKES, radar_display(m)):
            marks.append(marker(shape, x, y, colour, f"{m['label']} · {spoke.lower()}: {shown}"))
    return "".join(polys) + "".join(marks)


def compare_radar(selected: list[dict]) -> str:
    W, H, cx, cy, R, n = CMP["W"], CMP["H"], CMP["cx"], CMP["cy"], CMP["R"], len(SPOKES)
    names = [m["label"] for m in selected]
    who = (", ".join(names[:-1]) + " and " + names[-1]) if len(names) > 1 else (names[0] if names else "no models")
    o = [f'<svg data-lb-radar viewBox="0 0 {W} {H}" role="img" aria-label="Radar chart comparing {esc(who)} on six axes. '
         'The values are in the table beside it." style="display: block; width: 100%; height: auto">']
    o += _rings(cx, cy, R, n)
    for i, (_, label, rim, _) in enumerate(SPOKES):
        x, y = _pt(cx, cy, R + 14, i, n)
        if i == 0:
            y1, y2 = y - 20, y - 3
        elif i == 3:
            y1, y2 = y + 12, y + 29
        else:
            y1, y2 = y - 2, y + 15
        a = _anchor(i, n)
        o.append(f'<text x="{x:.1f}" y="{y1:.1f}" text-anchor="{a}" font-size="15" font-weight="600" fill="{CHALK}">'
                 f'{label}</text>')
        o.append(f'<text x="{x:.1f}" y="{y2:.1f}" text-anchor="{a}" font-size="12" font-family="{MONO}" '
                 f'fill="{FAINT}">{esc(rim)}</text>')
    o.append(f'<g data-lb-series>{compare_series(selected)}</g></svg>')
    return "".join(o)


_TH = (f'scope="col" style="padding: 0 12px 8px 0; text-align: left; font-family: {MONO}; font-size: 11px; '
       f'font-weight: 500; letter-spacing: 0.06em; color: {FAINT}"')
_TD = f'padding: 9px 12px 9px 0; border-top: 1px solid {HAIRLINE}; vertical-align: top'
_TDM = f"{_TD}; font-family: {MONO}; font-size: 13px; white-space: nowrap"


def values_head(selected: list[dict]) -> str:
    cols = "".join(
        f'<th {_TH}><span style="display: inline-flex; align-items: center; gap: 6px; font-family: {SANS}; '
        f'font-size: 14px; font-weight: 600; letter-spacing: 0; color: {CHALK}">{swatch(slot)}{esc(m["label"])}</span></th>'
        for slot, m in enumerate(selected))
    return f"<tr><th {_TH}>VALUES SHOWN</th>{cols}</tr>"


def values_body(selected: list[dict]) -> str:
    shown = [radar_display(m) for m in selected]
    rows = []
    for i, (_, _, _, row_label) in enumerate(SPOKES):
        cells = "".join(f'<td style="{_TDM}">{esc(v[i])}</td>' for v in shown)
        rows.append(f'<tr><th scope="row" style="text-align: left; font-weight: 500; {_TD}">{row_label}</th>{cells}</tr>')
    return "".join(rows)


def values_table(selected: list[dict]) -> str:
    return (f'<table data-lb-values style="width: 100%; border-collapse: collapse; font-size: 14px">'
            f'<thead>{values_head(selected)}</thead><tbody>{values_body(selected)}</tbody></table>')


def spoke_table(briefs: int) -> str:
    rows = "".join(
        f'<tr><th scope="row" style="text-align: left; font-weight: 600; {_TD}">{a}</th>'
        f'<td style="{_TD}; color: {MUTED}">{b}</td><td style="{_TDM}">{c}</td><td style="{_TDM}">{d}</td></tr>'
        for a, b, c, d in spoke_defs(briefs))
    return (f'<table style="width: 100%; border-collapse: collapse; font-size: 14px"><thead><tr><th {_TH}>AXIS</th>'
            f'<th {_TH}>WHAT IT MEASURES</th><th {_TH}>CENTRE</th><th {_TH}>RIM</th></tr></thead><tbody>{rows}</tbody></table>')


# ---------------------------------------------------------------- methodology

def request_settings(params: dict) -> str:
    """The settings actually sent, in the board's words; token ceilings are stated once above the table."""
    params = dict(params or {})
    out = []
    if "temperature" in params:
        out.append(f"temperature {params.pop('temperature'):g}")
    if "reasoning_effort" in params:
        out.append(f"reasoning_effort {params.pop('reasoning_effort')}")
    effort = (params.pop("output_config", None) or {}).get("effort")
    if effort:
        out.append(f"output effort {effort}")
    thinking = ((params.pop("extra_body", None) or {}).get("thinking") or {}).get("type")
    if thinking:
        out.append(f"thinking {thinking}")
    for key in ("max_tokens", "max_completion_tokens"):
        params.pop(key, None)
    for key, value in sorted(params.items()):
        out.append(f"{key} {value}")
    return " · ".join(out) or "provider defaults"


def token_ceiling(params: dict) -> int | None:
    params = params or {}
    return params.get("max_tokens") or params.get("max_completion_tokens")
