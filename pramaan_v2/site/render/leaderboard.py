"""The leaderboard page `/leaderboard` (``Leaderboard.dc.html``, spec 8.4 and 9).

Every count, name and sentence that states a fact is generated from ``leaderboard.json``,
``wall.json``, ``kolam.json`` and ``meta.json``. The query string (``q``, ``prov``, ``sort``,
``compare``) only filters, orders and selects; it never changes a number.
"""
from __future__ import annotations

from pramaan_v2.constants import number_word

from . import charts
from .charts import FAINT, HAIRLINE, MONO, esc
from ..shell import footer, json_for_script, page

from ..labels import short_rule  # one label mapping for check ids (spec 3.1)


def chart_label(check_id: str) -> str:
    """Noun label for a failing check in a chart ("Symmetry", "Dot spacing")."""
    words = short_rule(check_id)
    return words[:1].upper() + words[1:]

IMPOSSIBLE_KOLAM = "py_kolam-as-practised-in-puducherry-easy-35"
DEFAULT_COMPARE = ("gpt-6-astra", "claude-opus-5-5", "beam-hf")
SORTS = (("within3", "Within three"), ("first", "First try"), ("evaluated", "Evaluated briefs"), ("cost", "Cost"))
PROVENANCE = (("all", "All runs"), ("canonical", "Canonical"), ("community", "Community"))
CABOVE = 40  # the cost call-out names the cheapest model above this many briefs
DATASET = "https://huggingface.co/datasets/"
CODE = "https://github.com/Mandark-droid/Proof-of-Art"

H2 = "margin: 0 0 8px; font-size: 34px; line-height: 1.08; font-weight: 700; font-stretch: 114%"
LEDE = "margin: 0 0 16px; max-width: 900px; font-size: 15px; color: #B8ADA1; text-wrap: pretty"
NOTE = "margin: 14px 0 0; max-width: 900px; font-size: 14px; color: #948980; text-wrap: pretty"
FIELD = ("min-height: 44px; padding: 0 12px; border: 1px solid #4A3F38; border-radius: 6px; background: #1D1815; "
         "color: #F4EFE6; font-family: inherit; font-size: 15px")
NAV_LINK = ("display: inline-flex; align-items: center; min-height: 44px; padding: 0 10px; color: #F4EFE6; "
            "text-decoration: none")


# ---------------------------------------------------------------- helpers

def join_names(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def short_names(models: list[dict]) -> dict[str, str]:
    """Short model names for dense lists ("Opus", "Fable 5.1"): family word dropped, version kept only when needed."""
    def parts(label: str) -> list[str]:
        words = label.split()
        if len(words) > 1 and (words[0] == "Claude" or words[0].startswith("GPT")):
            words = words[1:]
        return words

    bases = [parts(m["label"])[0] for m in models]
    out = {}
    for m, base in zip(models, bases):
        words = parts(m["label"])
        out[m["key"]] = base if bases.count(base) == 1 or len(words) == 1 else f"{base} {words[1]}"
    return out


def counts_list(rows: list[dict], field: str, short: dict[str, str]) -> str:
    """'Opus 52, Sonnet 42, ...' in descending order, ties in leaderboard order."""
    picked = sorted((m for m in rows if m.get(field)), key=lambda m: -m[field])
    # "Fable 5.1 17" would misread, so a colon separates name and count when a short name ends in a digit.
    sep = ": " if any(short[m["key"]][-1:].isdigit() for m in picked) else " "
    return ", ".join(f"{short[m['key']]}{sep}{m[field]}" for m in picked)


def matches(m: dict, q: str, prov: str) -> bool:
    if prov == "community" and m.get("provenance", "canonical") != "community":
        return False
    if prov == "canonical" and m.get("provenance", "canonical") != "canonical":
        return False
    q = q.strip().lower()
    return not q or q in m["label"].lower() or q in (m.get("provider") or "").lower()


def sort_rows(rows: list[dict], sort: str, order: dict[str, int]) -> list[dict]:
    """Order rows; ties and the default keep leaderboard order (ranked models first, then unranked ones).

    Models without a published spend sort after every priced model under ``cost``.
    """
    def pos(m: dict) -> int:
        return order.get(m["key"], len(order))

    def spend(m: dict) -> tuple[bool, float]:
        usd = m.get("spend_usd")
        return (not isinstance(usd, (int, float)), usd if isinstance(usd, (int, float)) else 0.0)

    key = {
        "first": lambda m: (-m["first"]["passed"], pos(m)),
        "evaluated": lambda m: (-m["n"], pos(m)),
        "cost": lambda m: (*spend(m), pos(m)),
    }.get(sort, pos)
    return sorted(rows, key=key)


def parse_compare(raw: str | None, keys: list[str]) -> list[str]:
    picked: list[str] = []
    for key in (raw or "").split(","):
        key = key.strip()
        if key in keys and key not in picked:
            picked.append(key)
    picked = picked[-3:]
    return picked or [k for k in DEFAULT_COMPARE if k in keys]


def overlap_sentence(rows: list[dict], briefs: int) -> str:
    """Which models the benchmark cannot separate from the leader (intervals overlap the leader's)."""
    leader_lo = rows[0]["wilson95"][0]
    tied = [m for m in rows if m["wilson95"][1] >= leader_lo]
    names = join_names([m["label"] for m in tied])
    contiguous = tied == rows[:len(tied)]
    lead = (f"The top {number_word(len(tied))} intervals overlap." if contiguous
            else f"{number_word(len(tied) - 1).capitalize()} intervals overlap the leader's.")
    if len(tied) == 1:
        return f"{rows[0]['label']}'s interval overlaps no other model's. With {briefs} briefs and one run each, ranks below it are less certain."
    return f"{lead} With {briefs} briefs and one run each, this benchmark does not separate {names}."


# ---------------------------------------------------------------- sections

def intro(data, n_models: int) -> str:
    meta = data.meta
    eyebrow = f"LEADERBOARD · SUITE {meta['suite_id'].upper()} · VALIDATOR {meta['validator']}"
    return (
        '<section style="display: flex; flex-wrap: wrap; align-items: flex-end; gap: 20px 64px; padding: 36px 0 28px">'
        '<div style="flex: 1 1 520px; min-width: 0">'
        f'<p style="margin: 0 0 12px; font-family: {MONO}; font-size: 12px; letter-spacing: 0.12em; color: {FAINT}">{esc(eyebrow)}</p>'
        '<h1 style="margin: 0; font-size: 58px; line-height: 0.98; font-weight: 700; font-stretch: 118%; text-wrap: balance">'
        f'{number_word(n_models).capitalize()} models. One checker. Three tries.</h1></div>'
        '<p style="flex: 1 1 420px; min-width: 0; margin: 0; font-size: 18px; color: #B8ADA1; text-wrap: pretty">'
        f'Canonical results on the same {meta["totals"]["briefs"]} frozen briefs, with every chart tied to rows you can '
        'replay. Community runs sit beside them, never mixed in.</p></section>'
    )


def controls(q: str, prov: str, sort: str, shown: int, total: int, compare: list[str]) -> str:
    prov_opts = "".join(f'<option value="{v}"{" selected" if v == prov else ""}>{t}</option>' for v, t in PROVENANCE)
    sort_opts = "".join(f'<option value="{v}"{" selected" if v == sort else ""}>{t}</option>' for v, t in SORTS)
    keep = f'<input type="hidden" name="compare" value="{esc(",".join(compare))}">' if compare else ""
    return (
        '<form id="lb-controls" data-lb-controls method="get" action="/leaderboard#ranking" '
        'style="display: flex; flex-wrap: wrap; align-items: flex-end; gap: 12px 16px">'
        '<div style="display: flex; flex-direction: column; gap: 4px; flex: 1 1 220px; min-width: 0">'
        f'<label for="lb-find" style="font-size: 13px; color: {FAINT}">Find a model</label>'
        f'<input id="lb-find" name="q" type="search" value="{esc(q)}" placeholder="Name or provider" autocomplete="off" '
        f'style="{FIELD}; box-sizing: border-box"></div>'
        '<div style="display: flex; flex-direction: column; gap: 4px">'
        f'<label for="lb-prov" style="font-size: 13px; color: {FAINT}">Provenance</label>'
        f'<select id="lb-prov" name="prov" style="{FIELD}">{prov_opts}</select></div>'
        '<div style="display: flex; flex-direction: column; gap: 4px">'
        f'<label for="lb-sort" style="font-size: 13px; color: {FAINT}">Sort by</label>'
        f'<select id="lb-sort" name="sort" style="{FIELD}">{sort_opts}</select></div>'
        f'{keep}<button type="submit" data-lb-apply style="min-height: 44px; padding: 0 16px; border-radius: 22px; '
        'border: 1px solid #F4B63F; background: transparent; color: #F4B63F; font-family: inherit; font-size: 15px; '
        'font-weight: 600; cursor: pointer">Apply</button>'
        '<a href="/leaderboard" data-lb-reset role="button" style="display: inline-flex; align-items: center; '
        'box-sizing: border-box; min-height: 44px; padding: 0 16px; border-radius: 22px; border: 1px solid #4A3F38; '
        'background: transparent; color: #F4EFE6; font-size: 15px; font-weight: 600; text-decoration: none">Reset</a>'
        f'<span role="status" data-lb-count data-total="{total}" style="min-height: 44px; display: inline-flex; '
        f'align-items: center; font-size: 14px; color: {FAINT}">{shown} of {total} model rows shown</span></form>'
    )


def section_nav() -> str:
    items = (("#ranking", "Ranking"), ("#results", "Full results"), ("#matrix", "Results matrix"),
             ("#failing", "Failing checks"), ("#cost", "Cost"), ("#grammar", "By grammar"),
             ("#profiles", "Model profiles"), ("/replays", "Replays"), ("#method", "Methodology"),
             ("#community", "Community runs"), ("#gaps", "Known gaps"))
    links = "".join(f'<a href="{h}" style="{NAV_LINK}">{t}</a>' for h, t in items)
    return ('<nav aria-label="Leaderboard sections" style="display: flex; flex-wrap: wrap; gap: 4px 6px; margin-top: 16px; '
            f'padding: 8px 0; border-top: 1px solid {HAIRLINE}; border-bottom: 1px solid {HAIRLINE}; font-size: 15px">'
            f'{links}</nav>')


def ranking_section(rows, ordered, hidden, ranks, briefs: int) -> str:
    body = "".join(charts.ranking_row(m, ranks.get(m["key"]), hidden=m["key"] in hidden) for m in ordered)
    of = (f"{briefs} briefs" if all(m["n"] == briefs for m in rows) else
          "the briefs each model ran. Unranked models (–) did not run all " + str(briefs))
    return (
        '<section id="ranking" style="padding: 56px 0 0">'
        '<div style="display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 8px 32px; '
        'margin-bottom: 20px">'
        f'<h2 style="{H2}; margin: 0">Briefs passed within three tries</h2>'
        f'<span style="font-size: 14px; color: #B8ADA1">Bar: share of {of}. Line beneath: 95% Wilson interval.</span></div>'
        f'<div class="lb-ranking" data-lb-rows="ranking" style="display: flex; flex-direction: column; '
        f'border-bottom: 1px solid {HAIRLINE}">{body}</div>'
        f'{charts.ranking_table(rows, ranks)}'
        f'<p class="lb-caveat" style="margin: 14px 0 0; max-width: 900px; font-size: 15px; color: #B8ADA1; '
        f'text-wrap: pretty">{esc(overlap_sentence(rows, briefs))}</p>'
        '<!-- /ranking --></section>'
    )


def results_section(ordered, hidden, briefs: int) -> str:
    legend = "".join(
        f'<span style="display: inline-flex; align-items: center; gap: 8px"><span style="width: 22px; height: 12px; '
        f'background: {c}"></span>{t}</span>'
        for c, t in (("#F4B63F", "Passed every check"), ("#C8452B", "Readable, broke a rule"),
                     ("#5E554E", "Unreadable or cut off")))
    return (
        '<section id="results" style="padding: 88px 0 0">'
        f'<h2 style="{H2}">Full results</h2>'
        f'<p style="{LEDE}">First tries are split three ways, because a proposal the checker could not read and a pattern '
        'that broke a rule are different failures.</p>'
        f'<div style="display: flex; flex-wrap: wrap; gap: 8px 24px; margin-bottom: 12px; font-size: 14px; color: #B8ADA1">'
        f'{legend}</div>{charts.results_table(ordered, hidden, briefs)}'
        f'<p style="{NOTE}">Repaired: briefs that failed the first try and passed on try two or three. Tokens and latency '
        'are means per brief and per request. Spend is a rate-card estimate for the whole run; provider invoices were '
        'unavailable.</p><!-- /results --></section>'
    )


def matrix_section(data, rows) -> str:
    states = {s["code"]: s["name"] for s in data.meta["states"]}
    models = [{"key": m["key"], "label": m["label"]} for m in rows]
    return (
        '<section id="matrix" style="padding: 88px 0 0">'
        f'<h2 style="{H2}">Results matrix</h2>'
        f'<p style="{LEDE}">Every model on every brief, grouped by craft grammar. The number is the try that passed. '
        "Select a cell to open that run's replay.</p>"
        f'{charts.matrix(models, data.leaderboard["matrix"]["groups"], data.wall["runs"], states)}'
        f'{charts.matrix_legend(data.meta["totals"]["low_ink"])}<!-- /matrix --></section>'
    )


def failing_block(data) -> str:
    failing = data.leaderboard["failing_first_try"]
    first_tries = data.meta["totals"]["runs"]
    top = max(failing, key=lambda r: r["count"])
    if top["check"] == "candidate_parse":
        largest = f"An unreadable proposal is the largest single failure: {top['count']} of {first_tries} first tries."
    else:
        largest = f"The largest single failure is {short_rule(top['check'])}: {top['count']} of {first_tries} first tries."
    bars = charts.failing_bars(failing, first_tries, chart_label, lambda k: f"{number_word(k).capitalize()} other checks")
    return (
        '<div id="failing" style="flex: 1 1 460px; min-width: 0">'
        f'<h2 style="{H2}">Failing checks on the first try</h2>'
        f'<p style="margin: 0 0 18px; font-size: 15px; color: #B8ADA1; text-wrap: pretty">Counted over the {first_tries} '
        'first proposals. One proposal can fail several checks.</p>'
        f'{bars}<p style="margin: 16px 0 0; font-size: 14px; color: {FAINT}; text-wrap: pretty">Grey bars are format '
        f'failures, red bars are geometry. {esc(largest)}</p><!-- /failing --></div>'
    )


def cost_callout(rows: list[dict]) -> str:
    rows = charts.priced(rows)
    front = sorted(charts.frontier(rows), key=lambda m: -m["within3"])
    out = []
    if len(front) >= 2:
        top, second = front[0], front[1]
        out.append(f"{second['label']} passes {second['within3']} of {second['n']} for an estimated "
                   f"{charts.dollars(second['spend_usd'])}.")
        extra = top["within3"] - second["within3"]
        diff = round(top["spend_usd"], 2) - round(second["spend_usd"], 2)
        briefs = "The one extra brief" if extra == 1 else f"The {number_word(extra)} extra briefs"
        out.append(f"{briefs} {top['label']} passes {'costs' if extra == 1 else 'cost'} {charts.dollars(diff)} more.")
    above = [m for m in rows if m["within3"] > CABOVE]
    if above:
        cheap = min(above, key=lambda m: m["spend_usd"])
        out.append(f"{cheap['label']} is the cheapest model above {CABOVE} briefs, at {charts.dollars(cheap['spend_usd'])}.")
    return " ".join(out)


def cost_block(rows, briefs: int) -> str:
    return (
        '<div id="cost" style="flex: 1 1 460px; min-width: 0">'
        f'<h2 style="{H2}">Cost against briefs passed</h2>'
        '<p style="margin: 0 0 12px; font-size: 15px; color: #B8ADA1; text-wrap: pretty">Estimated spend for the whole run, '
        'on a log scale. The dashed line joins the models nothing cheaper beats.</p>'
        f'<div class="lb-scroll" style="overflow-x: auto">{charts.cost_scatter(rows, briefs)}</div>'
        f'<p style="margin: 10px 0 0; font-size: 14px; color: {FAINT}; text-wrap: pretty">{esc(cost_callout(rows))}</p>'
        '<!-- /cost --></div>'
    )


def kolam_grid_notes(data, rows) -> list[tuple[dict, list[str], int]]:
    """Per model, the kolam passes whose declared grid differs from the grid the brief asked for."""
    out = []
    for m in rows:
        notes = []
        for task, brief in data.kolam.items():
            tries = brief["models"].get(m["key"], [])
            passed = [t for t in tries if t.get("outcome") == "passed"]
            for t in passed:
                if (t.get("rows"), t.get("cols")) != (brief["rows"], brief["cols"]):
                    notes.append(f"{t['rows']} × {t['cols']} for a {brief['rows']} × {brief['cols']}")
        if notes:
            out.append((m, notes, m["by_grammar"].get("kolam_loop", 0)))
    return out


def kolam_note(data, rows) -> str:
    parts = []
    found = sorted(kolam_grid_notes(data, rows), key=lambda item: -len(item[1]))  # stable: leaderboard order on ties
    for k, (m, notes, passes) in enumerate(found):
        listed = join_names(notes).replace(" and ", ", and ") if len(notes) == 2 else join_names(notes)
        if len(notes) == passes == 2:
            lead = f"Both of {m['label']}'s kolam passes used"
        elif len(notes) == passes == 1:
            lead = f"{m['label']}'s only kolam pass used"
        elif len(notes) == passes:
            lead = f"All {number_word(passes)} of {m['label']}'s kolam passes used"
        else:
            lead = f"{number_word(len(notes)).capitalize()} of {m['label']}'s {number_word(passes)} kolam passes used"
        parts.append(f"{lead} a grid the brief did not ask for: {listed}." if k == 0 else
                     f"{lead.replace(' used', ' did the same')}: {listed}.")
    return " ".join(parts)


def grammar_section(data, rows) -> str:
    grammars = [(g["id"], g["briefs"]) for g in data.meta["grammars"]]
    jaali_n = data.leaderboard["grammar_briefs"]["jaali_lattice"]
    jaali = sum(m["by_grammar"]["jaali_lattice"] for m in rows)
    best = max(m["by_grammar"]["jaali_lattice"] for m in rows)
    leaders = [m["label"] for m in rows if m["by_grammar"]["jaali_lattice"] == best]
    best_line = (f"The best model manages {best} of {jaali_n}." if len(leaders) == 1 else
                 f"The best models, {join_names(leaders)}, manage {best} of {jaali_n}.")
    no_kolam = sum(1 for m in rows if m["by_grammar"]["kolam_loop"] == 0)
    big = "font-size: 44px; line-height: 1; font-weight: 700; font-stretch: 112%"
    return (
        '<section id="grammar" style="padding: 88px 0 0">'
        f'<h2 style="{H2}; margin: 0 0 16px">By craft grammar</h2>'
        '<div style="display: flex; flex-wrap: wrap; gap: 28px 56px; align-items: flex-start">'
        f'{charts.grammar_heatmap(rows, grammars)}'
        '<div style="flex: 1 1 320px; min-width: 0; display: flex; flex-direction: column; gap: 20px">'
        f'<div><div style="{big}">{jaali} of {len(rows) * jaali_n}</div>'
        '<p style="margin: 8px 0 0; color: #B8ADA1; text-wrap: pretty">Jaali lattice results that pass. The screen must '
        f'stay one connected piece with holes inside the frame. {esc(best_line)}</p></div>'
        f'<div><div style="{big}">{no_kolam} of {len(rows)}</div>'
        f'<p style="margin: 8px 0 0; color: #B8ADA1; text-wrap: pretty">models pass no kolam brief. {esc(kolam_note(data, rows))}'
        '</p></div></div></div><!-- /grammar --></section>'
    )


def chip(m: dict, slot: int | None, selection: list[str], keep: dict) -> str:
    keys = [k for k in selection if k != m["key"]] if slot is not None else (selection + [m["key"]])[-3:]
    pressed = slot is not None
    inner = (charts.swatch(slot) if pressed else "") + esc(m["label"])
    return (f'<button type="submit" class="lb-chip" name="compare" value="{esc(",".join(keys))}" '
            f'data-key="{esc(m["key"])}" aria-pressed="{"true" if pressed else "false"}">{inner}</button>')


def profiles_section(data, rows, selection: list[str], keep: dict) -> str:
    grammars = [(g["id"], g["briefs"]) for g in data.meta["grammars"]]
    by = {m["key"]: m for m in rows}
    selected = [by[k] for k in selection]
    key_line = " · ".join(f"{charts.heat_header(g)} {n}" for g, n in grammars).upper()
    cards = "".join(charts.radar_card(m, grammars) for m in rows)
    chips = "".join(chip(m, selection.index(m["key"]) if m["key"] in selection else None, selection, keep) for m in rows)
    hidden_inputs = "".join(f'<input type="hidden" name="{k}" value="{esc(v)}">' for k, v in keep.items() if v)
    sizes = sorted({n for _, n in grammars if n <= 5})
    step = (f"Spokes with {sizes[0]} to {sizes[-1]} briefs move in steps of {round(100 / sizes[-1])} to "
            f"{round(100 / sizes[0])} points, so read the shape and not the decimals." if len(sizes) > 1 else
            "Read the shape and not the decimals.")
    payload = {m["key"]: {"label": m["label"], "values": [round(v, 6) for v in charts.radar_values(m)],
                          "shown": charts.radar_display(m)} for m in rows}
    h3 = "margin: 0 0 6px; font-size: 24px; line-height: 1.15; font-weight: 700"
    return (
        '<section id="profiles" data-lb-profiles style="padding: 88px 0 0">'
        f'<h2 style="{H2}">Model profiles</h2>'
        f'<p style="{LEDE}; margin: 0 0 32px">Two radar views of the same frozen run. The first shows where each model passes, '
        'by craft grammar. The second sets quality against tokens, speed and spend for three models at a time. Outward is '
        'better on every axis.</p>'
        f'<h3 style="{h3}">Shape by grammar</h3>'
        f'<p style="{LEDE}; margin: 0 0 6px">Each spoke is the share of that grammar\'s briefs passed within three tries. '
        'Every chart uses the same spokes in the same order, so shapes compare directly.</p>'
        f'<p style="margin: 0 0 18px; font-family: {MONO}; font-size: 12px; color: {FAINT}">BRIEFS PER SPOKE · {esc(key_line)}</p>'
        '<div class="lb-radar-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(min(240px, 100%), 1fr)); '
        f'gap: 12px">{cards}</div>'
        f'<p style="{NOTE}">{step} The heatmap above has the counts.</p>'
        f'<h3 style="{h3}; margin: 56px 0 6px">Quality against economy</h3>'
        f'<p style="{LEDE}">Pick up to three models. The three quality axes are plain percentages. The three economy axes '
        'are log scaled between fixed ends, so one ring is not one dollar or one second.</p>'
        '<form method="get" action="/leaderboard#profiles" data-lb-compare>'
        f'{hidden_inputs}<div role="group" aria-label="Models to compare" style="display: flex; flex-wrap: wrap; gap: 8px; '
        f'margin-bottom: 20px">{chips}</div></form>'
        '<div class="lb-compare" style="display: flex; flex-wrap: wrap; gap: 32px 56px; align-items: flex-start">'
        f'<div style="flex: 1 1 520px; min-width: 0; max-width: 700px">{charts.compare_radar(selected)}</div>'
        '<div style="flex: 1 1 440px; min-width: 0; display: flex; flex-direction: column; gap: 28px">'
        f'<div class="lb-scroll" style="overflow-x: auto">{charts.values_table(selected)}</div>'
        f'<div class="lb-scroll" style="overflow-x: auto">{charts.spoke_table(data.meta["totals"]["briefs"])}</div>'
        f'<p style="margin: 0; font-size: 14px; color: {FAINT}; text-wrap: pretty">Readable at try 1 is held down for the '
        'Claude models by the untyped anchor field. <a href="#method">Methodology</a> has the split.</p></div></div>'
        f'{script_json("lb-radar-data", payload)}<!-- /profiles --></section>'
    )


def method_section(data, rows) -> str:
    meta = data.meta
    ceilings = {charts.token_ceiling(m.get("request_params")) for m in rows} - {None}
    ceiling = f"a {ceilings.pop():,}-token output ceiling, " if len(ceilings) == 1 else "the output ceilings listed below, "
    th = f"padding: 8px 16px; font-weight: 500"
    body = "".join(
        f'<tr class="lb-method-row" data-model="{esc(m["key"])}" style="border-top: 1px solid {HAIRLINE}">'
        f'<th scope="row" style="padding: 10px 16px 10px 0; font-weight: 600">{esc(m["label"])}</th>'
        f'<td style="padding: 10px 16px; color: #B8ADA1">{esc(m.get("provider") or "")}</td>'
        f'<td style="padding: 10px 16px; font-family: {MONO}; font-size: 13px">{esc(charts.request_settings(m.get("request_params")))}</td>'
        f'<td style="padding: 10px 0 10px 16px; font-family: {MONO}; text-align: right">'
        f'{charts.pct(charts.share(m["readable"], m["n"]))}</td></tr>'
        for m in rows)
    totals = meta["totals"]
    return (
        '<section id="method" style="padding: 88px 0 0">'
        f'<h2 style="{H2}">Methodology</h2>'
        f'<p style="{LEDE}; margin: 0 0 20px">Same {totals["briefs"]} brief IDs, same deterministic verifier, {ceiling}up to '
        'three proposals per brief, zero provider retries. Providers do not expose identical sampling controls, so the '
        'settings actually sent are listed per model.</p>'
        '<div class="lb-scroll lb-sticky" style="overflow-x: auto">'
        '<table style="width: 100%; min-width: 820px; border-collapse: collapse; font-size: 15px; text-align: left"><thead>'
        f'<tr style="font-family: {MONO}; font-size: 11px; letter-spacing: 0.08em; color: {FAINT}">'
        '<th scope="col" style="padding: 8px 16px 8px 0; font-weight: 500; width: 200px">MODEL</th>'
        f'<th scope="col" style="{th}; width: 280px">PROVIDER</th>'
        f'<th scope="col" style="{th}">REQUEST SETTINGS</th>'
        '<th scope="col" style="padding: 8px 0 8px 16px; font-weight: 500; width: 150px; text-align: right">READABLE AT TRY 1</th>'
        f'</tr></thead><tbody style="border-bottom: 1px solid {HAIRLINE}">{body}</tbody></table></div>'
        f'<p style="{NOTE}">Scores are deterministic and replay-audited: all {totals["proposals"]:,} saved proposals were '
        f're-checked before a chart on this page was built, and the {totals["rendered"]:,} that rendered matched their '
        'published checks. Provider errors are excluded from evaluated counts. Third-party model outputs are evaluation '
        f'data only and are not training-eligible. <a href="{CODE}">Benchmark protocol</a></p><!-- /method --></section>'
    )


def community_rows(rows: list[dict]) -> str:
    """One aggregate row per community model (``community.aggregate``), marked Community, never ranked.

    "k of n" uses that run's own brief count.
    """
    body = []
    for r in rows:
        label = r.get("label") or r.get("key") or "Unnamed model"
        within, n = r.get("within3"), r.get("n")
        first = r.get("first_try")
        body.append(
            f'<tr style="border-top: 1px solid {HAIRLINE}"><th scope="row" style="padding: 10px 16px 10px 0; text-align: left; '
            f'font-weight: 600">{esc(label)} <span style="margin-left: 8px; font-family: {MONO}; font-size: 11px; '
            f'letter-spacing: 0.08em; color: {FAINT}">COMMUNITY</span></th>'
            f'<td style="padding: 10px 16px; color: #B8ADA1">{esc(r.get("provider") or "")}</td>'
            f'<td style="padding: 10px 16px; font-family: {MONO}">{"" if within is None or n is None else esc(f"{within} of {n}")}</td>'
            f'<td style="padding: 10px 0 10px 16px; font-family: {MONO}">{"" if first is None else esc(first)}</td></tr>')
    return "".join(body)


def community_section(n_models: int) -> str:
    try:
        from ..community import community_snapshot

        snap = community_snapshot()
    except Exception:  # the page never fails because the community snapshot is unavailable
        snap = {"rows": []}
    rows = snap.get("rows") or []
    th = (f'style="padding: 0 16px 8px 0; text-align: left; font-family: {MONO}; font-size: 11px; font-weight: 500; '
          f'letter-spacing: 0.08em; color: {FAINT}"')
    table = (f'<div data-lb-community-table class="lb-scroll" style="overflow-x: auto; margin-top: 20px"'
             f'{"" if rows else " hidden"}><table style="width: 100%; border-collapse: collapse; font-size: 15px">'
             f'<thead><tr><th scope="col" {th}>MODEL</th><th scope="col" {th}>PROVIDER</th><th scope="col" {th}>WITHIN THREE</th>'
             f'<th scope="col" {th}>FIRST TRY PASSED</th></tr></thead><tbody data-lb-community-rows>{community_rows(rows)}'
             '</tbody></table></div>')
    status = (f"{len(rows)} community {'run has' if len(rows) == 1 else 'runs have'} been merged." if rows
              else "No community runs have been merged yet.")
    return (
        '<section id="community" style="padding: 88px 0 0">'
        '<div style="display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 8px 32px; '
        'margin-bottom: 16px">'
        f'<h2 style="{H2}; margin: 0">Community runs</h2>'
        '<span style="font-size: 14px; color: #B8ADA1">Latest consented run per model · replayed before merging · kept apart '
        f'from the canonical {number_word(n_models)}</span></div>'
        '<div style="display: flex; flex-wrap: wrap; align-items: center; gap: 20px 48px; padding: 28px; '
        'border: 1px dashed #6B625B; border-radius: 6px">'
        '<div style="flex: 1 1 420px; min-width: 0">'
        f'<p role="status" data-lb-community-status style="margin: 0 0 8px; font-size: 22px; font-weight: 700">{esc(status)}</p>'
        '<p style="margin: 0; font-size: 15px; color: #B8ADA1; text-wrap: pretty">The first accepted run appears here as its '
        'own row, marked Community, with the same charts and replays as the canonical models. Model and provider names on '
        'community rows are self-reported.</p></div>'
        '<form method="get" action="/leaderboard#community" style="display: flex; flex-wrap: wrap; gap: 12px">'
        '<a href="/submit" style="display: inline-flex; align-items: center; min-height: 48px; padding: 0 24px; '
        'border-radius: 24px; background: #F4B63F; color: #15110F; font-weight: 700; text-decoration: none">Submit a run</a>'
        '<button type="submit" data-lb-refresh data-src="/api/community.json" style="min-height: 48px; padding: 0 20px; '
        'border-radius: 24px; border: 1px solid #4A3F38; background: transparent; color: #F4EFE6; font-family: inherit; '
        'font-size: 16px; font-weight: 600; cursor: pointer">Refresh community runs</button></form></div>'
        f'{table}<!-- /community --></section>'
    )


def gaps_section(data, rows) -> str:
    totals = data.meta["totals"]
    short = short_names(rows)
    ceilings = {charts.token_ceiling(m.get("request_params")) for m in rows} - {None}
    limit = f"{ceilings.pop():,}-token limit" if len(ceilings) == 1 else "token limit"
    kolam = data.kolam.get(IMPOSSIBLE_KOLAM, {})
    state = next((s["name"] for s in data.meta["states"] if IMPOSSIBLE_KOLAM in (s["easy"], s["hard"])), "")
    accepted = []
    for m in rows:
        for t in kolam.get("models", {}).get(m["key"], []):
            if t.get("outcome") == "passed":
                accepted.append((t["n"], f"{m['label']} (try {t['n']})", f"{t['rows']} × {t['cols']}"))
    accepted = [(who, grid) for _, who, grid in sorted(accepted, key=lambda a: a[0])]
    grids = {g for _, g in accepted}
    if not accepted:
        who = "No model was accepted on it."
    elif len(accepted) == 1:
        who = f"{accepted[0][0]} was still accepted, by declaring a {accepted[0][1]} grid."
    elif len(grids) == 1:
        both = "both" if len(accepted) == 2 else "all"
        who = f"{join_names([a for a, _ in accepted])} were still accepted, {both} by declaring a {grids.pop()} grid."
    else:
        who = "Still accepted: " + ", ".join(f"{a} ({g})" for a, g in accepted) + "."
    loops = kolam.get("required_loops", 1)
    ask = (f"The {state} easy brief asks for {number_word(loops)} loop{'s' if loops != 1 else ''} on a "
           f"{kolam.get('rows')} × {kolam.get('cols')} grid, which allows only zero or two.")
    reference = data.replay(IMPOSSIBLE_KOLAM)
    ref_note = " and has no reference pattern" if reference is not None and reference.get("reference") is None else ""
    items = (
        ("First tries measure one field",
         "The prompt names layers[].anchor without a type. First proposals rejected for that field alone: "
         f"{counts_list(rows, 'anchor_only', short)}. Use the within-three score to compare models."),
        ("The token ceiling cut some off",
         f"First proposals that hit the {limit} and could not be certified: {counts_list(rows, 'truncated', short)}."),
        ("Blank tiles can pass",
         f"{totals['low_ink']} of {totals['passed']} accepted designs show under 2% visible ink: "
         f"{counts_list(rows, 'low_ink', short)}."),
        ("One brief cannot be met as written",
         f"{ask} {who} It stays in every model's denominator{ref_note}."),
    )
    cards = "".join(
        f'<div style="background: #15110F; padding: 24px"><h3 style="margin: 0 0 8px; font-size: 20px; font-weight: 700">{t}</h3>'
        f'<p style="margin: 0; font-size: 15px; color: #B8ADA1; text-wrap: pretty">{esc(p)}</p></div>' for t, p in items)
    return (
        '<section id="gaps" style="padding: 88px 0 0">'
        f'<p style="margin: 0 0 12px; font-family: {MONO}; font-size: 12px; letter-spacing: 0.12em; color: {FAINT}">'
        'KNOWN GAPS · PUBLISHED, NOT PATCHED</p>'
        f'<h2 style="{H2}; margin: 0 0 24px">Read this before quoting a number.</h2>'
        '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(min(300px, 100%), 1fr)); gap: 1px; '
        f'background: {HAIRLINE}; border: 1px solid {HAIRLINE}">{cards}</div><!-- /gaps --></section>'
    )


def script_json(element_id: str, payload) -> str:
    text = json_for_script(payload, separators=(",", ":"))
    return f'<script type="application/json" id="{element_id}">{text}</script>'


def facts_json(rows: list[dict], ranks: dict[str, int | None], order: dict[str, int]) -> str:
    """What the live filter and sort need, so JavaScript applies exactly the server's rules.

    ``order`` is the leaderboard position (the default sort and every tie-break); ``rank`` is the
    displayed rank, null for a model that did not run every brief. ``spend`` is null when unpublished.
    """
    return script_json("lb-facts", {
        m["key"]: {"label": m["label"], "provider": m.get("provider") or "", "prov": m.get("provenance", "canonical"),
                   "rank": ranks.get(m["key"]), "order": order.get(m["key"], len(order)),
                   "first": m["first"]["passed"], "n": m["n"],
                   "spend": m["spend_usd"] if isinstance(m.get("spend_usd"), (int, float)) else None} for m in rows})


# ---------------------------------------------------------------- page

def render_leaderboard(data, base_url: str, query: dict) -> str:
    query = query or {}
    rows = [m for m in data.leaderboard["models"]]
    ranks = {m["key"]: m["rank"] for m in data.models}
    order = {m["key"]: i for i, m in enumerate(data.models)}
    rows.sort(key=lambda m: order.get(m["key"], len(order)))
    q = str(query.get("q", ""))[:80]
    prov = query.get("prov", "all") if query.get("prov") in {v for v, _ in PROVENANCE} else "all"
    sort = query.get("sort", "within3") if query.get("sort") in {v for v, _ in SORTS} else "within3"
    selection = parse_compare(query.get("compare"), [m["key"] for m in rows])
    hidden = {m["key"] for m in rows if not matches(m, q, prov)}
    ordered = sort_rows(rows, sort, order)
    n = len(rows)
    briefs = data.meta["totals"]["briefs"]
    keep = {"q": q, "prov": "" if prov == "all" else prov, "sort": "" if sort == "within3" else sort}

    body = (
        f'<div data-lb data-briefs="{briefs}" data-compare-default="{",".join(DEFAULT_COMPARE)}">'
        + intro(data, n)
        + controls(q, prov, sort, n - len(hidden), n, selection if query.get("compare") else [])
        + section_nav()
        + ranking_section(rows, ordered, hidden, ranks, briefs)
        + results_section(ordered, hidden, briefs)
        + matrix_section(data, rows)
        + '<section style="padding: 88px 0 0; display: flex; flex-wrap: wrap; gap: 48px 56px; align-items: flex-start">'
        + failing_block(data) + cost_block(rows, briefs) + "</section>"
        + grammar_section(data, rows)
        + profiles_section(data, rows, selection, keep)
        + method_section(data, rows)
        + community_section(n)
        + gaps_section(data, rows)
        + facts_json(rows, ranks, order)
        + "</div>"
    )
    meta = data.meta
    foot = footer(
        [("Results dataset", DATASET + meta["dataset"]),
         ("Community dataset", DATASET + "kshitijthakkar/pramaan-community-runs"), ("Code", CODE)],
        "One observed run per model, not a confidence claim about future performance.", mono=False, margin_top=72)
    return page(
        title="Leaderboard · Pramaan",
        description=(f"{number_word(n).capitalize()} models on the same {meta['totals']['briefs']} frozen briefs, one "
                     "checker, three tries: ranking with intervals, first-try splits, cost, and every result linked to its replay."),
        path="/leaderboard", body=body, footer=foot, current="leaderboard", base_url=base_url)
