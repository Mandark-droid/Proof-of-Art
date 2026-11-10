"""Visitor sentences built from the site data, so they stay right when models are added.

Small pure functions: each takes plain values (or the data dicts) and returns text. No HTML here.
"""
from __future__ import annotations

from ..constants import number_word

VERDICTS = {
    "1": "PASSED ON THE FIRST TRY", "2": "PASSED ON TRY 2", "3": "PASSED ON TRY 3",
    "x": "BROKE A RULE ON EVERY TRY", "-": "NEVER READABLE",
}
VERDICT_TONES = {"1": "#F4B63F", "2": "#F4B63F", "3": "#F4B63F", "x": "#E8613C", "-": "#B8ADA1"}
TRANSFORMS = {
    "F_vertical": "a left-right flip", "F_horizontal": "a top-bottom flip", "R90": "a quarter turn",
    "R180": "a half turn", "R270": "a three-quarter turn", "F_diagonal": "a diagonal flip",
    "F_antidiagonal": "an anti-diagonal flip",
}
ORDINALS = {1: "first", 2: "second", 3: "third"}


def n(value: int) -> str:
    """Thousands separators, as the boards print counts."""
    return f"{value:,}"


def word(value: int) -> str:
    return number_word(value)


def Word(value: int) -> str:  # noqa: N802 - sentence-initial form of word()
    return number_word(value).capitalize()


def join_names(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def plural(count: int, one: str, many: str | None = None) -> str:
    return one if count == 1 else (many or one + "s")


# Hero and wall -----------------------------------------------------------------------------------

def hero_eyebrow(totals: dict, states: int) -> str:
    return f"{totals['models']} MODELS · {totals['briefs']} BRIEFS · {states} STATES AND UNION TERRITORIES"


def phone_lede(totals: dict) -> str:
    return (f"{Word(totals['models'])} models, {totals['briefs']} briefs from Indian craft traditions. "
            "Every tile is what a model actually drew. Bright ones passed every check.")


def phone_top(label: str, within3: int, briefs: int) -> str:
    return f"TOP: {label.upper()}, {within3} OF {briefs}"


def wall_label(totals: dict) -> str:
    return f"The wall of {n(totals['runs'])} runs"


def verdict(status: str) -> str:
    return VERDICTS[status]


def run_name(model: str, state: str, level: str, status: str) -> str:
    """Accessible name of a wall tile: "{Model}, {State}, {easy|hard}: {verdict}"."""
    return f"{model}, {state}, {level}: {VERDICTS[status].lower()}"


def legend(totals: dict) -> list[tuple[str, str]]:
    return [
        ("passed", f"{n(totals['passed'])} passed every check"),
        ("broke", f"{n(totals['broke'])} drew a pattern that broke a rule"),
        ("never", f"{n(totals['never'])} never produced a readable pattern"),
    ]


WALL_ROWS_NOTE = "Each model has two rows: its easy brief above its hard one."


def four_numbers(totals: dict) -> list[tuple[str, str, bool]]:
    """(value, caption, evidence colour) for the stat row under the wall."""
    return [
        (n(totals["runs"]), f"runs: {word(totals['models'])} models on the same {totals['briefs']} frozen briefs", False),
        (n(totals["proposals"]), "proposals, up to three tries per brief", False),
        (n(totals["pixel_evidence"]), "failed tries where the checker names the exact pixel", True),
        ("0", f"mismatches when all {n(totals['rendered'])} rendered proposals are re-verified", False),
    ]


# Kolam -------------------------------------------------------------------------------------------

def reward_text(reward: float) -> str:
    if reward >= 1:
        return "1.0"
    if reward <= 0:
        return "0"
    return f"{reward:.2f}"


def loops_phrase(loops: int) -> str:
    return f"{loops} {plural(loops, 'LOOP', 'LOOPS')}"


def kolam_badge(item: dict, *, last: bool) -> tuple[str, str]:
    """Top-right label of a try card and its colour."""
    if item["outcome"] == "unreadable":
        return "UNREADABLE", "#B8ADA1"
    if item["outcome"] == "passed":
        return f"{loops_phrase(item['loops'] or 1)} · PASSED", "#F4B63F"
    text = loops_phrase(item["loops"]) if item["loops"] is not None else "BROKE A RULE"
    return (text + " · FAILED" if last else text), "#E8613C"


def mirror_gaps(rows: int, cols: int) -> int:
    """Places a mirror can go between neighbouring dots."""
    return rows * (cols - 1) + cols * (rows - 1)


def kolam_caption(item: dict, previous: dict | None, *, anchor_error: bool) -> str:
    """One plain sentence per try, from its mirror and loop counts."""
    if item["outcome"] == "unreadable":
        if anchor_error:
            return "One field held a string where the schema expects [x, y]."
        return "The proposal could not be read."
    mirrors, loops = item["mirrors"] or 0, item["loops"]
    count = f"{Word(mirrors)} {plural(mirrors, 'mirror')}" if mirrors else "No mirrors"
    if item["outcome"] == "passed":
        if item["n"] == 1:
            return f"{count}, placed once. One line."
        return f"{count}. One line around every dot."
    if loops is None:
        return f"{count}. The line cannot be traced."
    if loops == 0:
        if item["rows"] and item["cols"] and mirrors == mirror_gaps(item["rows"], item["cols"]):
            return "A mirror in every gap. Nothing can run."
        return f"{count}. Nothing can run."
    drawn = previous if previous and previous.get("loops") else None
    if drawn and mirrors > (drawn["mirrors"] or 0) and loops > drawn["loops"]:
        extra = mirrors - (drawn["mirrors"] or 0)
        return f"{Word(extra)} more {plural(extra, 'mirror')} made it worse."
    if previous and previous.get("mirrors") is not None and mirrors < previous["mirrors"]:
        return f"Fewer mirrors. {Word(loops)} {plural(loops, 'loop')}."
    if loops == 1:
        return f"{count}. One line."
    return f"{count}. {Word(loops)} separate lines."


def reward_trail(rewards: list[float]) -> str:
    return "reward " + " → ".join(reward_text(r) for r in rewards)


def kolam_first_try_note(others: list[str], never: int, models: int) -> str:
    """The dashed note beside a first-try pass."""
    lead = f"{join_names(others)} also passed first time. " if others else ""
    return f"{lead}{Word(never)} of the {word(models)} models never did."


NO_MIRRORS_NOTE = ("With no mirrors at all, the number of loops is gcd(rows, columns). "
                   "The same tracer gives one loop on a 5 × 6 grid and six on a 6 × 6.")


# Spot the break ----------------------------------------------------------------------------------

def pair_text(pair: dict) -> str:
    (x, y), (tx, ty) = pair["original"], pair["transformed"]
    return f"({x}, {y}) ≠ ({tx}, {ty}) under {TRANSFORMS.get(pair.get('transform'), 'the required symmetry')}"


def spot_after(model: str, status: str) -> str:
    if status in ("2", "3"):
        missed = "missed again, and " if status == "3" else ""
        return f"{model} got that pair as feedback, {missed}passed on its {ORDINALS[int(status)]} try."
    if status == "1":
        return f"{model} passed on its first try."
    return f"{model} got that pair as feedback and never passed."


# Findings ----------------------------------------------------------------------------------------

def impossible_kolam(asked: tuple[int, int], accepted: list[tuple[str, int, int]], failed: int) -> str:
    """accepted: (model label, declared rows, declared cols) in leaderboard order."""
    rows, cols = asked
    head = (f"The Puducherry brief asks for one unbroken line on a {rows} × {cols} dot grid. "
            "That grid can only give zero or two loops. "
            f"{Word(failed)} {plural(failed, 'model')} failed. ")
    if not accepted:
        return head + "None was accepted."
    grids = {(r, c) for _, r, c in accepted}
    names = join_names([label for label, _, _ in accepted])
    if len(grids) == 1:
        r, c = next(iter(grids))
        both = "" if len(accepted) == 1 else (" both" if len(accepted) == 2 else " all")
        middle = f"{names}{both} passed by drawing a {r} × {c} grid instead. "
    else:
        middle = names + " passed by drawing " + join_names([f"a {r} × {c} grid" for _, r, c in accepted]) + " instead. "
    return head + middle + "The checker counted loops, not dots."


def blank_tile(blank: int, models: int, low_ink: int, passed: int) -> str:
    return ("The Manipur brief asks for black ink and the host fixes the ground to the same black. "
            f"{Word(blank)} of {word(models)} models were accepted with an empty tile. "
            f"Across the benchmark, {n(low_ink)} of {n(passed)} accepted designs show under 2% visible ink.")


# Leaderboard strip and open problems -------------------------------------------------------------

def strip_heading(briefs: int) -> str:
    return f"Same {briefs} briefs. Three tries each."


def strip_caption(briefs: int) -> str:
    return (f"Briefs passed within three tries, out of {briefs}. "
            "One observed run per model, not a confidence claim about future performance.")


def open_problems(*, jaali_passed: int, jaali_pairs: int, jaali_best: int, jaali_briefs: int,
                  kolam_none: int, models: int, kolam_briefs: int) -> list[tuple[str, str]]:
    return [
        ("Train the first policy.", "The reward function is tested and shipped. Nothing has been trained on it yet."),
        ("Jaali lattices.", f"{jaali_passed} of {jaali_pairs} model and brief pairs pass. "
                            f"The best model manages {jaali_best} of {jaali_briefs}."),
        ("Kolam loops.", f"{Word(kolam_none)} of {word(models)} models pass none of the "
                         f"{word(kolam_briefs)} briefs."),
        ("Close the reward gaps.", "Add a visible-ink check and a brief-fidelity check without breaking "
                                   "replay of the frozen runs."),
        ("Shape the partial credit.", "Two loops and five loops both score 0.75 today. Reward counts checks "
                                      "passed, not how near a miss was."),
    ]


def footer_right(meta: dict) -> str:
    return f"Validator {meta['validator']} · suite {meta['suite_id']}"
