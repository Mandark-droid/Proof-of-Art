"""Build verifier-certified procedural references for the eval-v2.1 tasks."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pramaan.certificate import make_certificate, verify_certificate
from pramaan.checks.core import all_pass, verify_pattern
from pramaan.dsl import parse_pattern
from pramaan.export import render_png
from pramaan.oracles import find_oracle_candidate
from pramaan.orchestrate import _parse_task_candidate, apply_host_profile
from pramaan.render import mask_digest
from pramaan.tasks import Task, load_suite


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "benchmarks/data/eval-v2.1-references.jsonl"
DEFAULT_THUMBNAILS = ROOT / "assets/reference-thumbnails"
DEFAULT_CONTACT_SHEET = ROOT / "artifacts/easy-reference-contact-sheet.png"
DEFAULT_GAP_TABLE = ROOT / "benchmarks/data/eval-v2.1-reference-profile-gaps.md"
KNOWN_UNSATISFIABLE_TASK_ID = "py_kolam-as-practised-in-puducherry-easy-35"
EXISTING_ORACLE_PATH = ROOT / "benchmarks/data/eval-v2.1-procedural-oracles.jsonl"

EASY_REFERENCE_NOTES = {
    "ap_mangalagiri-sarees-and-fabrics-easy-0": "A 7-by-7 window check with paired gold edge bands. The brief's bottom-only border is mirrored to the top by D2.",
    "ar_apatani-textiles-easy-1": "A centered white band over the host's indigo background carries an x-period-8 mirrored stripe sequence; yellow lines flank the red band. The coverage cap leaves indigo above and below.",
    "as_gamosa-red-on-white-border-and-motif-band-easy-2": "A red cell-grid border and centered filled diamond. D2 requires the fourth side to mirror the requested three-side border.",
    "br_madhubani-easy-3": "A double yellow outline, centered diamond and black diagonal crosshatch, mirrored across both axes.",
    "cg_bastar-wooden-craft-easy-4": "A wood-ground center diamond with a zigzag border reflected across left-right and top-bottom by D2.",
    "ga_kunbi-check-sari-easy-5": "A centered 7-by-7 red-and-white window check with black bands on both opposite edges to satisfy D2.",
    "gj_bandhani-easy-6": "Twelve white center dots form a diamond, with one white dot in each corner.",
    "hr_phulkari-and-bagh-counted-thread-embroidery-easy-7": "A yellow lozenge is built from aligned cell runs, with a small white center mark.",
    "hp_kinnauri-shawl-geometric-border-easy-8": "A centered stepped white hexagonal motif with a yellow core on the brick-red field.",
    "jh_sohrai-khovar-wall-painting-easy-9": "Four outer and one central triple-color disk motifs. C4 symmetry permits four outer circles, not the six requested; filled nested disks approximate triple-line circles.",
    "ka_kasuti-embroidery-easy-10": "A red, cell-aligned chain of paired stepped diamonds evokes double-running stitch and is mirrored at the center.",
    "kl_pookalam-easy-11": "Nested white, orange and yellow diamond bands surround a green center. Diamond bands approximate rings; the requested red band is omitted because the DSL caps a pattern at four layers including the center.",
    "mp_bagh-hand-block-print-easy-12": "A black checker lattice carries a small red diamond in each cell on an off-white ground.",
    "mh_warli-painting-easy-13": "A dotted square outline and four corner triangles appear on an ochre inset panel over the host's white background; the coverage cap requires a narrow white margin.",
    "mn_moirang-phee-moirang-pheejin-stepped-triangle-border-easy-14": "A white 5-step cell staircase on the host's black background is mirrored along both edges; palette order prevents the requested black marks on white ground.",
    "ml_garo-dakmanda-easy-15": "Four nested red-and-white diamonds sit between paired red edge strips, using the task's 4-pixel cell grid.",
    "mz_mizo-puanchei-easy-16": "A centered white horizontal band over the host's red background carries a central red pair, green separators and black end bands; the coverage cap leaves red above and below.",
    "nl_chakhesang-shawls-easy-17": "A centered red horizontal band over the host's black background carries four black bands and a centered row of white squares repeating at the declared x-period.",
    "od_sambalpuri-bandha-easy-18": "A centered 7-by-7 red-and-black window check with a white cell border on all four sides.",
    "pb_phulkari-and-bagh-counted-thread-embroidery-easy-19": "Four yellow corner lozenges frame a white center lozenge, all on the 4-pixel stitch grid.",
    "rj_bagru-hand-block-print-easy-20": "Four red diamond petals with black outlines and indigo corner dots sit on a cream inset panel over the host's indigo background; the coverage cap leaves an indigo margin.",
    "sk_lepcha-thara-weave-stripes-easy-21": "An 8-stripe mirrored sequence uses four inks. The profile's color cap and D2 symmetry exclude the requested fifth black stripe and exact order.",
    "tn_kolam-easy-22": "One dark closed loop surrounds the 3-by-3 dot grid on the host's white background. Palette order does not allow the requested white loop on dark ground; the frozen profile also enforces D2, not the brief's D4.",
    "tg_pochampally-ikat-easy-23": "The host fixes a red ground; a larger cream square frame encloses a clearly stepped cream diamond, with a red gap between them. The prompt's D4 request is stronger than the frozen D2 profile.",
    "tr_risa-easy-24": "A mirrored red-and-white stripe sequence with thin white outer lines repeats every 8 pixels on the black field.",
    "up_mughal-pierced-stone-jaali-screens-of-agra-and-fatehpur-sikri-easy-25": "A 5-by-5 square-hole grid in sandstone passes the openness and web checks. The brief asks for a 6-pixel frame, while the frozen profile fixes frame 4; the certified witness follows the profile.",
    "uk_aipan-easy-26": "A C4 four-point rosette with a center mark stands in for the requested five-point star; the exact five-point motif conflicts with C4 symmetry.",
    "wb_nakshi-kantha-easy-27": "A red eight-point center star and four smaller corner stars are cell-aligned and mirrored on both axes.",
    "an_nicobari-plaited-pandanus-mat-easy-28": "A centered 7-by-7 alternating two-tone plaited check leaves a narrow symmetric margin.",
    "ch_phulkari-easy-29": "A centered yellow lozenge and four white corner marks use the 4-pixel counted-stitch grid.",
    "dd_warli-painting-easy-30": "Eight white dots outline a square, with eight small white triangles on a centered brown panel over the host's white background; the coverage cap leaves a white margin.",
    "dl_mughal-sandstone-and-marble-jaali-screens-of-the-humayun-s-tomb-complex-easy-31": "A red-sandstone web with cream square openings uses the frozen 4-pixel frame profile; the brief asks for a 6-pixel frame.",
    "jk_khatamband-geometric-wooden-ceiling-easy-32": "A 4-by-4 square-hole jaali meets D2, openness and web checks. Eight-point star cut-outs are not expressible as supported jaali holes in this verifier.",
    "la_thigma-resist-dyed-circles-easy-33": "Sixteen white dots form a symmetric 4-by-4 lattice. Ring outlines and maroon center dots are omitted because the dot-clearance verifier treats those coincident marks as collisions.",
    "ld_lakshadweep-coir-and-screw-pine-mat-weave-easy-34": "A centered 7-by-7 cream-and-coir check leaves a narrow margin for exact D2 symmetry.",
}

EASY_REFERENCE_GAPS = {
    "as_gamosa-red-on-white-border-and-motif-band-easy-2": "Three-side border conflicts with D2; witness has four sides.",
    "cg_bastar-wooden-craft-easy-4": "D2 adds top-bottom reflection to the brief's left-right symmetry.",
    "jh_sohrai-khovar-wall-painting-easy-9": "Six outer circles conflict with C4 orbit size; witness has four plus center.",
    "kl_pookalam-easy-11": "Ring geometry is approximated by nested diamonds.",
    "mn_moirang-phee-moirang-pheejin-stepped-triangle-border-easy-14": "Bottom-only row conflicts with D2; witness mirrors it on top.",
    "sk_lepcha-thara-weave-stripes-easy-21": "Requested stripe order uses five colors and is not D2-palindromic; witness uses four-color palindrome.",
    "tn_kolam-easy-22": "Brief asks D4; frozen profile enforces D2.",
    "tg_pochampally-ikat-easy-23": "The red ground is fixed by the host; a larger cream square encloses the stepped cream diamond with a visible red gap. Brief asks D4; frozen profile enforces D2.",
    "up_mughal-pierced-stone-jaali-screens-of-agra-and-fatehpur-sikri-easy-25": "Brief asks frame 6; frozen profile sets frame 4. A 5x5 hole grid passes the frozen openness and minimum-web checks.",
    "ar_apatani-textiles-easy-1": "Host fixes background to palette index 0 (indigo) and the coverage cap is 70%; a centered white horizontal band preserves indigo margins rather than making a full white tile.",
    "rj_bagru-hand-block-print-easy-20": "Host fixes background to palette index 0 (indigo) and coverage is capped at 70%; an inset cream panel is used instead of a full cream tile.",
    "mh_warli-painting-easy-13": "Host fixes background to palette index 0 (white) and coverage is capped at 70%; an inset ochre ground is used.",
    "mn_moirang-phee-moirang-pheejin-stepped-triangle-border-easy-14": "Host fixes background to palette index 0 (black); the witness uses white marks on black rather than black marks on white.",
    "mz_mizo-puanchei-easy-16": "Host fixes background to palette index 0 (red); a centered white horizontal band is used to show the stated white ground.",
    "nl_chakhesang-shawls-easy-17": "Host fixes background to palette index 0 (black); a centered red horizontal band is used to show the requested red textile ground.",
    "tn_kolam-easy-22": "Host fixes background to palette index 0 (white); the witness uses a dark curve on white rather than the requested white curve on dark ground.",
    "dd_warli-painting-easy-30": "Host fixes background to palette index 0 (white) and coverage is capped at 70%; a brown inset ground is used.",
    "uk_aipan-easy-26": "Five-point star conflicts with C4; witness uses a four-point rosette.",
    "dl_mughal-sandstone-and-marble-jaali-screens-of-the-humayun-s-tomb-complex-easy-31": "Red sandstone is used for the web; brief asks frame 6 while the frozen profile sets frame 4.",
    "jk_khatamband-geometric-wooden-ceiling-easy-32": "Eight-point star openings are outside the supported jaali-hole grammar; witness uses square openings.",
    "la_thigma-resist-dyed-circles-easy-33": "Ring centers collide under the verifier's min-clearance rule; witness uses white dots without maroon center marks.",
}

HARD_REFERENCE_NOTES = {
    "ap_mangalagiri-sarees-and-fabrics-hard-0": "White-and-maroon body checks sit between gold selvedges with red check marks. The checker proves the 4-pixel grid and D2 raster; it does not measure the requested inverted-check and dot weaving technique.",
    "ar_idu-mishmi-textiles-hard-1": "Three repeated nested-diamond motifs use red, blue and cream with mirrored top-edge lozenges. D2 and the cell grid are checked; the exact triangular weave and four-ink order are not.",
    "as_gamosa-red-on-white-border-and-motif-band-hard-2": "Red edge stripes frame a stepped central diamond band with alternating blue marks. D2 mirrors the requested lower band onto the upper edge; the checker does not certify counted-thread stitch order.",
    "br_madhubani-hard-3": "Rice-paper ground carries a double border, saw-tooth edge, disk-like fish-eye motifs and crosshatching. The raster check does not prove 3-pixel hatch pitch or enclosed-region fill semantics.",
    "cg_bastar-wooden-craft-hard-4": "Burnt-line chevrons frame a central lozenge and sun. The declared D2 symmetry is exact; the verifier does not measure the requested 20–35% line coverage as a cultural or material property.",
    "ga_kunbi-check-sari-hard-5": "Two-cell red-and-white checks are overlaid with black warp pinstripes and mirrored dobby bands. The 8-pixel host cell grid is a coarser procedural reading of the requested 4-pixel boundaries.",
    "gj_patan-patola-double-ikat-grid-hard-6": "Alternating yellow and green stepped lozenges sit on red with a black-and-white check frame. The frozen profile enforces D2 rather than the brief's stronger D4 symmetry.",
    "hr_phulkari-and-bagh-counted-thread-embroidery-hard-7": "Nested gold, orange and cream lozenges are framed by small white border stitches. The host profile is D2 and the verifier does not encode the requested three-band embroidery technique or red one-cell outlines.",
    "hp_kullu-shawl-end-bands-hard-8": "A mirrored stack of seven black-ground colour bands provides the shawl-end structure. The stripe profile repeats every 8 pixels across x, so the stated 1:2:1:4:1:2:1 widths and zigzag details are approximate.",
    "jh_sohrai-khovar-wall-painting-hard-9": "Fourfold groups of nested cream, red and charcoal disks form inner and outer rings around the centre; gold dots mark the edges. Filled disks approximate concentric circle bands, and exact 6-pixel perimeter spacing is not checked.",
    "ka_ilkal-sarees-hard-10": "Mustard and indigo pinstripes carry red, cream and green selvedge colours. The frozen x-period and D2 symmetry repeat the selvedge sequence around the tile; the requested right-only 16-pixel border cannot be represented under those rules.",
    "kl_balaramapuram-kasavu-handloom-hard-11": "Cream field and mirrored gold-cream-red border bands include green edge lines. D2 and x-period are checked; the 1-pixel edge detail is rasterized with the available motif widths.",
    "mp_maheshwari-sari-border-bands-hard-12": "Peacock-blue ground and a mirrored gold-crimson reversible band use a small check accent. The host's D2 profile places the band symmetrically rather than only in the brief's bottom 20 rows.",
    "mh_warli-painting-hard-13": "The visible rice-white dotted double square and triangle frieze sit on the brief's red-ochre ground. The replay host fixes model-output background to palette index 0; this reference restores the brief's red-ochre ground while retaining the checked D2 mask. The verifier does not validate D4, the 3-pixel dot pitch, or the stated coverage percentage.",
    "mn_moirang-phee-moirang-pheejin-stepped-triangle-border-hard-14": "A cream field with mirrored black stepped border marks and thin inner lines follows D2. The cell-grid checker does not verify seven-step triangle counts or textile technique.",
    "ml_garo-dakmanda-hard-15": "Dark ground, paired red border strips, white zigzags and nested red-gold-cream diamonds create a D2 border sample. Period-16 placement and crosses on the brief's 5-cell grid are approximated by the 4-pixel host cell grid.",
    "mz_mizo-puanchei-hard-16": "A mirrored sequence of coloured lengthwise bands is crossed by two black transverse bands. The host's four-colour cap omits the green and blue accents; red, yellow, black and white remain in the witness.",
    "nl_chakhesang-shawls-hard-17": "An orange-white-black stripe field carries a centred white check row and selvedge dots. The host enforces an 8-pixel x-period; the brief's 24-pixel vertical repeat and exact band widths are not expressible together here.",
    "od_sambalpuri-bandha-hard-18": "A centred check field and alternating red-and-white frame evoke the saktapar layout. The frozen host profile checks D2 rather than D4, and dots are approximated by counted cells.",
    "pb_phulkari-and-bagh-counted-thread-embroidery-hard-19": "Four repeated patch motifs show nested lozenges, chevrons, stitch dashes and dots inside yellow cell borders. The frozen 4-pixel cell profile is coarser than the brief's 2-pixel stitch cell.",
    "rj_leheriya-diagonal-stripe-tie-dye-hard-20": "Yellow, green, magenta and blue stripes form a mirror-closed eight-pixel repeat. The four-colour ceiling omits white, while the frozen x-period and D2 constraints prevent the requested five-colour 24-pixel diagonal cycle.",
    "sk_lepcha-thara-weave-stripes-hard-21": "White, red, yellow, green and black warp colours repeat with short stepped chevron accents. The 8-pixel host period cannot encode the requested ten-stripe group, so the motif is a compact periodic reading.",
    "tn_toda-embroidery-hard-22": "Three red-and-black bands sit on a white field with mirrored stepped marks. The verifier checks the 4-pixel grid and D2 raster, not the full 16-cell motif period or Toda stitch form.",
    "tg_pochampally-ikat-hard-23": "Repeated red diamonds with black stepped square frames sit on white. The host profile enforces D2 rather than the brief's D4 symmetry, and the full period-16 feathered edge is simplified.",
    "tr_risa-hard-24": "A black-ground mirrored stripe sequence is interrupted by a central white check row. The host's 8-pixel x-period differs from the brief's 9-pixel repeat; the raster checker does not certify the exact sequence widths.",
    "up_lucknow-chikan-jaali-hard-25": "A blue diamond-like paisley outline and white crossed net marks approximate the chikan panel with corner butis. The DSL has no paisley or net primitive; the frozen 4-pixel grid also differs from the brief's 3-pixel cell and uses two inks.",
    "uk_rangwali-pichhoda-of-kumaon-hard-26": "A staggered maroon lattice surrounds a dot-free centre with twelve smaller maroon dots in a ring. The host checks D2 rather than the brief's C4 set symmetry; dots approximate the requested radius-1 ring marks.",
    "wb_nakshi-kantha-hard-27": "A central stepped star-like lozenge and paired border stitches use a cream ground. The host enforces D2 rather than D4; teardrop kalka motifs and empty-cell running stitches are outside the counted-grid verifier.",
    "an_nicobari-plaited-pandanus-mat-hard-28": "A mirrored two-tone twill check is framed by a plain border. The host enforces D2 and cell alignment; it does not verify the requested C2 terminology or exact 45–55% light/dark balance.",
    "ch_phulkari-hard-29": "Nested yellow, white and orange lozenges sit on red with a green accent edge. The D2 checker accepts the cell raster; half-lozenges at tile edges and the brief's finer stitch grid are approximated.",
    "dd_warli-painting-hard-30": "Visible white dotted-square marks and a ring of white triangles sit on the brief's mud-brown ground, with a yellow centre. The replay host fixes model-output background to palette index 0; this reference restores the brief's mud-brown ground while retaining the checked D2 mask. The verifier does not validate D4 or the edge dash-dot period.",
    "dl_mughal-sandstone-and-marble-jaali-screens-of-the-humayun-s-tomb-complex-hard-31": "A connected sandstone web with four large square openings is attached to the frame and passes the 40–50% openness rule. Hexagonal holes and the central star are not supported by the jaali-hole grammar; the verifier proves the declared web and openness checks only.",
    "jk_khatamband-geometric-wooden-ceiling-hard-32": "A connected brown web with a 4-by-4 grid of square openings meets the frozen openness and web rules. The brief's eight-point stars and D4 symmetry are outside the supported square-hole/D2 profile.",
    "la_thigma-resist-dyed-circles-hard-33": "Two offset rows of twelve pale dots include four gold cross motifs rendered as surrounding dots. Filled dots stand in for rings; the host checks minimum centre spacing and D2, not stitch or ring outlines.",
    "ld_lakshadweep-coir-and-screw-pine-mat-weave-hard-34": "A coir-toned check is crossed by red-green stripe pairs and a centred blue twill diamond. The 4-pixel D2 cell raster is verified; the checker does not assess physical weave or colour balance.",
    "py_kolam-as-practised-in-puducherry-hard-35": "The verifier-certified witness has one closed loop under the frozen D2 profile. The brief asks for two strokes and D4; the profile instead requires one loop, so the reference documents the host constraints rather than claiming a match to that wording.",
}


def build_reference_rows(tasks: list[Task], suite_sha256: str) -> list[dict]:
    """Generate one verified reference per satisfiable task; reject all unknown gaps."""
    rows = []
    unresolved = []
    for task in tasks:
        candidate = find_oracle_candidate(task)
        if candidate is None:
            if task.id == KNOWN_UNSATISFIABLE_TASK_ID:
                continue
            unresolved.append(task.id)
            continue
        if task.id in {"mh_warli-painting-hard-13", "dd_warli-painting-hard-30"}:
            # These two briefs specify a coloured ground, while the frozen replay
            # host forces palette index 0 as the model-output background. Preserve
            # the task-owned geometry/palette, then restore the brief's ground on
            # this procedural reference only.
            candidate = apply_host_profile(candidate, task)
            candidate["background_index"] = task.palette.index(
                "#9A4A2A" if task.id.startswith("mh_warli") else "#8B5A3C"
            )
            pattern = parse_pattern(json.dumps(candidate))
        else:
            pattern = _parse_task_candidate(json.dumps(candidate), task)
        checks, masks, grid = verify_pattern(pattern)
        certificate = make_certificate(pattern, checks, masks, grid)
        if not all_pass(checks) or certificate["all_hard_constraints_pass"] is not True:
            failed = [item["id"] for item in checks if item.get("status") != "PASS"]
            unresolved.append(f"{task.id} (checks: {', '.join(failed)})")
            continue
        fresh = make_certificate(pattern, checks, masks, grid)
        certificate_ok, reason = verify_certificate(certificate, fresh)
        if not certificate_ok:
            unresolved.append(f"{task.id} (certificate: {reason})")
            continue
        rows.append({
            "schema_version": "1.0",
            "suite_id": task.suite_id,
            "suite_sha256": suite_sha256,
            "task_id": task.id,
            "state": task.state,
            "tradition_id": task.tradition_id,
            "grammar": task.grammar,
            "level": task.level,
            "brief": task.brief,
            "source": "procedural_reference",
            "data_role": "reference",
            "training_eligible": False,
            "candidate_pattern": pattern.model_dump(mode="json"),
            "checks": checks,
            "mask_sha256": mask_digest(masks, grid),
            "certificate": certificate,
            "all_hard_constraints_pass": True,
            **({"reference_notes": EASY_REFERENCE_NOTES[task.id]}
               if task.level == "easy" and task.id in EASY_REFERENCE_NOTES else {}),
            **({"reference_notes": HARD_REFERENCE_NOTES[task.id]}
               if task.level == "hard" and task.id in HARD_REFERENCE_NOTES else {}),
        })
    if unresolved:
        raise ValueError("No verified reference for: " + "; ".join(unresolved))
    return rows


def build_eval_references() -> list[dict]:
    tasks, digest = load_suite("eval-v2.1")
    task_ids = {task.id for task in tasks}
    if KNOWN_UNSATISFIABLE_TASK_ID not in task_ids:
        raise ValueError(f"Expected known unsatisfiable task is missing from eval-v2.1: {KNOWN_UNSATISFIABLE_TASK_ID}")
    oracle_rows = [json.loads(line) for line in EXISTING_ORACLE_PATH.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    unsat_rows = [row for row in oracle_rows if row.get("task_id") == KNOWN_UNSATISFIABLE_TASK_ID
                  and row.get("suite_id") == "eval-v2.1" and row.get("oracle_status") == "unsatisfiable"]
    if len(unsat_rows) != 1:
        raise ValueError(f"Expected exactly one existing unsatisfiable oracle row for {KNOWN_UNSATISFIABLE_TASK_ID}")
    rows = build_reference_rows(tasks, digest)
    states = {task.state for task in tasks}
    referenced_ids = {row["task_id"] for row in rows}
    missing = task_ids - referenced_ids
    if len(rows) != 71 or len(states) != 36 or missing != {KNOWN_UNSATISFIABLE_TASK_ID}:
        raise ValueError(f"Expected 71 verified references across 36 states and only the documented unsatisfiable exception; "
                         f"found {len(rows)} rows, {len(states)} states, missing={sorted(missing)}")
    return rows


def write_reference_thumbnails(rows: list[dict], output_dir: Path = DEFAULT_THUMBNAILS) -> list[Path]:
    """Pre-render every replay-verified reference for fast Home and state-sheet rendering."""
    selected: dict[str, dict] = {}
    for row in rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str):
            continue
        if task_id in selected:
            raise ValueError(f"Duplicate reference task for thumbnail: {task_id}")
        selected[task_id] = row
    if len(selected) != 71:
        raise ValueError(f"Expected 71 verified task references for thumbnails, found {len(selected)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for row in selected.values():
        task_id = row["task_id"]
        if not re.fullmatch(r"[a-z0-9_-]{1,120}", task_id):
            raise ValueError(f"Unsafe reference task ID for thumbnail: {task_id!r}")
        pattern = parse_pattern(json.dumps(row["candidate_pattern"], ensure_ascii=False))
        checks, masks, grid = verify_pattern(pattern)
        certificate = make_certificate(pattern, checks, masks, grid)
        if (not all_pass(checks) or certificate.get("mask_sha256") != row.get("mask_sha256")
                or certificate.get("all_hard_constraints_pass") is not True):
            raise ValueError(f"Reference failed thumbnail replay: {task_id}")
        path = output_dir / f"{task_id}.png"
        image = render_png(grid, pattern.palette, scale=3, repeats=1)
        image.save(path, format="PNG", optimize=True)
        written.append(path)
    return written


def write_easy_reference_profile_gaps(tasks: list[Task], output: Path = DEFAULT_GAP_TABLE) -> Path:
    """Write the brief-versus-frozen-profile disclosures for the dataset card."""
    easy = [task for task in tasks if task.level == "easy" and task.id in EASY_REFERENCE_NOTES]
    if len(easy) != 35:
        raise ValueError(f"Expected 35 easy reference notes, found {len(easy)}")
    lines = ["# Eval v2.1 easy references: brief/profile gap table", "",
             "The procedural references are verifier-certified constraint witnesses. This table records where the frozen brief, profile, or supported visual grammar limits exact motif fidelity.",
             "", "| Task ID | Brief/profile or representation gap |", "|---|---|"]
    for task in easy:
        gap = EASY_REFERENCE_GAPS.get(task.id, "No explicit brief/profile conflict identified; the procedural reference is an illustrative geometric witness, not an authentic craft artifact.")
        lines.append(f"| `{task.id}` | {gap} |")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def write_easy_reference_contact_sheet(rows: list[dict], output: Path = DEFAULT_CONTACT_SHEET) -> Path:
    """Pair every satisfiable easy brief with its replay-verified rendered image."""
    from pramaan.checks.core import verify_pattern

    selected = [row for row in rows if row.get("level") == "easy"]
    if len(selected) != 35:
        raise ValueError(f"Expected 35 easy reference rows, found {len(selected)}")
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 12)
        small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 10)
    except OSError:
        font = small = ImageFont.load_default()
    columns, card_width, image_size = 4, 360, 132
    wrapped = {row["task_id"]: textwrap.wrap(row["brief"], width=48) for row in selected}
    max_lines = max(len(lines) for lines in wrapped.values())
    card_height = 32 + max_lines * 15 + image_size + 24
    rows_n = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * card_width, rows_n * card_height), "#e8e5df")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected):
        left, top = (index % columns) * card_width, (index // columns) * card_height
        draw.rounded_rectangle((left+6, top+6, left+card_width-6, top+card_height-6),
                               radius=8, fill="#fffdf8", outline="#c8c1b5", width=1)
        draw.text((left+14, top+13), row["task_id"], fill="#263243", font=small)
        brief_y = top + 31
        for line in wrapped[row["task_id"]]:
            draw.text((left+14, brief_y), line, fill="#34312c", font=font)
            brief_y += 15
        pattern = parse_pattern(json.dumps(row["candidate_pattern"], ensure_ascii=False))
        checks, _masks, grid = verify_pattern(pattern)
        if not all_pass(checks):
            raise ValueError(f"Contact-sheet reference no longer verifies: {row['task_id']}")
        thumb = render_png(grid, pattern.palette, scale=2, repeats=1).convert("RGB")
        thumb.thumbnail((image_size, image_size), Image.Resampling.NEAREST)
        x = left + (card_width - thumb.width) // 2
        y = brief_y + 8
        sheet.paste(thumb, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG", optimize=True)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thumbnails-dir", type=Path, default=DEFAULT_THUMBNAILS)
    parser.add_argument("--contact-sheet", type=Path, default=DEFAULT_CONTACT_SHEET)
    parser.add_argument("--gap-table", type=Path, default=DEFAULT_GAP_TABLE)
    args = parser.parse_args(argv)
    rows = build_eval_references()
    tasks, _digest = load_suite("eval-v2.1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in rows), encoding="utf-8",
    )
    thumbnails = write_reference_thumbnails(rows, args.thumbnails_dir)
    sheet = write_easy_reference_contact_sheet(rows, args.contact_sheet)
    gaps = write_easy_reference_profile_gaps(tasks, args.gap_table)
    counts = {}
    for row in rows:
        counts[row["grammar"]] = counts.get(row["grammar"], 0) + 1
    print(f"wrote {len(rows)} verified references for {len({row['state'] for row in rows})} states to {args.output}")
    print("per-grammar references: " + ", ".join(f"{key}={counts[key]}" for key in sorted(counts)))
    print(f"unsatisfiable exception: {KNOWN_UNSATISFIABLE_TASK_ID}; no reference emitted; see {EXISTING_ORACLE_PATH.relative_to(ROOT)}")
    print(f"wrote {len(thumbnails)} cached task-reference thumbnails to {args.thumbnails_dir}")
    print(f"wrote easy-reference contact sheet to {sheet}")
    print(f"wrote brief/profile gaps table to {gaps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
