from collections import Counter
from itertools import combinations
import json

from pramaan.checks.core import all_pass, verify_pattern
from pramaan.certificate import make_certificate, verify_certificate
from pramaan.kolam import loop_count, mirror_symmetry_orbits
from pramaan.oracles import find_oracle_candidate
from pramaan.orchestrate import _parse_task_candidate
from pramaan.tasks import load_suite


def test_simple_grammar_references_pass_the_real_verifier_for_every_eval_task():
    tasks, _digest = load_suite("eval-v2.1")
    simple = [task for task in tasks if task.grammar in {
        "counted_grid", "stripe_band", "periodic_raster", "dot_field",
    }]
    assert Counter(task.grammar for task in simple) == {
        "counted_grid": 31, "stripe_band": 14, "periodic_raster": 11, "dot_field": 4,
    }
    for task in simple:
        candidate = find_oracle_candidate(task)
        assert candidate is not None, task.id
        pattern = _parse_task_candidate(json.dumps(candidate), task)
        checks, masks, grid = verify_pattern(pattern)
        assert all_pass(checks), (task.id, [check for check in checks if check["status"] != "PASS"])
        certificate = make_certificate(pattern, checks, masks, grid)
        assert certificate["all_hard_constraints_pass"] is True
        assert verify_certificate(certificate, make_certificate(pattern, checks, masks, grid))[0]


def test_reference_rows_are_marked_nontraining_and_carry_certificate_evidence():
    from scripts.build_references import build_reference_rows

    tasks, digest = load_suite("eval-v2.1")
    selected = [task for task in tasks if task.grammar in {"counted_grid", "stripe_band", "periodic_raster", "dot_field"}]
    rows = build_reference_rows(selected, digest)
    assert len(rows) == 60
    assert all(row["source"] == "procedural_reference" for row in rows)
    assert all(row["data_role"] == "reference" for row in rows)
    assert all(row["training_eligible"] is False for row in rows)
    assert all(row["all_hard_constraints_pass"] is True for row in rows)
    assert all(row["suite_sha256"] == digest for row in rows)
    assert all(row["certificate"]["mask_sha256"] == row["mask_sha256"] for row in rows)


def test_full_eval_reference_set_covers_36_states_and_documents_only_known_unsat_task():
    from scripts.build_references import KNOWN_UNSATISFIABLE_TASK_ID, build_eval_references

    tasks, _digest = load_suite("eval-v2.1")
    rows = build_eval_references()
    expected_ids = {task.id for task in tasks} - {KNOWN_UNSATISFIABLE_TASK_ID}
    assert len(rows) == 71
    assert {row["task_id"] for row in rows} == expected_ids
    assert len({row["state"] for row in rows}) == 36
    assert all(row["source"] == "procedural_reference" for row in rows)
    assert all(row["data_role"] == "reference" for row in rows)
    assert all(row["training_eligible"] is False for row in rows)
    assert all(row["all_hard_constraints_pass"] is True for row in rows)


def test_reference_builder_writes_one_replay_verified_thumbnail_per_reference(tmp_path):
    from scripts.build_references import build_eval_references, write_reference_thumbnails

    rows = build_eval_references()
    output = tmp_path / "reference-thumbnails"
    paths = write_reference_thumbnails(rows, output)

    assert len(paths) == 71
    assert len({path.stem for path in paths}) == 71
    assert all(path.parent == output and path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
               for path in paths)
    first = {path.name: path.read_bytes() for path in paths}
    write_reference_thumbnails(rows, output)
    assert first == {path.name: path.read_bytes() for path in paths}


def test_puducherry_two_by_two_d2_brief_cannot_have_one_loop():
    orbits = mirror_symmetry_orbits(2, 2, "D2")
    assert set(orbits) == {
        ((0, 1), (2, 1)),
        ((1, 0), (1, 2)),
    }
    counts = set()
    for number in range(len(orbits) + 1):
        for selected in combinations(orbits, number):
            mirrors = frozenset(point for orbit in selected for point in orbit)
            actual = loop_count(2, 2, mirrors)
            counts.add(actual)
            assert actual != 1
    assert counts == {0, 2}


def test_gujarat_bandhani_reference_matches_twelve_white_dots_and_four_corners():
    from pramaan.tasks import load_suite

    tasks, _digest = load_suite("eval-v2.1")
    task = next(item for item in tasks if item.id == "gj_bandhani-easy-6")
    candidate = find_oracle_candidate(task)
    pattern = _parse_task_candidate(json.dumps(candidate), task)
    layer = next(item for item in pattern.layers if item.id == "reference-dots")
    centres = {tuple(motif.center) for motif in layer.motifs}
    expected_centre = {
        (28, 22), (35, 22),
        (22, 28), (28, 28), (35, 28), (41, 28),
        (22, 35), (28, 35), (35, 35), (41, 35),
        (28, 41), (35, 41),
    }
    expected_corners = {(6, 6), (57, 6), (6, 57), (57, 57)}
    assert centres == expected_centre | expected_corners
    assert layer.colour_index == task.palette.index("#F6EFDD")
    checks, _masks, _grid = verify_pattern(pattern)
    assert all_pass(checks)


def test_every_satisfiable_easy_reference_has_task_specific_notes_and_profile_gaps(tmp_path):
    from scripts.build_references import (
        EASY_REFERENCE_NOTES, build_eval_references, write_easy_reference_profile_gaps,
    )

    tasks, _digest = load_suite("eval-v2.1")
    rows = build_eval_references()
    easy_tasks = [task for task in tasks if task.level == "easy" and task.id != "py_kolam-as-practised-in-puducherry-easy-35"]
    easy_rows = [row for row in rows if row["level"] == "easy"]
    assert len(easy_tasks) == len(easy_rows) == 35
    assert {task.id for task in easy_tasks} == set(EASY_REFERENCE_NOTES)
    assert {row["task_id"] for row in easy_rows} == set(EASY_REFERENCE_NOTES)
    assert all(len(row.get("reference_notes", "")) >= 30 for row in easy_rows)

    gap_table = write_easy_reference_profile_gaps(tasks, tmp_path / "gaps.md")
    content = gap_table.read_text(encoding="utf-8")
    assert "brief/profile" in content.lower()
    assert all(task.id in content for task in easy_tasks)
    assert "rj_bagru-hand-block-print-easy-20" in content
    assert "Host fixes background to palette index 0 (indigo)" in content


def test_hard_references_have_distinct_masks_and_task_specific_notes():
    from scripts.build_references import HARD_REFERENCE_NOTES, build_eval_references

    hard_rows = [row for row in build_eval_references() if row["level"] == "hard"]
    assert len(hard_rows) == 36
    assert {row["task_id"] for row in hard_rows} == set(HARD_REFERENCE_NOTES)
    assert all(len(row.get("reference_notes", "")) >= 30 for row in hard_rows)
    assert len({row["mask_sha256"] for row in hard_rows}) == len(hard_rows)


def test_every_reference_thumbnail_contains_visible_pattern_ink():
    from pathlib import Path
    from PIL import Image

    root = Path(__file__).resolve().parents[1]
    thumbnails = sorted((root / "assets/reference-thumbnails").glob("*.png"))
    assert len(thumbnails) == 71
    for path in thumbnails:
        with Image.open(path) as image:
            counts = image.convert("RGB").getcolors(image.width * image.height)
        assert counts, f"{path.name} is not a readable raster"
        dominant_share = max(count for count, _colour in counts) / sum(count for count, _colour in counts)
        assert dominant_share <= 0.95, f"{path.name} is {dominant_share:.1%} one colour"


def test_warli_hard_references_use_the_briefs_ground_and_visible_ink():
    from pathlib import Path
    from pramaan.dsl import parse_pattern
    from pramaan.checks.core import verify_pattern, all_pass

    root = Path(__file__).resolve().parents[1]
    rows = [json.loads(line) for line in (root / "benchmarks/data/eval-v2.1-references.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    for task_id, ground in (
        ("mh_warli-painting-hard-13", "#9A4A2A"),
        ("dd_warli-painting-hard-30", "#8B5A3C"),
    ):
        row = next(item for item in rows if item["task_id"] == task_id)
        pattern = parse_pattern(json.dumps(row["candidate_pattern"]))
        assert pattern.palette[pattern.background_index] == ground
        checks, _masks, grid = verify_pattern(pattern)
        assert all_pass(checks)
        assert int((grid != pattern.background_index).sum()) / grid.size > 0.05
        assert len(row.get("reference_notes", "")) >= 30


def test_bagru_reference_uses_cream_inset_ground_under_block_print_motifs():
    tasks, _digest = load_suite("eval-v2.1")
    task = next(item for item in tasks if item.id == "rj_bagru-hand-block-print-easy-20")
    candidate = find_oracle_candidate(task)
    pattern = _parse_task_candidate(json.dumps(candidate), task)
    ground = next(layer for layer in pattern.layers if layer.id == "reference-cream-field")
    assert ground.colour_index == task.palette.index("#F1E3C3")
    checks, _masks, _grid = verify_pattern(pattern)
    assert all_pass(checks)


def test_telangana_reference_visibly_separates_stepped_diamond_from_square():
    from pramaan.render import render

    tasks, _digest = load_suite("eval-v2.1")
    task = next(item for item in tasks if item.id == "tg_pochampally-ikat-easy-23")
    pattern = _parse_task_candidate(json.dumps(find_oracle_candidate(task)), task)
    _masks, grid = render(pattern)
    red = task.palette.index("#B22222")
    cream = task.palette.index("#F3EFE6")
    assert pattern.background_index == red
    assert int(grid[32, 32]) == cream  # stepped diamond centre
    assert int(grid[10, 32]) == cream  # square frame
    assert int(grid[14, 32]) == red  # visible gap between them
    assert all_pass(verify_pattern(pattern)[0])


def test_easy_jaali_references_use_brief_color_and_five_by_five_holes_when_supported():
    tasks, _digest = load_suite("eval-v2.1")
    by_id = {task.id: task for task in tasks}

    delhi = by_id["dl_mughal-sandstone-and-marble-jaali-screens-of-the-humayun-s-tomb-complex-easy-31"]
    delhi_pattern = _parse_task_candidate(json.dumps(find_oracle_candidate(delhi)), delhi)
    assert delhi_pattern.palette[delhi_pattern.background_index] == "#A3492B"
    assert delhi_pattern.palette[delhi_pattern.jaali.open_colour_index] == "#F2EEE6"
    assert all_pass(verify_pattern(delhi_pattern)[0])

    up = by_id["up_mughal-pierced-stone-jaali-screens-of-agra-and-fatehpur-sikri-easy-25"]
    up_pattern = _parse_task_candidate(json.dumps(find_oracle_candidate(up)), up)
    assert len(up_pattern.jaali.holes) == 25
    assert all_pass(verify_pattern(up_pattern)[0])


def test_tracked_reference_thumbnails_match_the_current_reference_jsonl(tmp_path):
    from pathlib import Path
    from scripts.build_references import write_reference_thumbnails

    root = Path(__file__).resolve().parents[1]
    rows = [json.loads(line) for line in (root / "benchmarks/data/eval-v2.1-references.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    generated = write_reference_thumbnails(rows, tmp_path / "reference-thumbnails")
    tracked_dir = root / "assets/reference-thumbnails"
    expected_names = {path.name for path in generated}
    assert {path.name for path in tracked_dir.glob("*.png")} == expected_names
    assert all((tracked_dir / path.name).read_bytes() == path.read_bytes() for path in generated)
