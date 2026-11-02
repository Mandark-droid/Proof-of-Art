import json
import hashlib
import re
from pathlib import Path

import pytest

from scripts.build_suites import build_suites, validate_constraint_profile
from scripts.build_common47 import (
    COMMON47_SEED, build_common47_rows, serialize_common47, write_common47,
)
from pramaan.tasks import canonical_jsonl_bytes
from scripts.plot_benchmark_summary import render_svg
from pramaan.traditions import ALL_STATES, load_registry

ROOT = Path(__file__).parents[1]
EVAL_PATH = ROOT / "benchmarks" / "suites" / "eval-v2.1.jsonl"
TRAIN_PATH = ROOT / "benchmarks" / "suites" / "train-v1.2.jsonl"


def read_rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_eval_covers_all_states_both_levels():
    rows = read_rows(EVAL_PATH)
    assert len(rows) == 72
    assert {row["state"] for row in rows} == set(ALL_STATES)
    for state in ALL_STATES:
        assert {row["level"] for row in rows if row["state"] == state} == {"easy", "hard"}


def test_no_brief_overlap_between_eval_and_train():
    eval_briefs = {row["brief"] for row in read_rows(EVAL_PATH)}
    train_briefs = {row["brief"] for row in read_rows(TRAIN_PATH)}
    assert len(eval_briefs) == 72
    assert len(train_briefs) == 144
    assert eval_briefs.isdisjoint(train_briefs)


def test_suite_briefs_fit_orchestrator_limit():
    rows = read_rows(EVAL_PATH) + read_rows(TRAIN_PATH)
    assert all(len(row["brief"]) <= 500 for row in rows)


def test_rows_validate_against_constraints_v2():
    for row in read_rows(EVAL_PATH) + read_rows(TRAIN_PATH):
        validate_constraint_profile(row["grammar"], row["constraint_profile"])


def test_suite_palettes_fit_pattern_v2_schema():
    rows = read_rows(EVAL_PATH) + read_rows(TRAIN_PATH)
    assert all(1 <= len(row["palette"]) <= 6 for row in rows)


def test_brief_symmetry_matches_profile():
    rows = read_rows(EVAL_PATH) + read_rows(TRAIN_PATH)
    forbidden = re.compile(r"(?<![A-Za-z0-9])C(?:3|5|6|7|8|9|10|11|12)(?![A-Za-z0-9])")
    n_fold = re.compile(r"\b(\d+)-fold\b", re.IGNORECASE)
    for row in rows:
        brief = row["brief"]
        assert not forbidden.search(brief), row["id"]
        assert not any(
            phrase in brief.lower()
            for phrase in ("8-fold rotational", "10-fold rotational", "12-fold rotational")
        ), row["id"]
        assert "N-fold" not in brief, row["id"]
        for order in n_fold.findall(brief):
            assert int(order) in {2, 4}, row["id"]
            assert row["symmetry"] in {f"C{order}", f"D{order}"}, row["id"]


def test_one_sided_feature_briefs_state_mirroring():
    rows = {row["id"]: row for row in read_rows(EVAL_PATH)}
    train_rows = {row["id"]: row for row in read_rows(TRAIN_PATH)}
    corrected_ids = {
        "ap_mangalagiri-sarees-and-fabrics-easy-0",
        "ap_mangalagiri-sarees-and-fabrics-hard-0",
        "as_gamosa-red-on-white-border-and-motif-band-hard-2",
        "ga_kunbi-check-sari-easy-5",
        "ga_kunbi-check-sari-hard-5",
        "kl_balaramapuram-kasavu-handloom-hard-11",
        "mn_moirang-phee-moirang-pheejin-stepped-triangle-border-easy-14",
        "ml_garo-dakmanda-easy-15",
        "ml_garo-dakmanda-hard-15",
    }
    for task_id in corrected_ids:
        assert "mirrored onto the opposite edge" in rows[task_id]["brief"], task_id
    assert "exact D2 symmetry (mirror symmetric about both midlines)" in rows[
        "ml_garo-dakmanda-hard-15"
    ]["brief"]
    garo_train = train_rows["ml_garo-dakmanda-train-1"]
    assert garo_train["constraint_profile"]["symmetry"] == "D2"
    assert "mirrored onto the opposite edge to satisfy host D2 symmetry" in garo_train["brief"]


def test_every_jaali_suite_task_has_bounded_open_fraction():
    rows = read_rows(EVAL_PATH) + read_rows(TRAIN_PATH)
    jaali_rows = [row for row in rows if row["grammar"] == "jaali_lattice"]
    assert len(jaali_rows) == 17
    for row in jaali_rows:
        profile = row["constraint_profile"]
        assert 0 < profile["min_open_fraction"] < profile["max_open_fraction"] < 1
    hard_interval = next(row for row in jaali_rows if "40-50% open area" in row["brief"])
    assert hard_interval["constraint_profile"]["min_open_fraction"] == 0.4
    assert hard_interval["constraint_profile"]["max_open_fraction"] == 0.5


def test_builder_adds_documented_jaali_default_bounds(tmp_path):
    build_suites(tmp_path)
    rows = read_rows(tmp_path / "train-v1.2.jsonl")
    jaali_rows = [row for row in rows if row["grammar"] == "jaali_lattice"]
    assert jaali_rows
    assert all(row["constraint_profile"]["min_open_fraction"] == 0.25 for row in jaali_rows)
    assert all(row["constraint_profile"]["max_open_fraction"] == 0.75 for row in jaali_rows)


def test_builder_garo_train_variant_mirrors_bottom_border_for_d2(tmp_path):
    build_suites(tmp_path, train_suite_id="train-v1.2")
    rows = read_rows(tmp_path / "train-v1.2.jsonl")
    garo = next(row for row in rows if row["id"] == "ml_garo-dakmanda-train-1")
    assert garo["suite_id"] == "train-v1.2"
    assert garo["constraint_profile"]["symmetry"] == "D2"
    assert "mirrored onto the opposite edge to satisfy host D2 symmetry" in garo["brief"]


def test_benchmark_chart_uses_summary_prompt_count():
    svg = render_svg({"by_model": {"beam-hf": {
        "overall": {"n": 72, "validity_at_1": 0.2, "validity_at_3": 0.4},
        "by_level": {},
    }}})
    assert "72 prompts" in svg


def test_builder_refuses_overwrite(tmp_path):
    build_suites(tmp_path)
    with pytest.raises(FileExistsError):
        build_suites(tmp_path)


def test_builder_digest_matches_bytes_consumed_by_runner(tmp_path):
    digests = build_suites(tmp_path)
    for name, digest in digests.items():
        actual = hashlib.sha256(canonical_jsonl_bytes((tmp_path / f"{name}.jsonl").read_bytes())).hexdigest()
        assert digest == actual


def test_common47_is_source_bound_balanced_deterministic_and_byte_frozen():
    source_bytes = EVAL_PATH.read_bytes()
    source_rows = read_rows(EVAL_PATH)
    rows = build_common47_rows(source_rows)
    content = serialize_common47(rows)

    assert COMMON47_SEED == "pramaan-eval-v2.1-common47-v1"
    assert hashlib.sha256(canonical_jsonl_bytes(source_bytes)).hexdigest() == "b04b568d84d2568fd261789819d4e925844dba5afe9c5962275ee571d84e4e3a"
    assert len(rows) == 47
    assert len({row["state"] for row in rows}) == 36
    assert sum(row["level"] == "easy" for row in rows) == 25
    assert sum(row["level"] == "hard" for row in rows) == 22
    assert sum(sum(row["state"] == state for row in rows) == 2 for state in ALL_STATES) == 11
    assert hashlib.sha256(canonical_jsonl_bytes(content)).hexdigest() == "dd3c9ce96a22deb1af94a1db22dc2a5016d1b1ad1c9c4a12befcd6690a6f7603"
    assert content == serialize_common47(build_common47_rows(source_rows))


def test_common47_builder_refuses_to_overwrite(tmp_path):
    target = tmp_path / "eval-v2.1-common47.jsonl"
    write_common47(EVAL_PATH, target)
    with pytest.raises(FileExistsError):
        write_common47(EVAL_PATH, target)


def test_builder_caps_palettes_using_only_registry_colours(tmp_path):
    registry = load_registry()
    build_suites(tmp_path)
    for path in (tmp_path / "eval-v2.1.jsonl", tmp_path / "train-v1.2.jsonl"):
        for row in read_rows(path):
            sourced = registry[row["tradition_id"]].palette_presets["traditional"]
            assert 1 <= len(row["palette"]) <= 6
            assert set(row["palette"]).issubset(sourced)

def test_jaali_frame_is_validated_outside_constraints_v2():
    validate_constraint_profile("jaali_lattice", {"symmetry": "D2", "frame": 4, "min_web": 2})
    with pytest.raises(ValueError, match="frame"):
        validate_constraint_profile("jaali_lattice", {"symmetry": "D2", "frame": 13, "min_web": 2})
    with pytest.raises(ValueError, match="frame"):
        validate_constraint_profile("dot_field", {"symmetry": "D2", "frame": 4})


def test_procedural_oracles_cover_all_current_hard_grammar_tasks():
    from pramaan.oracles import build_oracle_rows
    from pramaan.tasks import load_suite

    suites = {name: load_suite(name) for name in ("eval-v2.1", "train-v1.2")}
    expected = {
        task.id
        for tasks, _digest in suites.values()
        for task in tasks
        if task.grammar in {"rotational_field", "kolam_loop", "jaali_lattice"}
    }

    rows = build_oracle_rows(suites=suites)

    assert {row["task_id"] for row in rows} == expected
    assert all(row["source"] == "procedural_oracle" for row in rows)
    unsatisfiable = {row["task_id"] for row in rows if row["oracle_status"] == "unsatisfiable"}
    assert unsatisfiable == {"py_kolam-as-practised-in-puducherry-easy-35"}
    assert all(row["all_hard_constraints_pass"] is True for row in rows if row["oracle_status"] == "verified")
    assert all(row["suite_sha256"] == suites[row["suite_id"]][1] for row in rows)




def test_no_sacred_terms_in_benchmark_briefs():
    rows = read_rows(EVAL_PATH) + read_rows(TRAIN_PATH)
    banned = re.compile(
        r"\b(?:swastik|swastika|deity|idol|ganesha)\b|\bOm\s+symbol\b|\bLakshmi footprints\b",
        re.IGNORECASE,
    )
    for row in rows:
        assert not banned.search(row["brief"]), row["id"]


def test_oracle_cli_accepts_train_v1_2(tmp_path):
    from pramaan.oracles import main

    output = tmp_path / "train-oracles.jsonl"
    assert main(["--suite", "train-v1.2", "--output", str(output)]) == 0
    assert output.exists()
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert {row["suite_id"] for row in rows} == {"train-v1.2"}
