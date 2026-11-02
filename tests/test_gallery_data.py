import base64
import io
import json
from pathlib import Path

from PIL import Image
import pytest

from scripts.build_gallery_data import build_gallery_data
from scripts.build_gallery_data import _validate_gallery_summary
from pramaan.tasks import load_suite

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def gallery(tmp_path_factory):
    output = tmp_path_factory.mktemp("gallery")
    index_path = build_gallery_data(output_dir=output)
    return output, json.loads(index_path.read_text(encoding="utf-8"))


def test_index_json_matches_contract(gallery):
    output, index = gallery
    assert index["suite_id"] == "eval-v2.1"
    assert len(index["suite_sha256"]) == 64
    assert index["validator_version"]
    assert len(index["states"]) == 36
    assert len({tuple(item["grid"]) for item in index["states"]}) == 36
    state = next(item for item in index["states"] if item["state"] == "Jharkhand")
    assert state["diwali_tie"]["claim"]
    assert state["diwali_tie"]["source_url"].startswith("https://")
    arm = index["arms"][0]
    assert {"key", "label", "provider", "run_kind", "request_params", "n_tasks",
            "pass_at_1", "pass_at_3", "ci_pass_at_1", "ci_pass_at_3", "rescue_rate",
            "mean_reward", "valid_schema_at_1", "tokens_per_task", "est_cost_per_verified",
            "median_latency_ms", "length_rate", "near_bound_rate", "normalizations", "normalization_calls",
            "normalization_rates", "by_grammar"} <= set(arm)
    assert arm["n_tasks"] == 72
    assert len(list(output.glob("*.json"))) == 2


def test_gallery_digest_is_stable_for_lf_suite_copy(tmp_path):
    source = ROOT / "benchmarks" / "suites" / "eval-v2.1.jsonl"
    lf_suite = tmp_path / "eval-v2.1.jsonl"
    lf_suite.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))

    index_path = build_gallery_data(
        output_dir=tmp_path / "gallery", suite_path=lf_suite,
        generated_at="2026-10-31T00:00:00+00:00",
    )
    _, expected_digest = load_suite("eval-v2.1")

    assert json.loads(index_path.read_text(encoding="utf-8"))["suite_sha256"] == expected_digest


def test_baseline_rows_present(gallery):
    _, index = gallery
    baselines = {item["key"]: item for item in index["baselines"]}
    assert {"empty-tile", "single-centred-dot"} <= set(baselines)
    assert all(item["n_tasks"] == 72 for item in baselines.values())
    assert all(0.0 <= item["pass_at_1"] <= 1.0 for item in baselines.values())


def test_per_arm_file_size_under_400kb(gallery):
    output, index = gallery
    for arm in index["arms"]:
        assert (output / f"{arm['key']}.json").stat().st_size < 400_000


def test_arm_payload_includes_replay_verified_attempt_images(gallery):
    output, index = gallery
    arm = index["arms"][0]
    payload = json.loads((output / f"{arm['key']}.json").read_text(encoding="utf-8"))
    assert payload["key"] == arm["key"]
    assert len(payload["tasks"]) == 72
    accepted = next(task for task in payload["tasks"].values() if task["passed_at"] is not None)
    assert accepted["cert_sha256"] and accepted["mask_sha256"]
    attempt = next(item for item in accepted["attempts"] if item["passed"])
    image = Image.open(io.BytesIO(base64.b64decode(attempt["tile_png_b64"])))
    assert image.size == (64, 64)
    assert image.mode == "P"


def test_partial_arms_mark_unrun_tasks_and_publish_common_prompt_ids(tmp_path):
    source = ROOT / "benchmarks" / "data" / "eval-v2.1-trajectories.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    first, second, third = rows[:3]
    arm_a_rows = []
    for row in (first, second):
        arm_a_rows.append({**row, "model_key": "arm-a", "provider": "fixture-a"})
    arm_b_rows = []
    for row in (second, third):
        arm_b_rows.append({**row, "model_key": "arm-b", "provider": "fixture-b"})
    arm_a = tmp_path / "arm-a.jsonl"
    arm_b = tmp_path / "arm-b.jsonl"
    arm_a.write_text("".join(json.dumps(row) + "\n" for row in arm_a_rows), encoding="utf-8")
    arm_b.write_text("".join(json.dumps(row) + "\n" for row in arm_b_rows), encoding="utf-8")

    output = tmp_path / "out"
    index_path = build_gallery_data(
        output_dir=output,
        trajectory_paths=(arm_a, arm_b),
        generated_at="2026-10-30T00:00:00+00:00",
    )
    index = json.loads(index_path.read_text(encoding="utf-8"))

    assert index["common_prompt_ids"] == [second["prompt_id"]]
    for key, completed in (("arm-a", {first["prompt_id"], second["prompt_id"]}),
                           ("arm-b", {second["prompt_id"], third["prompt_id"]})):
        payload = json.loads((output / f"{key}.json").read_text(encoding="utf-8"))
        assert len(payload["tasks"]) == 72
        assert {task_id for task_id, task in payload["tasks"].items()
                if task["status"] == "completed"} == completed
        assert all(task["status"] == "not_run" for task_id, task in payload["tasks"].items()
                   if task_id not in completed)

def test_arm_summary_counts_normalization_actions_and_publishes_denominator():
    from scripts.build_gallery_data import _arm_summary

    rows = [{
        "prompt_id": "prompt-1", "grammar": "counted_grid",
        "validity_at_1": False, "validity_at_3": False, "valid_schema_at_1": True,
        "repair_rescued": False,
        "calls": [
            {"reward": 0.2, "metadata": {"normalizations": ["strip_code_fence"]}},
            {"reward": 0.3, "metadata": {"normalizations": ["strip_think_block", "strip_code_fence", "strip_code_fence"]}},
            {"reward": 0.4, "metadata": {"normalizations": []}},
        ],
    }]

    summary = _arm_summary("fixture-model", rows)

    assert summary["normalizations"] == {"strip_code_fence": 2, "strip_think_block": 1}
    assert summary["normalization_calls"] == 3
    assert summary["normalization_rates"] == {
        "strip_code_fence": pytest.approx(2 / 3),
        "strip_think_block": pytest.approx(1 / 3),
    }


def test_gallery_summary_guard_checks_all_ten_models_and_rejects_count_drift():
    keys = [
        "beam-hf", "claude-fable-5-1", "claude-haiku-5-5", "claude-opus-5-5",
        "claude-sonnet-5-5", "deepseek-v4.1-flash", "gpt-6-astra", "gpt-6-luna",
        "gpt-6.1-sol", "gpt-6.1-terra",
    ]
    arms = [{"key": key, "n_tasks": 72, "pass_at_1": 0.5, "pass_at_3": 0.75}
            for key in keys]
    summary = {"by_model": {key: {"overall": {
        "n": 72, "scored_n": 72, "passed_at_1": 36, "passed_at_3": 54,
    }} for key in keys}}

    _validate_gallery_summary(arms, summary)

    summary["by_model"]["gpt-6.1-terra"]["overall"]["passed_at_3"] = 53
    with pytest.raises(ValueError, match="gpt-6.1-terra.*passed_at_3"):
        _validate_gallery_summary(arms, summary)


def test_gallery_summary_guard_rejects_missing_or_extra_model():
    arms = [{"key": "beam-hf", "n_tasks": 1, "pass_at_1": 1.0, "pass_at_3": 1.0}]
    summary = {"by_model": {"beam-hf": {"overall": {
        "n": 1, "scored_n": 1, "passed_at_1": 1, "passed_at_3": 1,
    }}}}

    summary["by_model"]["unmatched-model"] = {"overall": {"n": 1}}
    with pytest.raises(ValueError, match="model keys differ"):
        _validate_gallery_summary(arms, summary)
