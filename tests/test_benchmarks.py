import json
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.run import BENCHMARK_VERSION, load_model_registry, load_resume_pairs, run_prompt
from pramaan import VALIDATOR_VERSION
from benchmarks.summarize import summarize
from tests.helpers import pattern_obj


class RepairFixture:
    def __init__(self):
        self.calls = 0
        self.valid = json.loads(pattern_obj().model_dump_json())
        self.invalid = json.loads(json.dumps(self.valid))
        self.invalid["layers"][0]["motifs"] = self.invalid["layers"][0]["motifs"][:-1]

    def complete(self, system, prompt):
        self.calls += 1
        candidate = self.invalid if self.calls == 1 else self.valid
        return json.dumps(candidate), {"model_id": "fixture", "provider": "test"}


def test_benchmark_pairs_one_shot_and_repair_on_the_same_initial_candidate():
    client = RepairFixture()
    record = run_prompt({
        "id": "easy-01", "level": "easy", "symmetry": "D4", "brief": "balanced tile"
    }, client=client)

    assert client.calls == 2
    assert record["attempt_count"] == 2
    assert record["validity_at_1"] is False
    assert record["validity_at_3"] is True
    assert record["repair_rescued"] is True
    assert record["attempts"][0]["candidate"]["title"] == "Test"


def test_benchmark_summary_marks_partial_prompt_coverage_and_repair_rescue():
    record = {
        "prompt_id": "easy-01", "level": "easy", "validity_at_1": False,
        "validity_at_3": True, "repair_rescued": True, "attempt_count": 2,
        "elapsed_ms": 12, "total_tokens": 50, "provider_error": None,
    }
    result = summarize([record], expected_prompt_ids=["easy-01", "easy-02"])

    assert result["study"]["complete"] is False
    assert result["study"]["missing_prompt_ids"] == ["easy-02"]
    assert result["overall"]["validity_at_1"] == 0
    assert result["overall"]["validity_at_3"] == 1
    assert result["overall"]["repair_rescue_rate"] == 1


def test_benchmark_summary_rejects_duplicate_prompt_records():
    row = {"prompt_id": "easy-01", "level": "easy"}
    try:
        summarize([row, dict(row)], expected_prompt_ids=["easy-01"])
    except ValueError as exc:
        assert "duplicate model/prompt records" in str(exc)
    else:
        raise AssertionError("duplicate prompt rows were accepted")


def test_same_prompt_is_a_separate_comparison_for_each_model():
    first = {"model_key": "model-a", "prompt_id": "easy-01", "level": "easy", "validity_at_1": True, "elapsed_ms": 1}
    second = {"model_key": "model-b", "prompt_id": "easy-01", "level": "easy", "validity_at_1": False, "elapsed_ms": 1}
    result = summarize([first, second], ["easy-01"], ["model-a", "model-b"])

    assert set(result["by_model"]) == {"model-a", "model-b"}
    assert result["by_model"]["model-a"]["study"]["complete"] is True
    assert result["by_model"]["model-b"]["study"]["complete"] is True
    assert result["by_model"]["model-a"]["overall"]["validity_at_1"] == 1
    assert result["by_model"]["model-b"]["overall"]["validity_at_1"] == 0
    assert "overall" not in result  # no misleading pooled quality score


def test_duplicate_is_rejected_only_for_same_model_prompt_pair():
    row = {"model_key": "model-a", "prompt_id": "easy-01"}
    try:
        summarize([row, dict(row)], ["easy-01"])
    except ValueError as exc:
        assert "model-a/easy-01" in str(exc)
    else:
        raise AssertionError("duplicate model/prompt row was accepted")


def test_registry_has_model_neutral_credential_configuration():
    registry = load_model_registry(Path(__file__).parents[1] / "benchmarks" / "models.json")
    assert list(registry) == ["beam-hf"]
    assert registry["beam-hf"]["api_key_env"] == "HF_TOKEN"
    assert "test-token" not in json.dumps(registry)


def test_registry_rejects_non_https_or_credential_overriding_models():
    base = load_model_registry(Path(__file__).parents[1] / "benchmarks" / "models.json")["beam-hf"]
    for field, value in (("base_url", "http://example.test/v1"),
                         ("api_key_env", "hf-token"),
                         ("request_params", {"stream": True})):
        entry = dict(base)
        entry[field] = value
        with TemporaryDirectory(dir=Path(__file__).parents[1]) as temp_dir:
            path = Path(temp_dir) / "models.json"
            path.write_text(json.dumps({"schema_version": 1, "models": [entry]}), encoding="utf-8")
            try:
                load_model_registry(path)
            except ValueError:
                pass
            else:
                raise AssertionError(f"unsafe registry configuration was accepted: {field}")


def test_resume_validates_existing_rows_and_returns_completed_pairs():
    registry = load_model_registry(Path(__file__).parents[1] / "benchmarks" / "models.json")
    config = registry["beam-hf"]
    row = {
        "benchmark_version": BENCHMARK_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "prompt_suite_sha256": "suite-hash",
        "model_key": "beam-hf",
        "model_id": config["model_id"],
        "provider": config["provider"],
        "api_style": config["api_style"],
        "request_params": config["request_params"],
        "prompt_id": "easy-01",
    }
    with TemporaryDirectory(dir=Path(__file__).parents[1]) as temp_dir:
        path = Path(temp_dir) / "results.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")

        assert load_resume_pairs(path, {"beam-hf": config}, ["easy-01", "easy-02"], "suite-hash") == {
            ("beam-hf", "easy-01")
        }
        try:
            load_resume_pairs(path, {"beam-hf": config}, ["easy-01"], "different-suite")
        except ValueError as exc:
            assert "mismatched" in str(exc)
        else:
            raise AssertionError("resume accepted a different prompt suite")
