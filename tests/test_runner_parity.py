import json
from pathlib import Path
import pytest

from benchmarks.run import (AnthropicMessagesClient, BudgetExceeded, CometOpenAICompatibleClient,
                            CostLedger, GuardedCompletionClient, ModelPrice, SecretRedactingClient, load_model_registry,
                            make_client, paid_run_freeze, preflight_cost, preflight_selected_models, run_prompt,
                            save_or_validate_paid_freeze)
from scripts.export_preference_pairs import export_pairs, export_repair_examples, export_sft_examples
from scripts.push_hub_datasets import (compact_suite_row, dataset_card, enforce_data_role,
                                       make_suite_trajectory, publish_community_eval_run,
                                       sanitize_community_eval_rows, stage_reference_bundle, stage_repository)
from scripts.push_hub_datasets import validate_eval_v21_pair
from pramaan.checks.core import all_pass, verify_pattern
from pramaan.dsl import parse_pattern
from pramaan.tasks import load_suite
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


def test_v2_trajectory_export_preserves_exact_messages_and_training_role():
    messages = [{"role": "system", "content": "exact system"},
                {"role": "user", "content": "exact task prompt"}]
    telemetry = {"latency_ms": {}, "token_usage": {}, "cost": {}, "provider_attempts": 1}
    row = {
        "model_key": "beam-hf", "model_id": "beam", "provider": "hf", "api_style": "chat",
        "request_params": {}, "pricing": None, "prompt_suite_sha256": "a" * 64,
        "prompt_id": "tn-kolam-train-1", "suite_id": "train-v1", "state": "Tamil Nadu",
        "tradition_id": "tn_kolam", "grammar": "kolam_loop", "level": "train",
        "brief": "brief", "symmetry": "D2", "validator_version": "0.3.0",
        "valid_schema_at_1": False, "validity_at_1": False, "validity_at_3": False,
        "repair_rescued": False, "attempts": [], "elapsed_ms": 12, "created_at": "now",
        "model_calls": [{"call_index": 1, "phase": "proposal", "messages": messages,
                         "raw_response": "{}", "metadata": {}, "telemetry": telemetry}],
    }
    task = {"constraint_profile": {"symmetry": "D2"}, "palette": ["#000000"]}

    result = make_suite_trajectory(row, task, "training")

    assert result["data_role"] == "training"
    assert result["training_eligible"] is True
    assert result["calls"][0]["messages"] == messages


def test_sft_export_preserves_versioned_scalar_reward_and_skips_eval():
    records = [{"trajectory_id": "train:1", "data_role": "training", "training_eligible": True,
                "reward_version": "pramaan-reward-v1", "calls": [{
                    "messages": [{"role": "user", "content": "task"}], "raw_response": "{}",
                    "call_index": 1, "phase": "proposal", "reward": 0.5}]},
               {"trajectory_id": "eval:1", "data_role": "benchmark_evaluation", "training_eligible": False,
                "calls": [{"messages": [], "raw_response": "{}", "reward": 1.0}]}]
    result = export_sft_examples(records)
    assert len(result) == 1
    assert result[0]["reward"] == 0.5
    assert result[0]["reward_version"] == "pramaan-reward-v1"
    assert result[0]["training_eligible"] is True
    assert [message["role"] for message in result[0]["messages"]] == ["user", "assistant"]
    assert result[0]["messages"][-1]["content"] == result[0]["completion"] == "{}"


def test_compact_evaluation_export_removes_duplicate_raw_completions():
    row = {"prompt_id": "eval-1", "model_calls": [{"raw_response": "private duplicate"}],
           "attempts": [{"raw": "private duplicate", "candidate": {"ok": True}}]}
    result = compact_suite_row(row, "benchmark_evaluation")
    assert result["data_role"] == "benchmark_evaluation"
    assert result["training_eligible"] is False
    assert "model_calls" not in result
    assert "raw" not in result["attempts"][0]


def _without_timing(value):
    if isinstance(value, dict):
        return {
            key: _without_timing(item)
            for key, item in value.items()
            if key not in {"created_at", "started_at", "completed_at", "verified_at"}
            and "elapsed" not in key
            and key != "latency_ms"
        }
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


def test_via_env_and_direct_produce_identical_rows():
    tasks, _ = load_suite("eval-v2")
    prompt = next(task.model_dump(mode="json") for task in tasks if task.id == "br_madhubani-easy-3")
    direct = run_prompt(prompt, client=RepairFixture())
    through_env = run_prompt(prompt, client=RepairFixture(), via_env=True)

    assert _without_timing(through_env) == _without_timing(direct)


def test_via_env_parity_covers_every_periodic_raster_eval_task():
    tasks, _ = load_suite("eval-v2")
    for task in tasks:
        if task.grammar != "periodic_raster":
            continue
        prompt = task.model_dump(mode="json")
        direct = run_prompt(prompt, client=RepairFixture())
        through_env = run_prompt(prompt, client=RepairFixture(), via_env=True)
        assert _without_timing(through_env) == _without_timing(direct), task.id


def test_summary_groups_tasks_by_state_grammar_and_level():
    from benchmarks.summarize import summarize

    summary = summarize([{
        "prompt_id": "tn-kolam-easy", "model_key": "fixture", "state": "Tamil Nadu",
        "grammar": "kolam_loop", "level": "easy", "attempt_count": 1,
        "validity_at_1": True, "validity_at_3": True, "elapsed_ms": 1,
    }])
    model = summary["by_model"]["fixture"]
    assert model["by_state"]["Tamil Nadu"]["n"] == 1
    assert model["by_grammar"]["kolam_loop"]["n"] == 1
    assert model["by_level"]["easy"]["n"] == 1


def test_preference_export_excludes_evaluation_and_keeps_training_pair():
    messages = [{"role": "user", "content": "make a tile"}]
    training = {
        "data_role": "training", "training_eligible": True, "trajectory_id": "train:1",
        "calls": [
            {"phase": "proposal", "call_index": 1, "messages": messages, "raw_response": "bad",
             "candidate": {}, "accepted": False, "verifier_report": {"checks": [{"id": "motif_count", "status": "FAIL"}]}},
            {"phase": "repair", "call_index": 2, "messages": messages, "raw_response": "good",
             "candidate": {}, "accepted": True},
        ],
    }
    evaluation = {**training, "data_role": "benchmark_evaluation", "training_eligible": False}
    pairs = export_pairs([evaluation, training])
    assert len(pairs) == 1
    assert pairs[0]["chosen"] == "good"
    assert pairs[0]["rejected"] == "bad"
    assert pairs[0]["chosen_messages"][-1] == {"role": "assistant", "content": "good"}
    assert pairs[0]["rejected_messages"][-1] == {"role": "assistant", "content": "bad"}
    assert pairs[0]["data_role"] == "training"
    assert pairs[0]["training_eligible"] is True
    repairs = export_repair_examples([evaluation, training])
    assert len(repairs) == 1
    assert repairs[0]["completion"] == "good"
    assert repairs[0]["messages"][-1] == {"role": "assistant", "content": "good"}
    assert repairs[0]["data_role"] == "training"


def test_training_staging_exposes_assistant_turns_in_chat_messages(tmp_path):
    sources = {}
    rows = {
        "trajectories": [{"trajectory_id": "t1", "calls": [{
            "messages": [{"role": "system", "content": "rules"}, {"role": "user", "content": "task"}],
            "raw_response": "answer"}]}],
        "sft": [{"messages": [{"role": "system", "content": "rules"},
                              {"role": "user", "content": "task"}], "completion": "answer"}],
        "repair": [{"messages": [{"role": "user", "content": "repair"}], "completion": "fixed"}],
        "preference": [{"prompt": [{"role": "user", "content": "task"}],
                        "chosen": "good", "rejected": "bad"}],
    }
    for name, records in rows.items():
        source = tmp_path / f"{name}.jsonl"
        source.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        sources[name] = source

    staged = stage_repository("train", sources, suite_name="train-v1")

    sft = json.loads((staged / "sft" / "train.jsonl").read_text(encoding="utf-8"))
    repair = json.loads((staged / "repair" / "train.jsonl").read_text(encoding="utf-8"))
    preference = json.loads((staged / "preference" / "train.jsonl").read_text(encoding="utf-8"))
    trajectory = json.loads((staged / "trajectories" / "train.jsonl").read_text(encoding="utf-8"))
    assert sft["messages"][-1] == {"role": "assistant", "content": "answer"}
    assert repair["messages"][-1] == {"role": "assistant", "content": "fixed"}
    assert preference["chosen_messages"][-1] == {"role": "assistant", "content": "good"}
    assert preference["rejected_messages"][-1] == {"role": "assistant", "content": "bad"}
    call = trajectory["calls"][0]
    assert [message["role"] for message in call["messages"]] == ["system", "user"]
    assert call["conversation"][-1] == {"role": "assistant", "content": "answer"}


def test_hub_upload_refuses_cross_role_files(tmp_path):
    path = tmp_path / "evaluation.jsonl"
    path.write_text(json.dumps({"data_role": "training", "training_eligible": True}) + "\n", encoding="utf-8")
    try:
        enforce_data_role(path, "eval")
    except ValueError as exc:
        assert "conflicts" in str(exc)
    else:
        raise AssertionError("evaluation upload accepted training data")


def test_hub_staging_labels_untagged_rows_by_dataset_role(tmp_path):
    source = tmp_path / "compact.jsonl"
    source.write_text(json.dumps({"prompt_id": "p1"}) + "\n", encoding="utf-8")
    staged = stage_repository("eval", {"compact": source})
    row = json.loads((staged / "compact" / "test.jsonl").read_text(encoding="utf-8"))
    assert row["data_role"] == "benchmark_evaluation"
    assert row["training_eligible"] is False


def test_hub_staging_accepts_pretty_json_summary(tmp_path):
    source = tmp_path / "summary.json"
    source.write_text(json.dumps({"by_model": {"beam-hf": {"overall": {"n": 72}}}}, indent=2), encoding="utf-8")
    staged = stage_repository("eval", {"summary": source})
    row = json.loads((staged / "summary" / "test.jsonl").read_text(encoding="utf-8"))
    assert row["by_model"]["beam-hf"]["overall"]["n"] == 72
    assert row["data_role"] == "benchmark_evaluation"


def test_eval_v21_stages_small_leaderboard_and_plain_json_summary(tmp_path):
    from scripts.push_hub_datasets import _leaderboard_rows

    root = Path(__file__).parents[1]
    summary = root / "benchmarks/data/eval-v2.1-summary.json"
    rows = _leaderboard_rows(summary)
    assert len(rows) == 10
    fable = next(row for row in rows if row["model"] == "claude-fable-5-1")
    assert fable["evaluated"] == 72
    assert fable["cap_usd"] == 22.0
    assert fable["within_three"] == 63
    assert fable["data_role"] == "benchmark_evaluation"
    assert fable["training_eligible"] is False

    compact = tmp_path / "compact.jsonl"
    trajectories = tmp_path / "trajectories.jsonl"
    model_keys = ["beam-hf", "claude-fable-5-1", "claude-haiku-5-5", "claude-opus-5-5",
                  "claude-sonnet-5-5", "deepseek-v4.1-flash", "gpt-6-astra", "gpt-6-luna",
                  "gpt-6.1-sol", "gpt-6.1-terra"]
    compact_rows, trajectory_rows = [], []
    for model_key in model_keys:
        for task_index in range(72):
            base = {"model_key": model_key, "prompt_id": f"task-{task_index}",
                    "suite_id": "eval-v2.1", "data_role": "benchmark_evaluation",
                    "training_eligible": False}
            compact_rows.append({**base, "attempts": []})
            trajectory_rows.append({**base, "trajectory_schema_version": "1.1",
                                    "calls": [{"messages": [{"role": "system", "content": "rules"},
                                                               {"role": "user", "content": "task"}],
                                                   "phase": "proposal", "metadata": {}, "verifier_report": {}}]})
    compact.write_text("".join(json.dumps(row) + "\n" for row in compact_rows), encoding="utf-8")
    trajectories.write_text("".join(json.dumps(row) + "\n" for row in trajectory_rows), encoding="utf-8")
    stage = stage_repository("eval", {"compact": compact, "trajectories": trajectories, "summary": summary},
                             include_references=True, suite_name="eval-v2.1")
    card = (stage / "README.md").read_text(encoding="utf-8")
    staged_rows = [json.loads(line) for line in
                   (stage / "leaderboard/test.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(staged_rows) == 10
    assert (stage / "summary/eval-v2.1-summary.json").is_file()
    assert not (stage / "summary/test.jsonl").exists()
    assert "config_name: leaderboard" in card
    assert "config_name: summary" not in card
    assert "independent USD 10 cap" in card
    assert "USD 22 cap" in card


def test_eval_v21_publish_gate_rejects_beam_only_or_compact_as_trajectories(tmp_path):
    compact = tmp_path / "compact.jsonl"
    trajectories = tmp_path / "trajectories.jsonl"
    compact_rows = []
    trajectory_rows = []
    for task_index in range(72):
        common = {"model_key": "beam-hf", "prompt_id": f"task-{task_index}",
                  "suite_id": "eval-v2.1", "data_role": "benchmark_evaluation",
                  "training_eligible": False}
        compact_rows.append({**common, "attempts": []})
        trajectory_rows.append({**common, "trajectory_schema_version": "1.1",
                                "calls": [{"phase": "proposal", "messages": [{"role": "system", "content": "rules"},
                                                                                     {"role": "user", "content": "task"}],
                                           "metadata": {}, "verifier_report": {}}]})
    compact.write_text("".join(json.dumps(row) + "\n" for row in compact_rows), encoding="utf-8")
    trajectories.write_text("".join(json.dumps(row) + "\n" for row in trajectory_rows), encoding="utf-8")
    with pytest.raises(ValueError, match="720 rows across ten models"):
        validate_eval_v21_pair(compact, trajectories)


def test_dataset_cards_parse_and_separate_eval_and_training_splits():
    import hashlib
    from pathlib import Path

    from huggingface_hub import DatasetCard

    eval_card = DatasetCard(dataset_card("eval"))
    train_card = DatasetCard(dataset_card("train"))
    assert eval_card.data.license == train_card.data.license == "cc-by-4.0"
    assert eval_card.data.task_categories == ["text-generation"]
    assert eval_card.data.configs[0]["data_files"] == [{"split": "test", "path": "trajectories/test.jsonl"}]
    assert train_card.data.configs[0]["data_files"] == [{"split": "train", "path": "sft/train.jsonl"}]
    assert train_card.data.configs[1]["data_files"] == [{"split": "train", "path": "trajectories/train.jsonl"}]
    assert 'load_dataset("kshitijthakkar/pramaan-proof-of-art-train-v1", "sft", split="train")' in train_card.content
    assert "calls[].conversation" in train_card.content
    assert "reflection/Beam-1.0-501B-A23B" in eval_card.content
    assert "inspired by" in eval_card.content
    assert "Tradition registry source list" in eval_card.content
    assert "pramaan-reward-v1" in eval_card.content
    for suite in ("eval-v2", "train-v1"):
        from pramaan.tasks import canonical_jsonl_bytes
        digest = hashlib.sha256(canonical_jsonl_bytes(Path(f"benchmarks/suites/{suite}.jsonl").read_bytes())).hexdigest()
        card = eval_card if suite == "eval-v2" else train_card
        assert digest in card.content


def test_eval_v21_card_digest_is_stable_for_lf_suite_checkout(tmp_path, monkeypatch):
    import hashlib
    from scripts import push_hub_datasets
    from pramaan.tasks import canonical_jsonl_bytes

    suite = tmp_path / "benchmarks" / "suites" / "eval-v2.1.jsonl"
    suite.parent.mkdir(parents=True)
    payload = b'{"id":"task-1"}\n{"id":"task-2"}\n'
    suite.write_bytes(payload)
    summary = tmp_path / "benchmarks" / "data" / "eval-v2.1-summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"by_model": {"beam-hf": {"overall": {
        "n": 72, "valid_schema_at_1": .5, "validity_at_1": .25, "validity_at_3": .75,
    }}}}), encoding="utf-8")
    monkeypatch.setattr(push_hub_datasets, "ROOT", tmp_path)

    card = push_hub_datasets.dataset_card("eval", suite_name="eval-v2.1")

    assert hashlib.sha256(canonical_jsonl_bytes(payload)).hexdigest() in card


def test_trajectory_export_accepts_lf_copy_of_frozen_suite(tmp_path):
    import hashlib
    from scripts.push_hub_datasets import export_suite_trajectories
    from pramaan.tasks import canonical_jsonl_bytes

    suite_source = Path("benchmarks/suites/eval-v2.1.jsonl")
    suite_copy = tmp_path / "eval-v2.1.jsonl"
    suite_copy.write_bytes(suite_source.read_bytes().replace(b"\r\n", b"\n"))
    result_path = Path("benchmarks/results/eval-v2.1-beam-20261030.jsonl")
    output_path = tmp_path / "trajectories.jsonl"

    result = export_suite_trajectories(result_path, suite_copy, output_path,
                                       role="benchmark_evaluation")

    expected = hashlib.sha256(canonical_jsonl_bytes(suite_copy.read_bytes())).hexdigest()
    assert result["suite_sha256"] == expected
    assert result["records"] == 72

class DotFieldFixture:
    def complete(self, system, prompt):
        candidate = {
            "schema_version": "0.2",
            "grammar": "dot_field",
            "tradition_id": "candidate-controlled",
            "title": "Centered dot",
            "tile": {"width": 64, "height": 64, "wrap_x": True, "wrap_y": True},
            "palette": ["#000000", "#FFFFFF"],
            "background_index": 0,
            "layers": [{
                "id": "dots", "colour_index": 1, "anchor": [0, 0],
                "motifs": [{"kind": "dot", "center": [31, 31], "radius": 1}],
            }],
            "constraints": {"symmetry": "None", "min_clearance": 0},
        }
        candidate.update({f"unknown_{index}": "x" * 100 for index in range(20)})
        return json.dumps(candidate), {"model_id": "fixture-v2", "provider": "test"}


def test_non_g1_task_runner_matches_direct_and_environment_paths():
    tasks, _ = load_suite("eval-v2")
    prompt = next(task.model_dump(mode="json") for task in tasks if task.grammar == "dot_field")
    direct = run_prompt(prompt, client=DotFieldFixture())
    through_env = run_prompt(prompt, client=DotFieldFixture(), via_env=True)
    assert _without_timing(through_env) == _without_timing(direct)
    assert direct["attempt_count"] == 3
    assert direct["grammar"] == "dot_field"
    assert all(len(check["message"]) <= 1000
               for attempt in direct["attempts"] for check in attempt["report"]["checks"]
               if check["id"] == "candidate_parse")


def test_via_env_normalizes_fenced_model_json_like_direct_runner():
    tasks, _ = load_suite("eval-v2")
    prompt = next(task.model_dump(mode="json") for task in tasks if task.grammar == "dot_field")

    class FencedDotFieldFixture(DotFieldFixture):
        def complete(self, system, prompt):
            raw, metadata = super().complete(system, prompt)
            candidate = json.loads(raw)
            candidate = {key: value for key, value in candidate.items() if not key.startswith("unknown_")}
            raw = json.dumps(candidate)
            return f"```json\n{raw}\n```", metadata

    direct = run_prompt(prompt, client=FencedDotFieldFixture())
    through_env = run_prompt(prompt, client=FencedDotFieldFixture(), via_env=True)

    assert _without_timing(through_env) == _without_timing(direct)
    assert through_env["attempts"][0]["normalizations"] == ["strip_code_fence"]

def test_runner_uses_cached_hf_token_when_env_is_unset(monkeypatch):
    from pathlib import Path

    from benchmarks.run import load_model_registry, make_client

    config = load_model_registry(Path("benchmarks/models.json"))["beam-hf"]
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_COMMUNITY_WRITE_TOKEN", raising=False)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "cached-test-token")
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("benchmarks.run.OpenAICompatibleClient", FakeClient)
    make_client(config)
    assert captured["token"] == "cached-test-token"

class GrammarFixture:
    def __init__(self, grammar):
        self.grammar = grammar

    def complete(self, system, prompt):
        bounded = self.grammar in {"rotational_field", "kolam_loop", "jaali_lattice"}
        candidate = {
            "schema_version": "0.2",
            "grammar": self.grammar,
            "tradition_id": "candidate-controlled",
            "title": "Parity fixture",
            "tile": {"width": 64, "height": 64, "wrap_x": not bounded, "wrap_y": not bounded},
            "palette": ["#000000", "#FFFFFF"],
            "background_index": 0,
            "layers": [{
                "id": "marks", "colour_index": 1, "anchor": [0, 0],
                "motifs": [{"kind": "dot", "center": [31, 31], "radius": 1}],
            }],
            "constraints": {"symmetry": "None"},
        }
        if self.grammar == "kolam_loop":
            candidate["kolam"] = {
                "rows": 3, "cols": 3, "spacing": 8, "mirrors": [], "stroke": 1, "dot_radius": 1,
            }
        if self.grammar == "jaali_lattice":
            # The model-provided frame is a deliberate wrong value. The host must
            # replace it with the suite-owned task profile before verification.
            candidate["jaali"] = {"frame": 0, "open_colour_index": 1, "holes": []}
        return json.dumps(candidate), {"model_id": "fixture-v2", "provider": "test"}


def test_direct_env_parity_covers_implemented_v2_grammars():
    tasks, _ = load_suite("eval-v2")
    v2_grammars = {
        "dot_field", "stripe_band", "counted_grid", "rotational_field",
        "kolam_loop", "jaali_lattice",
    }
    representatives = {
        grammar: next(
            task for task in tasks
            if task.grammar == grammar
            and (grammar != "jaali_lattice" or len(task.tradition_id) <= 40)
        )
        for grammar in v2_grammars
    }
    assert set(representatives) == v2_grammars
    for grammar, task in representatives.items():
        prompt = task.model_dump(mode="json")
        if grammar == "kolam_loop":
            assert task.id == "tn_kolam-easy-22"
        direct = run_prompt(prompt, client=GrammarFixture(grammar))
        through_env = run_prompt(prompt, client=GrammarFixture(grammar), via_env=True)
        assert _without_timing(through_env) == _without_timing(direct), grammar
        if grammar == "jaali_lattice":
            expected_frame = prompt["constraint_profile"]["frame"]
            assert expected_frame != 0
            assert direct["attempts"][-1]["candidate"]["jaali"]["frame"] == expected_frame


def test_via_env_and_direct_match_invalid_json_diagnostics():
    tasks, _ = load_suite("eval-v2")
    prompt = next(task.model_dump(mode="json") for task in tasks)

    class InvalidJSONFixture:
        def complete(self, system, prompt):
            return '{"schema_version":', {
                "model_id": "fixture",
                "provider": "test",
                "finish_reason": "length",
            }

    direct = run_prompt(prompt, client=InvalidJSONFixture())
    through_env = run_prompt(prompt, client=InvalidJSONFixture(), via_env=True)
    assert _without_timing(through_env) == _without_timing(direct)


def test_trajectory_calls_export_scalar_reward_and_version():
    from scripts.push_hub_datasets import make_suite_trajectory

    from pathlib import Path

    pattern_json = (Path(__file__).parents[1] / "examples/ajrakh_starfield.pattern.json").read_text(encoding="utf-8")
    known_checks, _masks, _grid = verify_pattern(parse_pattern(pattern_json))
    passing_report = {"checks": known_checks}
    partial_report = {"checks": [
        {**check, "status": "PASS" if check["id"] == "schema_valid" else "FAIL"}
        for check in known_checks
    ]}
    calls = []
    for index, report in enumerate((passing_report, partial_report, None), 1):
        calls.append({"call_index": index, "phase": "proposal", "messages": [
            {"role": "system", "content": "sys"}, {"role": "user", "content": "task"}],
            "telemetry": {}, "raw_response": "{}"})
    attempts = [
        {"call_index": 1, "index": 1, "candidate": {}, "report": passing_report},
        {"call_index": 2, "index": 2, "candidate": {}, "report": partial_report},
    ]
    row = {
        "model_key": "fixture", "prompt_id": "fixture-1", "prompt_suite_sha256": "a" * 64,
        "suite_id": "train-v1", "state": "Tamil Nadu", "tradition_id": "tn_kolam",
        "grammar": "kolam_loop", "level": "train", "brief": "brief", "symmetry": "D2",
        "validator_version": "0.3.0", "elapsed_ms": 1, "attempts": attempts,
        "model_calls": calls, "valid_schema_at_1": True, "validity_at_1": True,
        "validity_at_3": False,
    }
    task = {"constraint_profile": {}, "palette": []}

    trajectory = make_suite_trajectory(row, task, "training")

    assert trajectory["reward_version"] == "pramaan-reward-v1"
    assert [call["reward"] for call in trajectory["calls"]] == [1.0, 0.2, 0.0]


def test_paid_model_preflight_enforces_ten_dollar_per_model_at_4096():
    models = [ModelPrice("haiku", "anthropic", .1, .5),
              ModelPrice("sonnet", "anthropic", 2, 10),
              ModelPrice("opus", "anthropic", 4, 20),
              ModelPrice("comet", "cometapi", .24, .96)]
    plan = preflight_cost(models, prompts=72, attempts_per_prompt=3, max_input_bytes=16000,
                          input_token_overhead=256, max_output_tokens=4096)
    assert plan["calls_per_model"] == 216
    assert plan["anthropic_usd"] == pytest.approx(48.4033536)
    assert plan["cometapi_usd"] == pytest.approx(1.6920576)
    assert plan["model_cap_usd"] == {"haiku": 10.0, "sonnet": 10.0, "opus": 10.0, "comet": 10.0}
    assert plan["model_fits"] == {"haiku": True, "sonnet": False, "opus": False, "comet": True}
    assert not plan["anthropic_fits"] and plan["cometapi_fits"]


def test_paid_model_preflight_rejects_each_model_over_ten_dollars():
    models = [ModelPrice("haiku", "anthropic", .1, .5),
              ModelPrice("sonnet", "anthropic", 2, 10),
              ModelPrice("opus", "anthropic", 4, 20)]
    plan = preflight_cost(models, prompts=72, attempts_per_prompt=3, max_input_bytes=16000,
                          input_token_overhead=256, max_output_tokens=4096)
    assert plan["anthropic_usd"] == pytest.approx(48.4033536)
    assert plan["model_fits"] == {"haiku": True, "sonnet": False, "opus": False}
    assert not plan["anthropic_fits"]


def test_selected_model_preflight_reports_partial_coverage_under_independent_caps(tmp_path):
    def config(key, in_rate, out_rate):
        return {
            "key": key, "budget_provider": "anthropic",
            "budget": {"cap_usd": 10.0, "max_input_bytes": 16000, "input_token_overhead": 256},
            "request_params": {"max_tokens": 4096},
            "pricing": {"per_million_tokens": {"input": in_rate, "output": out_rate}},
        }
    selected = {
        "sonnet": config("sonnet", 2.0, 10.0),
        "opus": config("opus", 4.0, 20.0),
    }
    ledger = CostLedger(tmp_path / "ledger.jsonl", caps={"sonnet": 10.0, "opus": 10.0})
    plan = preflight_selected_models(selected, {"sonnet": 72, "opus": 72}, ledger)
    assert plan["model_fits_full_suite"] == {"sonnet": False, "opus": False}
    assert plan["model_cap_usd"] == {"sonnet": 10.0, "opus": 10.0}
    assert plan["assumptions"]["sonnet"]["cap_limited_full_prompts"] == 45
    assert plan["assumptions"]["opus"]["cap_limited_full_prompts"] == 22


def test_aggregate_cost_ledger_reserves_across_model_keys(tmp_path):
    ledger = CostLedger(tmp_path / "aggregate.jsonl", caps={"anthropic-aggregate": 1.0})
    ledger.reserve("haiku-1", provider="anthropic-aggregate", model_key="claude-haiku-5-5", amount_usd=.7)

    assert ledger.committed_usd("anthropic-aggregate") == pytest.approx(.7)
    assert ledger.committed_usd("anthropic-aggregate", model_key="claude-haiku-5-5") == pytest.approx(.7)
    assert ledger.committed_usd("anthropic-aggregate", model_key="claude-opus-5-5") == 0
    with pytest.raises(BudgetExceeded, match="anthropic-aggregate cap"):
        ledger.reserve("opus-1", provider="anthropic-aggregate", model_key="claude-opus-5-5", amount_usd=.31)


def test_prior_scope_ledger_import_preserves_settled_and_orphan_exposure(tmp_path):
    source_path = tmp_path / "old" / "opus.spend.jsonl"
    source = CostLedger(source_path, caps={"claude-opus-5-5": 1.0})
    source.reserve("settled", provider="claude-opus-5-5", amount_usd=.5)
    source.reconcile("settled", provider="claude-opus-5-5", actual_usd=None,
                     estimated_usd=.2, usage_reported=True)
    source.reserve("orphan", provider="claude-opus-5-5", amount_usd=.3)
    original_bytes = source_path.read_bytes()

    target = CostLedger(tmp_path / "new" / "aggregate.spend.jsonl",
                        caps={"anthropic-aggregate": 1.0})
    count = target.import_prior_ledger(
        source_path, source_provider="claude-opus-5-5", budget_key="anthropic-aggregate",
        model_key="claude-opus-5-5", scope="aborted-prior-scope",
    )

    assert count == 2
    assert source_path.read_bytes() == original_bytes
    assert target.committed_usd("anthropic-aggregate") == pytest.approx(.5)
    assert target.committed_usd("anthropic-aggregate", model_key="claude-opus-5-5") == pytest.approx(.5)
    events = target._events()
    assert all(event["scope"] == "aborted-prior-scope" for event in events)
    assert len([event for event in events if event["event"] == "reserve"]) == 2
    assert len([event for event in events if event["event"] == "settle"]) == 1


def test_selected_preflight_applies_independent_ten_dollar_anthropic_caps(tmp_path):
    def config(key, in_rate, out_rate):
        return {
            "key": key, "budget_provider": "anthropic", "budget_group": key,
            "budget": {"cap_usd": 10.0, "max_input_bytes": 16000, "input_token_overhead": 256},
            "request_params": {"max_tokens": 4096},
            "pricing": {"per_million_tokens": {"input": in_rate, "output": out_rate}},
        }
    selected = {
        "haiku": config("haiku", .1, .5),
        "sonnet": config("sonnet", 2, 10),
        "opus": config("opus", 4, 20),
        "fable": config("fable", 10, 50),
    }
    ledger = CostLedger(tmp_path / "per-model.jsonl", caps={key: 10.0 for key in selected})
    plan = preflight_selected_models(selected, {key: 47 for key in selected}, ledger)

    assert set(plan["budget_groups"]) == set(selected)
    assert all(group["cap_usd"] == 10.0 for group in plan["budget_groups"].values())
    assert all(group["model_keys"] == [key] for key, group in plan["budget_groups"].items())
    assert plan["model_fits_full_suite"] == {"haiku": True, "sonnet": False, "opus": False, "fable": False}
    assert plan["assumptions"]["haiku"]["conservative_complete_prompts_remaining"] == 47
    assert plan["assumptions"]["sonnet"]["conservative_complete_prompts_remaining"] > 0
    assert plan["assumptions"]["opus"]["conservative_complete_prompts_remaining"] > 0
    assert plan["assumptions"]["fable"]["conservative_complete_prompts_remaining"] > 0
    assert plan["anthropic_fits"] is False


def test_independent_cap_preflight_contains_aborted_scope_exposure_to_model(tmp_path):
    def config(key, in_rate, out_rate):
        return {
            "key": key, "budget_provider": "anthropic", "budget_group": key,
            "budget": {"cap_usd": 10.0, "max_input_bytes": 16000, "input_token_overhead": 256},
            "request_params": {"max_tokens": 4096},
            "pricing": {"per_million_tokens": {"input": in_rate, "output": out_rate}},
        }
    selected = {
        "haiku": config("haiku", .1, .5), "sonnet": config("sonnet", 2, 10),
        "opus": config("opus", 4, 20), "fable": config("fable", 10, 50),
    }
    ledger = CostLedger(tmp_path / "per-model.jsonl", caps={key: 10.0 for key in selected})
    ledger.reserve("aborted-run", provider="opus", model_key="claude-opus-5-5",
                   amount_usd=2.472579, scope="aborted-prior-scope")
    plan = preflight_selected_models(selected, {key: 47 for key in selected}, ledger)

    group = plan["budget_groups"]["opus"]
    assert group["committed_usd"] == pytest.approx(2.472579)
    assert group["committed_by_scope_usd"]["aborted-prior-scope"]["claude-opus-5-5"] == pytest.approx(2.472579)
    assert group["projected_usd"] == pytest.approx(2.472579 + plan["model_usd"]["opus"])
    assert plan["budget_groups"]["haiku"]["committed_usd"] == 0
    assert plan["budget_groups"]["sonnet"]["committed_usd"] == 0
    assert group["fits"] is False
    assert plan["model_fits_full_suite"]["haiku"] is True
    assert plan["model_fits_full_suite"]["sonnet"] is False
    assert plan["model_fits_full_suite"]["opus"] is False
    assert plan["anthropic_fits"] is False


def test_arm_preflight_freezes_exact_common47_ids_and_independent_caps(tmp_path):
    from pathlib import Path
    from scripts.run_arms import DEFAULT_MODELS, _load_plan

    root = Path(__file__).parents[1]
    registry, plan, digest = _load_plan(
        root / "benchmarks" / "models.json",
        root / "benchmarks" / "suites" / "eval-v2.1-common47.jsonl",
        list(DEFAULT_MODELS), tmp_path / "new-ledgers", None,
        prior_ledger_dir=tmp_path / "no-prior-ledgers",
    )

    assert digest == "dd3c9ce96a22deb1af94a1db22dc2a5016d1b1ad1c9c4a12befcd6690a6f7603"
    assert plan["suite_id"] == "eval-v2.1"
    assert plan["protocol_artifact"] == "eval-v2.1-common47"
    assert plan["task_count"] == 47
    assert len(plan["ordered_task_ids"]) == 47
    assert all(plan["assumptions"][key]["max_input_bytes_per_call"] == 100000 for key in DEFAULT_MODELS)
    assert all(plan["assumptions"][key]["input_token_upper_bound_per_call"] == 100256 for key in DEFAULT_MODELS)
    assert plan["ordered_task_ids"][0] == "ap_mangalagiri-sarees-and-fabrics-easy-0"
    assert set(registry) >= set(DEFAULT_MODELS)
    assert set(plan["budget_groups"]) == set(DEFAULT_MODELS)
    assert all(plan["budget_groups"][key]["model_keys"] == [key] for key in DEFAULT_MODELS)
    assert all(plan["budget_groups"][key]["cap_usd"] == 10.0 for key in DEFAULT_MODELS)
    assert plan["model_fits_full_suite"] == {
        "claude-haiku-5-5": True, "claude-sonnet-5-5": False,
        "claude-opus-5-5": False, "claude-fable-5-1": False,
        "deepseek-v4.1-flash": True,
    }


def test_delta_suite_is_the_exact_ordered_difference_from_full_eval_v21(tmp_path):
    from pathlib import Path
    from scripts.run_arms import build_delta_suite, _load_plan

    root = Path(__file__).parents[1]
    full = root / "benchmarks" / "suites" / "eval-v2.1.jsonl"
    common = root / "benchmarks" / "suites" / "eval-v2.1-common47.jsonl"
    delta = tmp_path / "eval-v2.1-delta25.jsonl"
    digest = build_delta_suite(full, common, delta)
    assert digest == "26c4866f44a8e8314de067ef9b6aaac821219914414aace11c0c288616eb1590"

    rows = [json.loads(line) for line in delta.read_text(encoding="utf-8").splitlines()]
    all_rows = [json.loads(line) for line in full.read_text(encoding="utf-8").splitlines()]
    common_ids = {json.loads(line)["id"] for line in common.read_text(encoding="utf-8").splitlines()}
    expected = [row for row in all_rows if row["id"] not in common_ids]
    assert rows == expected
    assert len(rows) == 25
    assert len({row["id"] for row in rows}) == 25

    _registry, plan, loaded_digest = _load_plan(
        root / "benchmarks" / "models.json", delta, ["claude-haiku-5-5"],
        tmp_path / "ledgers", None, prior_ledger_dir=tmp_path / "no-prior")
    assert loaded_digest == digest
    assert plan["protocol_artifact"] == "eval-v2.1-delta25"
    assert plan["task_count"] == 25


def test_arm_launcher_allows_partial_runs_under_independent_caps(monkeypatch, tmp_path):
    from scripts import run_arms

    key = "claude-haiku-5-5"
    registry = {key: {"key": key, "api_key_env": "ANTHROPIC_API_KEY",
                      "budget_group": key, "budget": {"cap_usd": 10.0}}}
    plan = {"budget_groups": {key: {"fits": False, "projected_usd": 12.0}},
            "assumptions": {key: {"conservative_complete_prompts_remaining": 12}}}
    launched = []
    monkeypatch.setattr(run_arms, "_load_plan", lambda *_args, **_kwargs: (registry, plan, "suite-sha"))
    monkeypatch.setattr(run_arms.runner, "configured_token", lambda _name: "test-key")
    monkeypatch.setattr(run_arms, "launch_parallel", lambda commands: launched.extend(commands) or 0)

    assert run_arms.main([
        "--model", key, "--execute",
        "--models", str(tmp_path / "models.json"),
        "--suite", str(tmp_path / "suite.jsonl"),
        "--output-dir", str(tmp_path / "outputs"),
        "--ledger-dir", str(tmp_path / "ledgers"),
        "--prior-ledger-dir", str(tmp_path / "prior"),
    ]) == 0
    assert len(launched) == 1


def test_arm_launcher_skips_model_that_cannot_afford_one_complete_task(monkeypatch, tmp_path):
    from scripts import run_arms

    key = "claude-fable-5-1"
    registry = {key: {"key": key, "api_key_env": "ANTHROPIC_API_KEY",
                      "budget_group": key, "budget": {"cap_usd": 10.0}}}
    plan = {"budget_groups": {key: {"fits": False, "projected_usd": 51.0}},
            "assumptions": {key: {"conservative_complete_prompts_remaining": 0}}}
    monkeypatch.setattr(run_arms, "_load_plan", lambda *_args, **_kwargs: (registry, plan, "suite-sha"))
    monkeypatch.setattr(run_arms.runner, "configured_token",
                        lambda _name: pytest.fail("must stop before checking credentials"))

    with pytest.raises(SystemExit):
        run_arms.main([
            "--model", key, "--execute",
            "--models", str(tmp_path / "models.json"),
            "--suite", str(tmp_path / "suite.jsonl"),
            "--output-dir", str(tmp_path / "outputs"),
            "--ledger-dir", str(tmp_path / "ledgers"),
            "--prior-ledger-dir", str(tmp_path / "prior"),
        ])


def test_direct_paid_cli_is_rejected_before_using_an_unaccounted_ledger(monkeypatch, tmp_path, capsys):
    from benchmarks import run as runner
    from pathlib import Path
    import sys

    root = Path(__file__).parents[1]
    ledger = tmp_path / "unaccounted-ledger.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "benchmarks.run", "--model", "claude-haiku-5-5",
        "--suite", str(root / "benchmarks" / "suites" / "eval-v2.1-common47.jsonl"),
        "--ledger", str(ledger), "--preflight-only",
    ])

    with pytest.raises(SystemExit) as error:
        runner.main()

    assert error.value.code == 2
    assert "scripts/run_arms.py" in capsys.readouterr().err
    assert not ledger.exists()


def test_arm_command_marks_paid_child_as_managed(tmp_path):
    from scripts.run_arms import arm_command

    command = arm_command(
        models_path=tmp_path / "models.json", suite_path=tmp_path / "suite.jsonl",
        output_dir=tmp_path / "outputs", ledger_dir=tmp_path / "ledgers",
        model_key="claude-haiku-5-5", budget_key="claude-haiku-5-5", cap_usd=10.0,
    )

    assert "--managed-paid-run" in command


def test_cost_projection_from_fake_usage():
    from scripts.run_arms import project_cost_from_usage

    projected = project_cost_from_usage(
        {"prompt_tokens": 1200, "completion_tokens": 350},
        {"input": 2.0, "output": 10.0},
    )

    assert projected == pytest.approx((1200 * 2.0 + 350 * 10.0) / 1_000_000)


def test_cap_aborts_before_overspend():
    from scripts.run_arms import effective_model_cap

    model = {"key": "sonnet", "budget": {"cap_usd": 10.0}}
    assert effective_model_cap(model, 7.5) == 7.5
    with pytest.raises(ValueError, match="cannot exceed the configured per-model cap"):
        effective_model_cap(model, 10.01)


def test_paid_suite_validation_accepts_only_canonical_full_common_and_delta_sets():
    from benchmarks import run as runner

    suite_dir = Path(__file__).parents[1] / "benchmarks" / "suites"
    def records(name):
        return [json.loads(line) for line in (suite_dir / name).read_text(encoding="utf-8").splitlines() if line.strip()]

    full = records("eval-v2.1.jsonl")
    common = records("eval-v2.1-common47.jsonl")
    delta = records("eval-v2.1-delta25.jsonl")
    assert runner.validate_paid_eval_suite(full) == "eval-v2.1"
    assert runner.validate_paid_eval_suite(common) == "eval-v2.1-common47"
    assert runner.validate_paid_eval_suite(delta) == "eval-v2.1-delta25"
    assert runner.validate_paid_eval_suite(list(reversed(full))) is None


def test_arm_coordinator_starts_all_processes_before_waiting(monkeypatch):
    from scripts.run_arms import launch_parallel

    events = []

    class Process:
        def __init__(self, command, cwd):
            events.append(("start", command[0]))
        def wait(self):
            events.append(("wait",))
            return 0

    monkeypatch.setattr("scripts.run_arms.subprocess.Popen", Process)

    assert launch_parallel([["arm-a"], ["arm-b"]]) == 0
    assert events == [("start", "arm-a"), ("start", "arm-b"), ("wait",), ("wait",)]


def test_budget_ledger_reservation_survives_restart_and_enforces_cap(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = CostLedger(path, caps={"haiku": 1.0})
    ledger.reserve("request-1", provider="haiku", amount_usd=.75)
    restarted = CostLedger(path, caps={"haiku": 1.0, "sonnet": 1.0})
    assert restarted.committed_usd("haiku") == pytest.approx(.75)
    with pytest.raises(BudgetExceeded):
        restarted.reserve("request-2", provider="haiku", amount_usd=.30)
    restarted.reserve("request-3", provider="sonnet", amount_usd=.90)


def test_failed_before_send_releases_reservation_and_keeps_exposure_unchanged(tmp_path):
    ledger_path = tmp_path / "fast-failure.jsonl"

    class FailedBeforeSend:
        def complete(self, _system, _prompt):
            exc = RuntimeError("local client was already closed")
            exc.provider_elapsed_ms = 2
            raise exc

    ledger = CostLedger(ledger_path, caps={"gpt-6-luna": 1.0})
    guarded = GuardedCompletionClient(
        FailedBeforeSend(), ledger=ledger, model_key="gpt-6-luna", provider="openai",
        model_price=ModelPrice("gpt-6-luna", "openai", .1, .5),
        max_input_bytes=1000, input_token_overhead=0, max_output_tokens=10)
    before = ledger.available_usd("gpt-6-luna")

    with pytest.raises(RuntimeError, match="already closed"):
        guarded.complete("system", "prompt")

    events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == ["reserve", "release"]
    assert events[-1]["reason"] == "provider_call_failed_before_send"
    assert events[-1]["evidence"]["elapsed_ms"] < 50
    assert events[-1]["evidence"]["usage_reported"] is False
    assert ledger.available_usd("gpt-6-luna") == pytest.approx(before)


def test_guard_retains_reservation_when_failed_call_may_have_reached_provider(tmp_path):
    ledger_path = tmp_path / "slow-failure.jsonl"

    class SlowFailure:
        def complete(self, _system, _prompt):
            exc = RuntimeError("request outcome is unknown")
            exc.provider_elapsed_ms = 80
            raise exc

    ledger = CostLedger(ledger_path, caps={"gpt-6-luna": 1.0})
    guarded = GuardedCompletionClient(
        SlowFailure(), ledger=ledger, model_key="gpt-6-luna", provider="openai",
        model_price=ModelPrice("gpt-6-luna", "openai", .1, .5),
        max_input_bytes=1000, input_token_overhead=0, max_output_tokens=10)

    with pytest.raises(RuntimeError):
        guarded.complete("system", "prompt")

    events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == ["reserve"]
    assert ledger.committed_usd("gpt-6-luna") == pytest.approx(events[0]["amount_usd"])


def test_resume_ignores_provider_error_rows_but_preserves_them_as_provenance(tmp_path):
    from benchmarks import run as runner

    path = tmp_path / "resume.jsonl"
    common = {"model_key": "fixture", "prompt_id": "task-1", "benchmark_version": runner.BENCHMARK_VERSION,
              "validator_version": runner.VALIDATOR_VERSION, "prompt_suite_sha256": "a" * 64,
              "model_id": "fixture-model", "provider": "test", "api_style": "test",
              "request_params": {}}
    good = dict(common, provider_error=None, attempt_count=1)
    failed = dict(common, prompt_id="task-2", provider_error={"type": "RuntimeError"}, attempt_count=0)
    path.write_text("\n".join(json.dumps(row) for row in (good, failed)) + "\n", encoding="utf-8")
    config = {"fixture": {"model_id": "fixture-model", "provider": "test", "api_style": "test",
                          "request_params": {}}}

    completed = runner.load_resume_pairs(path, config, ["task-1", "task-2"], "a" * 64)
    assert completed == {("fixture", "task-1")}
    recovered = dict(failed, provider_error=None, attempt_count=1)
    with path.open("a", encoding="utf-8") as target:
        target.write(json.dumps(recovered) + "\n")
    completed = runner.load_resume_pairs(path, config, ["task-1", "task-2"], "a" * 64)
    assert completed == {("fixture", "task-1"), ("fixture", "task-2")}
    retained = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(retained) == 3 and retained[1]["provider_error"]


def test_resume_releases_only_unmatched_fast_failed_reservations(tmp_path):
    from benchmarks import run as runner

    output_path = tmp_path / "arm.jsonl"
    ledger = CostLedger(tmp_path / "ledger.jsonl", caps={"gpt-6-luna": 1.0})
    ledger.reserve("fast-id", provider="gpt-6-luna", model_key="gpt-6-luna", amount_usd=.25)
    ledger.reserve("slow-id", provider="gpt-6-luna", model_key="gpt-6-luna", amount_usd=.25)
    def error_row(prompt_id, elapsed):
        return {"model_key": "gpt-6-luna", "prompt_id": prompt_id, "created_at": prompt_id,
                "elapsed_ms": elapsed,
                "provider_error": {"phase": "proposal", "type": "RuntimeError"},
                "model_calls": [{"error": {"type": "RuntimeError"}, "telemetry": {
                    "latency_ms": {"request_elapsed": elapsed},
                    "token_usage": {"prompt_tokens": None, "completion_tokens": None,
                                     "total_tokens": None}}}]}
    output_path.write_text("\n".join(json.dumps(error_row(pid, elapsed))
                                       for pid, elapsed in (("task-fast", 2), ("task-slow", 80))) + "\n",
                           encoding="utf-8")
    configs = {"gpt-6-luna": {"budget_group": "gpt-6-luna"}}

    assert runner.release_fast_failed_call_reservations(output_path, ledger, configs) == 1
    events = [json.loads(line) for line in ledger.path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == ["reserve", "reserve", "release"]
    assert events[-1]["reservation_id"] == "fast-id"
    assert ledger.committed_usd("gpt-6-luna") == pytest.approx(.25)


def test_resume_preserves_orphan_reservation_when_interrupted_run_has_no_result_row(tmp_path):
    from benchmarks import run as runner

    output_path = tmp_path / "arm.jsonl"
    ledger = CostLedger(tmp_path / "ledger.jsonl", caps={"gpt-6-luna": 1.0})
    ledger.reserve("slow-id", provider="gpt-6-luna", model_key="gpt-6-luna", amount_usd=.25)
    ledger.reserve("interrupted-id", provider="gpt-6-luna", model_key="gpt-6-luna", amount_usd=.25)
    failed = {"model_key": "gpt-6-luna", "prompt_id": "task-slow", "created_at": "task-slow",
              "elapsed_ms": 80, "provider_error": {"phase": "repair", "type": "RuntimeError"},
              "model_calls": [{"error": {"type": "RuntimeError"}, "telemetry": {
                  "latency_ms": {"request_elapsed": 6},
                  "token_usage": {"prompt_tokens": None, "completion_tokens": None,
                                  "total_tokens": None}}}]}
    output_path.write_text(json.dumps(failed) + "\n", encoding="utf-8")
    configs = {"gpt-6-luna": {"budget_group": "gpt-6-luna"}}

    assert runner.release_fast_failed_call_reservations(output_path, ledger, configs) == 0
    assert ledger.committed_usd("gpt-6-luna") == pytest.approx(.5)
    events = [json.loads(line) for line in ledger.path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == ["reserve", "reserve"]


def test_guard_only_starts_questions_with_room_for_three_maximum_requests(tmp_path):
    price = ModelPrice("haiku", "anthropic", .1, .5)
    enough = GuardedCompletionClient(
        object(), ledger=CostLedger(tmp_path / "enough.jsonl", caps={"haiku": .000045}),
        model_key="haiku", provider="anthropic", model_price=price,
        max_input_bytes=100, input_token_overhead=0, max_output_tokens=10)
    short = GuardedCompletionClient(
        object(), ledger=CostLedger(tmp_path / "short.jsonl", caps={"haiku": .000044}),
        model_key="haiku", provider="anthropic", model_price=price,
        max_input_bytes=100, input_token_overhead=0, max_output_tokens=10)
    assert enough.can_reserve_complete_question("x" * 100, "")
    assert not short.can_reserve_complete_question("x" * 100, "")


def test_guard_task_gate_uses_actual_request_size_with_full_output_reserve(tmp_path):
    price = ModelPrice("opus", "anthropic", 25.0, 75.0)
    ledger = CostLedger(tmp_path / "actual-size.jsonl", caps={"opus": 1.0})
    guarded = GuardedCompletionClient(
        object(), ledger=ledger, model_key="opus", provider="anthropic", model_price=price,
        max_input_bytes=100_000, input_token_overhead=256, max_output_tokens=4096)

    assert guarded.can_reserve_complete_question("system", "small task prompt")
    assert not guarded.can_reserve_complete_question("system", "x" * 20_000)


def test_community_eval_export_keeps_gallery_proof_and_removes_private_fields(tmp_path):
    tasks, suite_sha = load_suite("eval-v2.1")
    task = next(task for task in tasks if task.grammar == "periodic_raster")
    secret = "sk-test-never-publish"
    candidate = pattern_obj().model_dump(mode="json")
    checks, _, _ = verify_pattern(parse_pattern(json.dumps(candidate)))
    passed = all_pass(checks)
    row = {
        "model_key": "anthropic/claude-sonnet-4-5", "model_id": "claude-sonnet-4-5",
        "provider": "Anthropic", "api_style": "litellm-chat-completions",
        "request_params": {"temperature": 0.4, "max_tokens": 4096},
        "pricing": {"currency": "USD", "source": "test"},
        "resolved_model_ids": ["claude-sonnet-4-5-20261001"],
        "prompt_suite_sha256": suite_sha, "prompt_id": task.id,
        "suite_id": task.suite_id, "state": task.state, "tradition_id": task.tradition_id,
        "grammar": task.grammar, "level": task.level, "brief": task.brief,
        "symmetry": task.symmetry, "valid_schema_at_1": True,
        "validity_at_1": passed, "validity_at_3": passed, "repair_rescued": False,
        "accepted_mask_sha256": "a" * 64, "max_attempts": 3, "attempt_count": 1,
        "elapsed_ms": 1234, "total_tokens": 20,
        "token_usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
        "cost": {"amount": .00002, "currency": "USD", "source": "rate_card_estimate"},
        "latency_ms": {"request_total": 1000, "verification_total": 12},
        "provider_attempts": 1, "provider_retries": 0, "verification_elapsed_ms": 12,
        "provider_error": secret,
        "attempts": [{"index": 1, "candidate": candidate, "raw": secret,
                      "report": {"all_hard_constraints_pass": passed, "checks": checks},
                      "verified_at": "2026-10-30T00:00:00Z", "metadata": {"api_key": secret}}],
        "model_calls": [{"messages": [{"role": "user", "content": secret}],
                         "raw_response": secret, "metadata": {"authorization": secret}}],
    }

    exported = sanitize_community_eval_rows([row], run_id="run-123", expected_suite_sha256=suite_sha,
                                             provider_api_key=secret, submitted_by="kshitijthakkar")
    serialized = json.dumps(exported)
    assert secret not in serialized
    assert exported[0]["data_role"] == "community_evaluation"
    assert exported[0]["training_eligible"] is False
    assert exported[0]["submitted_by"] == "kshitijthakkar"
    assert exported[0]["attempts"][0]["candidate"] == row["attempts"][0]["candidate"]
    assert exported[0]["attempts"][0]["report"] == row["attempts"][0]["report"]
    assert "model_calls" not in exported[0]
    assert "provider_error" not in exported[0]
    resanitized = sanitize_community_eval_rows(
        exported, run_id="run-123", expected_suite_sha256=suite_sha, provider_api_key=secret,
        submitted_by="kshitijthakkar")
    assert resanitized[0]["run_id"] == "run-123"
    assert resanitized[0]["model_key"] == "anthropic/claude-sonnet-4-5"
    assert resanitized[0]["model_id"] == "claude-sonnet-4-5"
    assert secret not in json.dumps(resanitized)


def test_community_eval_upload_uses_space_token_and_unique_dataset_path(tmp_path, monkeypatch):
    tasks, suite_sha = load_suite("eval-v2.1")
    task = tasks[0]
    row = {
        "model_key": "deepseek-v4.1-flash", "model_id": "deepseek-v4.1-flash",
        "provider": "CometAPI", "api_style": "openai-compatible-chat-completions",
        "request_params": {"temperature": .4, "max_tokens": 4096,
                        "extra_body": {"thinking": {"type": "disabled"}}},
        "prompt_suite_sha256": suite_sha, "prompt_id": task.id,
        "suite_id": task.suite_id, "state": task.state, "tradition_id": task.tradition_id,
        "grammar": task.grammar, "level": task.level, "brief": task.brief,
        "symmetry": task.symmetry, "valid_schema_at_1": False,
        "validity_at_1": False, "validity_at_3": False, "repair_rescued": False,
        "attempts": [{"index": 1, "candidate": None,
                      "report": {"all_hard_constraints_pass": False,
                                 "checks": [{"id": "candidate_parse", "status": "FAIL"}]}}],
        "model_calls": [],
    }

    class FakeHub:
        def __init__(self): self.calls = []
        def create_repo(self, **kwargs): self.calls.append(("create_repo", kwargs))
        def list_repo_files(self, **kwargs): return []
        def upload_file(self, **kwargs):
            self.calls.append(("upload_file", {**kwargs, "data": kwargs["path_or_fileobj"].read()}))

    api = FakeHub()
    monkeypatch.setenv("HF_COMMUNITY_WRITE_TOKEN", "hf-secret")
    result = publish_community_eval_run([row], run_id="run-123", api=api,
                                        submitted_by="kshitijthakkar", expected_suite_sha256=suite_sha,
                                        provider_api_key="sk-private-provider-key")
    upload = next(payload for name, payload in api.calls
                  if name == "upload_file" and payload["path_in_repo"].endswith(".jsonl"))
    assert result["path_in_repo"] == "community/test/run-123.jsonl"
    assert upload["repo_id"] == "kshitijthakkar/pramaan-community-runs"
    assert upload["repo_type"] == "dataset"
    assert b"hf-secret" not in upload["data"]
    assert json.loads(upload["data"])["run_id"] == "run-123"
    assert any(payload["path_in_repo"].startswith("community/index/")
               for name, payload in api.calls if name == "upload_file")


def test_v21_community_eval_uses_separate_dataset_and_anthropic_protocol(monkeypatch):
    from scripts.push_hub_datasets import COMMUNITY_REPO, publish_community_eval_run

    tasks, suite_sha = load_suite("eval-v2.1")
    task = tasks[0]
    row = {
        "model_key": "claude-haiku-5-5", "model_id": "claude-haiku-5-5",
        "provider": "Anthropic", "api_style": "anthropic-messages",
        "request_params": {"max_tokens": 4096, "output_config": {"effort": "low"}},
        "prompt_suite_sha256": suite_sha, "prompt_id": task.id,
        "suite_id": task.suite_id, "state": task.state, "tradition_id": task.tradition_id,
        "grammar": task.grammar, "level": task.level, "brief": task.brief,
        "symmetry": task.symmetry, "valid_schema_at_1": False,
        "validity_at_1": False, "validity_at_3": False, "repair_rescued": False,
        "attempts": [{"index": 1, "candidate": None,
                      "report": {"all_hard_constraints_pass": False,
                                 "checks": [{"id": "candidate_parse", "status": "FAIL"}]}}],
        "model_calls": [],
    }

    class FakeHub:
        def __init__(self): self.calls = []
        def create_repo(self, **kwargs): self.calls.append(("create_repo", kwargs))
        def list_repo_files(self, **kwargs): return []
        def upload_file(self, **kwargs):
            self.calls.append(("upload_file", {**kwargs, "data": kwargs["path_or_fileobj"].read()}))

    api = FakeHub()
    monkeypatch.setenv("HF_TOKEN", "hf-space-token")
    result = publish_community_eval_run([row], run_id="run-v21", api=api,
                                        submitted_by="test-user", expected_suite_sha256=suite_sha,
                                        provider_api_key="sk-user-key")
    assert result["repo_id"] == COMMUNITY_REPO
    assert all(payload["repo_id"] == COMMUNITY_REPO for name, payload in api.calls
               if name == "upload_file")
    card = next(payload["data"].decode() for name, payload in api.calls
                if name == "upload_file" and payload["path_in_repo"] == "README.md")
    assert "suite identifier and SHA-256" in card and "HF_COMMUNITY_WRITE_TOKEN" in card
    assert "temperature" not in json.dumps(json.loads(next(payload["data"] for name, payload in api.calls
        if name == "upload_file" and payload["path_in_repo"].endswith(".jsonl"))))
    assert "hf-space-token" not in json.dumps(api.calls, default=str)


def test_community_upload_error_redacts_space_and_provider_credentials(monkeypatch):
    tasks, suite_sha = load_suite("eval-v2.1")
    task = tasks[0]
    row = {
        "model_key": "claude-haiku-5-5", "model_id": "claude-haiku-5-5",
        "provider": "Anthropic", "api_style": "anthropic-messages",
        "request_params": {"max_tokens": 4096, "output_config": {"effort": "low"}},
        "prompt_suite_sha256": suite_sha, "prompt_id": task.id,
        "suite_id": task.suite_id, "state": task.state, "tradition_id": task.tradition_id,
        "grammar": task.grammar, "level": task.level, "brief": task.brief,
        "symmetry": task.symmetry, "valid_schema_at_1": False,
        "validity_at_1": False, "validity_at_3": False,
        "attempts": [{"index": 1, "candidate": None,
                      "report": {"all_hard_constraints_pass": False,
                                 "checks": [{"id": "candidate_parse", "status": "FAIL"}]}}],
    }

    class FailingHub:
        def create_repo(self, **kwargs):
            pass

        def list_repo_files(self, **kwargs):
            return []

        def upload_file(self, **kwargs):
            raise RuntimeError("upload failed hf-private-token sk-provider-private")

    monkeypatch.setenv("HF_COMMUNITY_WRITE_TOKEN", "hf-private-token")
    with pytest.raises(RuntimeError) as caught:
        publish_community_eval_run([row], run_id="run-safe",
                                   api=FailingHub(), expected_suite_sha256=suite_sha,
                                   provider_api_key="sk-provider-private", submitted_by="test-user")
    assert "hf-private-token" not in str(caught.value)
    assert "sk-provider-private" not in str(caught.value)
    assert str(caught.value).count("[REDACTED]") == 2


def test_community_upload_requires_space_hf_token_not_a_cached_cli_token(monkeypatch):
    from scripts import push_hub_datasets

    tasks, suite_sha = load_suite("eval-v2.1")
    task = tasks[0]
    row = {
        "model_key": "deepseek-v4.1-flash", "model_id": "deepseek-v4.1-flash",
        "provider": "CometAPI", "api_style": "openai-compatible-chat-completions",
        "request_params": {"temperature": .4, "max_tokens": 4096,
                        "extra_body": {"thinking": {"type": "disabled"}}},
        "prompt_suite_sha256": suite_sha, "prompt_id": task.id,
        "suite_id": task.suite_id, "state": task.state, "tradition_id": task.tradition_id,
        "grammar": task.grammar, "level": task.level, "brief": task.brief,
        "symmetry": task.symmetry, "valid_schema_at_1": False,
        "validity_at_1": False, "validity_at_3": False,
        "attempts": [{"index": 1, "candidate": None,
                      "report": {"all_hard_constraints_pass": False,
                                 "checks": [{"id": "candidate_parse", "status": "FAIL"}]}}],
    }
    monkeypatch.delenv("HF_COMMUNITY_WRITE_TOKEN", raising=False)
    monkeypatch.setattr(push_hub_datasets, "get_token", lambda: "cached-cli-token")
    monkeypatch.setattr(push_hub_datasets, "HfApi",
                        lambda **kwargs: pytest.fail("community upload must not use cached CLI credentials"))

    with pytest.raises(RuntimeError, match="Space HF_COMMUNITY_WRITE_TOKEN"):
        push_hub_datasets.publish_community_eval_run(
            [row], run_id="run-no-space-token", expected_suite_sha256=suite_sha, submitted_by="test-user")


@pytest.mark.parametrize(("model_key", "expected_params"), [
    ("anthropic/claude-3-5-haiku-20241022", {"temperature": 0.4, "max_tokens": 4096}),
    ("deepseek/deepseek-chat", {"temperature": 0.4, "max_tokens": 4096}),
])
def test_community_evaluator_uses_transient_key_caps_per_run_and_stops_at_budget(
        monkeypatch, model_key, expected_params):
    from benchmarks import run as runner

    suite_digest = "b" * 64

    class Task:
        def __init__(self, task_id): self.task_id = task_id
        def model_dump(self, mode="json"): return {"id": self.task_id, "brief": self.task_id}

    class FakeClient:
        def __init__(self, api_key, ledger):
            self.api_key, self.ledger, self.checks = api_key, ledger, 0
        def can_reserve_complete_question(self, *_args, **_kwargs):
            self.checks += 1
            return self.checks <= 2

    captured = {}
    client = None
    sanitized = {}

    def fake_make_client(config, *, ledger, api_key):
        nonlocal client
        captured["config"] = config
        client = FakeClient(api_key, ledger)
        return client

    from types import SimpleNamespace
    monkeypatch.setitem(__import__("sys").modules, "litellm", SimpleNamespace(
        get_model_info=lambda *_args, **_kwargs: {
            "input_cost_per_token": 0.000001, "output_cost_per_token": 0.000002}))
    monkeypatch.setattr(runner, "_make_litellm_community_client", fake_make_client)
    monkeypatch.setattr(runner, "run_prompt", lambda prompt, **kwargs: {
        "prompt_id": prompt["id"], "raw_response": "private provider completion",
    })
    def fake_sanitize(rows, *, run_id, expected_suite_sha256, provider_api_key):
        sanitized.update(run_id=run_id, digest=expected_suite_sha256, key=provider_api_key,
                         raw_rows=rows)
        return [{"prompt_id": row["prompt_id"], "run_id": run_id,
                 "candidate": {"title": "safe gallery row"}} for row in rows]
    monkeypatch.setattr("scripts.push_hub_datasets.sanitize_community_eval_rows", fake_sanitize)
    def fake_load_suite(name):
        captured["suite"] = name
        return [Task("q1"), Task("q2"), Task("q3")], suite_digest
    monkeypatch.setattr("pramaan.tasks.load_suite", fake_load_suite)
    progress = []

    rows, digest = runner.run_community_evaluation(
        model_key, "sk-never-write-this", progress=lambda done, total: progress.append((done, total)))

    assert [row["prompt_id"] for row in rows] == ["q1", "q2"]
    assert all("sk-never-write-this" not in json.dumps(row) for row in rows)
    assert all("raw_response" not in row for row in rows)
    import uuid
    assert str(uuid.UUID(sanitized["run_id"])) == sanitized["run_id"]
    assert sanitized["key"] == "sk-never-write-this"
    assert all(row["raw_response"] == "private provider completion" for row in sanitized["raw_rows"])
    assert client.api_key == "sk-never-write-this"
    assert client.ledger.caps == {model_key: 10.0}
    assert captured["config"]["request_params"] == expected_params
    assert captured["suite"] == "eval-v2.1"
    assert digest == suite_digest
    assert progress == [(1, 3), (2, 3)]
    assert not client.ledger.path.exists(), "per-run ledger must be removed after the evaluation"


def _write_official_community_usage_fixture(path, *, duplicate_task=False, missing_usage_at=None,
                                            stale_digest=False):
    tasks, digest = load_suite("eval-v2.1")
    rows = []
    for index, task in enumerate(tasks):
        calls = [
            {"telemetry": {"token_usage": {"prompt_tokens": 100, "completion_tokens": 10}}},
            {"telemetry": {"token_usage": {"prompt_tokens": 50, "completion_tokens": 20}}},
        ]
        if index == missing_usage_at:
            calls = []
        prompt_id = tasks[0].id if duplicate_task and index == 1 else task.id
        rows.append({"suite_id": "eval-v2.1", "prompt_id": prompt_id,
                     "prompt_suite_sha256": "0" * 64 if stale_digest else digest,
                     "calls": calls})
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return digest


def test_community_full_run_cost_estimate_uses_official_mean_usage(tmp_path, monkeypatch):
    from benchmarks import run as runner
    path = tmp_path / "official.jsonl"
    suite_digest = _write_official_community_usage_fixture(path)
    monkeypatch.setattr(runner, "community_litellm_model_config", lambda _model: {
        "pricing": {"per_million_tokens": {"input": 1, "output": 2}}})
    estimate = runner.estimate_community_evaluation_cost("openai/gpt-4o-mini", 2, trajectory_path=path)
    assert estimate["mean_input_tokens_per_task"] == 150
    assert estimate["mean_output_tokens_per_task"] == 30
    assert estimate["estimated_usd"] == 0.00042
    assert estimate["reference_suite"] == "eval-v2.1"
    assert estimate["reference_tasks"] == 72
    assert estimate["reference_suite_sha256"] == suite_digest
    assert estimate["estimated_cost_exceeds_per_run_cap"] is False
    assert estimate["per_run_cap_usd"] == 10.0
    monkeypatch.setattr(runner, "community_litellm_model_config", lambda _model: {
        "pricing": {"per_million_tokens": {"input": 100000, "output": 100000}}})
    expensive = runner.estimate_community_evaluation_cost("anthropic/claude-fable-5-1", 72, trajectory_path=path)
    assert expensive["estimated_cost_exceeds_per_run_cap"] is True
    assert expensive["per_run_cap_usd"] == 10.0


def test_community_cost_estimate_rejects_duplicate_official_task_ids(tmp_path, monkeypatch):
    from benchmarks import run as runner
    path = tmp_path / "duplicate.jsonl"
    _write_official_community_usage_fixture(path, duplicate_task=True)
    monkeypatch.setattr(runner, "community_litellm_model_config", lambda _model: {
        "pricing": {"per_million_tokens": {"input": 1, "output": 2}}})
    with pytest.raises(ValueError, match="duplicate task ID"):
        runner.estimate_community_evaluation_cost("openai/gpt-4o-mini", trajectory_path=path)


def test_community_cost_estimate_requires_measured_usage_for_every_task(tmp_path, monkeypatch):
    from benchmarks import run as runner
    path = tmp_path / "missing-usage.jsonl"
    _write_official_community_usage_fixture(path, missing_usage_at=17)
    monkeypatch.setattr(runner, "community_litellm_model_config", lambda _model: {
        "pricing": {"per_million_tokens": {"input": 1, "output": 2}}})
    with pytest.raises(ValueError, match="measured token usage"):
        runner.estimate_community_evaluation_cost("openai/gpt-4o-mini", trajectory_path=path)


def test_community_cost_estimate_rejects_stale_suite_digest(tmp_path, monkeypatch):
    from benchmarks import run as runner
    path = tmp_path / "stale-digest.jsonl"
    _write_official_community_usage_fixture(path, stale_digest=True)
    monkeypatch.setattr(runner, "community_litellm_model_config", lambda _model: {
        "pricing": {"per_million_tokens": {"input": 1, "output": 2}}})
    with pytest.raises(ValueError, match="suite digest"):
        runner.estimate_community_evaluation_cost("openai/gpt-4o-mini", trajectory_path=path)


def test_community_runner_defaults_to_quick_run_and_requires_full_cost_ack(monkeypatch):
    from benchmarks import run as runner
    monkeypatch.setattr(runner, "community_litellm_model_config",
                        lambda _model: pytest.fail("must reject before provider config"))
    with pytest.raises(ValueError, match="explicit full-run cost acknowledgment"):
        runner.run_community_evaluation("openai/gpt-4o-mini", "sk-test", task_limit=72)


def test_community_runner_limits_quick_run_to_twelve_tasks(monkeypatch):
    from pathlib import Path
    from benchmarks import run as runner
    captured = {}

    class Task:
        def __init__(self, i): self.i = i
        def model_dump(self, mode="json"): return {"id": str(self.i), "brief": "test prompt"}

    class Client:
        def can_reserve_complete_question(self, *_args, **_kwargs): return True
        ledger = type("Ledger", (), {"path": Path("missing-ledger"), "caps": {}})()

    monkeypatch.setattr(runner, "community_litellm_model_config", lambda _model: {"budget": {}})
    monkeypatch.setattr("pramaan.tasks.load_suite", lambda _name: ([Task(i) for i in range(72)], "a" * 64))
    monkeypatch.setattr(runner, "_make_litellm_community_client", lambda *_a, **_k: Client())
    monkeypatch.setattr(runner, "run_prompt", lambda prompt, **_k: {"prompt_id": prompt["id"]})
    monkeypatch.setattr("scripts.push_hub_datasets.sanitize_community_eval_rows",
                        lambda rows, **_k: captured.setdefault("rows", rows))
    rows, _ = runner.run_community_evaluation("openai/gpt-4o-mini", "sk-test")
    assert len(rows) == 12
    assert [r["prompt_id"] for r in rows] == [str(i) for i in range(12)]


def test_multi_model_summary_scores_each_model_on_common_completed_questions():
    from benchmarks.summarize import summarize

    rows = [
        {"model_key": "a", "prompt_id": "q1", "attempt_count": 1, "elapsed_ms": 0,
         "validity_at_1": True, "validity_at_3": True},
        {"model_key": "a", "prompt_id": "q2", "attempt_count": 1, "elapsed_ms": 0,
         "validity_at_1": False, "validity_at_3": False},
        {"model_key": "b", "prompt_id": "q1", "attempt_count": 1, "elapsed_ms": 0,
         "validity_at_1": False, "validity_at_3": False},
        {"model_key": "b", "prompt_id": "q3", "attempt_count": 1, "elapsed_ms": 0,
         "validity_at_1": True, "validity_at_3": True},
    ]

    result = summarize(rows, expected_prompt_ids=["q1", "q2", "q3"],
                       expected_model_keys=["a", "b"])

    assert result["common_completed"]["prompt_ids"] == ["q1"]
    assert result["common_completed"]["prompt_count"] == 1
    assert result["common_completed"]["by_model"]["a"]["n"] == 1
    assert result["common_completed"]["by_model"]["a"]["validity_at_1"] == 1.0
    assert result["common_completed"]["by_model"]["b"]["n"] == 1
    assert result["common_completed"]["by_model"]["b"]["validity_at_1"] == 0.0


def test_community_dataset_card_exposes_a_separate_public_runs_config():
    card = dataset_card("eval", include_community=True)
    assert "config_name: community" in card
    assert "community/test/*.jsonl" in card
    assert "config_name: community" not in dataset_card("eval")
    assert "config_name: community" in dataset_card("eval", include_community=True, suite_name="eval-v2.1")


def test_eval_v21_reference_bundle_stages_verified_references_for_all_36_states():
    root = Path(__file__).resolve().parents[1]
    stage = stage_reference_bundle(
        root / "benchmarks/data/eval-v2.1-references.jsonl",
        root / "benchmarks/suites/eval-v2.1.jsonl",
        root / "assets/reference-thumbnails",
    )
    rows = [json.loads(line) for line in (stage / "references/test.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 71
    assert len({row["state"] for row in rows}) == 36
    assert all(row["all_hard_constraints_pass"] and row["data_role"] == "reference"
               and row["training_eligible"] is False for row in rows)
    assert len(list((stage / "references/thumbnails").glob("*.png"))) == 71
    card = (stage / "README.md").read_text(encoding="utf-8")
    assert "config_name: references" in card
    assert card.count("config_name: references") == 1
    assert "36 states and union territories" in card
    assert "not benchmark scores" in card


def test_reference_bundle_rejects_missing_state_coverage(tmp_path):
    root = Path(__file__).resolve().parents[1]
    refs = [json.loads(line) for line in (root / "benchmarks/data/eval-v2.1-references.jsonl").read_text(encoding="utf-8").splitlines()]
    missing_state = next(row["state"] for row in refs if sum(other["state"] == row["state"] for other in refs) == 1)
    refs = [row for row in refs if row["state"] != missing_state]
    refs_path = tmp_path / "references.jsonl"
    refs_path.write_text("".join(json.dumps(row) + "\n" for row in refs), encoding="utf-8")
    thumbs = tmp_path / "thumbs"
    thumbs.mkdir()
    for row in refs:
        (thumbs / f"{row['task_id']}.png").write_bytes(b"png")
    with pytest.raises(ValueError, match="all 36 states"):
        stage_reference_bundle(refs_path, root / "benchmarks/suites/eval-v2.1.jsonl", thumbs)


def test_user_provider_key_is_redacted_from_provider_errors():
    secret = "sk-provider-private"

    class FailingProvider:
        def complete(self, system, prompt):
            exc = RuntimeError(f"provider echoed {secret}")
            exc.provider_attempts = 1
            raise exc

    client = SecretRedactingClient(FailingProvider(), secret)
    with pytest.raises(RuntimeError) as caught:
        client.complete("system", "prompt")
    assert secret not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)
    assert caught.value.provider_attempts == 1


def test_provider_error_redaction_keeps_only_safe_probe_metadata():
    secret = "sk-provider-private"

    class FailingProvider:
        def complete(self, system, prompt):
            exc = RuntimeError(f"provider echoed {secret}")
            exc.status_code = 400
            exc.request_id = "req_safe_123"
            exc.provider_attempts = 1
            raise exc

    client = SecretRedactingClient(FailingProvider(), secret)
    with pytest.raises(RuntimeError) as caught:
        client.complete("system", "synthetic probe")
    assert secret not in str(caught.value)
    assert caught.value.provider_error_type == "RuntimeError"
    assert caught.value.http_status_code == 400
    assert caught.value.request_id == "req_safe_123"


def test_user_provider_key_is_redacted_from_provider_client_initialization_errors(monkeypatch):
    from benchmarks import run as runner

    secret = "sk-init-private"
    config = next(item for item in runner.PAID_MODELS if item["key"] == "claude-haiku-5-5")

    def fail_init(**kwargs):
        raise ValueError(f"provider client rejected {secret}")

    monkeypatch.setattr(runner, "AnthropicMessagesClient", fail_init)
    with pytest.raises(RuntimeError) as caught:
        runner.make_client(config, api_key=secret)

    assert secret not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)


def test_make_client_accepts_a_transient_user_key_without_mutating_model_config(tmp_path):
    from pathlib import Path
    key = "sk-transient-user-key"
    config = load_model_registry(Path(__file__).parents[1] / "benchmarks" / "models.json",
                                 include_paid=True)["claude-haiku-5-5"]
    original = json.loads(json.dumps(config))
    client = make_client(config, ledger=CostLedger(tmp_path / "ledger.jsonl",
                                                   caps={config["key"]: 10.0}), api_key=key)
    assert client.client.secret == key
    assert config == original
    assert key not in json.dumps(config)


def test_guarded_client_blocks_long_prompt_before_provider_or_ledger(tmp_path):
    class NoCall:
        calls = 0

        def complete(self, system, prompt):
            self.calls += 1
            return "unused", {}

    client = NoCall()
    ledger_path = tmp_path / "ledger.jsonl"
    guarded = GuardedCompletionClient(
        client, ledger=CostLedger(ledger_path, caps={"haiku": 1}), model_key="haiku",
        provider="anthropic", model_price=ModelPrice("haiku", "anthropic", .1, .5),
        max_input_bytes=3, input_token_overhead=0, max_output_tokens=10)
    with pytest.raises(BudgetExceeded):
        guarded.complete("é", "abc")
    assert client.calls == 0
    assert not ledger_path.exists()


def test_raised_paid_input_guard_sends_repair_sized_prompt_but_refuses_over_100kb(tmp_path):
    class FakeProvider:
        calls = 0

        def complete(self, system, prompt):
            self.calls += 1
            return "ok", {"usage": {"input_tokens": 1, "output_tokens": 1}}

    client = FakeProvider()
    guarded = GuardedCompletionClient(
        client, ledger=CostLedger(tmp_path / "ledger.jsonl", caps={"haiku": 10}),
        model_key="haiku", provider="anthropic",
        model_price=ModelPrice("haiku", "anthropic", .1, .5),
        max_input_bytes=100000, input_token_overhead=256, max_output_tokens=4096)
    response, _metadata = guarded.complete("system", "x" * 24000)
    assert response == "ok"
    assert client.calls == 1
    with pytest.raises(BudgetExceeded, match="cap is 100000"):
        guarded.complete("system", "x" * 100001)
    assert client.calls == 1


def test_guard_reserves_before_request_and_settles_reported_usd(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"

    class InspectLedger:
        def complete(self, system, prompt):
            events = [json.loads(line) for line in ledger_path.read_text().splitlines()]
            assert events[0]["event"] == "reserve"
            return "ok", {"usage": {"input_tokens": 2, "output_tokens": 3},
                           "cost": {"amount": .000004, "currency": "USD"}}

    guarded = GuardedCompletionClient(
        InspectLedger(), ledger=CostLedger(ledger_path, caps={"haiku": 1}), model_key="haiku",
        provider="anthropic", model_price=ModelPrice("haiku", "anthropic", .1, .5),
        max_input_bytes=100, input_token_overhead=0, max_output_tokens=10)
    content, metadata = guarded.complete("a", "b")
    events = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert content == "ok"
    assert [event["event"] for event in events] == ["reserve", "settle"]
    assert metadata["budget_guard"]["settlement_source"] == "provider_reported"
    assert metadata["budget_guard"]["settled_usd"] == .000004


def test_missing_usage_is_reconciled_as_full_reservation(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"

    class MissingUsage:
        def complete(self, system, prompt):
            return "ok", {}

    guarded = GuardedCompletionClient(
        MissingUsage(), ledger=CostLedger(ledger_path, caps={"haiku": 1}), model_key="haiku",
        provider="anthropic", model_price=ModelPrice("haiku", "anthropic", .1, .5),
        max_input_bytes=100, input_token_overhead=0, max_output_tokens=10)
    _, metadata = guarded.complete("a", "b")
    events = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert [event["event"] for event in events] == ["reserve", "settle"]
    assert events[-1]["usage_reported"] is False
    assert events[-1]["amount_usd"] == metadata["budget_guard"]["reserved_usd"]
    assert metadata["budget_guard"]["settlement_source"] == "reservation_retained"


def test_anthropic_adapter_normalizes_usage_and_resolves_model_without_retries(monkeypatch):
    from types import SimpleNamespace

    captured = {"calls": 0}
    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.messages = SimpleNamespace(create=self.create)
        def create(self, **kwargs):
            captured["calls"] += 1
            captured["params"] = kwargs
            return SimpleNamespace(
                model="claude-haiku-5-5-20261001", stop_reason="end_turn",
                _request_id="req_safe_456",
                content=[SimpleNamespace(type="text", text="{}")],
                usage=SimpleNamespace(input_tokens=11, output_tokens=7,
                                      cache_read_input_tokens=2, cache_creation_input_tokens=0),
            )

    monkeypatch.setattr("anthropic.Anthropic", FakeAnthropic)
    client = AnthropicMessagesClient(token="test-secret", model_id="claude-haiku-5-5", provider="Anthropic",
        base_url="https://api.anthropic.com", request_params={
            "max_tokens": 4096, "output_config": {"effort": "low"}})
    content, metadata = client.complete("sys", "user")
    assert captured["client_kwargs"] == {
        "api_key": "test-secret", "base_url": "https://api.anthropic.com",
        "max_retries": 0, "timeout": 75.0,
    }
    assert captured["calls"] == 1
    assert captured["params"] == {
        "model": "claude-haiku-5-5", "max_tokens": 4096, "system": "sys",
        "messages": [{"role": "user", "content": "user"}],
        "output_config": {"effort": "low"},
    }
    assert metadata["request_params"] == {"max_tokens": 4096, "output_config": {"effort": "low"}}
    assert metadata["content_block_types"] == ["text"]
    assert metadata["resolved_model_id"] == "claude-haiku-5-5-20261001"
    assert metadata["request_id"] == "req_safe_456"
    assert metadata["usage"]["total_tokens"] == 18
    assert metadata["provider_attempts"] == 1
    assert content == "{}"


def test_anthropic_max_tokens_stop_maps_to_length(monkeypatch):
    from types import SimpleNamespace

    class FakeAnthropic:
        def __init__(self, **_kwargs):
            self.messages = SimpleNamespace(create=lambda **_params: SimpleNamespace(
                model="claude-haiku-5-5-20261001", stop_reason="max_tokens", _request_id="req_len",
                content=[SimpleNamespace(type="text", text='{"partial":')],
                usage=SimpleNamespace(input_tokens=10, output_tokens=4096),
            ))

    monkeypatch.setattr("anthropic.Anthropic", FakeAnthropic)
    client = AnthropicMessagesClient(token="test-secret", model_id="claude-haiku-5-5", provider="Anthropic",
        base_url="https://api.anthropic.com", request_params={"max_tokens": 4096})

    _text, metadata = client.complete("sys", "user")

    assert metadata["finish_reason"] == "length"


def test_anthropic_refusal_raises_bounded_error_with_category(monkeypatch):
    from types import SimpleNamespace
    from benchmarks.run import AnthropicRefusalError

    class FakeAnthropic:
        def __init__(self, **_kwargs):
            self.messages = SimpleNamespace(create=lambda **_params: SimpleNamespace(
                model="claude-opus-5-5-20261001", stop_reason="refusal", _request_id="req_refusal",
                stop_details=SimpleNamespace(category="safety"),
                content=[SimpleNamespace(type="text", text="provider refusal details are not exposed")],
                usage=SimpleNamespace(input_tokens=10, output_tokens=4),
            ))

    monkeypatch.setattr("anthropic.Anthropic", FakeAnthropic)
    client = AnthropicMessagesClient(token="test-secret", model_id="claude-opus-5-5", provider="Anthropic",
        base_url="https://api.anthropic.com", request_params={"max_tokens": 4096})

    with pytest.raises(AnthropicRefusalError) as caught:
        client.complete("sys", "user")

    assert caught.value.category == "safety"
    assert "provider refusal details" not in str(caught.value)


def test_anthropic_adapter_rejects_temperature_setting():
    with pytest.raises(ValueError, match="does not accept temperature"):
        AnthropicMessagesClient(token="test-secret", model_id="claude-haiku-5-5", provider="Anthropic",
            base_url="https://api.anthropic.com", request_params={"max_tokens": 16, "temperature": .4})


def test_anthropic_model_registry_rejects_temperature(tmp_path):
    from benchmarks.run import load_model_registry

    path = tmp_path / "models.json"
    path.write_text(json.dumps({"schema_version": 1, "models": [{
        "key": "anthropic-probe", "provider": "Anthropic", "api_style": "anthropic-messages",
        "model_id": "claude-probe", "base_url": "https://api.anthropic.com",
        "api_key_env": "ANTHROPIC_API_KEY",
        "request_params": {"temperature": 0.4, "max_tokens": 64},
    }]}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not accept temperature"):
        load_model_registry(path)


def test_registry_accepts_extra_body_thinking(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"schema_version": 1, "models": [{
        "key": "deepseek-test", "provider": "CometAPI", "api_style": "openai-compatible-chat-completions",
        "model_id": "deepseek-v4.1-flash", "base_url": "https://api.cometapi.com/v1",
        "api_key_env": "COMETAPI_KEY",
        "request_params": {"temperature": 0.4, "max_tokens": 4096,
                            "extra_body": {"thinking": {"type": "disabled"}}},
    }]}), encoding="utf-8")

    model = load_model_registry(path)["deepseek-test"]

    assert model["request_params"]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_registry_still_rejects_model_messages_stream_overrides(tmp_path):
    for forbidden in ("model", "messages", "stream"):
        path = tmp_path / f"models-{forbidden}.json"
        path.write_text(json.dumps({"schema_version": 1, "models": [{
            "key": "openai-compatible-test", "provider": "CometAPI",
            "api_style": "openai-compatible-chat-completions", "model_id": "test-model",
            "base_url": "https://api.cometapi.com/v1", "api_key_env": "COMETAPI_KEY",
            "request_params": {forbidden: [] if forbidden == "messages" else "bad"},
        }]}), encoding="utf-8")
        with pytest.raises(ValueError, match="cannot override model/messages or enable streaming"):
            load_model_registry(path)


def test_comet_adapter_disables_retries_and_records_upstream_alias(monkeypatch):
    from types import SimpleNamespace

    captured = {"calls": 0}
    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        def create(self, **kwargs):
            captured["calls"] += 1
            captured["request_params"] = kwargs
            return SimpleNamespace(model="deepseek-v4.1-flash-1017",
                choices=[SimpleNamespace(message=SimpleNamespace(content="{}"), finish_reason="stop")],
                usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 2, "completion_tokens": 1,
                                                          "total_tokens": 3}))

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    client = CometOpenAICompatibleClient(token="test-secret", model_id="deepseek-v4.1-flash",
        provider="CometAPI", base_url="https://api.cometapi.com/v1",
        request_params={"temperature": .4, "max_tokens": 4096,
                        "extra_body": {"thinking": {"type": "disabled"}}})
    content, metadata = client.complete("sys", "user")
    assert captured["client_kwargs"]["max_retries"] == 0
    assert captured["calls"] == metadata["provider_attempts"] == 1
    assert metadata["resolved_model_id"] == "deepseek-v4.1-flash-1017"
    assert captured["request_params"]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert metadata["request_params"]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert content == "{}"


def test_paid_models_are_available_only_to_explicit_paid_run_registry():
    from pathlib import Path
    registry_path = Path(__file__).parents[1] / "benchmarks" / "models.json"
    assert list(load_model_registry(registry_path)) == ["beam-hf"]
    paid = load_model_registry(registry_path, include_paid=True)
    assert {"claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5",
            "claude-fable-5-1", "deepseek-v4.1-flash", "gpt-6.1-sol", "gpt-6.1-terra",
            "gpt-6-luna", "gpt-6-astra"}.issubset(paid)
    assert paid["claude-fable-5-1"]["model_id"] == "claude-fable-5-1"
    assert all(item["budget_group"] == item["key"] for item in paid.values() if item.get("budget_provider"))
    assert all(item["budget"]["cap_usd"] == 10.0 for item in paid.values() if item.get("budget_provider"))
    assert paid["deepseek-v4.1-flash"]["budget"]["cap_usd"] == 10.0
    from benchmarks.run import output_token_limit
    assert all(output_token_limit(item["request_params"]) == 4096
               for item in paid.values() if item.get("budget_provider"))
    expected_openai = {
        "gpt-6.1-sol": ("low", 2.0, 10.0, 0.10),
        "gpt-6.1-terra": ("none", 1.0, 5.0, 0.05),
        "gpt-6-luna": ("none", 0.10, 0.50, 0.01),
        "gpt-6-astra": ("low", 10.0, 50.0, 1.0),
    }
    for key, (effort, input_rate, output_rate, cached_rate) in expected_openai.items():
        item = paid[key]
        assert item["api_key_env"] == "OPENAI_API_KEY"
        assert item["request_params"] == {"reasoning_effort": effort, "max_completion_tokens": 4096}
        assert item["budget_provider"] == "openai" and item["budget"]["cap_usd"] == 10.0
        assert item["pricing"]["per_million_tokens"] == {
            "input": input_rate, "cached_input": cached_rate, "output": output_rate}
    deepseek = paid["deepseek-v4.1-flash"]
    assert deepseek["request_params"] == {
        "temperature": 0.4, "max_tokens": 4096,
        "extra_body": {"thinking": {"type": "disabled"}},
    }


def test_openai_paid_arm_uses_api_key_env_and_redacts_the_key_from_errors(monkeypatch, tmp_path):
    from benchmarks import run as runner

    secret = "sk-openai-do-not-log"
    config = load_model_registry(runner.ROOT / "benchmarks" / "models.json", include_paid=True)["gpt-6.1-sol"]
    captured = {}

    class FakeAdapter:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def complete(self, _system, _prompt):
            exc = RuntimeError("provider failure for " + secret)
            exc.status_code = 503
            exc.provider_elapsed_ms = 2
            raise exc

    monkeypatch.setattr(runner, "configured_token", lambda name: secret if name == "OPENAI_API_KEY" else None)
    monkeypatch.setattr(runner, "OpenAICompatibleClient", FakeAdapter)
    guarded = runner.make_client(config, ledger=CostLedger(tmp_path / "openai.spend.jsonl", caps={"gpt-6.1-sol": 10.0}))
    assert captured["token"] == secret
    assert captured["request_params"] == {"reasoning_effort": "low", "max_completion_tokens": 4096}
    with pytest.raises(RuntimeError) as exc:
        guarded.complete("system", "prompt")
    assert secret not in str(exc.value)
    assert exc.value.provider_error_type == "RuntimeError"
    assert exc.value.status_code == 503
    assert exc.value.provider_elapsed_ms == 2
    from pramaan.orchestrate import _provider_error
    error = _provider_error(exc.value, "proposal")
    assert error["message"] == "provider failure for [REDACTED]"
    assert error["status_code"] == 503
    assert error["budget_guard"]["released"] is True


def test_paid_run_freeze_is_durable_and_resume_requires_identical_settings(tmp_path):
    from pathlib import Path
    config = load_model_registry(Path(__file__).parents[1] / "benchmarks" / "models.json",
                                 include_paid=True)["claude-haiku-5-5"]
    freeze = paid_run_freeze({"claude-haiku-5-5": config}, "suite-digest")
    plan = {"anthropic_future_usd": 1.0}
    path = tmp_path / "run.preflight.json"
    digest = save_or_validate_paid_freeze(path, freeze, plan, resume=False)
    loaded = json.loads(path.read_text())
    assert loaded["freeze_sha256"] == digest
    assert loaded["freeze"] == freeze
    assert save_or_validate_paid_freeze(path, freeze, {"anthropic_future_usd": .5}, resume=True) == digest
    changed = dict(freeze, suite_sha256="different")
    with pytest.raises(ValueError, match="settings differ"):
        save_or_validate_paid_freeze(path, changed, plan, resume=True)


def test_openenv_export_replays_reset_and_each_step_as_role_separated_transitions(tmp_path):
    from scripts.push_hub_datasets import export_openenv_episodes

    tasks, suite_sha = load_suite("eval-v2.1")
    task = next(item for item in tasks if item.grammar == "periodic_raster")
    candidate = pattern_obj().model_dump(mode="json")
    from pramaan.orchestrate import apply_host_profile
    candidate = apply_host_profile(candidate, task)
    checks, _, _ = verify_pattern(parse_pattern(json.dumps(candidate)))
    passed = all_pass(checks)
    row = {
        "model_key": "beam-hf", "model_id": "reflection/Beam-1.0-501B-A23B:baseten",
        "provider": "huggingface", "api_style": "openai-compatible-chat-completions",
        "resolved_model_ids": [],
        "prompt_suite_sha256": suite_sha, "prompt_id": task.id, "suite_id": task.suite_id,
        "state": task.state, "tradition_id": task.tradition_id, "grammar": task.grammar,
        "level": task.level, "brief": task.brief, "symmetry": task.symmetry,
        "validator_version": "0.3.0", "elapsed_ms": 500, "total_tokens": 23,
        "token_usage": {"input_tokens": 10, "output_tokens": 13, "total_tokens": 23},
        "attempts": [{"index": 1, "candidate": candidate,
                      "report": {"checks": checks, "all_hard_constraints_pass": passed},
                      "call_index": 1, "metadata": {"finish_reason": "stop"}}],
        "model_calls": [{"call_index": 1, "telemetry": {"token_usage": {"total_tokens": 23},
                          "cost": {"amount": 0.01, "currency": "USD"}}}],
    }
    suite_file = tmp_path / "eval-v2.1.jsonl"
    suite_bytes = __import__("pathlib").Path("benchmarks/suites/eval-v2.1.jsonl").read_bytes()
    suite_file.write_bytes(suite_bytes.replace(b"\r\n", b"\n"))
    target = tmp_path / "transitions.jsonl"

    result = export_openenv_episodes([row], suite_path=suite_file, output_path=target,
                                     role="benchmark_evaluation")

    records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert result["episodes"] == 1
    assert [item["event_type"] for item in records] == ["reset", "step"]
    assert records[0]["next_observation"]["task_id"] == task.id
    step = records[1]
    assert step["action"]["pattern_json"] == json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
    assert step["checks"] == checks
    assert step["reward"] == 1.0
    assert step["terminated"] is True and step["truncated"] is False
    assert step["reward_version"] == "pramaan-reward-v1"
    assert step["data_role"] == "benchmark_evaluation"
    assert step["training_eligible"] is False
    assert step["provenance"]["provider"] == "huggingface"
    assert step["provenance"]["resolved_model_ids"] is None
    assert step["usage"]["total_tokens"] == 23
    assert len(step["episode_id"]) == 64


def test_openenv_export_rejects_reward_or_check_mismatch_and_never_overwrites(tmp_path):
    from scripts.push_hub_datasets import export_openenv_episodes

    tasks, suite_sha = load_suite("eval-v2.1")
    task = next(item for item in tasks if item.grammar == "periodic_raster")
    candidate = pattern_obj().model_dump(mode="json")
    checks, _, _ = verify_pattern(parse_pattern(json.dumps(candidate)))
    row = {
        "model_key": "beam-hf", "model_id": "beam", "provider": "huggingface",
        "prompt_suite_sha256": suite_sha, "prompt_id": task.id, "suite_id": task.suite_id,
        "state": task.state, "tradition_id": task.tradition_id, "grammar": task.grammar,
        "level": task.level, "brief": task.brief, "symmetry": task.symmetry,
        "attempts": [{"index": 1, "candidate": candidate,
                      "report": {"checks": checks, "all_hard_constraints_pass": False}}],
    }
    suite_file = tmp_path / "suite.jsonl"
    suite_file.write_text(__import__("pathlib").Path("benchmarks/suites/eval-v2.1.jsonl").read_text(), encoding="utf-8")
    with pytest.raises(ValueError, match="replay"):
        export_openenv_episodes([row], suite_path=suite_file,
                                output_path=tmp_path / "bad.jsonl", role="benchmark_evaluation")
    target = tmp_path / "existing.jsonl"
    target.write_text("keep\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_openenv_episodes([row], suite_path=suite_file, output_path=target,
                                role="benchmark_evaluation")
    assert target.read_text(encoding="utf-8") == "keep\n"


def test_community_index_and_recent_run_loader_are_bounded_and_scrub_private_fields(monkeypatch):
    from scripts.push_hub_datasets import load_recent_community_runs

    class FakeHub:
        def list_repo_files(self, **kwargs):
            return ["community/index/20261030T120000000000Z_run-a.json",
                    "community/index/20261029T120000000000Z_run-b.json"]
        def download_file(self, *, path_in_repo, **kwargs):
            if path_in_repo.endswith("run-a.json"):
                return json.dumps({"run_id": "run-a", "model_key": "openai/gpt-4o",
                                   "submitted_at": "2026-10-30T12:00:00Z",
                                   "row_path": "community/test/run-a.jsonl"})
            return json.dumps({"run_id": "run-b", "model_key": "deepseek-v4.1-flash",
                               "submitted_at": "2026-10-29T12:00:00Z",
                               "row_path": "community/test/run-b.jsonl"})
        def download_rows(self, *, path_in_repo, **kwargs):
            return json.dumps({"run_id": path_in_repo.split("/")[-1].split(".")[0],
                               "model_key": "openai/gpt-4o", "prompt_id": "q1",
                               "candidate": {"layers": []}, "raw_response": "secret completion",
                               "provider_error": "secret error", "api_key": "secret key"})

    runs = load_recent_community_runs(limit=1, api=FakeHub())
    assert len(runs) == 1 and runs[0]["run_id"] == "run-a"
    assert runs[0]["model_key"] == "openai/gpt-4o"
    serialized = json.dumps(runs)
    assert "secret completion" not in serialized and "secret error" not in serialized
    assert "secret key" not in serialized
    with pytest.raises(ValueError, match="limit"):
        load_recent_community_runs(limit=0, api=FakeHub())


def test_openenv_dataset_publisher_is_append_only_and_checks_roles(tmp_path, monkeypatch):
    from scripts.push_hub_datasets import OPENENV_REPO, publish_openenv_episodes

    source = tmp_path / "beam-run.jsonl"
    row = {"data_role": "benchmark_evaluation", "training_eligible": False,
           "event_type": "step", "reward": 1.0,
           "provenance": {"resolved_model_ids": []}}
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setenv("HF_COMMUNITY_WRITE_TOKEN", "hf-secret")

    class FakeHub:
        def __init__(self): self.files = set(); self.uploads = []
        def create_repo(self, **kwargs): pass
        def list_repo_files(self, **kwargs): return sorted(self.files)
        def upload_file(self, **kwargs):
            self.files.add(kwargs["path_in_repo"])
            self.uploads.append((kwargs["path_in_repo"], kwargs["path_or_fileobj"].read()))

    api = FakeHub()
    result = publish_openenv_episodes(eval_path=source, api=api)
    assert result["repo_id"] == OPENENV_REPO
    path, payload = next(item for item in api.uploads if item[0].endswith("beam-run.jsonl"))
    assert path == "transitions-v1.1/test/beam-run.jsonl"
    published = json.loads(payload)
    assert published["data_role"] == "benchmark_evaluation"
    assert published["provenance"]["resolved_model_ids"] is None
    assert any(item[0] == "README.md" for item in api.uploads)
    with pytest.raises(ValueError, match="append-only"):
        publish_openenv_episodes(eval_path=source, api=api)

    wrong = tmp_path / "wrong.jsonl"
    wrong.write_text(json.dumps({**row, "training_eligible": True}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="role metadata"):
        publish_openenv_episodes(eval_path=wrong, api=api)



def test_openenv_schema_normalizer_canonicalizes_empty_resolved_model_ids():
    from scripts.push_hub_datasets import normalize_openenv_transition

    empty = {"provenance": {"resolved_model_ids": []}, "event_type": "step"}
    populated = {"provenance": {"resolved_model_ids": ["hf/model"]}, "event_type": "step"}
    normalized_empty = normalize_openenv_transition(empty)
    normalized_populated = normalize_openenv_transition(populated)

    assert empty["provenance"]["resolved_model_ids"] == []
    assert normalized_empty["provenance"]["resolved_model_ids"] is None
    assert normalized_populated["provenance"]["resolved_model_ids"] == ["hf/model"]
    with pytest.raises(ValueError, match="resolved model IDs"):
        normalize_openenv_transition({"provenance": {"resolved_model_ids": [None]}})


def test_openenv_export_records_provider_truncation_separately_from_environment_reward(tmp_path):
    from scripts.push_hub_datasets import export_openenv_episodes
    from pramaan.orchestrate import apply_host_profile

    task = next(task for task in load_suite("eval-v2.1")[0] if task.grammar == "periodic_raster")
    suite_sha = load_suite("eval-v2.1")[1]
    candidate = apply_host_profile(pattern_obj().model_dump(mode="json"), task)
    checks, _, _ = verify_pattern(parse_pattern(json.dumps(candidate)))
    assert all_pass(checks)
    truncated_check = {"id": "candidate_truncated", "status": "FAIL", "assurance": "EXACT_DISCRETE"}
    row = {"model_key": "beam-hf", "prompt_suite_sha256": suite_sha,
           "prompt_id": task.id, "suite_id": task.suite_id, "state": task.state,
           "tradition_id": task.tradition_id, "grammar": task.grammar, "level": task.level,
           "brief": task.brief, "symmetry": task.symmetry,
           "attempts": [{"index": 1, "candidate": candidate,
                         "raw": json.dumps(candidate), "metadata": {"finish_reason": "length"},
                         "report": {"checks": checks + [truncated_check],
                                    "all_hard_constraints_pass": False}}]}
    suite_file = tmp_path / "eval-v2.1.jsonl"
    suite_file.write_bytes(__import__("pathlib").Path("benchmarks/suites/eval-v2.1.jsonl").read_bytes())
    target = tmp_path / "truncated-transitions.jsonl"

    export_openenv_episodes([row], suite_path=suite_file, output_path=target,
                            role="benchmark_evaluation")

    step = next(json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()
                if json.loads(line)["event_type"] == "step")
    assert step["reward"] == 1.0
    assert step["source_benchmark_truncated"] is True
    assert step["source_benchmark_reward"] == 0.0
    assert step["source_finish_reason"] == "length"

@pytest.mark.parametrize("base_url", [
    "https://" + ".".join(("127", "0", "0", "1")) + "/v1",
    "https://" + ".".join(("169", "254", "169", "254")) + "/latest/meta-data",
    "https://attacker.example/v1",
    "https://user:pass" + "@" + "api.anthropic.com",
    "https://api.anthropic.com/?redirect=1",
])
def test_byo_evaluation_rejects_endpoint_in_model_field_before_client_init(monkeypatch, base_url):
    from benchmarks import run as runner

    monkeypatch.setattr(runner, "_make_litellm_community_client",
                        lambda *_args, **_kwargs: pytest.fail("provider client was initialized"))
    with pytest.raises(ValueError, match="provider/model"):
        runner.run_community_evaluation(base_url, "sk-canary-never-log")
def test_community_model_identity_accepts_self_reported_and_huggingface_names():
    from scripts.push_hub_datasets import community_model_identity

    assert community_model_identity("anthropic/claude-sonnet-4-5") == {
        "provider": "Anthropic", "model_id": "claude-sonnet-4-5",
        "api_style": "litellm-chat-completions",
    }
    assert community_model_identity("openai/gpt-4o")["provider"] == "OpenAI"
    assert community_model_identity("huggingface/together/meta-llama/Llama-3.3-70B-Instruct") == {
        "provider": "Hugging Face", "model_id": "together/meta-llama/Llama-3.3-70B-Instruct",
        "api_style": "litellm-chat-completions",
    }
    assert community_model_identity("unknown/provider-model")["provider"] == "Self-reported"
    assert community_model_identity("my-model") == {
        "provider": "Self-reported", "model_id": "my-model",
        "api_style": "litellm-chat-completions",
    }
    assert community_model_identity("https://attacker.example/v1") is None
    assert community_model_identity(["openai", "gpt-4o"]) is None
    assert community_model_identity("bad provider/model name") is None
    assert community_model_identity("provider/../secret") is None


def _minimal_self_reported_archive_row(model_key, model_id, provider, task, digest):
    return {
        "model_key": model_key, "model_id": model_id, "provider": provider,
        "api_style": "litellm-chat-completions",
        "request_params": {"temperature": 0.4, "max_tokens": 4096},
        "prompt_suite_sha256": digest, "prompt_id": task.id, "suite_id": task.suite_id,
        "state": task.state, "tradition_id": task.tradition_id, "grammar": task.grammar,
        "level": task.level, "brief": task.brief, "symmetry": task.symmetry,
        "valid_schema_at_1": False, "validity_at_1": False, "validity_at_3": False,
        "attempts": [{"index": 1, "candidate": None, "report": {
            "all_hard_constraints_pass": False,
            "checks": [{"id": "candidate_parse", "status": "FAIL"}],
        }}],
    }


def test_huggingface_model_identity_packs_and_replay_audits_without_invented_host(tmp_path):
    from pramaan.community import audit_submission_archive
    from pramaan.tasks import load_suite
    from scripts.pack_submission import pack_submission

    tasks, digest = load_suite("eval-v2.1")
    key = "huggingface/together/meta-llama/Llama-3.3-70B-Instruct"
    row = _minimal_self_reported_archive_row(
        key, "together/meta-llama/Llama-3.3-70B-Instruct", "Hugging Face", tasks[0], digest)
    rows_path = tmp_path / "hf-run.jsonl"
    rows_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    archive_path = tmp_path / "hf-run.zip"
    pack_submission(rows_path, archive_path)

    report = audit_submission_archive(archive_path, run_id="hf-self-report")

    assert report.accepted is True
    assert report.accepted_rows[0]["provider_host"] is None
    assert report.accepted_rows[0]["model_label"].endswith("self-reported, unverified)")
    assert report.accepted_rows[0]["training_eligible"] is False


def test_community_audit_rejects_failed_design_claimed_as_passing():
    from pramaan.community import audit_trajectory_rows
    from pramaan.tasks import load_suite

    tasks, digest = load_suite("eval-v2.1")
    row = _minimal_self_reported_archive_row("my-model", "my-model", "Self-reported", tasks[0], digest)
    row["validity_at_1"] = True
    row["validity_at_3"] = True
    row["attempts"][0]["report"]["all_hard_constraints_pass"] = True

    report = audit_trajectory_rows([row], run_id="false-pass", expected_suite_sha256=digest)

    assert report.accepted is False
    assert "acceptance" in report.reasons[0].lower() or "scores" in report.reasons[0].lower()


def test_approved_community_model_set_matches_user_budget_decision():
    from benchmarks import run as runner
    from scripts.push_hub_datasets import COMMUNITY_MODEL_PROVIDERS

    expected_paid = {"claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5",
                     "claude-fable-5-1", "deepseek-v4.1-flash", "gpt-6.1-sol", "gpt-6.1-terra",
                     "gpt-6-luna", "gpt-6-astra"}
    expected_community = {"claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5",
                          "claude-fable-5-1", "deepseek-v4.1-flash"}
    assert {item["key"] for item in runner.PAID_MODELS} == expected_paid
    assert set(COMMUNITY_MODEL_PROVIDERS) == expected_community

def test_eval_dataset_cards_label_v21_current_and_v2_superseded():
    from scripts.push_hub_datasets import dataset_card, openenv_dataset_card

    current = dataset_card("eval", suite_name="eval-v2.1")
    historical = dataset_card("eval", suite_name="eval-v2")
    openenv = openenv_dataset_card()
    training = dataset_card("train", suite_name="train-v1")

    assert "current corrected held-out suite" in current.lower()
    assert "23.6%" in current and "65.3%" in current and "USD 0.4382" in current
    assert "eval-v2 is historical and superseded" in current.lower()
    assert "historical and superseded by eval-v2.1" in historical.lower()
    assert "superseded" in openenv.lower()
    assert "Current eval-v2.1 suite SHA-256" in openenv
    assert "config_name: transitions-v1.1" in openenv
    assert "transitions-v1.1/test/*.jsonl" in openenv
    assert "transitions/test/*.jsonl" not in openenv
    assert "train-v1.2" in training and "superseded" in training.lower()
    assert "c8bc4c59c6738b755bee24d8ed5af7fbe90bb9ca2a843dcd47c4e68bd9ad773e" in openenv

def test_litellm_community_model_is_strict_and_requires_known_prices(monkeypatch):
    import sys
    from types import SimpleNamespace
    from benchmarks import run as runner

    calls = []
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(
        get_model_info=lambda model, custom_llm_provider: (
            calls.append((model, custom_llm_provider)) or {
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
                "output_cost_per_reasoning_token": 0.00002,
            }
        )
    ))
    config = runner.community_litellm_model_config("openai/gpt-4o")
    assert calls == [("gpt-4o", "openai")]
    assert config["key"] == "openai/gpt-4o"
    assert config["model_id"] == "gpt-4o"
    assert config["provider"] == "OpenAI"
    assert config["api_style"] == "litellm-chat-completions"
    assert config["pricing"]["per_million_tokens"] == {"input": 3.0, "output": 20.0}
    assert config["budget"]["cap_usd"] == 10.0
    assert config["request_params"]["max_tokens"] == 4096


def test_litellm_huggingface_route_requires_litellm_pricing(monkeypatch):
    import sys
    from types import SimpleNamespace
    from benchmarks import run as runner

    calls = []
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(
        get_model_info=lambda model, custom_llm_provider: (
            calls.append((model, custom_llm_provider)) or {
                "input_cost_per_token": 0.000001,
                "output_cost_per_token": 0.000002,
            }
        )
    ))
    config = runner.community_litellm_model_config(
        "huggingface/together/meta-llama/Llama-3.3-70B-Instruct")
    assert calls == [("together/meta-llama/Llama-3.3-70B-Instruct", "huggingface")]
    assert config["provider"] == "Hugging Face"
    assert config["budget"]["cap_usd"] == 10.0
    assert config["pricing"]["per_million_tokens"] == {"input": 1.0, "output": 2.0}


def test_model_registry_accepts_custom_endpoint_and_pricing(tmp_path):
    from benchmarks.run import load_model_registry

    path = tmp_path / "models.json"
    path.write_text(json.dumps({"schema_version": 1, "models": [{
        "key": "custom/model-v1", "provider": "Self-hosted", "api_style": "openai-compatible-chat-completions",
        "model_id": "model-v1", "base_url": "https://inference.example/v1",
        "api_key_env": "CUSTOM_MODEL_KEY", "request_params": {"temperature": 0.4, "max_tokens": 4096},
        "pricing": {"currency": "USD", "per_million_tokens": {"input": 0.2, "output": 0.6},
                    "source": "operator supplied", "source_url": "https://inference.example/pricing",
                    "observed_at": "2026-11-01"},
    }]}), encoding="utf-8")

    loaded = load_model_registry(path)

    assert loaded["custom/model-v1"]["base_url"] == "https://inference.example/v1"
    assert loaded["custom/model-v1"]["pricing"]["per_million_tokens"] == {"input": 0.2, "output": 0.6}


@pytest.mark.parametrize("model_id", [
    "https://attacker.example/v1", "openai/", "bad provider/model name",
    "openai/model?api_base=https://attacker.example", "openai/model with spaces",
    "openai/model\\path", "openai/model#fragment", "openai/" + "m" * 300,
])
def test_litellm_community_model_rejects_unsafe_or_unroutable_identifiers(monkeypatch, model_id):
    import sys
    from types import SimpleNamespace
    from benchmarks import run as runner

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(
        get_model_info=lambda **_kwargs: pytest.fail("invalid model must fail before price lookup")
    ))
    with pytest.raises(ValueError, match="provider/model"):
        runner.community_litellm_model_config(model_id)


def test_litellm_community_model_rejects_missing_or_zero_cost_before_provider_use(monkeypatch):
    import sys
    from types import SimpleNamespace
    from benchmarks import run as runner

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(
        get_model_info=lambda *_args, **_kwargs: {"input_cost_per_token": 0, "output_cost_per_token": None}
    ))
    with pytest.raises(ValueError, match="known non-zero token pricing"):
        runner.community_litellm_model_config("anthropic/claude-sonnet-5")


def test_litellm_completion_receives_key_and_disables_retry_fallback_and_streaming(monkeypatch):
    import sys
    from types import SimpleNamespace
    from benchmarks.run import LiteLLMCommunityClient

    recorded = {}
    class Message:
        content = "response"
    class Choice:
        message = Message()
        finish_reason = "stop"
    class Response:
        choices = [Choice()]
        model = "gpt-4o"
        usage = SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10)
    def completion(**kwargs):
        recorded.update(kwargs)
        return Response()
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    client = LiteLLMCommunityClient(
        token="visitor-secret", model_id="gpt-4o", provider="openai",
        request_params={"temperature": 0.4, "max_tokens": 4096})
    content, metadata = client.complete("system", "prompt")
    assert content == "response"
    assert recorded["api_key"] == "visitor-secret"
    assert recorded["model"] == "openai/gpt-4o"
    assert recorded["num_retries"] == 0
    assert recorded["max_retries"] == 0
    assert recorded["stream"] is False
    assert "fallbacks" not in recorded
    assert metadata["usage"]["prompt_tokens"] == 7
    assert metadata["provider_attempts"] == 1


def test_managed_paid_arm_continues_when_preflight_only_fits_part_of_suite(monkeypatch, tmp_path):
    import sys
    from benchmarks import run as runner

    registry = load_model_registry(runner.ROOT / "benchmarks" / "models.json", include_paid=True)
    monkeypatch.setattr(sys, "argv", ["run.py", "--suite", str(runner.ROOT / "benchmarks" / "suites" / "eval-v2.1-common47.jsonl"),
        "--model", "claude-opus-5-5", "--output", str(tmp_path / "opus.jsonl"),
        "--ledger", str(tmp_path / "opus.spend.jsonl"), "--cap-usd", "10", "--managed-paid-run"])
    monkeypatch.setattr(runner, "load_model_registry", lambda *_a, **_k: registry)
    monkeypatch.setattr(runner, "configured_token", lambda _name: "configured")
    monkeypatch.setattr(runner, "preflight_selected_models", lambda *_a, **_k: {
        "budget_groups": {"claude-opus-5-5": {"fits": False}}, "assumptions": {}})
    monkeypatch.setattr(runner, "save_or_validate_paid_freeze", lambda *_a, **_k: "freeze")

    class PartialCapClient(runner.GuardedCompletionClient):
        def __init__(self): self.checks = 0
        def can_reserve_complete_question(self, *_args, **_kwargs):
            self.checks += 1
            return self.checks == 1

    client = PartialCapClient()
    monkeypatch.setattr(runner, "make_client", lambda *_a, **_k: client)
    prompt_calls = []
    def fake_run_prompt(prompt, **_kwargs):
        prompt_calls.append(prompt["id"])
        return {"prompt_id": prompt["id"], "validity_at_1": 0, "validity_at_3": 0, "attempt_count": 1}
    monkeypatch.setattr(runner, "run_prompt", fake_run_prompt)

    assert runner.main() == 0
    assert len(prompt_calls) == 1
    rows = [json.loads(line) for line in (tmp_path / "opus.jsonl").read_text().splitlines()]
    assert [row["prompt_id"] for row in rows] == prompt_calls
    assert rows[0]["evaluation_freeze_sha256"] == "freeze"


def test_paid_arms_allow_long_repair_prompts_with_shared_100k_input_guard(tmp_path):
    from benchmarks import run as runner

    registry = load_model_registry(runner.ROOT / "benchmarks" / "models.json", include_paid=True)
    arm_keys = {"claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "deepseek-v4.1-flash"}
    assert {registry[key]["budget"]["max_input_bytes"] for key in arm_keys} == {100_000}

    calls = []
    class Client:
        def complete(self, system, prompt):
            calls.append((system, prompt))
            return "candidate", {"usage": {"input_tokens": len(prompt.encode()), "output_tokens": 1}}

    ledger = CostLedger(tmp_path / "long-input.spend.jsonl", caps={"claude-sonnet-5-5": 10.0})
    guarded = GuardedCompletionClient(
        Client(), ledger=ledger, model_key="claude-sonnet-5-5", provider="anthropic",
        model_price=ModelPrice("claude-sonnet-5-5", "anthropic", 2.0, 10.0),
        max_input_bytes=100_000, input_token_overhead=256, max_output_tokens=4096,
        budget_key="claude-sonnet-5-5")

    content, _metadata = guarded.complete("", "r" * 25_000)
    assert content == "candidate"
    assert len(calls) == 1
    with pytest.raises(BudgetExceeded, match="100000"):
        guarded.complete("", "x" * 100_001)
    assert len(calls) == 1


def test_paid_resume_accepts_only_a_uniform_input_guard_increase(tmp_path):
    from copy import deepcopy
    config = load_model_registry(Path(__file__).parents[1] / "benchmarks" / "models.json",
                                 include_paid=True)["claude-sonnet-5-5"]
    config["budget"]["max_input_bytes"] = 16_000
    old_freeze = paid_run_freeze({"claude-sonnet-5-5": config}, "suite-digest")
    path = tmp_path / "resume.preflight.json"
    original_hash = save_or_validate_paid_freeze(path, old_freeze, {"old": True}, resume=False)

    expanded = deepcopy(config)
    expanded["budget"]["max_input_bytes"] = 100_000
    new_freeze = paid_run_freeze({"claude-sonnet-5-5": expanded}, "suite-digest")
    assert save_or_validate_paid_freeze(path, new_freeze, {"new": True}, resume=True) == original_hash
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["freeze"] == old_freeze
    assert manifest["effective_input_guard_bytes_by_model"] == {"claude-sonnet-5-5": 100_000}
    assert manifest["input_guard_updates"][-1]["from_bytes_by_model"] == {"claude-sonnet-5-5": 16_000}
