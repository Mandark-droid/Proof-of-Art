import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.run import load_model_registry, run_prompt
from scripts.check_benchmark_results import audit
from scripts.export_public_benchmark import sanitize_rows
from scripts.plot_benchmark_summary import render_svg
from tests.test_benchmarks import RepairFixture


def test_benchmark_audit_replays_each_candidate_and_rejects_tampering():
    with TemporaryDirectory(dir=Path(__file__).parents[1]) as temp_dir:
        prompts_path = Path(temp_dir) / "prompts.jsonl"
        prompt = {"id": "easy-01", "level": "easy", "symmetry": "D4", "brief": "balanced tile"}
        prompt_bytes = (json.dumps(prompt) + "\n").encode("utf-8")
        prompts_path.write_bytes(prompt_bytes)
        suite_hash = hashlib.sha256(prompt_bytes).hexdigest()
        registry = load_model_registry(Path(__file__).parents[1] / "benchmarks" / "models.json")
        model = registry["beam-hf"]
        record = run_prompt(prompt, client=RepairFixture(), suite_sha256=suite_hash, model_config=model)

        report = audit([record], prompts_path, registry, ["beam-hf"])
        assert report["complete"] is True
        assert report["candidate_replays"] == 2

        tampered = json.loads(json.dumps(record))
        tampered["attempts"][0]["report"]["all_hard_constraints_pass"] = True
        try:
            audit([tampered], prompts_path, registry, ["beam-hf"])
        except ValueError as exc:
            assert "first-proposal score" in str(exc) or "does not replay" in str(exc)
        else:
            raise AssertionError("tampered benchmark report passed audit")


def test_public_result_export_drops_raw_completion_text_but_keeps_replay_data():
    row = {"elapsed_ms": 10, "attempts": [{"raw": "private duplicate text", "candidate": {"schema_version": "0.1"},
                         "report": {"all_hard_constraints_pass": True}, "metadata": {}}],
           "model_calls": [{"raw_response": "full response and request", "messages": [{"role": "user", "content": "brief"}]}]}
    clean = sanitize_rows([row])

    assert "raw" not in clean[0]["attempts"][0]
    assert clean[0]["attempts"][0]["candidate"] == row["attempts"][0]["candidate"]
    assert "model_calls" not in clean[0]
    assert clean[0]["cost"]["source"] == "not_reported"
    assert row["attempts"][0]["raw"] == "private duplicate text"


def test_benchmark_chart_uses_per_model_scores():
    svg = render_svg({"by_model": {"beam-hf": {"overall": {"validity_at_1": 0.5, "validity_at_3": 0.75},
                                                       "by_level": {}}}})
    assert "beam-hf" in svg
    assert "pass@1" in svg and "pass@3" in svg
    assert "50.0%" in svg and "75.0%" in svg
