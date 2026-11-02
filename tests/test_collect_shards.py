from __future__ import annotations

import json

import pytest

from benchmarks.collect_shards import append_shard, merge_training_shards
from pramaan.checks.core import all_pass, verify_pattern
from pramaan.tasks import canonical_jsonl_bytes
from tests.helpers import pattern_obj


def _batch(tmp_path, suite_id="train-v1", task_id="train-1", *, tokens=20, cost=0.01):
    pattern = pattern_obj()
    candidate = json.loads(pattern.model_dump_json())
    checks, _, _ = verify_pattern(pattern)
    task = {"id": task_id, "suite_id": suite_id, "state": "Tamil Nadu",
            "tradition_id": "tn_kolam", "grammar": "periodic_raster", "level": "train",
            "symmetry": "D2", "brief": "A simple test brief", "constraint_profile": {},
            "palette": candidate["palette"]}
    suite = tmp_path / f"{suite_id}.jsonl"
    suite.write_text(json.dumps(task) + "\n", encoding="utf-8")
    import hashlib
    digest = hashlib.sha256(suite.read_bytes()).hexdigest()
    report = {"checks": checks, "all_hard_constraints_pass": all_pass(checks)}
    row = {"prompt_id": task_id, "suite_id": suite_id, "state": task["state"],
           "tradition_id": task["tradition_id"], "grammar": task["grammar"], "level": "train",
           "symmetry": "D2", "brief": task["brief"], "prompt_suite_sha256": digest,
           "model_key": "fixture", "model_id": "fixture", "provider": "test",
           "api_style": "test", "request_params": {}, "pricing": None,
           "validator_version": "0.3.0", "valid_schema_at_1": True,
           "validity_at_1": all_pass(checks), "validity_at_3": all_pass(checks),
           "repair_rescued": False, "attempts": [{"index": 1, "call_index": 1,
           "candidate": candidate, "report": report, "verified_at": "now"}],
           "model_calls": [{"call_index": 1, "phase": "proposal", "messages": [
               {"role": "system", "content": "sys"}, {"role": "user", "content": task["brief"]}],
               "raw_response": json.dumps(candidate), "metadata": {},
               "telemetry": {"provider_attempts": 1, "token_usage": {"total_tokens": tokens},
                             "cost": {"estimated_amount": cost, "estimated_currency": "USD"}}}],
           "provider_attempts": 1, "total_tokens": tokens,
           "cost": {"estimated_amount": cost}, "elapsed_ms": 100,
           "created_at": "now", "max_attempts": 3, "accepted_mask_sha256": None}
    results = tmp_path / f"{task_id}.jsonl"
    results.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return suite, results


def _caps(**overrides):
    values = {"records": 2, "calls": 4, "tokens": 100, "cost_usd": 1.0, "seconds": 30}
    return values | overrides


def _append(tmp_path, shard_id, suite, results, *, role="training", caps=None):
    return append_shard(root=tmp_path / "store", shard_id=shard_id, role=role,
                        results_path=results, suite_path=suite,
                        generator_version="briefs-v1", seed="7", caps=caps or _caps())


def test_append_shard_records_provenance_and_cumulative_budget(tmp_path):
    suite, results = _batch(tmp_path)
    first = _append(tmp_path, "train-20261029-a", suite, results)
    resumed = _append(tmp_path, "train-20261029-a", suite, results)
    assert first["actual"] == {"records": 1, "calls": 1, "tokens": 20,
                                "cost_usd": 0.01, "seconds": 0.1}
    assert resumed["appended_records"] == 0
    assert resumed["actual"]["records"] == 1


def test_append_shard_accepts_lf_copy_of_frozen_suite(tmp_path):
    import hashlib

    suite, results = _batch(tmp_path)
    suite_bytes = suite.read_bytes().replace(b"\r\n", b"\n")
    suite.write_bytes(suite_bytes)
    row = json.loads(results.read_text(encoding="utf-8"))
    row["prompt_suite_sha256"] = hashlib.sha256(canonical_jsonl_bytes(suite_bytes)).hexdigest()
    results.write_text(json.dumps(row) + "\n", encoding="utf-8")

    result = _append(tmp_path, "lf-suite", suite, results)

    assert result["appended_records"] == 1


def test_append_shard_rejects_batches_over_user_budget(tmp_path):
    suite, results = _batch(tmp_path, tokens=101)
    with pytest.raises(ValueError, match="tokens=.*exceeds cap"):
        _append(tmp_path, "over-budget", suite, results, caps=_caps(tokens=100))


def test_evaluation_shards_never_enter_training_merge(tmp_path):
    suite, results = _batch(tmp_path, suite_id="eval-v2", task_id="eval-1")
    _append(tmp_path, "eval-only", suite, results, role="evaluation")
    with pytest.raises(ValueError, match="no training shards"):
        merge_training_shards(tmp_path / "store", tmp_path / "merged")
    assert not (tmp_path / "merged").exists()


def test_train_export_refuses_prompt_leakage_from_eval_shard(tmp_path):
    suite, results = _batch(tmp_path, suite_id="eval-v2", task_id="eval-1")
    _append(tmp_path, "eval", suite, results, role="evaluation")
    train_suite, train_results = _batch(tmp_path, suite_id="train-v1", task_id="train-1")
    # Same normalized prompt is an evaluation/training split leak.
    task = json.loads(train_suite.read_text(encoding="utf-8"))
    task["brief"] = "  A SIMPLE test brief  "
    train_suite.write_text(json.dumps(task) + "\n", encoding="utf-8")
    train_data = json.loads(train_results.read_text(encoding="utf-8"))
    train_data["brief"] = task["brief"]
    import hashlib
    train_data["prompt_suite_sha256"] = hashlib.sha256(train_suite.read_bytes()).hexdigest()
    train_results.write_text(json.dumps(train_data) + "\n", encoding="utf-8")
    _append(tmp_path, "train", train_suite, train_results)
    with pytest.raises(ValueError, match="contamination"):
        merge_training_shards(tmp_path / "store", tmp_path / "merged")

def test_training_merge_writes_trajectories_and_reward_bearing_exports(tmp_path):
    suite, results = _batch(tmp_path)
    _append(tmp_path, "train-a", suite, results)
    counts = merge_training_shards(tmp_path / "store", tmp_path / "merged")
    assert counts == {"trajectories": 1, "sft": 1, "repair": 0, "preference": 0}
    trajectory = json.loads((tmp_path / "merged" / "trajectories.jsonl").read_text(encoding="utf-8"))
    sft = json.loads((tmp_path / "merged" / "sft.jsonl").read_text(encoding="utf-8"))
    assert trajectory["data_role"] == "training" and trajectory["training_eligible"] is True
    assert trajectory["calls"][0]["reward"] == 1.0
    assert sft["reward_version"] == "pramaan-reward-v1" and sft["reward"] == 1.0
