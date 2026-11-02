from pathlib import Path

import pytest

from scripts.run_release_checks import build_checks, check_current_trajectory_export


def test_release_checks_replay_eval_v21_and_replay_v1_trajectory_only(tmp_path):
    (tmp_path / "benchmarks" / "suites").mkdir(parents=True)
    (tmp_path / "benchmarks" / "results").mkdir(parents=True)
    (tmp_path / "benchmarks" / "data").mkdir(parents=True)
    (tmp_path / "benchmarks" / "suites" / "eval-v2.1.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "benchmarks" / "results" / "eval-v2.1-beam-20261030.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "benchmarks" / "data" / "eval-v2.1-trajectories.jsonl").write_text("{}\n", encoding="utf-8")

    checks = build_checks(tmp_path)
    result_commands = [command for command in checks if any("check_benchmark_results.py" in item for item in command)]

    current = next(command for command in result_commands if any("eval-v2.1-beam-20261030.jsonl" in item for item in command))
    assert "benchmarks/suites/eval-v2.1.jsonl" in current
    assert "--require-complete" in current
    assert len(result_commands) == 1
    trajectory_command = next(command for command in checks if any("check_trajectory_dataset.py" in item for item in command))
    assert "benchmarks/data/beam-hf-trajectories-v1.jsonl" in trajectory_command
    assert not any("beam-hf-shared-first-v2.jsonl" in item for command in checks for item in command)


def test_release_checks_use_legacy_trajectory_replay_until_v21_artifacts_are_integrated(tmp_path):
    (tmp_path / "benchmarks" / "data").mkdir(parents=True)

    checks = build_checks(tmp_path)
    assert not any("check_benchmark_results.py" in item for command in checks for item in command)
    trajectory_command = next(command for command in checks if any("check_trajectory_dataset.py" in item for item in command))
    assert "benchmarks/data/beam-hf-trajectories-v1.jsonl" in trajectory_command


def test_release_checks_reject_partial_eval_v21_artifact_set(tmp_path):
    (tmp_path / "benchmarks" / "suites").mkdir(parents=True)
    (tmp_path / "benchmarks" / "suites" / "eval-v2.1.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="current eval-v2.1 release artifacts are incomplete"):
        build_checks(tmp_path)


def test_release_gate_runs_only_project_tests():
    pytest_command = build_checks(Path(__file__).resolve().parents[1])[0]
    assert pytest_command[-2:] == ["-q", "tests"]


def test_offline_app_configuration_uses_space_oauth_without_mock_network_call():
    source = (Path(__file__).resolve().parents[1] / "scripts/check_app_configuration.py").read_text(encoding="utf-8")
    assert 'os.environ.setdefault("SYSTEM", "spaces")' in source
    assert 'os.environ.setdefault("SPACE_ID", "pramaan-offline-release-check")' in source
    for variable in ("OAUTH_CLIENT_ID", "OAUTH_CLIENT_SECRET", "OAUTH_SCOPES", "OPENID_PROVIDER_URL"):
        assert f'os.environ.setdefault("{variable}"' in source


def test_current_trajectory_export_accepts_windows_line_endings(tmp_path, monkeypatch):
    import scripts.push_hub_datasets as publisher
    import scripts.run_release_checks as release_checks

    for relative in (release_checks.CURRENT_RESULT, release_checks.CURRENT_SUITE):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture\n")
    committed = tmp_path / release_checks.CURRENT_TRAJECTORY
    committed.parent.mkdir(parents=True, exist_ok=True)
    committed.write_bytes(b'{"prompt_id":"one"}\r\n')

    def export(_results, _suite, output, *, role):
        assert role == "benchmark_evaluation"
        output.write_bytes(b'{"prompt_id":"one"}\n')
        return {"records": 1}

    monkeypatch.setattr(publisher, "export_suite_trajectories", export)

    assert check_current_trajectory_export(tmp_path) == {"records": 1}
