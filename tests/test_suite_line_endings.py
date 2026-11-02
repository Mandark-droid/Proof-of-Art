from pathlib import Path

from pramaan import app


def _write_lf_suite(tmp_path: Path) -> None:
    source = app.ROOT / "benchmarks" / "suites" / "eval-v2.1.jsonl"
    payload = source.read_bytes().replace(b"\r\n", b"\n")
    target = tmp_path / "benchmarks" / "suites" / "eval-v2.1.jsonl"
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    app.ROOT = tmp_path


def test_frozen_eval_suite_accepts_hub_lf_line_endings(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(app, "ROOT", app.ROOT)
    _write_lf_suite(tmp_path)

    _payload, prompts = app._load_frozen_eval_suite()

    assert len(prompts) == 72


def test_home_wall_tasks_accept_hub_lf_line_endings(tmp_path: Path, monkeypatch):
    original_root = app.ROOT
    monkeypatch.setattr(app, "ROOT", original_root)
    _write_lf_suite(tmp_path)
    app._home_suite_tasks.cache_clear()
    try:
        tasks = app._home_suite_tasks()
    finally:
        app._home_suite_tasks.cache_clear()

    assert len(tasks) == 72
    assert {task["state"] for task in tasks} >= {"Gujarat", "Puducherry"}
