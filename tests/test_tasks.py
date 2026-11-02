from pramaan.tasks import SUITES, Task, load_suite
from pramaan.traditions import ALL_STATES

from pathlib import Path


def test_eval_suite_tasks_and_digest():
    tasks, digest = load_suite("eval-v2")
    assert len(tasks) == 72
    assert len(digest) == 64
    assert all(isinstance(task, Task) for task in tasks)
    assert {task.state for task in tasks} == set(ALL_STATES)


def test_corrected_eval_suite_has_its_own_identity_and_digest():
    tasks, digest = load_suite("eval-v2.1")
    assert len(tasks) == 72
    assert len(digest) == 64
    assert {task.suite_id for task in tasks} == {"eval-v2.1"}
    assert {task.state for task in tasks} == set(ALL_STATES)


def test_common47_suite_is_loadable_with_the_frozen_task_digest():
    tasks, digest = load_suite("eval-v2.1-common47")

    assert len(tasks) == 47
    assert digest == "dd3c9ce96a22deb1af94a1db22dc2a5016d1b1ad1c9c4a12befcd6690a6f7603"
    assert {task.state for task in tasks} == set(ALL_STATES)
    assert sum(task.level == "easy" for task in tasks) == 25
    assert sum(task.level == "hard" for task in tasks) == 22


def test_suite_digest_is_stable_across_windows_and_linux_line_endings(tmp_path, monkeypatch):
    source = SUITES["eval-v2.1"].read_bytes()
    lf_path = tmp_path / "eval-lf.jsonl"
    crlf_path = tmp_path / "eval-crlf.jsonl"
    lf_path.write_bytes(source.replace(b"\r\n", b"\n").replace(b"\n", b"\n"))
    crlf_path.write_bytes(source.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))

    monkeypatch.setitem(SUITES, "fixture", lf_path)
    _lf_tasks, lf_digest = load_suite("fixture")
    monkeypatch.setitem(SUITES, "fixture", crlf_path)
    _crlf_tasks, crlf_digest = load_suite("fixture")

    assert lf_digest == crlf_digest == "b04b568d84d2568fd261789819d4e925844dba5afe9c5962275ee571d84e4e3a"
