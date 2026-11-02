from pathlib import Path

from pramaan.app import resolve_project_root


def test_wall_data_root_falls_back_to_project_cwd_for_installed_package(tmp_path: Path):
    installed_app = tmp_path / "venv" / "Lib" / "site-packages" / "pramaan" / "app.py"
    installed_app.parent.mkdir(parents=True)
    installed_app.write_text("# installed package\n", encoding="utf-8")

    project_root = tmp_path / "app"
    (project_root / "benchmarks" / "suites").mkdir(parents=True)
    (project_root / "benchmarks" / "data").mkdir(parents=True)
    (project_root / "benchmarks" / "suites" / "eval-v2.1.jsonl").write_text("{}\n", encoding="utf-8")
    (project_root / "benchmarks" / "data" / "eval-v2.1-references.jsonl").write_text("{}\n", encoding="utf-8")

    assert resolve_project_root(installed_app, cwd=project_root) == project_root.resolve()


def test_project_working_root_wins_over_a_matching_installed_package_tree(tmp_path: Path):
    installed_root = tmp_path / "venv" / "Lib" / "site-packages"
    installed_app = installed_root / "pramaan" / "app.py"
    installed_app.parent.mkdir(parents=True)
    installed_app.write_text("# installed package\n", encoding="utf-8")
    misleading_suite = installed_root / "benchmarks" / "suites" / "eval-v2.1.jsonl"
    misleading_suite.parent.mkdir(parents=True)
    misleading_suite.write_text("stale package data\n", encoding="utf-8")

    project_root = tmp_path / "app"
    suite = project_root / "benchmarks" / "suites" / "eval-v2.1.jsonl"
    suite.parent.mkdir(parents=True)
    suite.write_text("current Space project data\n", encoding="utf-8")

    assert resolve_project_root(installed_app, cwd=project_root / ".venv" / "bin") == project_root.resolve()


def test_explicit_space_root_overrides_working_directory(tmp_path: Path, monkeypatch):
    module_file = tmp_path / "site-packages" / "pramaan" / "app.py"
    module_file.parent.mkdir(parents=True)
    module_file.write_text("# installed package\n", encoding="utf-8")
    project_root = tmp_path / "app"
    (project_root / "benchmarks" / "suites").mkdir(parents=True)
    (project_root / "benchmarks" / "suites" / "eval-v2.1.jsonl").write_text("{}\n", encoding="utf-8")
    other_project = tmp_path / "other"
    (other_project / "benchmarks" / "suites").mkdir(parents=True)
    (other_project / "benchmarks" / "suites" / "eval-v2.1.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("PRAMAAN_PROJECT_ROOT", str(project_root))

    assert resolve_project_root(module_file, cwd=other_project) == project_root.resolve()
