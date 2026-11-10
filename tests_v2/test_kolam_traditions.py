import pytest

TN_KOLAM = "tn_kolam-easy-22"
PY_EASY = "py_kolam-as-practised-in-puducherry-easy-35"


@pytest.fixture(scope="module")
def kolam(published):
    from pramaan.tasks import load_suite
    from pramaan_v2.build.kolam_paths import kolam_data
    from pramaan_v2.build.replay import replay_all

    runs, _ = replay_all(published.rows)
    tasks, _ = load_suite("eval-v2.1")
    return kolam_data(runs, tasks)


def test_gcd_rule_with_no_mirrors():
    from pramaan_v2.build.kolam_paths import loop_paths

    assert len(loop_paths(5, 6, [])) == 1
    assert len(loop_paths(6, 6, [])) == 6
    assert all(path.startswith("M") and path.endswith("Z") for path in loop_paths(6, 6, []))


def test_beam_kolam_tries_match_spec(kolam):
    tries = kolam[TN_KOLAM]["models"]["beam-hf"]
    assert kolam[TN_KOLAM]["rows"] == 3 and kolam[TN_KOLAM]["cols"] == 3
    assert kolam[TN_KOLAM]["required_loops"] == 1
    assert [t["mirrors"] for t in tries] == [12, 8, 4]
    assert [t["loops"] for t in tries] == [0, 2, 1]
    assert [round(t["reward"], 3) for t in tries] == [0.691, 0.745, 1.0]
    assert [len(t["paths"]) for t in tries] == [0, 2, 1]


def test_every_model_and_kolam_brief_is_present(kolam, published):
    models = {row["model_key"] for row in published.rows}
    assert len(kolam) == 3
    for brief in kolam.values():
        assert set(brief["models"]) == models


def test_impossible_brief_accepted_tries_declare_a_different_grid(kolam):
    accepted = {key: [t for t in tries if t["outcome"] == "passed"]
                for key, tries in kolam[PY_EASY]["models"].items()}
    accepted = {key: tries[0] for key, tries in accepted.items() if tries}
    assert set(accepted) == {"claude-opus-5-5", "claude-fable-5-2"}
    for t in accepted.values():
        assert (t["rows"], t["cols"], t["mirrors"], t["loops"]) == (2, 3, 0, 1)
    assert kolam[PY_EASY]["rows"] == 2 and kolam[PY_EASY]["cols"] == 2


def test_unreadable_kolam_try_has_no_drawing(kolam):
    opus = kolam[TN_KOLAM]["models"]["claude-opus-5-5"]
    assert opus[0]["outcome"] == "unreadable"
    assert opus[0]["paths"] == [] and opus[0]["loops"] is None


def test_traditions_totals():
    from pramaan_v2.build.traditions_data import traditions_data

    data = traditions_data()
    assert data["totals"] == {"traditions": 87, "sources": 271, "unverified": 43, "gi": 54, "multi_state": 45}
    assert list(data["by_grammar"].values()) == [26, 20, 17, 9, 7, 5, 3]
    assert list(data["by_grammar"]) == ["counted_grid", "stripe_band", "periodic_raster", "rotational_field",
                                        "jaali_lattice", "dot_field", "kolam_loop"]
    assert len(data["states"]) == 36 and all(state["traditions"] for state in data["states"])
    hp = next(state for state in data["states"] if state["code"] == "HP")
    kinnauri = next(t for t in hp["traditions"] if t["id"] == "hp_kinnauri-shawl-geometric-border")
    assert kinnauri["tasks"]["easy"] == "hp_kinnauri-shawl-geometric-border-easy-8"
    assert kinnauri["sources"] and all("supports" in source for source in kinnauri["sources"])
    assert kinnauri["cautions"].startswith("Border motifs include")
