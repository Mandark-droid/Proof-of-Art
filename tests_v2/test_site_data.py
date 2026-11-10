import base64
import io
import json

import pytest
from PIL import Image

KINNAURI = "hp_kinnauri-shawl-geometric-border-easy-8"
PY_EASY = "py_kolam-as-practised-in-puducherry-easy-35"


@pytest.fixture(scope="module")
def built(tmp_path_factory, published):
    from pramaan_v2.build.site_data import build

    out = tmp_path_factory.mktemp("site")
    hashes = build(out, published)
    return out, hashes


def _load(out, name):
    return json.loads((out / name).read_text(encoding="utf-8"))


def test_build_is_deterministic(built, published, tmp_path):
    from pramaan_v2.build.site_data import build

    _, first = built
    assert build(tmp_path / "again", published) == first


def test_wall_covers_every_run(built):
    out, _ = built
    wall = _load(out, "wall.json")
    meta = _load(out, "meta.json")
    assert len(wall["runs"]) == meta["totals"]["runs"] == 792
    assert wall["cols"] == 36 and wall["rows"] == 22 and wall["cell"] == 64
    counts = {key: sum(run["status"] == key for run in wall["runs"]) for key in ("1", "2", "3", "x", "-")}
    assert sum(counts.values()) == 792
    assert counts["1"] + counts["2"] + counts["3"] == meta["totals"]["passed"]
    assert Image.open(out / "wall.png").size == (2304, 1408)
    beam = next(run for run in wall["runs"] if run["model"] == "beam-hf" and run["task"] == KINNAURI)
    assert (beam["state"], beam["level"], beam["col"], beam["status"]) == ("HP", "easy", 8, "3")
    assert beam["tradition"] == "Kinnauri shawl border"


def test_meta_lists_models_in_leaderboard_order(built):
    out, _ = built
    meta = _load(out, "meta.json")
    keys = [m["key"] for m in meta["models"]]
    assert keys[:3] == ["gpt-6-astra", "gpt-6.1-sol", "claude-fable-5-2"]
    assert len(keys) == 11 and len(meta["states"]) == 36
    assert meta["revision"] and meta["suite_id"] == "eval-v2.1"
    assert "built_at" not in json.dumps(meta)


def test_one_replay_file_per_brief_with_the_featured_run(built):
    out, _ = built
    files = sorted((out / "replays").glob("*.json"))
    assert len(files) == 72
    replay = _load(out, f"replays/{KINNAURI}.json")
    assert replay["number"] == 17 and replay["state_code"] == "HP"
    beam = next(m for m in replay["models"] if m["key"] == "beam-hf")
    assert [t["mismatch_px"] for t in beam["tries"]] == [24, 24, None]
    assert beam["tries"][2]["pixels_sha256"] == "2a851a1377b4909d921d89000ff37d2b7446dc4d77bc87fceff200aa4f6cb141"
    tile = Image.open(io.BytesIO(base64.b64decode(beam["tries"][0]["tile_png"])))
    assert tile.size == (64, 64)
    assert replay["reference"] is not None and len(replay["models"]) == 11


def test_impossible_brief_has_no_reference(built):
    out, _ = built
    assert _load(out, f"replays/{PY_EASY}.json")["reference"] is None


def test_never_readable_runs_have_no_tiles(built):
    out, _ = built
    wall = _load(out, "wall.json")
    never = [run for run in wall["runs"] if run["status"] == "-"]
    assert never
    for run in never:
        replay = _load(out, f"replays/{run['task']}.json")
        model = next(m for m in replay["models"] if m["key"] == run["model"])
        assert all("tile_png" not in t for t in model["tries"])


def test_check_mode_accepts_a_fresh_build(built, published):
    from pramaan_v2.build.site_data import check

    out, _ = built
    assert check(out, published) == []
