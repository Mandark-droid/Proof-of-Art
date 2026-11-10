from collections import Counter

import pytest

KINNAURI = "hp_kinnauri-shawl-geometric-border-easy-8"


@pytest.fixture(scope="module")
def ten(ten_model_rows):
    from pramaan_v2.build.replay import replay_all

    return replay_all(ten_model_rows)


@pytest.fixture(scope="module")
def eleven(published):
    from pramaan_v2.build.replay import replay_all

    return replay_all(published.rows)


def test_replay_matches_every_stored_report(ten, eleven):
    runs, stats = ten
    assert (stats.proposals, stats.rendered, stats.mismatches) == (1516, 1092, 0)
    assert eleven[1].mismatches == 0
    assert len(eleven[0]) == 792


def test_run_status_totals_match_spec(ten):
    runs, _ = ten
    counts = Counter(run.status for run in runs.values())
    assert counts["1"] + counts["2"] + counts["3"] == 470
    assert counts["x"] == 212
    assert counts["-"] == 38


def test_pixel_evidence_and_low_ink_totals_match_spec(ten):
    from pramaan_v2.build.replay import low_ink

    runs, _ = ten
    evidence = sum(1 for run in runs.values() for t in run.tries if t.outcome != "passed" and t.pair)
    assert evidence == 430
    assert sum(1 for run in runs.values() if run.status in "123" and low_ink(run.shown)) == 44


def test_featured_run_matches_spec(ten):
    from pramaan_v2.build.replay import clean_rgba, pixels_sha256

    run = ten[0][("beam-hf", KINNAURI)]
    assert run.status == "3"
    assert [t.outcome for t in run.tries] == ["broke", "broke", "passed"]
    assert [t.reward for t in run.tries] == pytest.approx([0.725, 0.725, 1.0])
    assert [t.rules_held for t in run.tries] == [(7, 8), (7, 8), (8, 8)]
    assert run.tries[0].pair == {"check": "symmetry_d2", "transform": "F_vertical",
                                 "original": [16, 21], "transformed": [47, 21]}
    assert run.tries[1].pair["original"] == [13, 20] and run.tries[1].pair["transformed"] == [50, 20]
    assert [None if t.mismatch is None else int(t.mismatch.sum()) for t in run.tries] == [24, 24, None]
    assert [t.tokens for t in run.tries] == [2521, 2756, 2810]
    assert [t.latency_ms for t in run.tries] == pytest.approx([3650, 2370, 2320], abs=10)
    assert pixels_sha256(run.tries[2]) == "2a851a1377b4909d921d89000ff37d2b7446dc4d77bc87fceff200aa4f6cb141"
    assert clean_rgba(run.tries[0]).shape == (64, 64, 4)
    assert run.shown is run.tries[2]


def test_unreadable_tries_have_no_tile_and_zero_reward(ten):
    runs, _ = ten
    unreadable = [t for run in runs.values() for t in run.tries if t.outcome == "unreadable"]
    assert unreadable
    assert all(t.grid is None and t.reward == 0.0 for t in unreadable)
    never = [run for run in runs.values() if run.status == "-"]
    assert all(run.shown is None for run in never)


def test_failures_without_pixels_carry_a_measured_value(ten):
    runs, _ = ten
    broke_no_pair = [t for run in runs.values() for t in run.tries if t.outcome == "broke" and not t.pair]
    assert broke_no_pair
    assert all(t.measured for t in broke_no_pair)


def test_replay_mismatch_is_a_hard_failure(ten_model_rows):
    import copy
    from pramaan_v2.build.replay import ReplayMismatch, replay_all

    row = copy.deepcopy(next(r for r in ten_model_rows if r["model_key"] == "beam-hf"
                             and r["prompt_id"] == KINNAURI))
    row["attempts"][0]["report"]["checks"][3]["status"] = "PASS"
    with pytest.raises(ReplayMismatch):
        replay_all([row])
