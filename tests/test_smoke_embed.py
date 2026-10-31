import pytest

from scripts.smoke_embed import (
    select_rl_code_snippet,
    validate_home_wall_rendering,
    validate_visitor_panels,
)


def test_rl_smoke_selects_the_first_code_block_when_multiple_are_present():
    snippets = [
        "from pramaan_env import PramaanEnv, PramaanAction",
        "python scripts/run_arms.py --suite benchmarks/suites/eval-v2.1.jsonl",
    ]
    assert select_rl_code_snippet(snippets) == snippets[0]


def test_home_wall_smoke_accepts_complete_reference_art_and_beam_results():
    validate_home_wall_rendering(
        state_count=36,
        art_count=36,
        loaded_art_count=36,
        placeholder_count=0,
        beam_selected=True,
        non_ring_mark_count=47,
    )


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"art_count": 0, "loaded_art_count": 0, "placeholder_count": 36}, "reference art"),
        ({"beam_selected": False}, "Beam"),
        ({"non_ring_mark_count": 0}, "non-ring"),
    ],
)
def test_home_wall_smoke_rejects_missing_visual_evidence(changes, message):
    values = {
        "state_count": 36,
        "art_count": 36,
        "loaded_art_count": 36,
        "placeholder_count": 0,
        "beam_selected": True,
        "non_ring_mark_count": 47,
    }
    values.update(changes)
    with pytest.raises(AssertionError, match=message):
        validate_home_wall_rendering(**values)


def test_visitor_panel_smoke_accepts_complete_public_panels():
    validate_visitor_panels(
        proof_step_count=6,
        repaired_tile_loaded=True,
        model_keys=[f"model-{index}" for index in range(10)],
        fable_cap_visible=True,
        models_signin_bar_present=False,
        try_state="Gujarat",
        try_example="beam_g2_gj_bandhani",
        rl_snippet="PramaanEnv env.reset env.step",
    )


def test_visitor_panel_smoke_rejects_partial_models():
    with pytest.raises(AssertionError, match="ten unique official model rows"):
        validate_visitor_panels(
            proof_step_count=6,
            repaired_tile_loaded=True,
            model_keys=["beam-hf"],
            fable_cap_visible=True,
            models_signin_bar_present=False,
            try_state="Gujarat",
            try_example="beam_g2_gj_bandhani",
            rl_snippet="PramaanEnv env.reset env.step",
        )


def test_visitor_panel_smoke_rejects_models_signin_bar():
    with pytest.raises(AssertionError, match="Sign in to publish"):
        validate_visitor_panels(
            proof_step_count=6,
            repaired_tile_loaded=True,
            model_keys=[f"model-{index}" for index in range(10)],
            fable_cap_visible=True,
            models_signin_bar_present=True,
            try_state="Gujarat",
            try_example="beam_g2_gj_bandhani",
            rl_snippet="PramaanEnv env.reset env.step",
        )
