import pytest

from scripts.smoke_embed import (
    is_known_hf_avatar_404,
    validate_home_wall_rendering,
    validate_kinnauri_replay,
    validate_visitor_panels,
)


def test_smoke_ignores_only_the_known_huggingface_organization_avatar_404():
    assert is_known_hf_avatar_404(404, "https://huggingface.co/api/organizations/kshitijthakkar/avatar")
    assert not is_known_hf_avatar_404(404, "https://kshitijthakkar-pramaan-proof-of-art.hf.space/favicon.ico")
    assert not is_known_hf_avatar_404(404, "https://huggingface.co/api/organizations/other/avatar")
    assert not is_known_hf_avatar_404(500, "https://huggingface.co/api/organizations/kshitijthakkar/avatar")


def test_kinnauri_replay_smoke_requires_real_place_and_witness_pixels():
    validate_kinnauri_replay("Himachal Pradesh · Kinnauri · Pixel (16, 21) and mirror (47, 21)")
    with pytest.raises(AssertionError, match="saved Kinnauri evidence"):
        validate_kinnauri_replay("Gujarat · Bandhani · pixel (2, 3)")


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
        proof_step_count=7,
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
            proof_step_count=7,
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
            proof_step_count=7,
            repaired_tile_loaded=True,
            model_keys=[f"model-{index}" for index in range(10)],
            fable_cap_visible=True,
            models_signin_bar_present=True,
            try_state="Gujarat",
            try_example="beam_g2_gj_bandhani",
            rl_snippet="PramaanEnv env.reset env.step",
        )
