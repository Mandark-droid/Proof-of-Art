from pramaan.app import _initial_custom_state, _render_custom_app


def test_build_reward_adapter_discloses_training_scope_and_test_status():
    rendered = _render_custom_app(_initial_custom_state())
    build = rendered.split('data-tab-panel="rl"', 1)[1]

    assert "GRPO reward adapter" in build
    assert "Reward function tested." in build
    assert "No model has been trained on the Pramaan environment yet." in build
    assert "train-v1" in build
    assert "held-out evaluation suite" in build
    assert "from pramaan_env.reward_fn import pramaan_reward" in build
