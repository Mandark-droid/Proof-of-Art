from dataclasses import fields

from openenv.core.client_types import StepResult

from pramaan_env import PramaanAction, PramaanObservation
from pramaan.ui.openenv_contract import render_openenv_contract


def test_contract_card_tracks_openenv_models_and_step_result_shape():
    rendered = render_openenv_contract()

    assert 'data-testid="openenv-contract"' in rendered
    for field in PramaanAction.model_fields:
        assert f'<code>{field}</code>' in rendered
    for field in PramaanObservation.model_fields:
        assert f'<code>{field}</code>' in rendered
    for field in fields(StepResult):
        assert f'<code>{field.name}</code>' in rendered
    assert "at most 3 attempts" in rendered
    assert "Reward is 0" in rendered
    assert "0.2 to under 0.8" in rendered
    assert "1.0 when every hard check passes" in rendered
