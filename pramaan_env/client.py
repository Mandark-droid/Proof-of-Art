from __future__ import annotations

from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import PramaanAction, PramaanObservation, PramaanState


class PramaanEnv(EnvClient[PramaanAction, PramaanObservation, PramaanState]):
    def _step_payload(self, action: PramaanAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[PramaanObservation]:
        observation_payload = payload.get("observation", payload)
        observation = PramaanObservation.model_validate(observation_payload)
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
            metadata=payload.get("metadata", observation.metadata),
        )

    def _parse_state(self, payload: dict[str, Any]) -> PramaanState:
        return PramaanState.model_validate(payload)
