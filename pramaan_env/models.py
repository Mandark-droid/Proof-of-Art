from __future__ import annotations

from typing import Any

from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class PramaanAction(Action):
    pattern_json: str = Field(max_length=20_000)


class PramaanObservation(Observation):
    task_id: str | None = None
    state: str | None = None
    tradition_id: str | None = None
    grammar: str | None = None
    brief: str | None = None
    palette: list[str] = Field(default_factory=list)
    constraint_profile: dict[str, Any] = Field(default_factory=dict)
    attempt: int = 0
    max_attempts: int = 3
    prompt_messages: list[dict[str, str]] = Field(default_factory=list)
    checks: list[dict[str, Any]] = Field(default_factory=list)
    all_hard_constraints_pass: bool | None = None
    certificate: dict[str, Any] | None = None
    parse_error: str | None = None


class PramaanState(State):
    task_id: str | None = None
    prompt_suite_sha256: str = ""
    validator_version: str = ""
    best_reward: float = 0.0
    accepted: bool = False
