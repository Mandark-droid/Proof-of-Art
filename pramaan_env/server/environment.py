from __future__ import annotations

import json
from uuid import uuid4

from pydantic import ValidationError
from openenv.core.env_server.interfaces import Environment

from pramaan import VALIDATOR_VERSION
from pramaan.certificate import make_certificate
from pramaan.checks.core import all_pass, verify_pattern
from pramaan.dsl import parse_pattern
from pramaan.orchestrate import apply_host_profile
from pramaan.prompts import design_prompt_for_task, repair_prompt_for_task, system_prompt_for_task
from pramaan.reward import reward_for_report
from pramaan.tasks import Task, load_suite

from ..models import PramaanAction, PramaanObservation, PramaanState


class PramaanEnvironment(Environment[PramaanAction, PramaanObservation, PramaanState]):
    SUPPORTS_CONCURRENT_SESSIONS = True
    MAX_ATTEMPTS = 3

    def __init__(self) -> None:
        super().__init__()
        self._task: Task | None = None
        self._suite_sha256 = ""
        self._state = PramaanState()
        self._attempt = 0
        self._prompt_messages: list[dict[str, str]] = []
        self._finished = False

    @property
    def state(self) -> PramaanState:
        return self._state.model_copy(deep=True)

    def _load_task(self, task_id: str | None, seed: int | None, suite: str) -> Task:
        tasks, digest = load_suite(suite)
        self._suite_sha256 = digest
        if task_id is not None:
            for task in tasks:
                if task.id == task_id:
                    return task
            raise ValueError(f"task_id {task_id!r} is not in suite {suite!r}")
        index = 0 if seed is None else seed % len(tasks)
        return tasks[index]

    def _design_messages(self) -> list[dict[str, str]]:
        assert self._task is not None
        return [
            {"role": "system", "content": system_prompt_for_task(self._task)},
            {"role": "user", "content": design_prompt_for_task(self._task)},
        ]

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        suite: str = "eval-v2.1",
        **kwargs,
    ) -> PramaanObservation:
        self._task = self._load_task(task_id, seed, suite)
        self._attempt = 0
        self._finished = False
        self._prompt_messages = self._design_messages()
        self._state = PramaanState(
            episode_id=episode_id or uuid4().hex,
            step_count=0,
            task_id=self._task.id,
            prompt_suite_sha256=self._suite_sha256,
            validator_version=VALIDATOR_VERSION,
            best_reward=0.0,
            accepted=False,
        )
        self._reset_rubric()
        return self._observation(reward=None, done=False)

    def _observation(
        self,
        *,
        reward: float | None,
        done: bool,
        checks: list[dict] | None = None,
        all_hard_constraints_pass: bool | None = None,
        certificate: dict | None = None,
        parse_error: str | None = None,
    ) -> PramaanObservation:
        assert self._task is not None
        return PramaanObservation(
            task_id=self._task.id,
            state=self._task.state,
            tradition_id=self._task.tradition_id,
            grammar=self._task.grammar,
            brief=self._task.brief,
            palette=self._task.palette,
            constraint_profile=self._task.constraint_profile,
            attempt=self._attempt,
            max_attempts=self.MAX_ATTEMPTS,
            prompt_messages=self._prompt_messages,
            reward=reward,
            done=done,
            checks=checks or [],
            all_hard_constraints_pass=all_hard_constraints_pass,
            certificate=certificate,
            parse_error=parse_error,
            metadata={"suite_id": self._task.suite_id},
        )

    def step(
        self,
        action: PramaanAction,
        timeout_s: float | None = None,
        **kwargs,
    ) -> PramaanObservation:
        if self._task is None:
            raise RuntimeError("call reset() before step()")
        if self._finished:
            raise RuntimeError("episode is done; call reset() before another step")

        self._attempt += 1
        self._state.step_count += 1
        parse_error = None
        certificate = None
        checks: list[dict] = []
        all_hard_pass = False
        pattern = None
        try:
            try:
                candidate = json.loads(action.pattern_json)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid PatternDSL JSON: {exc}") from exc
            candidate = apply_host_profile(candidate, self._task)
            pattern = parse_pattern(json.dumps(candidate))
            checks, masks, grid = verify_pattern(pattern)
            all_hard_pass = all_pass(checks)
            report = {"checks": checks, "all_hard_constraints_pass": all_hard_pass}
            reward = reward_for_report(report)
            if all_hard_pass:
                certificate = make_certificate(pattern, checks, masks, grid)
        except (ValueError, TypeError, json.JSONDecodeError, ValidationError) as exc:
            parse_error = str(exc)[:1000]
            checks = [{"id": "candidate_parse", "status": "FAIL", "assurance": "EXACT_DISCRETE", "message": parse_error}]
            reward = reward_for_report({"parse_error": parse_error, "checks": checks})

        self._state.best_reward = max(self._state.best_reward, reward)
        self._state.accepted = reward == 1.0
        done = self._state.accepted or self._attempt >= self.MAX_ATTEMPTS
        self._finished = done
        if not self._state.accepted:
            failures = [check for check in checks if check.get("status") == "FAIL"]
            if parse_error and not failures:
                failures = [{"id": "candidate_parse", "status": "FAIL", "message": parse_error}]
            previous = action.pattern_json
            if pattern is not None:
                previous = json.dumps(pattern.model_dump(mode="json"), sort_keys=True)
            self._prompt_messages = [
                {"role": "system", "content": system_prompt_for_task(self._task)},
                {"role": "user", "content": repair_prompt_for_task(self._task, previous, failures)},
            ]
        return self._observation(
            reward=reward,
            done=done,
            checks=checks,
            all_hard_constraints_pass=all_hard_pass,
            certificate=certificate,
            parse_error=parse_error,
        )
