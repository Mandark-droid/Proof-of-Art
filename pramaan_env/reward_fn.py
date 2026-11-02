"""TRL-compatible reward adapter for Pramaan's train-v1 environment."""

from __future__ import annotations

from pramaan_env import PramaanAction
from pramaan_env.server.environment import PramaanEnvironment


def pramaan_reward(completions, task_id, **kwargs):
    rewards = []
    for completion, task in zip(completions, task_id):
        text = completion if isinstance(completion, str) else completion[0]["content"]
        env = PramaanEnvironment()
        env.reset(task_id=task, suite="train-v1")
        rewards.append(env.step(PramaanAction(pattern_json=text)).reward)
    return rewards
