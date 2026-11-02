from __future__ import annotations

import json
from pathlib import Path

from pramaan_env.reward_fn import pramaan_reward


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROWS = ROOT / "benchmarks" / "data" / "train-v1-trajectories.jsonl"


def _saved_candidate(prompt_id: str, call_index: int) -> str:
    for line in TRAIN_ROWS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("prompt_id") != prompt_id or row.get("suite_id") != "train-v1":
            continue
        call = next(item for item in row["calls"] if item["call_index"] == call_index)
        return json.dumps(call["candidate"])
    raise AssertionError(f"saved train-v1 candidate not found: {prompt_id}/{call_index}")


def test_grpo_reward_runs_saved_train_v1_pass_partial_and_garbage():
    passing = _saved_candidate("ap_mangalagiri-sarees-and-fabrics-train-3", 2)
    failing = _saved_candidate("ap_mangalagiri-sarees-and-fabrics-train-1", 2)

    rewards = pramaan_reward(
        [passing, failing, "not JSON"],
        [
            "ap_mangalagiri-sarees-and-fabrics-train-3",
            "ap_mangalagiri-sarees-and-fabrics-train-1",
            "ap_mangalagiri-sarees-and-fabrics-train-1",
        ],
    )

    assert rewards[0] == 1.0
    assert 0.2 <= rewards[1] < 0.8
    assert rewards[2] == 0.0
