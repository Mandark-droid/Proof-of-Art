"""Site data loaded once per process; the data is frozen for the life of the container."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

SITE_DATA = Path(__file__).resolve().parents[1] / "site_data"


def _read(root: Path, name: str):
    return json.loads((Path(root) / name).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class SiteData:
    root: Path = SITE_DATA

    @cached_property
    def meta(self) -> dict:
        return _read(self.root, "meta.json")

    @cached_property
    def wall(self) -> dict:
        return _read(self.root, "wall.json")

    @cached_property
    def leaderboard(self) -> dict:
        return _read(self.root, "leaderboard.json")

    @cached_property
    def kolam(self) -> dict:
        return _read(self.root, "kolam.json")

    @cached_property
    def traditions(self) -> dict:
        return _read(self.root, "traditions.json")

    @cached_property
    def models(self) -> list[dict]:
        return self.meta["models"]

    @cached_property
    def model_index(self) -> dict[str, int]:
        return {m["key"]: i for i, m in enumerate(self.models)}

    @cached_property
    def task_ids(self) -> list[str]:
        return [task for state in self.meta["states"] for task in (state["easy"], state["hard"])]

    def replay(self, task_id: str) -> dict | None:
        return _replay(self.root, task_id) if task_id in self._task_set else None

    @cached_property
    def _task_set(self) -> frozenset[str]:
        return frozenset(self.task_ids)


@lru_cache(maxsize=256)
def _replay(root: Path, task_id: str) -> dict:
    return _read(root, f"replays/{task_id}.json")


@lru_cache(maxsize=1)
def site_data() -> SiteData:
    return SiteData()
