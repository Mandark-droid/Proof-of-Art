from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .paths import resolve_project_root

try:
    from pramaan.dsl import GRAMMARS
except ImportError:  # TODO: remove when the public v0.2 interface is available.
    GRAMMARS = (
        "periodic_raster",
        "dot_field",
        "stripe_band",
        "counted_grid",
        "rotational_field",
        "kolam_loop",
        "jaali_lattice",
    )

ALL_STATES = (
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chhattisgarh",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
    "Andaman and Nicobar Islands",
    "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi",
    "Jammu and Kashmir",
    "Ladakh",
    "Lakshadweep",
    "Puducherry",
)


class TraditionSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    title: str
    publisher: str
    supports: str = ""


class FestivalLens(BaseModel):
    model_config = ConfigDict(extra="forbid")

    surface: Literal["floor", "wall"] | None
    diwali_status: Literal["sourced_institutional", "sourced_secondary", "none"]
    claim_ids: list[str] = Field(default_factory=list)


class Tradition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    states: list[str]
    state_types: list[Literal["state", "union_territory"]]
    grammar: str
    category: str
    palette_presets: dict[str, list[str]]
    palette_notes: list[dict[str, str]]
    default_profile: dict
    motif_vocabulary: list[str]
    sources: list[TraditionSource]
    sources_unverified: list[TraditionSource] = Field(default_factory=list)
    cautions: str
    tier: Literal["A", "B"]
    gi_registered: bool | None
    priority: Literal["primary", "alternate"]
    confidence: Literal["high", "medium", "low"]
    draft_briefs: dict[str, str]
    primary_state: str
    festival_lens: FestivalLens | None = None

    @field_validator("grammar")
    @classmethod
    def grammar_is_known(cls, value: str) -> str:
        if value not in GRAMMARS:
            raise ValueError(f"unknown grammar: {value}")
        return value

    @field_validator("states")
    @classmethod
    def states_are_canonical(cls, value: list[str]) -> list[str]:
        if not value or any(state not in ALL_STATES for state in value):
            raise ValueError("tradition states must use canonical Census names")
        return value


@lru_cache(maxsize=1)
def load_registry() -> dict[str, Tradition]:
    package_path = Path(__file__).parent / "data" / "traditions.json"
    project_path = resolve_project_root(__file__) / "pramaan" / "data" / "traditions.json"
    path = package_path if package_path.is_file() else project_path
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = [Tradition.model_validate(item) for item in raw["entries"]]
    registry = {item.id: item for item in entries}
    if len(registry) != len(entries):
        raise ValueError("duplicate tradition id in registry")
    return registry


def traditions_for_state(state: str) -> list[Tradition]:
    if state not in ALL_STATES:
        return []
    return sorted(
        (tradition for tradition in load_registry().values() if tradition.primary_state == state),
        key=lambda tradition: (tradition.primary_state != state, tradition.priority != "primary", tradition.id),
    )
