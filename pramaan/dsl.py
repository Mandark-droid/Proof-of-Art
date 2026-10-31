"""Strict, non-executable PatternDSL v0.1 models."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator, model_validator

Coord = tuple[StrictInt, StrictInt]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Tile(StrictModel):
    width: StrictInt = 64
    height: StrictInt = 64
    wrap_x: bool = True
    wrap_y: bool = True

    @model_validator(mode="after")
    def fixed_grid(self):
        if self.width != 64 or self.height != 64:
            raise ValueError("PatternDSL v0.1 supports a fixed 64x64 tile")
        if not self.wrap_x or not self.wrap_y:
            raise ValueError("PatternDSL v0.1 requires wrapping on both axes")
        return self


class Motif(StrictModel):
    kind: Literal["dot", "diamond", "rectangle", "line", "polygon"]
    center: Coord | None = None
    radius: Annotated[StrictInt, Field(ge=1, le=64)] | None = None
    stroke: Annotated[StrictInt, Field(ge=1, le=16)] | None = None
    size: Coord | None = None
    points: list[Coord] | None = None

    @model_validator(mode="after")
    def valid_shape(self):
        allowed_fields = {
            "dot": {"center", "radius"},
            "diamond": {"center", "radius"},
            "rectangle": {"center", "size"},
            "line": {"points", "stroke"},
            "polygon": {"points"},
        }[self.kind]
        present_fields = {
            name for name in ("center", "radius", "stroke", "size", "points")
            if getattr(self, name) is not None
        }
        unsupported = present_fields - allowed_fields
        if unsupported:
            raise ValueError(f"{self.kind} does not support: {', '.join(sorted(unsupported))}")
        coords = ([self.center] if self.center is not None else []) + (self.points or [])
        if any(not (-64 <= p[0] <= 127 and -64 <= p[1] <= 127) for p in coords):
            raise ValueError("coordinates must be within the supported one-tile wrap range [-64, 127]")
        needs_center = self.kind in {"dot", "diamond", "rectangle"}
        if needs_center and self.center is None:
            raise ValueError(f"{self.kind} requires center")
        if self.kind in {"dot", "diamond"} and self.radius is None:
            raise ValueError(f"{self.kind} requires radius")
        if self.kind == "rectangle" and self.size is None:
            raise ValueError("rectangle requires size")
        if self.size is not None and any(not (1 <= value <= 128) for value in self.size):
            raise ValueError("rectangle dimensions must be between 1 and 128")
        if self.kind in {"line", "polygon"}:
            if self.points is None or len(self.points) < (2 if self.kind == "line" else 3):
                raise ValueError(f"{self.kind} requires at least {2 if self.kind == 'line' else 3} points")
            if len(self.points) > 12:
                raise ValueError("polygon/line supports at most 12 points")
        if self.kind == "line" and self.stroke is None:
            raise ValueError("line requires stroke")
        return self


class Layer(StrictModel):
    id: Annotated[StrictStr, Field(min_length=1, max_length=40)]
    colour_index: Annotated[StrictInt, Field(ge=0, le=5)]
    anchor: Coord = (0, 0)
    motifs: list[Motif] = Field(max_length=100)

    @model_validator(mode="after")
    def anchor_bounds(self):
        if any(not (-64 <= value <= 127) for value in self.anchor):
            raise ValueError("layer anchor must be within [-64, 127]")
        for motif in self.motifs:
            points = ([motif.center] if motif.center is not None else []) + (motif.points or [])
            if any(not (-64 <= p[0] + self.anchor[0] <= 127 and -64 <= p[1] + self.anchor[1] <= 127) for p in points):
                raise ValueError("motif coordinates plus layer anchor exceed the supported wrap range")
        return self


class Constraints(StrictModel):
    symmetry: Literal["None", "D2", "D4"] = "D4"
    max_colours: Annotated[StrictInt, Field(ge=1, le=6)] = 4
    max_primitives: Annotated[StrictInt, Field(ge=1, le=100)] = 100
    min_gap: Annotated[StrictInt, Field(ge=0, le=8)] = 0
    min_coverage: float = Field(default=0.02, ge=0, le=1)
    max_coverage: float = Field(default=0.70, ge=0, le=1)
    exclusive_layer_pairs: list[tuple[StrictStr, StrictStr]] = Field(default_factory=list, max_length=6)

    @model_validator(mode="after")
    def ordered_coverage(self):
        if self.min_coverage > self.max_coverage:
            raise ValueError("min_coverage must be <= max_coverage")
        return self


class Pattern(StrictModel):
    schema_version: Literal["0.1"]
    style: Literal["ajrakh_inspired"]
    title: Annotated[StrictStr, Field(min_length=1, max_length=80)]
    tile: Tile
    palette: list[Annotated[StrictStr, Field(pattern=r"^#[0-9A-Fa-f]{6}$")]] = Field(min_length=1, max_length=6)
    background_index: Annotated[StrictInt, Field(ge=0, le=5)]
    layers: list[Layer] = Field(min_length=1, max_length=4)
    constraints: Constraints

    @field_validator("palette")
    @classmethod
    def canonical_palette(cls, values):
        canonical = [value.upper() for value in values]
        if len(set(canonical)) != len(canonical):
            raise ValueError("palette colours must be unique")
        return canonical

    @model_validator(mode="after")
    def cross_validate(self):
        if self.background_index >= len(self.palette):
            raise ValueError("background_index must refer to a palette entry")
        if len({layer.id for layer in self.layers}) != len(self.layers):
            raise ValueError("layer ids must be unique")
        for layer in self.layers:
            if layer.colour_index >= len(self.palette):
                raise ValueError(f"layer {layer.id} has an unknown colour_index")
            if len(layer.motifs) > self.constraints.max_primitives:
                raise ValueError("layer exceeds max_primitives")
        if sum(len(layer.motifs) for layer in self.layers) > self.constraints.max_primitives:
            raise ValueError("pattern exceeds max_primitives")
        layer_ids = {layer.id for layer in self.layers}
        for a, b in self.constraints.exclusive_layer_pairs:
            if a not in layer_ids or b not in layer_ids or a == b:
                raise ValueError("exclusive_layer_pairs must name two different existing layers")
        return self


GRAMMARS = (
    "periodic_raster",
    "dot_field",
    "stripe_band",
    "counted_grid",
    "rotational_field",
    "kolam_loop",
    "jaali_lattice",
)


class TileV2(StrictModel):
    width: Literal[64] = 64
    height: Literal[64] = 64
    wrap_x: bool = True
    wrap_y: bool = True


class TranslationPeriod(StrictModel):
    axis: Literal["x", "y", "xy"]
    period: Annotated[StrictInt, Field(ge=1, le=64)]


class KolamSpec(StrictModel):
    rows: Annotated[StrictInt, Field(ge=1, le=9)]
    cols: Annotated[StrictInt, Field(ge=1, le=9)]
    spacing: Literal[6, 8, 10, 12]
    mirrors: list[Coord] = Field(max_length=40)
    stroke: Annotated[StrictInt, Field(ge=1, le=3)]
    dot_radius: Annotated[StrictInt, Field(ge=0, le=2)]

    @field_validator("mirrors")
    @classmethod
    def unique_mirrors(cls, mirrors):
        if len(set(mirrors)) != len(mirrors):
            raise ValueError("Kolam mirror positions must be unique")
        return mirrors

    @model_validator(mode="after")
    def fits_fixed_grid(self):
        if (self.cols + 1) * self.spacing > 64:
            raise ValueError("Kolam dot grid width exceeds the 64x64 canvas")
        if (self.rows + 1) * self.spacing > 64:
            raise ValueError("Kolam dot grid height exceeds the 64x64 canvas")
        return self


class JaaliSpec(StrictModel):
    frame: Annotated[StrictInt, Field(ge=0, le=12)]
    open_colour_index: Annotated[StrictInt, Field(ge=0, le=5)]
    holes: list[Motif] = Field(max_length=100)


class ConstraintsV2(Constraints):
    symmetry: Literal["None", "D2", "D4", "C2", "C4"] = "D4"
    min_clearance: Annotated[StrictInt, Field(ge=0, le=64)] | None = None
    translation_period: TranslationPeriod | None = None
    cell_size: Literal[2, 4, 8] | None = None
    required_loops: Annotated[StrictInt, Field(ge=1, le=81)] | None = None
    min_open_fraction: float | None = Field(default=None, ge=0, le=1)
    max_open_fraction: float | None = Field(default=None, ge=0, le=1)
    min_web: Annotated[StrictInt, Field(ge=0, le=32)] | None = None

    @model_validator(mode="after")
    def ordered_open_fraction(self):
        if (
            self.min_open_fraction is not None
            and self.max_open_fraction is not None
            and self.min_open_fraction > self.max_open_fraction
        ):
            raise ValueError("min_open_fraction must be <= max_open_fraction")
        return self


class PatternV2(StrictModel):
    schema_version: Literal["0.2"]
    grammar: Literal[
        "periodic_raster", "dot_field", "stripe_band", "counted_grid",
        "rotational_field", "kolam_loop", "jaali_lattice",
    ]
    tradition_id: Annotated[StrictStr, Field(min_length=1, max_length=80)]
    title: Annotated[StrictStr, Field(min_length=1, max_length=80)]
    tile: TileV2
    palette: list[Annotated[StrictStr, Field(pattern=r"^#[0-9A-Fa-f]{6}$")]] = Field(min_length=1, max_length=6)
    background_index: Annotated[StrictInt, Field(ge=0, le=5)]
    layers: list[Layer] = Field(min_length=1, max_length=4)
    constraints: ConstraintsV2
    kolam: KolamSpec | None = None
    jaali: JaaliSpec | None = None

    @field_validator("palette")
    @classmethod
    def canonical_palette(cls, values):
        canonical = [value.upper() for value in values]
        if len(set(canonical)) != len(canonical):
            raise ValueError("palette colours must be unique")
        return canonical

    @model_validator(mode="after")
    def cross_validate(self):
        if self.background_index >= len(self.palette):
            raise ValueError("background_index must refer to a palette entry")
        if len({layer.id for layer in self.layers}) != len(self.layers):
            raise ValueError("layer ids must be unique")
        for layer in self.layers:
            if layer.colour_index >= len(self.palette):
                raise ValueError(f"layer {layer.id} has an unknown colour_index")
        if sum(len(layer.motifs) for layer in self.layers) > self.constraints.max_primitives:
            raise ValueError("pattern exceeds max_primitives")
        layer_ids = {layer.id for layer in self.layers}
        for a, b in self.constraints.exclusive_layer_pairs:
            if a not in layer_ids or b not in layer_ids or a == b:
                raise ValueError("exclusive_layer_pairs must name two different existing layers")

        if self.grammar == "kolam_loop":
            if self.kolam is None or self.jaali is not None:
                raise ValueError("kolam_loop requires only a kolam block")
            if self.tile.wrap_x or self.tile.wrap_y:
                raise ValueError("kolam_loop uses a bounded canvas without wrapping")
        elif self.kolam is not None:
            raise ValueError("kolam block is only valid for kolam_loop")

        if self.grammar == "jaali_lattice":
            if self.jaali is None:
                raise ValueError("jaali_lattice requires a jaali block")
            if self.jaali.open_colour_index >= len(self.palette):
                raise ValueError("jaali open_colour_index must refer to a palette entry")
            if (self.tile.wrap_x or self.tile.wrap_y) and self.jaali.frame != 0:
                raise ValueError("a wrapped jaali lattice must have frame=0")
            if self.tile.wrap_x != self.tile.wrap_y:
                raise ValueError("jaali_lattice must wrap both axes or neither")
        elif self.jaali is not None:
            raise ValueError("jaali block is only valid for jaali_lattice")

        if not (self.tile.wrap_x and self.tile.wrap_y) and self.grammar not in {
            "rotational_field", "kolam_loop", "jaali_lattice"
        }:
            raise ValueError("unwrapped tiles are only valid for bounded grammars")
        if self.grammar == "dot_field" and any(
            motif.kind != "dot" for layer in self.layers for motif in layer.motifs
        ):
            raise ValueError("dot_field layers may contain dot motifs only")

        grammar_fields = {
            "min_clearance": "dot_field",
            "translation_period": "stripe_band",
            "cell_size": "counted_grid",
            "required_loops": "kolam_loop",
            "min_open_fraction": "jaali_lattice",
            "max_open_fraction": "jaali_lattice",
            "min_web": "jaali_lattice",
        }
        for field_name, grammar in grammar_fields.items():
            if getattr(self.constraints, field_name) is not None and self.grammar != grammar:
                raise ValueError(f"{field_name} is only valid for {grammar}")
        if (self.constraints.min_open_fraction is None) != (self.constraints.max_open_fraction is None):
            raise ValueError("min_open_fraction and max_open_fraction must be specified together")
        return self


def parse_pattern(text: str) -> Pattern | PatternV2:
    """Parse a supported PatternDSL version without allowing version fallthrough."""
    try:
        document = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid PatternDSL JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("PatternDSL document must be a JSON object")
    version = document.get("schema_version")
    if version == "0.1":
        return Pattern.model_validate_json(text)
    if version == "0.2":
        return PatternV2.model_validate_json(text)
    raise ValueError(f"unsupported schema_version: {version!r}")
