"""Deterministic integer-grid rasterization for PatternDSL."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from itertools import product

import numpy as np
from PIL import Image, ImageDraw

from .dsl import Layer, Motif, Pattern, PatternV2

N = 64


def _draw_motif(draw: ImageDraw.ImageDraw, motif: Motif, dx: int, dy: int) -> None:
    def xy(p):
        return int(p[0]) + dx, int(p[1]) + dy

    if motif.kind == "dot":
        x, y = xy(motif.center)
        r = motif.radius
        draw.ellipse((x-r, y-r, x+r, y+r), fill=1)
    elif motif.kind == "diamond":
        x, y = xy(motif.center)
        r = motif.radius
        draw.polygon([(x, y-r), (x+r, y), (x, y+r), (x-r, y)], fill=1)
    elif motif.kind == "rectangle":
        x, y = xy(motif.center)
        w, h = motif.size
        draw.rectangle((x-w//2, y-h//2, x+(w-1)//2, y+(h-1)//2), fill=1)
    elif motif.kind == "line":
        draw.line([xy(p) for p in motif.points], fill=1, width=motif.stroke)
    elif motif.kind == "polygon":
        draw.polygon([xy(p) for p in motif.points], fill=1)


def _motif_bounds(motif: Motif, anchor: tuple[int, int]) -> tuple[int, int, int, int]:
    """Return conservative inclusive raster bounds in the anchored coordinate frame."""
    if motif.kind in {"dot", "diamond"}:
        x, y = motif.center
        radius = motif.radius
        return (x + anchor[0] - radius, y + anchor[1] - radius,
                x + anchor[0] + radius, y + anchor[1] + radius)
    if motif.kind == "rectangle":
        x, y = motif.center
        width, height = motif.size
        return (x + anchor[0] - width // 2,
                y + anchor[1] - height // 2,
                x + anchor[0] + (width - 1) // 2,
                y + anchor[1] + (height - 1) // 2)

    xs = [point[0] + anchor[0] for point in motif.points]
    ys = [point[1] + anchor[1] for point in motif.points]
    padding = (motif.stroke + 1) // 2 if motif.kind == "line" else 0
    return min(xs) - padding, min(ys) - padding, max(xs) + padding, max(ys) + padding


def _periodic_offsets(minimum: int, maximum: int) -> range:
    """Return every tile translation whose interval can touch raster coordinates 0..63."""
    first_copy = -(maximum // N)  # ceil(-maximum / N), using integer floor division.
    last_copy = (N - 1 - minimum) // N
    return range(first_copy * N, (last_copy + 1) * N, N)


def _motif_translations(motif: Motif, anchor: tuple[int, int]) -> Iterator[tuple[int, int]]:
    min_x, min_y, max_x, max_y = _motif_bounds(motif, anchor)
    return product(_periodic_offsets(min_x, max_x), _periodic_offsets(min_y, max_y))


def render_motif(motif: Motif, anchor: tuple[int, int] = (0, 0)) -> np.ndarray:
    image = Image.new("1", (N, N), 0)
    draw = ImageDraw.Draw(image)
    for dx, dy in _motif_translations(motif, anchor):
        _draw_motif(draw, motif, dx + anchor[0], dy + anchor[1])
    return np.asarray(image, dtype=bool)


def render_layer(layer: Layer) -> np.ndarray:
    image = Image.new("1", (N, N), 0)
    draw = ImageDraw.Draw(image)
    for motif in layer.motifs:
        # Draw every translated copy whose full footprint can touch the canonical tile.
        for dx, dy in _motif_translations(motif, layer.anchor):
            _draw_motif(draw, motif, dx + layer.anchor[0], dy + layer.anchor[1])
    return np.asarray(image, dtype=bool)


def render(pattern: Pattern | PatternV2) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if isinstance(pattern, PatternV2) and pattern.grammar == "kolam_loop":
        from .render_kolam import render_kolam

        masks, _ = render_kolam(pattern.kolam, len(pattern.palette))
        grid = np.full((N, N), pattern.background_index, dtype=np.uint8)
        curve_colour = pattern.layers[0].colour_index
        dots_colour = pattern.layers[1].colour_index if len(pattern.layers) > 1 else curve_colour
        grid[masks["curve"]] = curve_colour
        grid[masks["dots"]] = dots_colour
        return masks, grid
    if isinstance(pattern, PatternV2) and pattern.grammar == "jaali_lattice":
        from .render_jaali import render_jaali

        masks, _ = render_jaali(pattern.jaali, pattern.layers, (pattern.tile.wrap_x, pattern.tile.wrap_y))
        grid = np.full((N, N), pattern.background_index, dtype=np.uint8)
        if not any(layer.motifs for layer in pattern.layers):
            grid[masks["solid"]] = pattern.layers[0].colour_index
        else:
            for layer in pattern.layers:
                grid[masks[layer.id] & masks["solid"]] = layer.colour_index
        grid[masks["holes"]] = pattern.jaali.open_colour_index
        return masks, grid
    masks = {layer.id: render_layer(layer) for layer in pattern.layers}
    grid = np.full((N, N), pattern.background_index, dtype=np.uint8)
    for layer in pattern.layers:
        grid[masks[layer.id]] = layer.colour_index
    return masks, grid


def tiled_preview(grid: np.ndarray, repeats: int = 3, scale: int = 4) -> Image.Image:
    tiled = np.tile(grid, (repeats, repeats))
    # Palette indices are mapped by export.render_png; this grayscale helper is for tests only.
    return Image.fromarray(tiled, mode="L").resize((N * repeats * scale, N * repeats * scale), Image.Resampling.NEAREST)


def mask_digest(masks: Mapping[str, np.ndarray], grid: np.ndarray) -> str:
    """Hash canonical raster arrays with length-delimited layer identities."""
    import hashlib

    digest = hashlib.sha256()
    digest.update(b"pramaan-mask-digest-v1\0")

    def write_bytes(value: bytes) -> None:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)

    def write_array(array: np.ndarray) -> None:
        canonical = np.ascontiguousarray(array)
        write_bytes(canonical.dtype.str.encode("ascii"))
        digest.update(canonical.ndim.to_bytes(4, "big"))
        for dimension in canonical.shape:
            digest.update(int(dimension).to_bytes(8, "big"))
        write_bytes(canonical.tobytes(order="C"))

    write_array(grid)
    keys = sorted(masks)
    digest.update(len(keys).to_bytes(8, "big"))
    for key in keys:
        write_bytes(key.encode("utf-8"))
        write_array(masks[key].astype(np.uint8))
    return digest.hexdigest()
