"""Rasterize a bounded or periodic jaali panel and its explicit holes."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .dsl import JaaliSpec, Layer
from .render import N, _draw_motif, render_layer, render_motif


def _bounded_layer(layer: Layer) -> np.ndarray:
    image = Image.new("1", (N, N), 0)
    draw = ImageDraw.Draw(image)
    for motif in layer.motifs:
        _draw_motif(draw, motif, layer.anchor[0], layer.anchor[1])
    return np.asarray(image, dtype=bool)


def _motif_mask(motif, wrap: bool) -> np.ndarray:
    if wrap:
        return render_motif(motif)
    image = Image.new("1", (N, N), 0)
    _draw_motif(ImageDraw.Draw(image), motif, 0, 0)
    return np.asarray(image, dtype=bool)


def render_jaali(spec: JaaliSpec, layers: list[Layer], wrap: bool | tuple[bool, bool]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    wrap_x, wrap_y = (wrap, wrap) if isinstance(wrap, bool) else wrap
    wrapped = wrap_x and wrap_y
    layer_masks = [render_layer(layer) if wrapped else _bounded_layer(layer) for layer in layers]
    if any(layer.motifs for layer in layers):
        base = np.zeros((N, N), dtype=bool)
        for mask in layer_masks:
            base |= mask
    else:
        base = np.ones((N, N), dtype=bool)

    holes = np.zeros((N, N), dtype=bool)
    for motif in spec.holes:
        holes |= _motif_mask(motif, wrapped)
    solid = base & ~holes

    grid = np.zeros((N, N), dtype=np.uint8)
    for layer, layer_mask in zip(layers, layer_masks):
        grid[layer_mask] = layer.colour_index
    grid[holes] = spec.open_colour_index
    masks = {"solid": solid, "holes": holes}
    masks.update({layer.id: layer_mask for layer, layer_mask in zip(layers, layer_masks)})
    return masks, grid
