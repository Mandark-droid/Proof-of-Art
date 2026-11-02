"""Saved Manipur example showing the benchmark's visibility gap."""

from __future__ import annotations

import base64
from html import escape
from pathlib import Path

from .visibility import accepted_pattern_is_low_visibility


_TASK_ID = "mn_moirang-phee-moirang-pheejin-stepped-triangle-border-easy-14"


def render_known_gap_panel(
    gallery_documents: object,
    task_backgrounds: object,
    reference_thumbnail: Path,
) -> str:
    documents = gallery_documents if isinstance(gallery_documents, dict) else {}
    backgrounds = task_backgrounds if isinstance(task_backgrounds, dict) else {}
    background = backgrounds.get(_TASK_ID)
    reference = ""
    try:
        reference_bytes = reference_thumbnail.read_bytes()
        reference = "data:image/png;base64," + base64.b64encode(reference_bytes).decode("ascii")
    except OSError:
        pass

    for model_key in sorted(documents):
        tasks = documents[model_key].get("tasks") if isinstance(documents[model_key], dict) else None
        task = tasks.get(_TASK_ID) if isinstance(tasks, dict) else None
        attempts = task.get("attempts") if isinstance(task, dict) else None
        accepted = next((row for row in reversed(attempts or [])
                         if isinstance(row, dict) and row.get("passed") is True), None)
        if accepted is None or not accepted_pattern_is_low_visibility(accepted, background):
            continue
        tile = accepted.get("tile_png_b64")
        reward = task.get("reward")
        if not isinstance(tile, str) or type(reward) not in (int, float):
            continue
        reference_html = (
            f'<figure><img src="{reference}" alt="Visible Manipur reference tile">'
            '<figcaption>Visible reference</figcaption></figure>' if reference else
            '<figure><figcaption>Visible reference thumbnail unavailable</figcaption></figure>'
        )
        return (
            '<article id="pramaan-known-gap" class="pramaan-ui-card pramaan-known-gap" '
            'data-testid="known-visibility-gap"><p class="pramaan-ui-eyebrow">Reward gap found before training · Manipur</p>'
            '<h3>Geometry can pass while the tile is nearly blank</h3>'
            '<p>None of these models was trained on this reward, so this is not learned gaming. '
            'It is the hole a trained policy could find. The brief asks for black ink, and the host fixes '
            'the background to the same colour. We kept the frozen scores; a later validator can add a visibility rule.</p>'
            '<div class="pramaan-known-gap-tiles">' + reference_html +
            f'<figure><img src="data:image/png;base64,{escape(tile, quote=True)}" '
            'alt="Accepted Manipur model design with little or no visible ink">'
            f'<figcaption>Accepted design · reward {reward:.1f}</figcaption></figure></div>'
            f'<p class="pramaan-ui-muted">Task <code>{escape(_TASK_ID)}</code></p></article>'
        )
    return (
        '<article id="pramaan-known-gap" class="pramaan-ui-card pramaan-known-gap" '
        'data-testid="known-visibility-gap"><p class="pramaan-ui-eyebrow">Reward gap found before training · Manipur</p>'
        '<p>The saved Manipur example is unavailable in this Space build.</p></article>'
    )
