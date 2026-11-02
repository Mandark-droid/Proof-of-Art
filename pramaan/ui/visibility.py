"""UI visibility warnings using the saved trajectory audit's threshold."""

from __future__ import annotations

import base64
from io import BytesIO
import re

from PIL import Image

from pramaan.visibility import MIN_VISIBLE_SHARE


_BASE64 = re.compile(r"[A-Za-z0-9+/]+={0,2}\Z")
_HEX_COLOUR = re.compile(r"#[0-9a-fA-F]{6}\Z")


def accepted_pattern_is_low_visibility(attempt: object, background_colour: object) -> bool:
    """Check the accepted tile's pixels against its host-owned palette-0 ground."""
    if not isinstance(attempt, dict) or attempt.get("passed") is not True:
        return False
    tile = attempt.get("tile_png_b64")
    if not isinstance(tile, str) or not 0 < len(tile) <= 50_000 or not _BASE64.fullmatch(tile):
        return False
    if not isinstance(background_colour, str) or not _HEX_COLOUR.fullmatch(background_colour):
        return False
    try:
        decoded = base64.b64decode(tile, validate=True)
        if not decoded.startswith(b"\x89PNG\r\n\x1a\n"):
            return False
        with Image.open(BytesIO(decoded)) as image:
            if image.width <= 0 or image.height <= 0 or image.width * image.height > 1_048_576:
                return False
            counts = image.convert("RGB").getcolors(image.width * image.height)
            if counts is None:
                return False
    except (ValueError, OSError, base64.binascii.Error):
        return False
    background = tuple(int(background_colour[index:index + 2], 16) for index in (1, 3, 5))
    total_pixels = sum(count for count, _colour in counts)
    visible_pixels = sum(count for count, colour in counts if colour != background)
    visible_share = visible_pixels / total_pixels if total_pixels else 0.0
    return visible_share < MIN_VISIBLE_SHARE


def accepted_visibility_counts(document: object, task_backgrounds: object) -> tuple[int, int] | None:
    """Return low-visibility and accepted counts from a saved model gallery."""
    tasks = document.get("tasks") if isinstance(document, dict) else None
    if not isinstance(tasks, dict) or not isinstance(task_backgrounds, dict):
        return None
    low_visibility = accepted = 0
    for task_id, task in tasks.items():
        attempts = task.get("attempts") if isinstance(task, dict) else None
        attempts = attempts if isinstance(attempts, list) else []
        accepted_attempt = next((attempt for attempt in reversed(attempts)
                                 if isinstance(attempt, dict) and attempt.get("passed") is True), None)
        if accepted_attempt is None:
            continue
        background = task_backgrounds.get(task_id)
        if (not isinstance(background, str) or not _HEX_COLOUR.fullmatch(background)
                or not isinstance(accepted_attempt.get("tile_png_b64"), str)):
            return None
        accepted += 1
        low_visibility += accepted_pattern_is_low_visibility(accepted_attempt, background)
    return low_visibility, accepted
