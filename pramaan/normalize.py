"""Strict, model-neutral cleanup for two common JSON presentation wrappers."""

from __future__ import annotations

import re


_OUTER_FENCE = re.compile(r"\A```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```[ \t]*\Z")
_LEADING_THINK = re.compile(r"\A(?:<think>[\s\S]*?</think>[ \t\r\n]*)+")


def normalize_model_text(raw: str) -> tuple[str, list[str]]:
    """Remove one whole-response JSON fence and leading think blocks only.

    This intentionally does not search for JSON inside prose, repair malformed
    JSON, or combine multiple objects. Callers still need to parse and validate
    the returned text.
    """
    if not isinstance(raw, str):
        raise TypeError("model response must be text")

    text = raw.strip()
    actions: list[str] = []

    fence = _OUTER_FENCE.fullmatch(text)
    if fence:
        text = fence.group(1).strip()
        actions.append("strip_code_fence")

    think = _LEADING_THINK.match(text)
    if think:
        remainder = text[think.end():].lstrip()
        # Remove reasoning blocks only when they directly precede one JSON
        # object. Otherwise preserve the response so ordinary parsing fails.
        if remainder.startswith("{"):
            text = remainder
            actions.append("strip_think_block")

    return text, actions
