"""Rendered pages kept in memory, gzip-compressed, so a request does no rendering.

The site data is frozen for the life of the container, so every read-only page can be
rendered once (ahead of time by ``routes.warm_pages``) and then served as stored bytes.
"""
from __future__ import annotations

import gzip
import threading
from collections import OrderedDict
from typing import Callable

MAX_ENTRIES = 4096
_pages: OrderedDict[str, bytes] = OrderedDict()
_lock = threading.Lock()


def clear() -> None:
    with _lock:
        _pages.clear()


def keys() -> list[str]:
    with _lock:
        return list(_pages)


def get_or_render(key: str, render: Callable[[], str | None]) -> bytes | None:
    """Gzip bytes of the page for ``key``; ``None`` (never cached) when the page does not exist."""
    with _lock:
        cached = _pages.get(key)
        if cached is not None:
            _pages.move_to_end(key)
            return cached
    content = render()
    if content is None:
        return None
    data = gzip.compress(content.encode("utf-8"), compresslevel=6, mtime=0)
    with _lock:
        _pages[key] = data
        while len(_pages) > MAX_ENTRIES:
            _pages.popitem(last=False)
    return data
