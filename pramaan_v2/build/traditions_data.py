"""The traditions registry reshaped for the browser. Registry text is shown as written."""
from __future__ import annotations

from collections import Counter

from pramaan.tasks import load_suite
from pramaan.traditions import load_registry

from ..constants import STATE_CODES, STATE_ORDER


def _source(source) -> dict:
    return {"title": source.title, "publisher": source.publisher, "url": source.url, "supports": source.supports}


def traditions_data() -> dict:
    registry = load_registry()
    tasks, _ = load_suite("eval-v2.1")
    briefs: dict[str, dict[str, str]] = {}
    for task in tasks:
        briefs.setdefault(task.tradition_id, {})[task.level] = task.id
    entries = list(registry.values())
    grammar_counts = Counter(t.grammar for t in entries)
    states = []
    for state in STATE_ORDER:
        rows = []
        for t in entries:
            if state not in t.states:
                continue
            rows.append({
                "id": t.id, "name": t.name, "grammar": t.grammar, "category": t.category,
                "gi_registered": t.gi_registered, "confidence": t.confidence, "tier": t.tier,
                "states": t.states, "primary_state": t.primary_state, "tasks": briefs.get(t.id, {}),
                "motif_vocabulary": t.motif_vocabulary, "cautions": t.cautions,
                "palette_notes": t.palette_notes,
                "sources": [_source(s) for s in t.sources],
                "sources_unverified": [_source(s) for s in t.sources_unverified],
                "festival_lens": t.festival_lens.model_dump() if t.festival_lens else None,
            })
        states.append({"name": state, "code": STATE_CODES[state], "traditions": rows})
    return {
        "totals": {"traditions": len(entries), "sources": sum(len(t.sources) for t in entries),
                   "unverified": sum(len(t.sources_unverified) for t in entries),
                   "gi": sum(t.gi_registered is True for t in entries),
                   "multi_state": sum(len(t.states) > 1 for t in entries)},
        "by_grammar": dict(sorted(grammar_counts.items(), key=lambda item: (-item[1], item[0]))),
        "states": states,
    }
