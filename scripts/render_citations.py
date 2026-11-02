"""Render registry sources marked verified into the checked-in citations index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "pramaan" / "data" / "traditions.json"
CITATIONS = ROOT / "CITATIONS.md"
BEGIN = "<!-- BEGIN GENERATED TRADITION SOURCES -->"
END = "<!-- END GENERATED TRADITION SOURCES -->"


def render_source_index(registry_path: Path = REGISTRY) -> str:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("tradition registry must contain an entries list")

    rows = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each tradition registry entry must be an object")
        name = entry.get("name")
        states = entry.get("states")
        sources = entry.get("sources", [])
        if not isinstance(name, str) or not isinstance(states, list) or not isinstance(sources, list):
            raise ValueError("tradition entries need a name, states list, and sources list")
        for source in sources:
            if not isinstance(source, dict) or source.get("verified", True) is not True:
                continue
            title, url = source.get("title"), source.get("url")
            publisher = source.get("publisher", "")
            parsed = urlsplit(url) if isinstance(url, str) else None
            if not isinstance(title, str) or not title.strip() or not parsed or parsed.scheme != "https" or not parsed.hostname:
                continue
            for state in states:
                if isinstance(state, str) and state.strip():
                    rows.add((state.strip(), name.strip(), title.strip(), url, publisher.strip() if isinstance(publisher, str) else ""))

    lines = ["### Registry-verified sources by state and tradition", ""]
    previous = None
    for state, name, title, url, publisher in sorted(rows, key=lambda row: (row[0].casefold(), row[1].casefold(), row[2].casefold(), row[3])):
        group = (state, name)
        if group != previous:
            lines.append(f"#### {state} — {name}")
            lines.append("")
            previous = group
        label = title.replace("[", "\\[").replace("]", "\\]")
        suffix = f" — {publisher}" if publisher else ""
        lines.append(f"- [{label}]({url}){suffix}")
    if len(lines) == 2:
        lines.append("_No verified tradition sources are present in the registry._")
    return "\n".join(lines)


def update_citations(registry_path: Path = REGISTRY, citations_path: Path = CITATIONS) -> None:
    document = citations_path.read_text(encoding="utf-8")
    if document.count(BEGIN) != 1 or document.count(END) != 1:
        raise ValueError("CITATIONS.md must contain exactly one pair of generated-source markers")
    start = document.index(BEGIN) + len(BEGIN)
    end = document.index(END, start)
    if end < start:
        raise ValueError("generated-source markers are out of order")
    rendered = render_source_index(registry_path)
    updated = document[:start] + "\n\n" + rendered + "\n\n" + document[end:]
    citations_path.write_text(updated, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--citations", type=Path, default=CITATIONS)
    args = parser.parse_args()
    update_citations(args.registry, args.citations)
    print(f"Updated {args.citations}")


if __name__ == "__main__":
    main()