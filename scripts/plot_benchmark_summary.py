"""Render a small, data-driven SVG chart for the benchmark report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render_svg(summary: dict) -> str:
    models = summary.get("by_model", {})
    if len(models) != 1:
        raise ValueError("the static report chart expects exactly one model summary")
    model_key, metrics = next(iter(models.items()))
    prompt_count = metrics["overall"].get("n")
    prompt_label = f"{prompt_count} prompts · " if isinstance(prompt_count, int) else ""
    levels = metrics.get("by_level", {})
    rows = [("Overall", metrics["overall"])] + [
        (level.title(), levels[level]) for level in ("easy", "medium", "hard") if level in levels
    ]
    width, height = 960, 390
    x0, plot_width = 250, 620
    y_first = 116
    scale = plot_width / 100
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Pramaan Beam hard-constraint success rates</title>',
        '<desc id="desc">First-proposal and up-to-three-proposal success percentages, overall and by difficulty.</desc>',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<text x="42" y="42" font-family="Arial,sans-serif" font-size="23" font-weight="700" fill="#172F4A">Hard constraints pass: first proposal vs repair loop</text>',
        f'<text x="42" y="70" font-family="Arial,sans-serif" font-size="14" fill="#4b5563">{model_key} · {prompt_label}one stochastic run</text>',
    ]
    for tick in range(0, 101, 20):
        x = x0 + tick * scale
        elements.append(f'<line x1="{x:.1f}" y1="92" x2="{x:.1f}" y2="330" stroke="#d7d7d2" stroke-width="1"/>')
        elements.append(f'<text x="{x:.1f}" y="350" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#50545a">{tick}%</text>')
    for index, (label, row) in enumerate(rows):
        y = y_first + index * 58
        elements.append(f'<text x="42" y="{y+15}" font-family="Arial,sans-serif" font-size="15" font-weight="600" fill="#202124">{label}</text>')
        for offset, key, color in ((0, "validity_at_1", "#315A83"), (23, "validity_at_3", "#A84436")):
            value = row.get(key)
            if value is None:
                continue
            percent = value * 100
            bar_width = percent * scale
            elements.append(f'<rect x="{x0}" y="{y+offset}" width="{bar_width:.1f}" height="16" rx="3" fill="{color}"/>')
            elements.append(f'<text x="{x0+bar_width+8:.1f}" y="{y+offset+13}" font-family="Arial,sans-serif" font-size="12" fill="#202124">{percent:.1f}%</text>')
    elements.extend([
        '<rect x="250" y="367" width="12" height="12" rx="2" fill="#315A83"/>',
        '<text x="268" y="378" font-family="Arial,sans-serif" font-size="12" fill="#34373b">pass@1</text>',
        '<rect x="350" y="367" width="12" height="12" rx="2" fill="#A84436"/>',
        '<text x="368" y="378" font-family="Arial,sans-serif" font-size="12" fill="#34373b">pass@3</text>',
        '</svg>',
    ])
    return "\n".join(elements) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_svg(summary), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
