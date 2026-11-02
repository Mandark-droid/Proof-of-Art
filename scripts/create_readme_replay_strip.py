"""Render the README Kinnauri replay strip from the saved, replay-checked trace."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pydantic import ValidationError

from pramaan.checks.core import all_pass, verify_pattern
from pramaan.dsl import parse_pattern
from pramaan.export import render_png
from pramaan.ui.check_labels import plain_check_label


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_ID = "beam-hf:hp_kinnauri-shawl-geometric-border-easy-8:b04b568d84d2"
OUTPUT_PATH = PROJECT_ROOT / "assets" / "kinnauri-proof-replay-strip.png"
TILE_SCALE = 2


def _load_trajectory() -> dict:
    path = PROJECT_ROOT / "benchmarks" / "data" / "eval-v2.1-trajectories.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("trajectory_id") == TRAJECTORY_ID:
            return record
    raise ValueError(f"Saved replay trajectory not found: {TRAJECTORY_ID}")


def _replay_call(call: dict) -> tuple[Image.Image, list[dict]]:
    candidate = call.get("candidate")
    saved = call.get("verifier_report")
    if not isinstance(candidate, dict) or not isinstance(saved, dict):
        raise ValueError("A saved replay call is missing its candidate or verifier report")
    pattern = parse_pattern(json.dumps(candidate, ensure_ascii=False))
    checks, _masks, grid = verify_pattern(pattern)
    accepted = all_pass(checks)
    if accepted is not (call.get("accepted") is True):
        raise ValueError("The saved acceptance flag does not match fresh replay")
    if saved.get("checks") != checks or saved.get("all_hard_constraints_pass") is not accepted:
        raise ValueError("The saved verifier report does not match fresh replay")
    return render_png(grid, pattern.palette, scale=TILE_SCALE), checks


def _reward_text(value: object) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("The saved replay call has no numeric reward")
    number = float(value)
    return f"{number:.1f}" if number.is_integer() else f"{number:.3f}"


def build_strip(record: dict) -> Image.Image:
    calls = record.get("calls")
    if not isinstance(calls, list):
        raise ValueError("Saved replay has no call list")
    calls = sorted((call for call in calls if isinstance(call, dict)),
                   key=lambda call: call.get("call_index", 0))
    if len(calls) != 3 or [call.get("call_index") for call in calls] != [1, 2, 3]:
        raise ValueError("The Kinnauri strip expects the saved three-call replay")

    font = ImageFont.load_default()
    canvas = Image.new("RGB", (560, 228), "#f7f3ec")
    draw = ImageDraw.Draw(canvas)
    panels = []
    for call in calls:
        tile, checks = _replay_call(call)
        panels.append((call, tile, checks))

    canvas_title = "HIMACHAL PRADESH  /  KINNAURI BORDER"
    draw.text((16, 8), canvas_title, font=font, fill="#172f4a")
    for index, (call, tile, checks) in enumerate(panels):
        left = 16 + index * 176
        draw.rounded_rectangle((left, 26, left + 160, 194), radius=8,
                               fill="#fffdf9", outline="#c9bba8", width=1)
        attempt = call["call_index"]
        draw.text((left + 8, 33), f"ATTEMPT {attempt}", font=font, fill="#176a66")

        if attempt == 1:
            symmetry = next((check for check in checks
                             if check.get("id") == "symmetry_d2" and check.get("status") == "FAIL"), None)
            witness = symmetry.get("counterexample", {}) if symmetry else {}
            original, transformed = witness.get("original"), witness.get("transformed")
            if (isinstance(original, list) and len(original) == 2
                    and isinstance(transformed, list) and len(transformed) == 2):
                marker_draw = ImageDraw.Draw(tile)
                for point, colour in ((original, "#ff2d95"), (transformed, "#00b8cc")):
                    x, y = point
                    cx, cy = x * TILE_SCALE + TILE_SCALE // 2, y * TILE_SCALE + TILE_SCALE // 2
                    radius = 6
                    marker_draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                                        outline=colour, width=2)
                if witness.get("transform") == "F_vertical":
                    mirror_x = tile.width // 2
                    marker_draw.line((mirror_x, 0, mirror_x, tile.height), fill="#d4a900", width=2)
        tile_x = left + (160 - tile.width) // 2
        canvas.paste(tile, (tile_x, 49))

        failed = [plain_check_label(check) for check in checks if check.get("status") == "FAIL"]
        result = "PASSES EVERY CHECK" if call.get("accepted") is True else "FAILS: " + ", ".join(failed)
        draw.text((left + 8, 181), result[:24], font=font, fill="#172f4a")
        draw.text((left + 8, 199), "Reward " + _reward_text(call.get("reward")),
                  font=font, fill="#172f4a")

    first_checks = panels[0][2]
    witness_check = next((check for check in first_checks
                          if check.get("status") == "FAIL" and check.get("counterexample")), None)
    witness = witness_check.get("counterexample", {}) if witness_check else {}
    original, transformed = witness.get("original"), witness.get("transformed")
    if not (isinstance(original, list) and len(original) == 2
            and isinstance(transformed, list) and len(transformed) == 2):
        raise ValueError("The first attempt has no saved pixel mirror witness")
    relation = "its mirror" if witness.get("transform") == "F_vertical" else "its transformed pixel"
    caption = (f"Pixel ({original[0]}, {original[1]}) and {relation} "
               f"({transformed[0]}, {transformed[1]}) differ")
    draw.text((16, 214), caption, font=font, fill="#172f4a")
    return canvas


def main() -> None:
    record = _load_trajectory()
    image = build_strip(record).quantize(colors=32, method=Image.Quantize.MEDIANCUT,
                                         dither=Image.Dither.NONE)
    image.save(OUTPUT_PATH, format="PNG", optimize=True)
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)} ({OUTPUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
