"""Build the Pramaan demo GIF and cover from the real accepted Kolam trajectory."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pramaan.branding import HEADLINE
from pramaan.dsl import parse_pattern
from pramaan.render import render
from pramaan.traditions import ALL_STATES

ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY = ROOT / "examples" / "beam_g6_tn_kolam_sikku.trajectory.json"
CERTIFICATE = ROOT / "examples" / "beam_g6_tn_kolam_sikku.certificate.json"
GALLERY = ROOT / "assets" / "gallery" / "beam-hf.json"
BG = (17, 35, 52)
PAPER = (248, 246, 239)
INK = (31, 45, 57)
TEAL = (39, 194, 198)
RUST = (183, 71, 54)


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def _candidate_image(candidate: dict, scale: int = 3) -> Image.Image:
    pattern = parse_pattern(json.dumps(candidate, ensure_ascii=False))
    _, grid = render(pattern)
    colors = []
    for color in pattern.palette:
        value = color.lstrip("#")
        colors.append(tuple(int(value[i:i + 2], 16) for i in (0, 2, 4)))
    image = Image.fromarray(np.asarray(colors, dtype=np.uint8)[grid], mode="RGB")
    return image.resize((64 * scale, 64 * scale), Image.Resampling.NEAREST)


def _frame(title: str, subtitle: str, image: Image.Image | None = None,
           badge: str | None = None, badge_color: tuple[int, int, int] = TEAL) -> Image.Image:
    canvas = Image.new("RGB", (1200, 700), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((58, 42), "PRAMAAN  /  PROOF OF ART", font=_font(22), fill=TEAL)
    draw.text((58, 100), title, font=_font(39), fill=PAPER)
    for index, line in enumerate(textwrap.wrap(subtitle, width=88)[:2]):
        draw.text((60, 151 + index * 23), line, font=_font(18), fill=(194, 208, 215))
    if image is not None:
        panel = Image.new("RGB", (760, 460), PAPER)
        art = image.copy()
        art.thumbnail((420, 420), Image.Resampling.NEAREST)
        panel.paste(art, ((760 - art.width) // 2, (460 - art.height) // 2))
        canvas.paste(panel, (370, 210))
    if badge:
        box = (58, 575, 1142, 648)
        draw.rounded_rectangle(box, radius=14, fill=(27, 52, 67), outline=badge_color, width=3)
        draw.text((82, 596), badge, font=_font(21), fill=PAPER)
    draw.text((58, 665), "Beam proposes  ·  deterministic checks decide", font=_font(15), fill=(153, 176, 188))
    return canvas


def build_state_quilt() -> tuple[Image.Image, dict[str, int]]:
    """Render one real passing Beam preview per state where the frozen gallery has one."""
    gallery = json.loads(GALLERY.read_text(encoding="utf-8"))
    task_rows = gallery.get("tasks")
    if not isinstance(task_rows, dict):
        raise ValueError("gallery data must contain task rows")
    preview_by_state = {}
    for _task_id, task in sorted(task_rows.items()):
        if not isinstance(task, dict) or task.get("state") not in ALL_STATES:
            continue
        if task["state"] in preview_by_state:
            continue
        attempts = task.get("attempts") if isinstance(task.get("attempts"), list) else []
        for attempt in attempts:
            if not isinstance(attempt, dict) or attempt.get("passed") is not True:
                continue
            encoded = attempt.get("tile_png_b64")
            if not isinstance(encoded, str):
                continue
            try:
                with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as tile:
                    preview_by_state[task["state"]] = tile.convert("RGB").copy()
            except (ValueError, OSError):
                continue
            break

    canvas = Image.new("RGB", (1200, 700), (50, 25, 31))
    draw = ImageDraw.Draw(canvas)
    draw.text((58, 32), "PRAMAAN  /  PROOF OF ART", font=_font(22), fill=(255, 196, 99))
    draw.text((58, 76), HEADLINE, font=_font(37), fill=PAPER)
    draw.text((60, 128), "Threshold art is one doorway into India's diverse geometric craft inspirations.",
              font=_font(19), fill=(242, 220, 193))
    left, top, gap_x, gap_y = 58, 178, 8, 8
    cell_w, cell_h = 174, 53
    abbreviations = {
        "Andaman and Nicobar Islands": "Andaman & Nicobar Islands",
        "Dadra and Nagar Haveli and Daman and Diu": "Dadra & Nagar Haveli · Daman & Diu",
        "Jammu and Kashmir": "Jammu & Kashmir",
    }
    for index, state in enumerate(ALL_STATES):
        col, row = index % 6, index // 6
        x, y = left + col * (cell_w + gap_x), top + row * (cell_h + gap_y)
        has_preview = state in preview_by_state
        fill = (252, 240, 213) if has_preview else (72, 51, 52)
        edge = (245, 170, 67) if has_preview else (135, 104, 91)
        draw.rounded_rectangle((x, y, x + cell_w, y + cell_h), radius=7, fill=fill, outline=edge, width=2)
        label = abbreviations.get(state, state)
        lines = textwrap.wrap(label, width=23)[:2]
        label_color = (67, 43, 38) if has_preview else (232, 208, 183)
        for line_index, line in enumerate(lines):
            draw.text((x + 8, y + 6 + 15 * line_index), line, font=_font(11), fill=label_color)
        if has_preview:
            tile = preview_by_state[state].resize((36, 36), Image.Resampling.NEAREST)
            canvas.paste(tile, (x + cell_w - 44, y + 8))
        else:
            draw.text((x + cell_w - 43, y + 18), "—", font=_font(16), fill=(226, 176, 113))
    empty = len(ALL_STATES) - len(preview_by_state)
    coverage = {"state_count": len(ALL_STATES), "verified_pattern_states": len(preview_by_state),
                "empty_states": empty}
    badge = (f"{coverage['verified_pattern_states']} state/UT entries show replay-verified Beam designs  ·  "
             f"{empty} have no passing tile in this snapshot")
    draw.rounded_rectangle((58, 592, 1142, 646), radius=12, fill=(72, 39, 40), outline=(245, 170, 67), width=2)
    draw.text((78, 609), badge, font=_font(17), fill=PAPER)
    draw.text((58, 665), "Beam proposes  ·  deterministic checks decide  ·  no tile implies cultural authenticity",
              font=_font(13), fill=(225, 199, 176))
    return canvas, coverage


def build_frames() -> list[Image.Image]:
    trajectory = json.loads(TRAJECTORY.read_text(encoding="utf-8"))
    certificate = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    if not trajectory.get("accepted") or trajectory.get("certificate_mask_sha256") != certificate.get("mask_sha256"):
        raise ValueError("saved Kolam trajectory and certificate do not agree")
    attempts = trajectory["attempts"]
    first = attempts[0]
    final = attempts[-1]
    loop_check = next((check for check in first["report"]["checks"]
                       if check.get("id") == "kolam_loop_count"), None)
    if not loop_check or loop_check.get("status") != "FAIL":
        raise ValueError("demo source must include a failed Kolam loop-count proposal")
    if not final["report"].get("all_hard_constraints_pass"):
        raise ValueError("demo source must end in a verified design")
    first_art = _candidate_image(first["candidate"])
    final_art = _candidate_image(final["candidate"])
    counterexample = loop_check["counterexample"]
    brief = trajectory["original_prompt"]
    digest = certificate["mask_sha256"]
    quilt, _coverage = build_state_quilt()
    return [
        quilt,
        _frame("A bounded Kolam design", f"Tamil Nadu · Sikku Kolam · brief excerpt: “{brief[:70].rstrip()}…”",
               final_art, "A 3×3 pattern inspired by a cited craft geometry"),
        _frame("Beam's first proposal", "The verifier traces the actual mirror-curve paths.",
               first_art, f"FAIL  ·  {counterexample['loops']} loops found  /  {counterexample['required']} required", RUST),
        _frame("Verifier feedback guides repair", "Beam receives the counterexample and proposes a new mirror set.",
               final_art, "The next proposal satisfies the requested loop count", TEAL),
        _frame("Verified 3×3 tile", "Exact discrete checks pass; the certificate replays from the saved DSL.",
               final_art, "PASS  ·  replayable certificate", TEAL),
        _frame("Proof travels with the art", "Deterministic raster digest from the saved certificate.",
               final_art, f"mask sha256  {digest}"),
    ]


def build_assets() -> tuple[Path, Path]:
    frames = build_frames()
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    gif_path = assets / "demo.gif"
    cover_path = assets / "cover.png"
    gif_frames = [frame.quantize(colors=128, dither=Image.Dither.NONE) for frame in frames]
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=1400,
                       loop=0, disposal=2, optimize=True)
    frames[0].save(cover_path, format="PNG", optimize=True)
    return gif_path, cover_path


if __name__ == "__main__":
    for asset in build_assets():
        print(f"Wrote {asset.relative_to(ROOT)} ({asset.stat().st_size:,} bytes)")
