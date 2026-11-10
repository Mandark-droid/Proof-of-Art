"""Link preview images, 1200 × 630, drawn with Pillow in the site's type and colours (spec 11)."""
from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..data import site_data

W, H = 1200, 630
FONTS = Path(__file__).resolve().parents[2] / "static" / "fonts"
GROUND = (0x15, 0x11, 0x0F)
CHALK = (0xF4, 0xEF, 0xE6)
MUTED = (0xB8, 0xAD, 0xA1)
FAINT = (0x94, 0x89, 0x80)
TURMERIC = (0xF4, 0xB6, 0x3F)
EVIDENCE = (0x4F, 0xD1, 0xC5)
BROKE = (0xE8, 0x61, 0x3C)
HAIRLINE = (0x2E, 0x26, 0x21)
DASHED = (0x6B, 0x62, 0x5B)


def _sans(size: int, weight: int = 500, width: int = 100) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(str(FONTS / "anek-latin-latin-wdth-normal.woff2"), size)
    axes = {axis["name"]: axis for axis in font.get_variation_axes()}
    values = []
    for name, value in ((b"Weight", weight), (b"Width", width)):
        if name in axes:
            values.append(value)
    font.set_variation_by_axes(values)
    return font


def _mono(size: int, weight: int = 500) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / f"jetbrains-mono-{weight}.woff2"), size)


def _spaced(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill, spacing: float) -> None:
    """Text with letter spacing (Pillow has none built in)."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += font.getlength(ch) + spacing


def _text_ne(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font, fill) -> None:
    """Text that may hold "≠", which the bundled fonts lack: drawn as "=" with a stroke through it."""
    x, y = xy
    for i, part in enumerate(text.split("≠")):
        if i:
            draw.text((x, y), "=", font=font, fill=fill)
            w = font.getlength("=")
            left, top, right, bottom = font.getbbox("=")
            mid = y + (top + bottom) / 2
            half = (bottom - top) * 0.8
            draw.line((x + w * 0.68, mid - half, x + w * 0.32, mid + half), fill=fill, width=max(2, font.size // 12))
            x += w
        draw.text((x, y), part, font=font, fill=fill)
        x += font.getlength(part)


def _wrap(text: str, font, width: int, max_lines: int = 3) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if font.getlength(trial) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        while font.getlength(lines[-1] + "…") > width and " " in lines[-1]:
            lines[-1] = lines[-1].rsplit(" ", 1)[0]
        lines[-1] += "…"
    return lines


def _mark(draw: ImageDraw.ImageDraw, x: int, y: int, scale: float = 1.5) -> None:
    """The four-dot mark (22 px design grid, dots at 6 and 16, 4.5 px across)."""
    r = 2.25 * scale
    for cx in (6, 16):
        for cy in (6, 16):
            px, py = x + cx * scale, y + cy * scale
            draw.ellipse((px - r, py - r, px + r, py + r), fill=TURMERIC)


def _brand(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    _mark(draw, x, y, 1.6)
    draw.text((x + 46, y + 17), "Pramaan", font=_sans(30, 700, 112), fill=CHALK, anchor="lm")


def _png(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _decode(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")


@lru_cache(maxsize=1)
def og_home() -> bytes:
    data = site_data()
    totals = data.meta["totals"]
    image = Image.new("RGB", (W, H), GROUND)
    draw = ImageDraw.Draw(image)
    _brand(draw, 60, 40)
    draw.text((60, 96), "AI drew it. Math checked it.", font=_sans(66, 700, 116), fill=CHALK)
    eyebrow = (f'{totals["runs"]:,} TILES · {totals["models"]} MODELS · {totals["briefs"]} BRIEFS · '
               f'{totals["passed"]:,} PASSED THE MATHS')
    _spaced(draw, (62, 186), eyebrow, _mono(17), MUTED, 1.6)

    wall = data.wall
    sprite = Image.open(data.root / wall["sprite"]).convert("RGBA")
    cell, step, x0, y0 = wall["cell"], 30, 60, 228
    status: dict[tuple[int, int], str] = {}
    for r in wall["runs"]:
        model_row = data.model_index.get(r["model"], 0) * 2 + (0 if r["level"] == "easy" else 1)
        status[(model_row, r["col"])] = r["status"]
    layer = Image.new("RGBA", (W, H), GROUND + (255,))
    rows = (H - y0) // step + 1
    for row in range(min(rows, wall["rows"])):
        for col in range(wall["cols"]):
            x, y = x0 + col * step, y0 + row * step
            code = status.get((row, col), "-")
            if code == "-":
                ImageDraw.Draw(layer).rectangle((x + 1, y + 1, x + step - 3, y + step - 3),
                                                outline=(0x3A, 0x34, 0x2F, 255), width=1)
                continue
            tile = sprite.crop((col * cell, row * cell, col * cell + cell, row * cell + cell))
            tile = tile.resize((step - 2, step - 2), Image.NEAREST)
            if code == "x":
                tile.putalpha(tile.getchannel("A").point(lambda a: int(a * 0.28)))
            layer.alpha_composite(tile, (x, y))
    # Fade the wall into the ground towards the bottom edge.
    fade = Image.new("L", (W, H), 0)
    fd = ImageDraw.Draw(fade)
    for i in range(160):
        fd.line((0, H - 160 + i, W, H - 160 + i), fill=int(255 * (i / 160) ** 1.4))
    ground = Image.new("RGBA", (W, H), GROUND + (255,))
    layer = Image.composite(ground, layer, fade)
    image.paste(layer.crop((0, y0, W, H)).convert("RGB"), (0, y0))
    return _png(image)


def _verdict(model: dict) -> tuple[str, tuple[int, int, int]]:
    status = model["status"]
    if status == "1":
        return "Passed on the first try.", TURMERIC
    if status in ("2", "3"):
        return f"Passed on try {status} of 3.", TURMERIC
    if status == "-":
        return "Never produced a readable pattern.", MUTED
    if all(t["outcome"] == "broke" for t in model["tries"]):
        return f'Broke a rule on all {len(model["tries"])} tries.', BROKE
    return "Never passed.", BROKE


def _ring(draw: ImageDraw.ImageDraw, cx: float, cy: float, radius: int = 20, width: int = 3) -> None:
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=EVIDENCE, width=width)


def og_replay(task_id: str, model_key: str) -> bytes | None:
    """The run's link image, or None. Unknown brief or model ids are refused before the cache,
    so arbitrary URLs never fill it and never cache a miss."""
    data = site_data()
    if task_id not in data.task_ids or model_key not in data.model_index:
        return None
    return _og_replay(task_id, model_key)


@lru_cache(maxsize=1024)  # at least one entry per run (11 models x 72 briefs today)
def _og_replay(task_id: str, model_key: str) -> bytes | None:
    data = site_data()
    replay = data.replay(task_id)
    if replay is None:
        return None
    model = next((m for m in replay["models"] if m["key"] == model_key), None)
    if model is None:
        return None
    image = Image.new("RGB", (W, H), GROUND)
    draw = ImageDraw.Draw(image)
    scale, x0, y0 = 8, 59, 59
    size = 64 * scale
    shown = next((t for t in model["tries"] if t.get("tile_png")), None)
    evidence_line = None
    if shown is None:
        # Never readable: an empty dashed square, no tile.
        for i in range(0, size, 12):
            for a, b in (((x0 + i, y0), (x0 + min(i + 6, size), y0)),
                         ((x0 + i, y0 + size - 1), (x0 + min(i + 6, size), y0 + size - 1)),
                         ((x0, y0 + i), (x0, y0 + min(i + 6, size))),
                         ((x0 + size - 1, y0 + i), (x0 + size - 1, y0 + min(i + 6, size)))):
                draw.line((a, b), fill=DASHED, width=2)
        _spaced(draw, (x0 + 150, y0 + size // 2 - 14), "UNREADABLE", _mono(24), FAINT, 3)
    else:
        tile = _decode(shown["tile_png"]).resize((size, size), Image.NEAREST)
        base = Image.new("RGBA", (size, size), GROUND + (255,))
        base.alpha_composite(tile)
        if shown["outcome"] == "broke" and shown.get("mask_png"):
            base.alpha_composite(Image.new("RGBA", (size, size), GROUND + (int(255 * 0.72),)))
            base.alpha_composite(_decode(shown["mask_png"]).resize((size, size), Image.NEAREST))
            bd = ImageDraw.Draw(base)
            pair = shown.get("pair") or {}
            for point in (pair.get("original"), pair.get("transformed")):
                if point is not None:
                    _ring(bd, (point[0] + 0.5) * scale, (point[1] + 0.5) * scale)
            evidence_line = f'Try {shown["n"]}: {shown["mismatch_px"]:,} of 4,096 pixels disagree'
        elif shown["outcome"] == "broke":
            evidence_line = f'Try {shown["n"]}: ' + (shown.get("measured") or "broke a rule").split("\n", 1)[0]
        image.paste(base.convert("RGB"), (x0, y0))
        outline = TURMERIC if shown["outcome"] == "passed" else (0x4A, 0x3F, 0x38)
        draw.rectangle((x0 - 1, y0 - 1, x0 + size, y0 + size), outline=outline, width=1)

    left, right = 627, W - 59
    width = right - left
    eyebrow = f'{replay["state"]} · {replay["level"]}'.upper()
    _spaced(draw, (left, 64), eyebrow, _mono(18), MUTED, 1.8)
    y = 104
    name_font = _sans(60, 700, 116)
    for line in _wrap(model["label"], name_font, width, 2):
        draw.text((left, y), line, font=name_font, fill=CHALK)
        y += 62
    y += 8
    trad_font = _sans(32, 500, 100)
    for line in _wrap(replay["tradition"], trad_font, width, 2):
        draw.text((left, y), line, font=trad_font, fill=MUTED)
        y += 40
    y += 22
    verdict, colour = _verdict(model)
    verdict_font = _sans(38, 700, 112)
    for line in _wrap(verdict, verdict_font, width, 2):
        draw.text((left, y), line, font=verdict_font, fill=colour)
        y += 46
    if evidence_line:
        y += 12
        mono = _mono(21)
        for line in _wrap(evidence_line, mono, width, 2):
            draw.text((left, y), line, font=mono, fill=EVIDENCE)
            y += 30
        pair = (shown or {}).get("pair") or {}
        if pair.get("original") is not None and pair.get("transformed") is not None:
            o, t = pair["original"], pair["transformed"]
            _text_ne(draw, (left, y + 2), f"({o[0]}, {o[1]}) ≠ ({t[0]}, {t[1]})", mono, EVIDENCE)
    draw.line((left, 528, right, 528), fill=HAIRLINE, width=2)
    _brand(draw, left, 548)
    draw.text((right, 565), "AI drew it. Math checked it.", font=_sans(22, 600, 108), fill=MUTED, anchor="rm")
    return _png(image)
