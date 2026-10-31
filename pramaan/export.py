from __future__ import annotations

import json
import atexit
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw
import numpy as np

def _export_directory() -> Path:
    try:
        candidate = Path(tempfile.mkdtemp(prefix="pramaan-exports-"))
        probe = candidate / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return candidate
    except OSError:
        export_root = Path.cwd() / ".pramaan-exports"
        export_root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="run-", dir=export_root))


_EXPORT_DIR = _export_directory()
atexit.register(shutil.rmtree, _EXPORT_DIR, ignore_errors=True)
_EXPORT_TTL_SECONDS = 60 * 60
_CLEANUP_INTERVAL_SECONDS = 60
_CLEANUP_STOP = threading.Event()
_CLEANUP_LOCK = threading.Lock()
_CLEANUP_THREAD: threading.Thread | None = None
atexit.register(_CLEANUP_STOP.set)


def cleanup_expired_exports(directory: Path = _EXPORT_DIR, now: float | None = None,
                           ttl_seconds: int = _EXPORT_TTL_SECONDS) -> int:
    """Remove expired generated bundles; return the number of files removed."""
    cutoff = (time.time() if now is None else now) - ttl_seconds
    try:
        paths = tuple(directory.glob("pramaan-*.zip"))
    except OSError:
        return 0
    removed = 0
    for path in paths:
        try:
            if path.stat().st_mtime <= cutoff:
                path.unlink()
                removed += 1
        except FileNotFoundError:
            continue
        except OSError:
            continue
    return removed


def _cleanup_loop() -> None:
    while not _CLEANUP_STOP.wait(_CLEANUP_INTERVAL_SECONDS):
        cleanup_expired_exports()


def _ensure_cleanup_thread() -> None:
    global _CLEANUP_THREAD
    with _CLEANUP_LOCK:
        if _CLEANUP_THREAD is None or not _CLEANUP_THREAD.is_alive():
            _CLEANUP_STOP.clear()
            _CLEANUP_THREAD = threading.Thread(target=_cleanup_loop, name="pramaan-export-cleanup", daemon=True)
            _CLEANUP_THREAD.start()


def render_png(grid: np.ndarray, palette: list[str], scale: int = 8, repeats: int = 1) -> Image.Image:
    tiled = np.tile(grid, (repeats, repeats))
    rgb = np.zeros((*tiled.shape, 3), dtype=np.uint8)
    colours = [tuple(bytes.fromhex(c.lstrip("#"))) for c in palette]
    for index, colour in enumerate(colours):
        rgb[tiled == index] = colour
    image = Image.fromarray(rgb, mode="RGB")
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def render_highlighted_png(grid: np.ndarray, palette: list[str], markers: list[tuple[int, int, str]],
                           scale: int = 5, repeats: int = 3) -> Image.Image:
    """Render a tile preview and ring each witness pixel in every repeated tile."""
    image = render_png(grid, palette, scale=scale, repeats=repeats)
    draw = ImageDraw.Draw(image)
    for x, y, colour in markers:
        if not (0 <= x < 64 and 0 <= y < 64):
            continue
        for tile_y in range(repeats):
            for tile_x in range(repeats):
                left = (tile_x * 64 + x) * scale
                top = (tile_y * 64 + y) * scale
                # A one-pixel outline is easy to miss in a 3x preview. Keep the exact
                # witness at its centre while using a larger ring as a visual pointer.
                centre_x = left + scale // 2
                centre_y = top + scale // 2
                radius = max(scale, 6)
                draw.ellipse((centre_x - radius, centre_y - radius,
                              centre_x + radius, centre_y + radius),
                             outline=colour, width=max(2, scale // 2))
    return image


_TRACE_FIELDS = (
    "trajectory_schema_version", "trajectory_id", "data_role", "training_eligible",
    "prompt_id", "suite_id", "prompt_suite_sha256", "state", "tradition_id", "grammar",
    "level", "symmetry", "original_prompt", "model", "created_at", "started_at",
    "completed_at", "constraint_profile", "request_params", "elapsed_ms", "telemetry",
    "provider_attempts", "total_tokens", "cost", "provider_error", "model_calls", "attempts",
)


def trajectory_bundle(trajectory: dict) -> str:
    """Build a private, downloadable trace archive without treating it as training data."""
    if not isinstance(trajectory, dict):
        raise TypeError("trajectory must be a JSON object")
    cleanup_expired_exports()
    _ensure_cleanup_thread()
    output = _EXPORT_DIR / f"pramaan-trace-{uuid.uuid4().hex}.zip"
    record = {key: trajectory[key] for key in _TRACE_FIELDS if key in trajectory}
    record.setdefault("data_role", "unassigned")
    record["training_eligible"] = False
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("trajectory.json", json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        archive.writestr(
            "README.txt",
            "Pramaan execution trace\n\n"
            "This archive preserves the recorded model calls, attempts, verifier reports, and telemetry. "
            "It does not itself certify the geometry; a verified design's artifact bundle includes its certificate.\n"
            "The record is marked data_role=unassigned and training_eligible=false. Review, consent, "
            "and replay validation are required before adding it to a training dataset.\n",
        )
    return str(output)


def artifact_bundle(pattern, certificate: dict, grid: np.ndarray, provenance: dict | None = None,
                    trajectory: dict | None = None) -> str:
    """Build a short-lived ZIP containing only validated, application-generated artifacts."""
    cleanup_expired_exports()
    _ensure_cleanup_thread()
    output = _EXPORT_DIR / f"pramaan-{uuid.uuid4().hex}.zip"
    pattern_json = json.dumps(pattern.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"
    certificate_json = json.dumps(certificate, indent=2, ensure_ascii=False) + "\n"
    image = render_png(grid, pattern.palette, scale=8, repeats=1)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("design.pattern.json", pattern_json)
        archive.writestr("certificate.json", certificate_json)
        if provenance is not None:
            archive.writestr("provenance.json", json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
        if trajectory is not None:
            record = {key: trajectory[key] for key in _TRACE_FIELDS if key in trajectory}
            record.setdefault("data_role", "unassigned")
            record["training_eligible"] = False
            archive.writestr("trajectory.json", json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        from io import BytesIO
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        archive.writestr("tile.png", buffer.getvalue())
        archive.writestr(
            "README.txt",
            "Pramaan verified-artifact bundle\n\n"
            "Replay with: python -m pramaan.verify --design design.pattern.json --certificate certificate.json\n"
            "The certificate records deterministic checks of a finite 64x64 raster model. It is not signed and does not establish cultural authenticity or manufacturing quality.\n"
            "If present, provenance.json records the generation request and structured attempt history; it is not part of the geometry proof.\n",
        )
    return str(output)
