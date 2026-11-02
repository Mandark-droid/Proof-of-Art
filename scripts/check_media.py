"""Automatic validation gates for the silent Pramaan demo media bundle."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image

from scripts.record_demo import PUBLIC_ORIGIN

MAX_VIDEO_BYTES = 25 * 1024 * 1024
MAX_GIF_BYTES = 5 * 1024 * 1024
MIN_DURATION_SECONDS = 45.0
MAX_DURATION_SECONDS = 75.0
MIN_WIDTH, MIN_HEIGHT = 1280, 720


def _probe_video(path: Path) -> dict:
    executable = shutil.which("ffprobe")
    if not executable:
        raise RuntimeError("ffprobe is required for automatic video metadata and duration checks")
    result = subprocess.run(
        [executable, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def _has_identifying_metadata(value) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            key_l = str(key).casefold()
            if key_l in {"artist", "author", "creator", "comment", "copyright", "location", "gps"}:
                if item not in (None, "", [], {}):
                    return True
            if _has_identifying_metadata(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_has_identifying_metadata(item) for item in value)
    return False


def _probe_shape(probe: dict) -> tuple[dict, dict]:
    video = next((stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"), {})
    fmt = probe.get("format", {})
    return video, fmt


def validate_media(video_path: str | Path, poster_path: str | Path, gif_path: str | Path,
                   url_log: list[str] | str | Path, *, probe: dict | None = None,
                   expected_origin: str = PUBLIC_ORIGIN,
                   max_video_bytes: int = MAX_VIDEO_BYTES,
                   max_gif_bytes: int = MAX_GIF_BYTES) -> list[str]:
    """Return every media-policy violation; inject ffprobe JSON to test without ffprobe."""
    video_path, poster_path, gif_path = Path(video_path), Path(poster_path), Path(gif_path)
    errors: list[str] = []
    for label, path in (("video", video_path), ("poster", poster_path), ("GIF", gif_path)):
        if not path.is_file():
            errors.append(f"{label} file missing: {path}")
    if errors:
        return errors

    if video_path.stat().st_size > max_video_bytes:
        errors.append(f"video size exceeds {max_video_bytes} bytes")
    try:
        with video_path.open("rb") as stream:
            header = stream.read(12)
        if video_path.suffix.casefold() == ".mp4" and b"ftyp" not in header:
            errors.append("video container signature is not MP4")
        elif video_path.suffix.casefold() == ".webm" and not header.startswith(b"\x1aE\xdf\xa3"):
            errors.append("video container signature is not WebM")
        elif video_path.suffix.casefold() not in {".mp4", ".webm"}:
            errors.append("video extension must be .mp4 or .webm")
    except OSError as exc:
        errors.append(f"video read failed: {exc}")

    try:
        probe = probe if probe is not None else _probe_video(video_path)
        stream, fmt = _probe_shape(probe)
        codec = str(stream.get("codec_name", "")).casefold()
        expected_codec = "h264" if video_path.suffix.casefold() == ".mp4" else "vp8"
        if codec != expected_codec and not (expected_codec == "vp8" and codec == "vp9"):
            errors.append(f"video codec must be {expected_codec}; got {codec or 'missing'}")
        try:
            duration = float(fmt.get("duration", stream.get("duration")))
            if not MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS:
                errors.append(f"video duration must be 45 to 75 seconds; got {duration:g}")
        except (TypeError, ValueError):
            errors.append("video duration metadata is missing")
        try:
            width, height = int(stream.get("width", 0)), int(stream.get("height", 0))
            if width < MIN_WIDTH or height < MIN_HEIGHT or abs(width / height - 16 / 9) > 0.01:
                errors.append(f"video resolution must be at least 1280x720 at 16:9; got {width}x{height}")
        except (TypeError, ValueError, ZeroDivisionError):
            errors.append("video resolution metadata is invalid")
        if any(s.get("codec_type") == "audio" for s in probe.get("streams", [])):
            errors.append("video must not contain voice or music audio")
        if _has_identifying_metadata({"format": fmt.get("tags", {}), "stream": stream.get("tags", {})}):
            errors.append("video contains identifying metadata")
    except (RuntimeError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        errors.append(f"video metadata check failed: {exc}")

    try:
        with Image.open(poster_path) as poster:
            poster.verify()
        with Image.open(poster_path) as poster:
            if poster.format != "PNG":
                errors.append("poster must be PNG")
            if poster.width < MIN_WIDTH or poster.height < MIN_HEIGHT or abs(poster.width / poster.height - 16 / 9) > 0.01:
                errors.append(f"poster resolution must be at least 1280x720 at 16:9; got {poster.width}x{poster.height}")
            if poster.info.get("exif") or poster.info.get("comment") or poster.info.get("Author"):
                errors.append("poster contains identifying metadata")
    except Exception as exc:
        errors.append(f"poster image check failed: {exc}")

    if gif_path.stat().st_size > max_gif_bytes:
        errors.append(f"GIF size exceeds {max_gif_bytes} bytes")
    try:
        with Image.open(gif_path) as gif:
            gif.verify()
        with Image.open(gif_path) as gif:
            if gif.format != "GIF":
                errors.append("demo animation must be GIF")
            if gif.width > 640 or gif.height > 360:
                errors.append(f"GIF resolution must be at most 640x360; got {gif.width}x{gif.height}")
            if gif.info.get("comment") or gif.info.get("loop_comment"):
                errors.append("GIF contains identifying metadata")
    except Exception as exc:
        errors.append(f"GIF image check failed: {exc}")

    if isinstance(url_log, (str, Path)):
        try:
            url_log = json.loads(Path(url_log).read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"URL origin log could not be read: {exc}")
            url_log = []
    expected = urlsplit(expected_origin)
    for entry in url_log:
        parsed = urlsplit(str(entry))
        if parsed.scheme != "https" or parsed.netloc != expected.netloc or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            errors.append("URL log contains a non-public origin or path")
            break
    if not url_log:
        errors.append("URL origin log is empty")
    return errors


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--poster", type=Path, default=Path("media/pramaan-demo-poster.png"))
    parser.add_argument("--gif", type=Path, required=True,
                        help="path to the approved demonstration GIF")
    parser.add_argument("--url-log", type=Path, default=Path("local/demo-url-origins.json"))
    args = parser.parse_args()
    video = args.video or next((p for p in (Path("media/pramaan-demo.mp4"), Path("media/pramaan-demo.webm")) if p.is_file()), Path("media/pramaan-demo.mp4"))
    errors = validate_media(video, args.poster, args.gif, args.url_log)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("Media gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

