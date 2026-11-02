from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from scripts.record_demo import (
    PUBLIC_ORIGIN,
    build_live_caption,
    contains_token_like_string,
    is_non_rangoli_state,
    validate_public_url,
    _wait_for_loaded_surface,
    _recording_origin,
    _validate_recording_url,
    _page_requires_sign_in,
)
from scripts.check_media import main as check_media_main, validate_media


def test_media_cli_requires_explicit_gif_after_demo_gif_retirement(monkeypatch):
    monkeypatch.setattr("sys.argv", ["check_media"])
    with pytest.raises(SystemExit) as exc:
        check_media_main()
    assert exc.value.code == 2


def test_public_url_is_exact_origin_and_https_only():
    assert validate_public_url(PUBLIC_ORIGIN + "/studio/") == PUBLIC_ORIGIN + "/studio/"
    for url in (
        "https://evil.example/studio/",
        PUBLIC_ORIGIN.replace("https:", "http:") + "/studio/",
        PUBLIC_ORIGIN + ".evil.example/studio/",
        "https://user:pass@" + PUBLIC_ORIGIN.removeprefix("https://") + "/",
    ):
        with pytest.raises(ValueError):
            validate_public_url(url)


def test_loopback_recording_requires_explicit_local_dry_run_flag():
    loopback = ".".join(("127", "0", "0", "1"))
    with pytest.raises(ValueError, match="explicitly allowed"):
        _recording_origin(f"http://{loopback}:8000")
    origin = _recording_origin(f"http://{loopback}:8000/", allow_local=True)
    assert origin == f"http://{loopback}:8000"
    assert _validate_recording_url(origin + "/web/", origin) == origin + "/web/"
    with pytest.raises(ValueError, match="outside allowed"):
        _validate_recording_url("https://example.com/", origin)


def test_recorder_allows_publication_signin_copy_but_detects_login_forms():
    class Locator:
        def __init__(self, count):
            self.value = count

        def count(self):
            return self.value

    class Page:
        def __init__(self, password=0, login_form=0):
            self.password = password
            self.login_form = login_form

        def locator(self, selector):
            return Locator(self.password if "password" in selector else self.login_form)

        def get_by_text(self, _pattern):
            raise AssertionError("visible copy must not be mistaken for a login gate")

    assert not _page_requires_sign_in(Page())
    assert _page_requires_sign_in(Page(password=1))
    assert _page_requires_sign_in(Page(login_form=1))


def test_token_like_detector_flags_provider_and_hf_secrets():
    assert contains_token_like_string("Authorization: Bearer hf_" + "a" * 30)
    assert contains_token_like_string("sk-ant-api03-" + "a" * 36)
    assert not contains_token_like_string("The model received 72 tasks and 4096 tokens")


def test_rangoli_filter_uses_state_and_tradition_labels():
    assert not is_non_rangoli_state("Tamil Nadu", "Kolam floor design")
    assert not is_non_rangoli_state("Kerala", "Pookalam")
    assert is_non_rangoli_state("Odisha", "Pattachitra border")


def test_live_caption_interpolates_only_current_summary_values():
    caption = build_live_caption("{model_count} models · {task_count} tasks", {
        "model_count": 3,
        "task_count": 47,
    })
    assert caption == "3 models · 47 tasks"
    assert build_live_caption("{model_count} models", {"model_count": 1}) != "3 models"
    with pytest.raises((KeyError, ValueError)):
        build_live_caption("{model_count} models", {})


def test_recorder_waits_for_interactive_surface_without_network_idle():
    calls = []

    class Locator:
        def wait_for(self, **kwargs):
            calls.append(kwargs)

    class Page:
        def locator(self, selector):
            calls.append(selector)
            return Locator()

    _wait_for_loaded_surface(Page(), "#poa-pattern", timeout=1200)

    assert calls == ["#poa-pattern", {"state": "visible", "timeout": 1200}]


def make_image(path: Path, size=(1280, 720), fmt="PNG"):
    Image.new("RGB", size, "white").save(path, format=fmt)


def valid_probe():
    return {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "60.0", "size": "2000000", "tags": {}},
        "streams": [{"codec_type": "video", "codec_name": "h264", "width": 1280, "height": 720, "tags": {}}],
    }


def test_media_gate_accepts_compliant_outputs_and_exact_origin(tmp_path):
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"x" * 1024)
    poster = tmp_path / "poster.png"
    gif = tmp_path / "demo.gif"
    make_image(poster)
    make_image(gif, (320, 180), "GIF")
    errors = validate_media(video, poster, gif, [PUBLIC_ORIGIN, PUBLIC_ORIGIN], probe=valid_probe())
    assert errors == []


def test_media_gate_reports_duration_resolution_size_gif_and_origin_failures(tmp_path):
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"x" * 128)
    poster = tmp_path / "poster.png"
    gif = tmp_path / "demo.gif"
    make_image(poster, (640, 480))
    make_image(gif, (1280, 720), "GIF")
    bad = valid_probe()
    bad["format"]["duration"] = "20"
    bad["streams"][0]["width"] = 640
    bad["streams"][0]["height"] = 480
    bad["streams"][0]["tags"] = {"artist": "someone" + "@" + "example.com"}
    errors = validate_media(video, poster, gif, [PUBLIC_ORIGIN, "https://elsewhere.example/"], probe=bad,
                            max_video_bytes=100, max_gif_bytes=32)
    joined = " ".join(errors).lower()
    for expected in ("duration", "resolution", "metadata", "url", "gif", "size"):
        assert expected in joined


def test_media_gate_rejects_external_url_origin_and_identifying_poster_metadata(tmp_path):
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"x" * 128)
    poster = tmp_path / "poster.png"
    gif = tmp_path / "demo.gif"
    Image.new("RGB", (1280, 720), "white").save(poster, pnginfo=None)
    make_image(gif, (320, 180), "GIF")
    bad = valid_probe()
    bad["format"]["tags"] = {"comment": "private machine"}
    errors = validate_media(video, poster, gif, ["https://kshitijthakkar-pramaan-proof-of-art.hf.space.evil/path"], probe=bad)
    joined = " ".join(errors).lower()
    assert "url" in joined and "metadata" in joined


def test_media_gate_rejects_voice_or_music_stream(tmp_path):
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"x" * 128)
    poster = tmp_path / "poster.png"
    gif = tmp_path / "demo.gif"
    make_image(poster)
    make_image(gif, (320, 180), "GIF")
    probe = valid_probe()
    probe["streams"].append({"codec_type": "audio", "codec_name": "aac"})
    errors = validate_media(video, poster, gif, [PUBLIC_ORIGIN], probe=probe)
    assert any("audio" in error.lower() for error in errors)
