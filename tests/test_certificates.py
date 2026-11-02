import numpy as np

from pramaan.certificate import make_certificate, verify_certificate
from pramaan.checks.core import verify_pattern
from pramaan.export import artifact_bundle, cleanup_expired_exports, render_highlighted_png
from pramaan.render import mask_digest
from pramaan.verify import replay
from tests.helpers import pattern_obj
from pathlib import Path
import os
import shutil
import time
import uuid
import zipfile


def test_certificate_replays_deterministically():
    pattern = pattern_obj()
    checks, masks, grid = verify_pattern(pattern)
    cert = make_certificate(pattern, checks, masks, grid)
    assert verify_certificate(cert, cert) == (True, "Certificate matches deterministic replay")


def test_modified_geometry_invalidates_certificate_hash():
    pattern = pattern_obj()
    checks, masks, grid = verify_pattern(pattern)
    cert = make_certificate(pattern, checks, masks, grid)
    altered = pattern.model_copy(deep=True)
    altered.layers[0].motifs[0].radius = 4
    checks2, masks2, grid2 = verify_pattern(altered)
    replayed = make_certificate(altered, checks2, masks2, grid2)
    assert verify_certificate(cert, replayed)[0] is False


def test_certificate_metadata_tampering_is_detected():
    pattern = pattern_obj()
    checks, masks, grid = verify_pattern(pattern)
    cert = make_certificate(pattern, checks, masks, grid)
    cert["validator_version"] = "9.9.9"
    assert verify_certificate(cert, make_certificate(pattern, checks, masks, grid)) == (False, "Validator version differs from replay")


def test_certificate_replay_is_portable_across_runtime_versions():
    pattern = pattern_obj()
    checks, masks, grid = verify_pattern(pattern)
    cert = make_certificate(pattern, checks, masks, grid)
    cert["runtime"] = {"numpy": "different", "pillow": "different", "pydantic": "different"}

    assert verify_certificate(cert, make_certificate(pattern, checks, masks, grid)) == (
        True,
        "Certificate matches deterministic replay",
    )


def test_mask_digest_delimits_layer_ids_and_array_boundaries():
    grid = np.zeros((64, 64), dtype=np.uint8)
    zeros = np.zeros((64, 64), dtype=bool)
    shifted_zeros = zeros.copy()
    shifted_zeros.flat[-1] = True
    ones = np.ones((64, 64), dtype=bool)
    first = {"\x00": zeros, "\x01\x01": ones}
    second = {"\x00\x00": shifted_zeros, "\x01": ones}
    assert mask_digest(first, grid) != mask_digest(second, grid)


def test_export_bundle_contains_png_source_and_certificate():
    pattern = pattern_obj()
    checks, masks, grid = verify_pattern(pattern)
    cert = make_certificate(pattern, checks, masks, grid)
    bundle_path = artifact_bundle(pattern, cert, grid)
    with zipfile.ZipFile(bundle_path) as archive:
        assert set(archive.namelist()) == {"design.pattern.json", "certificate.json", "tile.png", "README.txt"}
        assert archive.read("tile.png").startswith(b"\x89PNG")


def test_counterexample_highlight_is_rendered_on_each_repeated_tile():
    pattern = pattern_obj()
    _, _, grid = verify_pattern(pattern)
    image = render_highlighted_png(grid, pattern.palette, [(16, 16, "#FF2D95")], scale=4, repeats=3)
    # The larger ring stays centred on the exact witness pixel in every tile.
    assert image.getpixel((60, 66)) == (255, 45, 149)
    assert image.getpixel((316, 66)) == (255, 45, 149)


def test_export_cleanup_removes_only_expired_bundles():
    directory = Path.cwd() / ".pramaan-exports" / f"cleanup-test-{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    try:
        expired = directory / "pramaan-expired.zip"
        recent = directory / "pramaan-recent.zip"
        unrelated = directory / "keep.txt"
        expired.write_bytes(b"old")
        recent.write_bytes(b"new")
        unrelated.write_bytes(b"keep")
        now = time.time()
        os.utime(expired, (now - 7200, now - 7200))
        removed = cleanup_expired_exports(directory=directory, now=now, ttl_seconds=3600)
        assert removed == 1
        assert not expired.exists()
        assert recent.exists()
        assert unrelated.exists()
    finally:
        shutil.rmtree(directory)


def test_hand_authored_counterexample_replays_with_concrete_failure():
    root = Path(__file__).resolve().parents[1]
    code, report = replay(root / "examples" / "ajrakh_d4_counterexample.pattern.json",
                          root / "examples" / "ajrakh_d4_counterexample.certificate.json")
    symmetry = next(check for check in report["checks"] if check["id"] == "symmetry_d4")
    assert code == 1
    assert report["all_hard_constraints_pass"] is False
    assert report["certificate_replay"] == {"matches": True, "message": "Certificate matches deterministic replay"}
    assert symmetry["status"] == "FAIL"
    assert symmetry["counterexample"]["original"] != symmetry["counterexample"]["transformed"]
