from tests.helpers import pattern_obj
from pramaan.checks.core import verify_pattern


def test_d4_symmetric_orbit_passes():
    checks, _, _ = verify_pattern(pattern_obj())
    assert next(c for c in checks if c["id"] == "symmetry_d4")["status"] == "PASS"


def test_single_non_invariant_motif_fails_d4_with_witness():
    p = pattern_obj()
    p.layers[0].motifs = p.layers[0].motifs[:1]
    checks, _, grid = verify_pattern(p)
    check = next(c for c in checks if c["id"] == "symmetry_d4")
    assert check["status"] == "FAIL"
    assert check["counterexample"]["original"]
    witness = check["counterexample"]
    a = witness["original"]
    b = witness["transformed"]
    assert a != b
    assert int(grid[a[1], a[0]]) != int(grid[b[1], b[0]])


def test_d2_pattern_can_pass_d2_and_fail_d4():
    p = pattern_obj()
    p.layers[0].motifs = [m.model_copy(update={"center": c}) for m, c in zip(
        p.layers[0].motifs,
        [(16, 20), (47, 20), (47, 43), (16, 43)],
    )]
    p.constraints.symmetry = "D2"
    checks, _, _ = verify_pattern(p)
    assert next(c for c in checks if c["id"] == "symmetry_d2")["status"] == "PASS"
    p.constraints.symmetry = "D4"
    checks, _, grid = verify_pattern(p)
    failed = next(c for c in checks if c["id"] == "symmetry_d4")
    assert failed["status"] == "FAIL"
    witness = failed["counterexample"]
    assert witness["transform"] == "R90"
    x1, y1 = witness["original"]
    x2, y2 = witness["transformed"]
    assert int(grid[y1, x1]) != int(grid[y2, x2])
