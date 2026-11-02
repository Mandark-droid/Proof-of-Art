import pytest

from pramaan.stats import intervals_overlap, wilson_interval


@pytest.mark.parametrize(
    ("successes", "total", "expected"),
    [
        (0, 10, (0.0, 0.2775)),
        (5, 10, (0.2366, 0.7634)),
        (10, 10, (0.7225, 1.0)),
        (58, 72, (0.6997, 0.8805)),
    ],
)
def test_wilson_reference_values(successes, total, expected):
    low, high = wilson_interval(successes, total)
    assert low == pytest.approx(expected[0], abs=0.0001)
    assert high == pytest.approx(expected[1], abs=0.0001)


def test_tie_badge_when_intervals_overlap():
    assert intervals_overlap((0.20, 0.40), (0.35, 0.55))
    assert not intervals_overlap((0.10, 0.30), (0.31, 0.50))
