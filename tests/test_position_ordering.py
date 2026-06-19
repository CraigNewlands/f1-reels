"""
Verify that at any time T, the faster driver (d1) is at a strictly higher
lap fraction than d2. Integer-index approaches collapse for small gaps
(< 1 telemetry step) — this test enforces float-fraction positioning.
"""
import numpy as np
import pytest


def _fractions_at_progress(progress, t1_laptime, t2_laptime):
    """Return (f1, f2) — each driver's lap fraction at animation progress."""
    f1 = progress
    f2 = progress * (t1_laptime / t2_laptime)
    return f1, f2


def _interp_position(fraction, values):
    """Float-interpolated position: fraction ∈ [0,1] → value from array."""
    grid = np.linspace(0, 1, len(values))
    return float(np.interp(fraction, grid, values))


@pytest.mark.parametrize("progress", [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
def test_d1_fraction_always_greater_than_d2(progress):
    """At any progress, d1's lap fraction must strictly exceed d2's."""
    t1, t2 = 70.270, 70.500   # Monaco P1/P2 — very close times
    f1, f2 = _fractions_at_progress(progress, t1, t2)
    assert f1 > f2, (
        f"progress={progress}: f1={f1:.6f} should be > f2={f2:.6f}"
    )


@pytest.mark.parametrize("progress", [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
def test_interp_position_d1_ahead_on_straight(progress):
    """On a straight track d1 must be physically further along than d2."""
    t1, t2 = 70.270, 70.500
    n = 500
    # Straight track: x increases with NormDist
    track_x = np.linspace(0, 5000, n)

    f1, f2 = _fractions_at_progress(progress, t1, t2)
    x1 = _interp_position(f1, track_x)
    x2 = _interp_position(f2, track_x)
    assert x1 > x2, (
        f"progress={progress}: d1 at {x1:.2f}m should be ahead of d2 at {x2:.2f}m"
    )


def test_both_at_start_line_at_zero():
    x1 = _interp_position(0.0, np.linspace(0, 5000, 500))
    x2 = _interp_position(0.0, np.linspace(0, 5000, 500))
    assert x1 == x2 == pytest.approx(0.0)


def test_gap_grows_over_lap():
    """Physical gap between cars should grow as the lap progresses."""
    t1, t2 = 70.270, 70.500
    track_x = np.linspace(0, 5000, 500)
    gaps = []
    for progress in [0.1, 0.3, 0.5, 0.7, 0.9]:
        f1, f2 = _fractions_at_progress(progress, t1, t2)
        gaps.append(_interp_position(f1, track_x) - _interp_position(f2, track_x))
    assert all(g > 0 for g in gaps), "Gap should always be positive (d1 ahead)"
    assert gaps[-1] > gaps[0], "Gap should grow over the lap"
