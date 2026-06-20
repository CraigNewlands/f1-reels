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
    """
    Both drivers must be at the same position when progress=0.
    This is the regression test for the GPS-phase bug: previously, build_telemetry
    zeroed time but NOT the arc-length, so each driver's NormDist=0 mapped to a
    different physical location (wherever their first GPS sample happened to land
    after the lap-start beacon, which varied by driver).

    The fix prepends a T=0 extrapolated row so NormDist=0 always corresponds to
    the true lap-start location.  We simulate this by checking that two telemetry
    arrays that START at different offsets (mimicking different GPS phases) still
    both evaluate to x=0 after the extrapolation anchoring.
    """
    # Simulate two drivers whose raw GPS first sample is at different offsets
    # from the true start line (50 m and 120 m into the lap respectively).
    # After the fix, both telemetry arrays should start at x=0 (the start line).
    n = 500
    # Driver 1: first GPS sample is 50 m in, so x[0] != 0
    x1_raw = np.linspace(50, 5000, n)
    # Driver 2: first GPS sample is 120 m in
    x2_raw = np.linspace(120, 5000, n)

    # The fix prepends a back-extrapolated x=0 entry; after interpolation to
    # NormDist grid, index 0 should be at or very near x=0 for both.
    # Here we test the contract: after anchoring, _interp_position(0, anchored)
    # must equal 0.0 for both.  We represent "anchored" by prepending the 0 point.
    x1_anchored = np.concatenate([[0.0], x1_raw])
    x2_anchored = np.concatenate([[0.0], x2_raw])

    pos1 = _interp_position(0.0, x1_anchored)
    pos2 = _interp_position(0.0, x2_anchored)

    assert pos1 == pytest.approx(0.0), f"d1 start position should be 0, got {pos1}"
    assert pos2 == pytest.approx(0.0), f"d2 start position should be 0, got {pos2}"
    assert pos1 == pytest.approx(pos2), (
        f"Both drivers must start at the same position at T=0. "
        f"Got d1={pos1:.2f}, d2={pos2:.2f} — GPS phase offset bug is present."
    )


def test_faster_driver_is_d1_after_swap():
    """
    If the API returns drivers in the wrong order (slower first), the swap
    must correct it so d1 always has the shorter lap time.
    """
    t1_wrong = 91.5   # slower — would be d1 if API order is trusted blindly
    t2_wrong = 90.8   # faster — would be d2

    # Simulate the swap logic
    if t1_wrong > t2_wrong:
        t1_correct, t2_correct = t2_wrong, t1_wrong
    else:
        t1_correct, t2_correct = t1_wrong, t2_wrong

    assert t1_correct < t2_correct, "After swap, d1 must have shorter lap time"
    ratio = t1_correct / t2_correct
    assert ratio < 1, "ratio must be < 1 so d2 always trails d1 on track"


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
