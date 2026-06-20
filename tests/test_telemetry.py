import numpy as np
import pandas as pd
import pytest

from f1reels.data.telemetry import _interpolate_to_grid, build_telemetry, N_POINTS


def _make_tel(n: int = 120) -> pd.DataFrame:
    t = np.linspace(0, 90, n)
    return pd.DataFrame(
        {
            "Time": pd.to_timedelta(t, unit="s"),
            "X": np.cos(t / 90 * 2 * np.pi) * 1000,
            "Y": np.sin(t / 90 * 2 * np.pi) * 500,
            "Speed": 150 + 150 * np.abs(np.sin(t / 8)),
            "Distance": np.linspace(0, 4500, n),
        }
    )


def _make_mock_lap(lap_start_offset_s: float, n: int = 120):
    """
    Return a minimal mock lap object for testing build_telemetry.

    lap_start_offset_s: how many seconds AFTER the true lap-start beacon the
    first GPS sample arrives (simulates GPS phase offset).  The raw telemetry
    Time column starts at (lap_start_s + lap_start_offset_s).
    """
    lap_start_s = 3600.0  # arbitrary session-absolute start time

    # Session-absolute times: first sample is offset seconds after beacon
    t_abs = lap_start_s + lap_start_offset_s + np.linspace(0, 90 - lap_start_offset_s, n)
    tel_df = pd.DataFrame(
        {
            "Time": pd.to_timedelta(t_abs, unit="s"),
            # Straight track from x=lap_start_offset_s*50 (where car is at t_abs[0])
            # to x=5000.  The true start-line is at x=0.
            "X": np.linspace(lap_start_offset_s * 50, 5000, n),
            "Y": np.zeros(n),
            "Speed": np.full(n, 200.0),
        }
    )

    class MockLap:
        def get_telemetry(self):
            return tel_df.copy()

        def __getitem__(self, key):
            if key == "LapStartTime":
                return pd.to_timedelta(lap_start_s, unit="s")
            if key == "LapTime":
                return pd.to_timedelta(90.0, unit="s")
            raise KeyError(key)

    return MockLap()


def test_output_has_correct_length():
    result = _interpolate_to_grid(_make_tel())
    assert len(result) == N_POINTS


def test_norm_dist_range():
    result = _interpolate_to_grid(_make_tel())
    assert result["NormDist"].iloc[0] == pytest.approx(0.0)
    assert result["NormDist"].iloc[-1] == pytest.approx(1.0)


def test_output_columns():
    result = _interpolate_to_grid(_make_tel())
    assert {"X", "Y", "Speed", "TimeS", "NormDist"}.issubset(result.columns)


def test_custom_n_points():
    result = _interpolate_to_grid(_make_tel(), n_points=200)
    assert len(result) == 200


def test_time_is_monotone():
    result = _interpolate_to_grid(_make_tel())
    assert (result["TimeS"].diff().dropna() >= 0).all()


def test_speed_within_input_bounds():
    df = _make_tel()
    result = _interpolate_to_grid(df)
    assert result["Speed"].min() >= df["Speed"].min() - 1
    assert result["Speed"].max() <= df["Speed"].max() + 1


# ---------------------------------------------------------------------------
# Tests for build_telemetry — the GPS-phase start-alignment fix
# ---------------------------------------------------------------------------

def test_build_telemetry_time_starts_at_zero_when_no_offset():
    """When first GPS sample coincides with beacon, TimeS[0] should be ~0."""
    lap = _make_mock_lap(lap_start_offset_s=0.0)
    result = build_telemetry(lap, n_points=N_POINTS)
    assert result["TimeS"].iloc[0] == pytest.approx(0.0, abs=0.05)


def test_build_telemetry_extrapolates_start_when_gps_lags():
    """
    When the first GPS sample is offset from the beacon (simulating GPS phase lag),
    build_telemetry must back-extrapolate so TimeS[0] ≈ 0.
    Previously, time was zeroed by subtracting time_s[0] (which is the time of the
    FIRST GPS SAMPLE, not the beacon crossing), so TimeS[0] was always 0 but the
    position was already hundreds of metres into the lap.
    After the fix, TimeS[0] should be near 0 AND the position should be the
    extrapolated start-line location.
    """
    offset = 0.08  # 80 ms GPS lag — typical worst case at 10 Hz GPS
    lap = _make_mock_lap(lap_start_offset_s=offset)
    result = build_telemetry(lap, n_points=N_POINTS)
    # TimeS at first grid point must be ≤ the lap_start_offset (extrapolated back)
    assert result["TimeS"].iloc[0] == pytest.approx(0.0, abs=0.05), (
        f"TimeS[0]={result['TimeS'].iloc[0]:.4f} — should be ~0 after back-extrapolation"
    )


def test_build_telemetry_two_drivers_same_start_x():
    """
    Two drivers with DIFFERENT GPS phase offsets must produce telemetry that
    starts at the same X position (i.e., NormDist=0 maps to the same location).
    This is the regression test for the visual offset bug at T=0.
    """
    lap1 = _make_mock_lap(lap_start_offset_s=0.02)  # 20 ms GPS lag
    lap2 = _make_mock_lap(lap_start_offset_s=0.09)  # 90 ms GPS lag — much further in

    tel1 = build_telemetry(lap1, n_points=N_POINTS)
    tel2 = build_telemetry(lap2, n_points=N_POINTS)

    x1_start = tel1["X"].iloc[0]
    x2_start = tel2["X"].iloc[0]

    # After anchoring, both starts should be close to the extrapolated start line.
    # The two raw first-samples differ by (0.09-0.02)*50 = 3.5 m; after fix the
    # extrapolated starts should be within 1 m of each other.
    assert abs(x1_start - x2_start) < 5.0, (
        f"Start positions differ by {abs(x1_start - x2_start):.2f} m — "
        f"GPS phase offset not corrected (d1 x0={x1_start:.1f}, d2 x0={x2_start:.1f})"
    )


def test_build_telemetry_norm_dist_range():
    lap = _make_mock_lap(lap_start_offset_s=0.05)
    result = build_telemetry(lap, n_points=N_POINTS)
    assert result["NormDist"].iloc[0] == pytest.approx(0.0)
    assert result["NormDist"].iloc[-1] == pytest.approx(1.0)


def test_build_telemetry_output_length():
    lap = _make_mock_lap(lap_start_offset_s=0.05)
    result = build_telemetry(lap, n_points=N_POINTS)
    assert len(result) == N_POINTS


def test_build_telemetry_time_monotone():
    lap = _make_mock_lap(lap_start_offset_s=0.05)
    result = build_telemetry(lap, n_points=N_POINTS)
    assert (result["TimeS"].diff().dropna() >= 0).all()
