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


def _make_mock_lap(lap_start_offset_s: float = 0.05, n_car: int = 500, n_pos: int = 50):
    """
    Mock lap supplying get_car_data() and get_pos_data() separately.

    lap_start_offset_s: seconds after the timing beacon that the first GPS sample arrives.
    car data is at higher frequency (n_car samples), pos data at lower (n_pos samples).
    Distance in car data always starts at 0 (start/finish line).
    """
    lap_start_s = 3600.0
    lap_time = 90.0

    # Merged telemetry with Distance — mimics get_telemetry().add_distance()
    t_rel = lap_start_offset_s + np.linspace(0, lap_time - lap_start_offset_s, n_car)
    tel_df = pd.DataFrame({
        "Time":     pd.to_timedelta(t_rel, unit="s"),
        "X":        np.linspace(lap_start_offset_s * 60, 5000.0, n_car),
        "Y":        np.zeros(n_car),
        "Speed":    np.full(n_car, 200.0),
        "Distance": np.linspace(0.0, 5412.0, n_car),
    })

    class MockTelemetry(pd.DataFrame):
        def add_distance(self):
            return MockTelemetry(tel_df.copy())

    class MockLap:
        def get_telemetry(self):
            return MockTelemetry(tel_df.copy())

        def __getitem__(self, key):
            if key == "LapStartTime":
                return pd.to_timedelta(lap_start_s, unit="s")
            if key == "LapTime":
                return pd.to_timedelta(lap_time, unit="s")
            raise KeyError(key)

    return MockLap()


# ---------------------------------------------------------------------------
# _interpolate_to_grid (legacy helper, still used in tests)
# ---------------------------------------------------------------------------

def test_output_has_correct_length():
    assert len(_interpolate_to_grid(_make_tel())) == N_POINTS


def test_norm_dist_range():
    result = _interpolate_to_grid(_make_tel())
    assert result["NormDist"].iloc[0] == pytest.approx(0.0)
    assert result["NormDist"].iloc[-1] == pytest.approx(1.0)


def test_output_columns():
    assert {"X", "Y", "Speed", "TimeS", "NormDist"}.issubset(
        _interpolate_to_grid(_make_tel()).columns
    )


def test_custom_n_points():
    assert len(_interpolate_to_grid(_make_tel(), n_points=200)) == 200


def test_time_is_monotone():
    assert (_interpolate_to_grid(_make_tel())["TimeS"].diff().dropna() >= 0).all()


def test_speed_within_input_bounds():
    df = _make_tel()
    result = _interpolate_to_grid(df)
    assert result["Speed"].min() >= df["Speed"].min() - 1
    assert result["Speed"].max() <= df["Speed"].max() + 1


# ---------------------------------------------------------------------------
# build_telemetry — Distance-anchored, GPS interpolated
# ---------------------------------------------------------------------------

def test_build_telemetry_output_length():
    assert len(build_telemetry(_make_mock_lap())) == N_POINTS


def test_build_telemetry_norm_dist_range():
    result = build_telemetry(_make_mock_lap())
    assert result["NormDist"].iloc[0] == pytest.approx(0.0)
    assert result["NormDist"].iloc[-1] == pytest.approx(1.0)


def test_build_telemetry_time_starts_at_zero():
    """TimeS[0] should be ~0 — anchored to the timing beacon via LapStartTime."""
    result = build_telemetry(_make_mock_lap(lap_start_offset_s=0.05))
    assert result["TimeS"].iloc[0] == pytest.approx(0.0, abs=0.05)


def test_build_telemetry_time_monotone():
    assert (build_telemetry(_make_mock_lap())["TimeS"].diff().dropna() >= 0).all()


def test_build_telemetry_time_grid_is_even():
    """
    TimeS must be evenly spaced from 0 to t_max — this is the contract that lets
    qualifying_map use float fractions to place each driver on d1's reference track.
    """
    result = build_telemetry(_make_mock_lap(lap_start_offset_s=0.05))
    diffs = np.diff(result["TimeS"].values)
    assert np.allclose(diffs, diffs[0], rtol=1e-6), "TimeS should be evenly spaced"
    assert result["TimeS"].iloc[0] == pytest.approx(0.0, abs=0.1)
