import numpy as np
import pandas as pd
import pytest

from f1reels.data.telemetry import _interpolate_to_grid, N_POINTS


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
