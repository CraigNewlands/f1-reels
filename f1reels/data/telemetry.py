import numpy as np
import pandas as pd

N_POINTS = 500  # interpolation resolution along lap distance


def get_pole_laps(session, n: int = 2) -> list[tuple]:
    """
    Return (result_row, fastest_lap) for the top n qualifying finishers.
    Uses session.results for position ordering and picks the fastest timed lap per driver.
    """
    results = session.results.copy()
    results["Position"] = pd.to_numeric(results["Position"], errors="coerce")
    results = results.sort_values("Position").head(n)
    pairs = []
    for _, row in results.iterrows():
        driver_laps = session.laps.pick_drivers(row["Abbreviation"])
        if len(driver_laps) == 0:
            continue
        fastest = driver_laps.pick_fastest()
        pairs.append((row, fastest))
    return pairs


def _interpolate_to_grid(tel_df: pd.DataFrame, n_points: int = N_POINTS) -> pd.DataFrame:
    """
    Interpolate telemetry columns to n_points evenly spaced along lap distance.
    Input DataFrame must have: Time (timedelta), X, Y, Speed, Distance columns.
    """
    tel = tel_df.dropna(subset=["X", "Y", "Speed", "Distance"]).copy()
    tel = tel.sort_values("Distance").reset_index(drop=True)

    dist = tel["Distance"].values
    time_s = tel["Time"].dt.total_seconds().values

    dist_max = dist[-1]
    grid = np.linspace(0, dist_max, n_points)

    return pd.DataFrame(
        {
            "X": np.interp(grid, dist, tel["X"].values),
            "Y": np.interp(grid, dist, tel["Y"].values),
            "Speed": np.interp(grid, dist, tel["Speed"].values),
            "TimeS": np.interp(grid, dist, time_s),
            "NormDist": grid / dist_max,
        }
    )


def _smooth1d(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple edge-padded moving average."""
    kernel = np.ones(window) / window
    padded = np.pad(arr, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


def build_telemetry(lap, n_points: int = N_POINTS) -> pd.DataFrame:
    """
    Return telemetry for one lap interpolated to n_points evenly spaced in time.

    get_telemetry() already merges GPS and car data correctly.  We normalise
    time relative to LapStartTime (beacon crossing) and interpolate onto an
    even time grid so both drivers share a consistent time axis.
    """
    tel = lap.get_telemetry().dropna(subset=["X", "Y", "Speed"]).reset_index(drop=True)

    raw_t = tel["Time"].dt.total_seconds().values
    # Time in get_telemetry() is already lap-relative (starts near 0).
    # Subtract the first value to guarantee it starts at exactly 0.
    time_s = np.maximum.accumulate(raw_t - raw_t[0])
    x = _smooth1d(tel["X"].values, window=5)
    y = _smooth1d(tel["Y"].values, window=5)
    speed = tel["Speed"].values

    t_max = time_s[-1]
    grid = np.linspace(0, t_max, n_points)

    return pd.DataFrame(
        {
            "X": np.interp(grid, time_s, x),
            "Y": np.interp(grid, time_s, y),
            "Speed": np.interp(grid, time_s, speed),
            "TimeS": grid,
            "NormDist": grid / t_max,
        }
    )
