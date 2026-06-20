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
    Return telemetry interpolated to n_points evenly spaced along odometry Distance.

    get_telemetry().add_distance() gives GPS and wheel-speed-integrated distance
    in a single consistently-timed DataFrame.  Distance is smoother than GPS
    arc-length and starts at 0 at the lap start (timing beacon crossing),
    providing a common physical reference axis for comparing both drivers.

    TimeS is lap-relative (time_s - time_s[0]) so the delta between drivers at
    the same Distance point correctly reflects sectoral time differences.
    """
    tel = (
        lap.get_telemetry()
        .add_distance()
        .dropna(subset=["X", "Y", "Speed", "Distance"])
        .reset_index(drop=True)
    )

    raw_t  = tel["Time"].dt.total_seconds().values
    time_s = np.maximum.accumulate(raw_t - raw_t[0])       # lap-relative
    dist   = np.maximum.accumulate(tel["Distance"].values)  # monotonic odometry
    x      = _smooth1d(tel["X"].values, window=5)
    y      = _smooth1d(tel["Y"].values, window=5)
    speed  = tel["Speed"].values

    dist_max = dist[-1]
    grid = np.linspace(0, dist_max, n_points)

    return pd.DataFrame(
        {
            "X":        np.interp(grid, dist, x),
            "Y":        np.interp(grid, dist, y),
            "Speed":    np.interp(grid, dist, speed),
            "TimeS":    np.interp(grid, dist, time_s),
            "NormDist": grid / dist_max,
        }
    )
