import numpy as np
import pandas as pd

N_POINTS = 500  # interpolation resolution along lap distance


def get_pole_laps(session, n: int = 2) -> list[tuple]:
    """
    Return (result_row, fastest_lap) for the top n qualifying finishers.
    Uses session.results for position ordering and picks the fastest timed lap per driver.
    """
    results = session.results.sort_values("Position").head(n)
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


def build_telemetry(lap, n_points: int = N_POINTS) -> pd.DataFrame:
    """
    Return telemetry interpolated to n_points evenly spaced along GPS arc length.
    Arc length (not odometry Distance) is used so dots move at visually constant speed.
    """
    tel = lap.get_telemetry().dropna(subset=["X", "Y", "Speed"]).reset_index(drop=True)

    x = tel["X"].values
    y = tel["Y"].values
    time_s = tel["Time"].dt.total_seconds().values
    speed = tel["Speed"].values

    # Arc length from GPS positions — this is what drives visual speed, not odometry
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    arc = np.cumsum(np.sqrt(dx**2 + dy**2))

    arc_max = arc[-1]
    grid = np.linspace(0, arc_max, n_points)

    return pd.DataFrame(
        {
            "X": np.interp(grid, arc, x),
            "Y": np.interp(grid, arc, y),
            "Speed": np.interp(grid, arc, speed),
            "TimeS": np.interp(grid, arc, time_s),
            "NormDist": grid / arc_max,
        }
    )
