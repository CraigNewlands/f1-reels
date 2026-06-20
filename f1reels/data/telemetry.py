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
    Return telemetry interpolated to n_points evenly spaced along GPS arc length.
    Arc length (not odometry Distance) is used so dots move at visually constant speed.

    The first row is extrapolated back to the true lap-start time (T=0 relative to
    the lap-start crossing) so that NormDist=0 corresponds to the same physical
    location for every driver, regardless of GPS sampling phase at the lap boundary.
    """
    tel = lap.get_telemetry().dropna(subset=["X", "Y", "Speed"]).reset_index(drop=True)

    x = tel["X"].values
    y = tel["Y"].values
    speed = tel["Speed"].values

    # Normalize to lap-relative time.
    # FastF1 Time is session-absolute; subtracting LapStartTime (the beacon crossing)
    # gives the true elapsed time within the lap.  The first raw GPS sample may be
    # several hundred milliseconds AFTER the beacon crossing, so time_s[0] > 0.
    lap_start_s = lap["LapStartTime"].total_seconds()
    time_s = tel["Time"].dt.total_seconds().values - lap_start_s
    # Guard against any tiny backwards glitches in the raw time channel
    time_s = np.maximum.accumulate(time_s)

    # Smooth GPS positions before computing arc length — removes sensor noise
    # that would otherwise cause jumpy dot motion
    x = _smooth1d(x, window=9)
    y = _smooth1d(y, window=9)

    # Prepend a synthetic T=0 row by linear extrapolation from the first two GPS
    # samples.  This anchors NormDist=0 to the true start/finish crossing rather
    # than to whichever GPS sample happened to fall first after the beacon event.
    # If the first sample is already at T≈0 (within 20 ms) skip extrapolation to
    # avoid amplifying any noise in the first GPS reading.
    _T0_THRESHOLD = 0.02  # seconds — GPS at 10 Hz → max 100 ms gap
    if time_s[0] > _T0_THRESHOLD and len(x) >= 2:
        dt10 = time_s[1] - time_s[0]  # time between first two samples
        if dt10 > 0:
            # linear back-extrapolation: position at t=0
            frac = time_s[0] / dt10
            x0 = x[0] - frac * (x[1] - x[0])
            y0 = y[0] - frac * (y[1] - y[0])
            s0 = speed[0]  # carry the first known speed
        else:
            x0, y0, s0 = x[0], y[0], speed[0]
        x = np.concatenate([[x0], x])
        y = np.concatenate([[y0], y])
        speed = np.concatenate([[s0], speed])
        time_s = np.concatenate([[0.0], time_s])

    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    arc = np.cumsum(np.sqrt(dx**2 + dy**2))
    arc = np.maximum.accumulate(arc)  # ensure strictly non-decreasing

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
