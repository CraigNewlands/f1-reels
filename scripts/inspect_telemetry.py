"""
Visualise raw FastF1 telemetry in Rerun before any processing.

Run:
    python scripts/inspect_telemetry.py
    python scripts/inspect_telemetry.py --year 2025 --round Bahrain --n 3

Saves output/inspect_telemetry.rrd — open with the Rerun desktop app
(https://rerun.io/download) or drag onto the app window.

Layout
──────
  Left:  2-D track view  (raw GPS points + line, smoothed overlay)
  Right: time-series panels stacked vertically
           • speed (kph) vs lap distance
           • GPS fix-to-fix step size vs lap distance  (red = artifact >100u)
"""

import argparse
import sys
from pathlib import Path

import fastf1
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

sys.path.insert(0, str(Path(__file__).parent.parent))

from f1reels.config import CACHE_DIR
from f1reels.data.telemetry import get_pole_laps, _smooth1d

DRIVER_COLORS = [
    [255, 136,   0],   # orange  P1
    [ 39, 244, 210],   # cyan    P2
    [255, 135, 188],   # pink    P3
]


def load_session(year: int, round_name: str, n: int):
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    session = fastf1.get_session(year, round_name, "Q")
    session.load()
    return get_pole_laps(session, n=n)


def log_driver(pos: int, abbr: str, tel, suffix: str = ""):
    color = DRIVER_COLORS[min(pos, len(DRIVER_COLORS) - 1)]
    tag = f"{abbr}_{suffix}" if suffix else abbr

    dist   = np.maximum.accumulate(tel["Distance"].values).astype(float)
    x_raw  = tel["X"].values.astype(float)
    y_raw  = tel["Y"].values.astype(float)
    speed  = tel["Speed"].values.astype(float)
    dates  = tel["Date"]   # UTC datetime per sample

    steps    = np.sqrt(np.diff(x_raw) ** 2 + np.diff(y_raw) ** 2)
    dates_mid = dates.iloc[:-1]

    # Interpolate to 2000 evenly-spaced distance points — no smoothing
    grid = np.linspace(0, dist[-1], 2000)
    xi = np.interp(grid, dist, x_raw)
    yi = np.interp(grid, dist, y_raw)

    # Time at each grid point: interpolate lap-relative seconds against distance,
    # then convert back to UTC timestamps
    import pandas as pd
    t_raw_s = (dates - dates.iloc[0]).dt.total_seconds().values
    t_grid_s = np.interp(grid, dist, t_raw_s)
    dates_grid = dates.iloc[0] + pd.to_timedelta(t_grid_s, unit="s")

    # Colour raw GPS points by step size to show freezes vs jumps:
    #   green  = step < 30  (normal movement)
    #   yellow = step 30-100  (fast section or moderate jump)
    #   red    = step > 100  (jump artifact — GPS freeze then snap)
    # First point has no predecessor, colour it green.
    step_per_point = np.concatenate([[0], steps])
    def step_color(s):
        if s < 30:   return [0, 220, 80]    # green
        if s < 100:  return [255, 200, 0]   # yellow
        return [255, 40, 40]                 # red

    point_colors = [step_color(s) for s in step_per_point]

    # ── Static: raw GPS points coloured by step size ───────────────────────
    rr.log(f"track/raw/{tag}",
           rr.Points2D(np.column_stack([x_raw, y_raw]),
                       colors=point_colors, radii=5),
           static=True)
    rr.log(f"track/raw/{tag}_line",
           rr.LineStrips2D([np.column_stack([x_raw, y_raw])],
                           colors=[color], radii=1),
           static=True)

    # ── Static: interpolated line + points (no smoothing) ─────────────────
    rr.log(f"track/interpolated/{tag}_line",
           rr.LineStrips2D([np.column_stack([xi, yi])],
                           colors=[[255, 255, 255]], radii=1),
           static=True)
    rr.log(f"track/interpolated/{tag}_points",
           rr.Points2D(np.column_stack([xi, yi]),
                       colors=[[180, 180, 180]] * len(xi), radii=3),
           static=True)

    # ── Animated: raw GPS dot ──────────────────────────────────────────────
    for ts, x, y in zip(dates, x_raw, y_raw):
        rr.set_time("time", timestamp=ts)
        rr.log(f"track/car_raw/{tag}",
               rr.Points2D([[x, y]], colors=[color], radii=12))

    # ── Animated: interpolated dot ─────────────────────────────────────────
    for ts, x, y in zip(dates_grid, xi, yi):
        rr.set_time("time", timestamp=ts)
        rr.log(f"track/car_interp/{tag}",
               rr.Points2D([[x, y]], colors=[[255, 255, 255]], radii=12))

    # ── Time-series: speed vs time ─────────────────────────────────────────
    for ts, v in zip(dates, speed):
        rr.set_time("time", timestamp=ts)
        rr.log(f"plots/speed/{tag}", rr.Scalars([v]))

    # ── Time-series: GPS step size vs time ────────────────────────────────
    for ts, s in zip(dates_mid, steps):
        rr.set_time("time", timestamp=ts)
        rr.log(f"plots/gps_step/{tag}", rr.Scalars([s]))

    lap_s = (dates.iloc[-1] - dates.iloc[0]).total_seconds()
    print(f"       [{suffix}] step mean={steps.mean():.0f}  "
          f"max={steps.max():.0f}  jumps>100: {(steps > 100).sum()}")


def log_driver_gps_only(pos: int, abbr: str, gps):
    """Log just the raw GPS position fixes, with no forward-filling."""
    color = DRIVER_COLORS[min(pos, len(DRIVER_COLORS) - 1)]
    x = gps["X"].values.astype(float)
    y = gps["Y"].values.astype(float)
    dates = gps["Date"]

    steps = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    step_per_point = np.concatenate([[0], steps])
    def step_color(s):
        if s < 30:  return [0, 220, 80]
        if s < 100: return [255, 200, 0]
        return [255, 40, 40]
    point_colors = [step_color(s) for s in step_per_point]

    # Static: true GPS fixes coloured by step size
    rr.log(f"track/gps_only/{abbr}",
           rr.Points2D(np.column_stack([x, y]),
                       colors=point_colors, radii=8),
           static=True)
    rr.log(f"track/gps_only/{abbr}_line",
           rr.LineStrips2D([np.column_stack([x, y])],
                           colors=[color], radii=2),
           static=True)

    # Animated: dot at each true GPS fix timestamp
    for ts, xi, yi in zip(dates, x, y):
        rr.set_time("time", timestamp=ts)
        rr.log(f"track/car_gps/{abbr}",
               rr.Points2D([[xi, yi]], colors=[color], radii=16))

    dt = np.diff(gps["Time"].dt.total_seconds().values)
    print(f"       gps_only:  steps mean={steps.mean():.0f}  "
          f"max={steps.max():.0f}  jumps>100: {(steps > 100).sum()}")


def make_blueprint(abbrs: list[str]) -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(
                name="GPS track",
                origin="/track",
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(
                    name="Speed (kph)",
                    origin="/plots/speed",
                ),
                rrb.TimeSeriesView(
                    name="GPS step size (artifacts)",
                    origin="/plots/gps_step",
                ),
            ),
            column_shares=[1, 1],
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Inspect F1 GPS telemetry in Rerun")
    parser.add_argument("--year",  type=int, default=2025)
    parser.add_argument("--round", default="Bahrain")
    parser.add_argument("--n",     type=int, default=2,
                        help="number of drivers (default 2)")
    args = parser.parse_args()

    abbrs = []

    print(f"Loading {args.year} {args.round} qualifying …")
    pairs = load_session(args.year, args.round, args.n)

    # Find the rerun viewer binary — it ships inside the rerun-sdk package
    _viewer = Path(sys.executable).parent / "rerun"
    rr.init("f1_telemetry_inspect")
    rr.spawn(executable_path=str(_viewer))

    for pos, (row, lap) in enumerate(pairs):
        abbr = row["Abbreviation"]
        abbrs.append(abbr)
        print(f"P{pos + 1} {abbr}")

        # The two raw streams before FastF1 merges them
        car = lap.get_car_data()   # speed/RPM/gear — no X,Y
        gps = lap.get_pos_data()   # X,Y only
        dt_car = np.diff(car["Time"].dt.total_seconds().values)
        dt_gps = np.diff(gps["Time"].dt.total_seconds().values)

        # Merged stream (what get_telemetry() gives): union of both timestamps,
        # gaps filled by forward-filling — X,Y is stale between position updates
        merged = (
            lap.get_telemetry()
            .add_distance()
            .dropna(subset=["X", "Y", "Speed", "Distance"])
            .reset_index(drop=True)
        )
        dt_merged = np.diff(merged["Time"].dt.total_seconds().values)

        print(f"       car data:  {len(car)} samples  {1/dt_car.mean():.1f} Hz avg")
        print(f"       gps:       {len(gps)} samples  {1/dt_gps.mean():.1f} Hz avg  "
              f"(true position updates)")
        print(f"       merged:    {len(merged)} samples  {1/dt_merged.mean():.1f} Hz avg  "
              f"(forward-filled between gps updates)")

        # Log merged (what we currently use — includes stale X,Y rows)
        log_driver(pos, abbr, merged, suffix="merged")

        # Log true GPS fixes only — these are the real position measurements
        gps_clean = gps.dropna(subset=["X", "Y"]).reset_index(drop=True)
        log_driver_gps_only(pos, abbr, gps_clean)

    blueprint = make_blueprint(abbrs)
    rr.send_blueprint(blueprint)
    print("\nData sent — Rerun viewer should be open.")


if __name__ == "__main__":
    main()
