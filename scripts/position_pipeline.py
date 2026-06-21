"""
Rerun inspection script for the F1 position pipeline.

Loads a qualifying session, builds the track shape and driver positions,
then streams everything to the Rerun viewer for interactive exploration.

Run:
    python scripts/position_pipeline.py
    python scripts/position_pipeline.py --year 2025 --round Bahrain
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fastf1
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

sys.path.insert(0, str(Path(__file__).parent.parent))
from f1reels.colors import driver_color
from f1reels.config import CACHE_DIR
from f1reels.pipeline import (
    DriverFrames,
    TrackShape,
    build_driver_frames,
    build_track_shape,
    extract_gps_fixes,
)


# ── Rerun logging ──────────────────────────────────────────────────────────────

def log_track(track: TrackShape, circuit_info) -> None:
    rr.log("track/shape",
           rr.LineStrips2D([track.all_points()], colors=[[255, 255, 255]], radii=2),
           static=True)
    corners = circuit_info.corners
    rr.log("track/corners",
           rr.Points2D(corners[["X", "Y"]].values,
                       colors=[[255, 200, 0]] * len(corners), radii=10),
           static=True)

    # Show start/finish marker — all_points() closes the loop so this is just
    # a reference marker, not a gap indicator
    start = np.array([track.x[0], track.y[0]])
    rr.log("track/start_finish",
           rr.Points2D([start], colors=[[0, 255, 0]], radii=20), static=True)
    print(f"  Start/finish: ({start[0]:.0f}, {start[1]:.0f})")


def log_driver(driver: DriverFrames) -> None:
    rgb = [int(driver.color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)]

    rr.log(f"car/{driver.abbr}",
           rr.SeriesPoints(colors=[rgb], names=[driver.abbr]), static=True)
    rr.log(f"plots/speed/{driver.abbr}",
           rr.SeriesLines(colors=[rgb], names=[driver.abbr]), static=True)
    rr.log(f"plots/norm_dist/{driver.abbr}",
           rr.SeriesLines(colors=[rgb], names=[driver.abbr]), static=True)

    for pos in driver.positions:
        rr.set_time("session_time_s", duration=pos.session_time_s)
        rr.log(f"car/{driver.abbr}",
               rr.Points2D([[pos.x, pos.y]], colors=[rgb], radii=14))

    for pkt in driver.packets:
        rr.set_time("session_time_s", duration=pkt.session_time_s)
        rr.log(f"plots/speed/{driver.abbr}", rr.Scalars([pkt.speed_kph]))

    nd_arr = [(pos.norm_dist, pos.session_time_s) for pos in driver.positions]
    for nd, t in nd_arr:
        rr.set_time("session_time_s", duration=t)
        rr.log(f"plots/norm_dist/{driver.abbr}", rr.Scalars([nd]))


def make_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(name="Track", origin="/"),
            rrb.Vertical(
                rrb.TimeSeriesView(name="Speed (kph)", origin="/plots/speed",
                                   axis_y=rrb.ScalarAxis(zoom_lock=True)),
                rrb.TimeSeriesView(name="Lap progress (0→1)", origin="/plots/norm_dist",
                                   axis_y=rrb.ScalarAxis(zoom_lock=True)),
            ),
            column_shares=[3, 2],
        ),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",  type=int, default=2025)
    parser.add_argument("--round", default="Bahrain")
    args = parser.parse_args()

    fastf1.Cache.enable_cache(str(CACHE_DIR))
    print(f"Loading {args.year} {args.round} Q …")
    session = fastf1.get_session(args.year, args.round, "Q")
    session.load()

    q3_drivers = session.results[session.results["Q3"].notna()]["DriverNumber"].tolist()

    # Build TrackShape from all Q3 GPS fixes
    print("Collecting Q3 GPS fixes …")
    all_fixes = []
    for drv in q3_drivers:
        try:
            lap   = session.laps.pick_drivers(drv).pick_fastest()
            fixes = extract_gps_fixes(lap)
            if fixes:
                all_fixes.append(fixes)
                print(f"  {session.get_driver(drv)['Abbreviation']}: {len(fixes)} fixes")
        except Exception:
            pass
    print(f"  {len(all_fixes)} laps collected")
    track = build_track_shape(all_fixes)
    print(f"  TrackShape: {len(track.x)} points")

    # Build positions for each driver
    print("Building driver positions …")
    drivers = []
    for drv in q3_drivers:
        try:
            info  = session.get_driver(drv)
            abbr  = info["Abbreviation"]
            color = driver_color(abbr, info.get("TeamName", ""))
            lap        = session.laps.pick_drivers(drv).pick_fastest()
            q3_row     = session.results[session.results["DriverNumber"] == str(drv)].iloc[0]
            official_s = float(q3_row["Q3"].total_seconds())
            df         = build_driver_frames(lap, track, color, abbr, official_laptime_s=official_s)
            drivers.append(df)
            print(f"  {abbr}  {len(df.packets)} packets → {len(df.positions)} @ 30 Hz")
        except Exception as e:
            print(f"  {drv}: skipped ({e})")

    # Stream to Rerun
    viewer = Path(sys.executable).parent / "rerun"
    rr.init("f1_position_pipeline")
    rr.spawn(executable_path=str(viewer))

    log_track(track, session.get_circuit_info())
    for driver in drivers:
        log_driver(driver)

    rr.send_blueprint(make_blueprint())
    print("Done — Rerun viewer should be open.")


if __name__ == "__main__":
    main()
