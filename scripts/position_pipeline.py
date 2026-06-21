"""
Position pipeline: raw FastF1 data → accurate driver position in Rerun.

Models
──────
  GpsFix            one GPS position packet  (get_pos_data, Source=='pos')
  CarPacket         one car telemetry sample (get_car_data)
  OdometryPoint     wheel-speed-integrated distance at one point in time
  NormalisedPoint   0→1 fraction along lap   + session time
  TrackShape        circuit outline from multi-lap GPS median, parameterised 0→1
  CarPositionAtTime X,Y on the track at a session time  → goes to Rerun

Pipeline
────────
  [all Q3 laps] → extract_gps_fixes() → build_track_shape() → TrackShape
                                                                    ↓
  [fastest lap] → extract_car_packets()                             │
                       ↓                                            │
                  build_odometry()                                  │
                       ↓                                            │
                  normalise_distance()  → list[NormalisedPoint]     │
                                               ↓                    │
                                       compute_positions(TrackShape)
                                               ↓
                                     list[CarPositionAtTime]  → Rerun
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import fastf1
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(Path(__file__).parent.parent))
from f1reels.config import CACHE_DIR

# ── Models ────────────────────────────────────────────────────────────────────

@dataclass
class GpsFix:
    """One raw GPS position from pos_data (Source=='pos' rows only — no forward-fill)."""
    x: float
    y: float
    z: float
    distance_m: float       # wheel-speed distance at this broadcast time (from FastF1 merge)
    norm_dist: float        # distance_m / total_lap_distance  (0→1)


@dataclass
class CarPacket:
    """One raw car telemetry sample. Speed is reliable (wheel sensor); session_time is the
    relative clock from the F1 timing beacon."""
    speed_kph: float
    session_time_s: float   # seconds from lap start (reliable relative clock)


@dataclass
class OdometryPoint:
    """Wheel-speed-integrated cumulative distance at one point in time."""
    distance_m: float
    session_time_s: float


@dataclass
class NormalisedPoint:
    """How far around the lap (0→1) and when."""
    norm_dist: float
    session_time_s: float


@dataclass
class TrackShape:
    """Circuit outline, built once from the median of all Q3 GPS fixes.
    Parameterised uniformly 0→1 by arc-length so lookup() returns X,Y,Z
    at any normalised lap position."""
    x: np.ndarray   # (N,)
    y: np.ndarray   # (N,)
    z: np.ndarray   # (N,)

    def lookup(self, norm_dist: float) -> tuple[float, float, float]:
        t = np.linspace(0, 1, len(self.x))
        return (
            float(np.interp(norm_dist, t, self.x)),
            float(np.interp(norm_dist, t, self.y)),
            float(np.interp(norm_dist, t, self.z)),
        )

    def all_points(self) -> np.ndarray:
        return np.column_stack([self.x, self.y])


@dataclass
class CarPositionAtTime:
    """Plot-ready: where the car is on the track and when."""
    x: float
    y: float
    z: float
    session_time_s: float


# ── Pure functions ─────────────────────────────────────────────────────────────

def extract_gps_fixes(lap) -> list[GpsFix]:
    """
    Extract true GPS position updates from a lap.
    Uses Source=='pos' rows from the merged telemetry — these have fresh X,Y,Z
    and a distance_m computed from wheel speed at the time of the broadcast.
    Ignores 'interpolation' boundary rows added by FastF1.
    """
    tel = (
        lap.get_telemetry()
        .add_distance()
        .query("Source == 'pos'")
        .dropna(subset=["X", "Y", "Z", "Distance"])
        .reset_index(drop=True)
    )
    total = lap.get_telemetry().add_distance()["Distance"].max()
    return [
        GpsFix(
            x=float(row.X),
            y=float(row.Y),
            z=float(row.Z),
            distance_m=float(row.Distance),
            norm_dist=float(row.Distance) / total,
        )
        for _, row in tel.iterrows()
    ]


def extract_car_packets(lap) -> list[CarPacket]:
    """Extract car telemetry packets. Speed comes from the wheel sensor (reliable).
    session_time_s is the timing-beacon-based relative clock."""
    car = lap.get_car_data().reset_index(drop=True)
    return [
        CarPacket(
            speed_kph=float(row.Speed),
            session_time_s=float(row.SessionTime.total_seconds()),
        )
        for _, row in car.iterrows()
    ]


def build_odometry(packets: list[CarPacket]) -> list[OdometryPoint]:
    """Integrate wheel speed over session_time to get cumulative distance.
    This is reliable because the speed sensor is accurate and packet order is preserved."""
    points: list[OdometryPoint] = []
    dist = 0.0
    for i, p in enumerate(packets):
        if i > 0:
            dt = p.session_time_s - packets[i - 1].session_time_s
            dist += (p.speed_kph / 3.6) * max(dt, 0.0)
        points.append(OdometryPoint(distance_m=dist, session_time_s=p.session_time_s))
    return points


def normalise_distance(points: list[OdometryPoint]) -> list[NormalisedPoint]:
    """Scale odometry 0→1 as a fraction of total lap distance."""
    total = points[-1].distance_m
    return [
        NormalisedPoint(norm_dist=p.distance_m / total, session_time_s=p.session_time_s)
        for p in points
    ]


def build_track_shape(
    all_fixes: list[list[GpsFix]],
    n_bins: int = 500,
    n_out: int = 2000,
) -> TrackShape:
    """
    Build circuit outline in two steps:

    1. Median pass — bin GPS fixes into n_bins buckets and take the median
       X,Y,Z in each.  n_bins is chosen so each bucket has ~5-7 fixes on
       average (10 laps × ~337 fixes / 500 bins ≈ 6.7).  The median then
       has enough samples to reject freeze-snap outliers.

    2. Upsample — interpolate the n_bins median points up to n_out points
       so TrackShape.lookup() has fine resolution.
    """
    xs: list[list[float]] = [[] for _ in range(n_bins)]
    ys: list[list[float]] = [[] for _ in range(n_bins)]
    zs: list[list[float]] = [[] for _ in range(n_bins)]

    for fixes in all_fixes:
        for fix in fixes:
            idx = min(int(fix.norm_dist * (n_bins - 1)), n_bins - 1)
            xs[idx].append(fix.x)
            ys[idx].append(fix.y)
            zs[idx].append(fix.z)

    x_med = np.array([np.median(v) if v else np.nan for v in xs])
    y_med = np.array([np.median(v) if v else np.nan for v in ys])
    z_med = np.array([np.median(v) if v else np.nan for v in zs])

    # Fill any empty bins by interpolating from neighbours
    t_bins = np.arange(n_bins)
    for arr in (x_med, y_med, z_med):
        nans = np.isnan(arr)
        if nans.any():
            arr[nans] = np.interp(t_bins[nans], t_bins[~nans], arr[~nans])

    # Upsample to n_out points on a uniform bin grid first
    t_fine = np.linspace(0, n_bins - 1, n_out)
    x_out = np.interp(t_fine, t_bins, x_med)
    y_out = np.interp(t_fine, t_bins, y_med)
    z_out = np.interp(t_fine, t_bins, z_med)

    # Re-parameterise by arc length so consecutive points are evenly spaced
    # in physical/visual distance.  Without this, GPS bin gaps are uneven
    # (0 to 5.3m per step) and the dot appears to speed up and slow down
    # purely due to sparse GPS coverage — nothing to do with actual car speed.
    arc = np.concatenate([[0.0], np.cumsum(np.sqrt(np.diff(x_out)**2 + np.diff(y_out)**2))])
    arc_norm = arc / arc[-1]
    t_uniform = np.linspace(0.0, 1.0, n_out)

    return TrackShape(
        x=np.interp(t_uniform, arc_norm, x_out),
        y=np.interp(t_uniform, arc_norm, y_out),
        z=np.interp(t_uniform, arc_norm, z_out),
    )


def resample(
    packets: list[CarPacket],
    total_dist_m: float,
    rate_hz: float,
) -> list[NormalisedPoint]:
    """
    Resample onto a uniform time grid by PCHIP-interpolating speed then
    integrating at rate_hz.  Interpolating norm_dist directly created
    speed humps at large gaps (up to 1240ms); interpolating speed first
    avoids this because speed is a first-order signal with no such artefact.
    """
    t_in   = np.array([p.session_time_s for p in packets])
    spd_in = np.array([p.speed_kph      for p in packets]) / 3.6  # m/s

    spd_spline = PchipInterpolator(t_in, spd_in)

    t_out   = np.arange(t_in[0], t_in[-1], 1.0 / rate_hz)
    spd_out = np.clip(spd_spline(t_out), 0.0, None)

    dt_fine  = np.diff(t_out)
    dist_inc = (spd_out[:-1] + spd_out[1:]) / 2.0 * dt_fine
    dist_out = np.concatenate([[0.0], np.cumsum(dist_inc)])

    nd_out = np.clip(dist_out / total_dist_m, 0.0, 1.0)

    return [
        NormalisedPoint(norm_dist=float(nd), session_time_s=float(t))
        for nd, t in zip(nd_out, t_out)
    ]


def compute_positions(
    norm_points: list[NormalisedPoint],
    track: TrackShape,
) -> list[CarPositionAtTime]:
    """Map normalised lap positions onto the track shape to get X,Y at each moment."""
    return [
        CarPositionAtTime(
            *track.lookup(p.norm_dist),
            session_time_s=p.session_time_s,
        )
        for p in norm_points
    ]


# ── Rerun logging ──────────────────────────────────────────────────────────────

def log_track(track: TrackShape, circuit_info) -> None:
    """Log the track outline and corner markers as static entities."""
    rr.log(
        "track/shape",
        rr.LineStrips2D([track.all_points()], colors=[[255, 255, 255]], radii=2),
        static=True,
    )
    corners = circuit_info.corners
    rr.log(
        "track/corners",
        rr.Points2D(
            corners[["X", "Y"]].values,
            colors=[[255, 200, 0]] * len(corners),
            radii=10,
        ),
        static=True,
    )


def log_driver(
    positions: list[CarPositionAtTime],
    packets: list[CarPacket],
    norm_pts: list[NormalisedPoint],
    color: list[int],
    abbr: str,
) -> None:
    """Log animated car dot at 30 Hz and speed + lap progress at native ~3.7 Hz."""
    for pos in positions:
        rr.set_time("session_time_s", duration=pos.session_time_s)
        rr.log(f"car/{abbr}", rr.Points2D([[pos.x, pos.y]], colors=[color], radii=14))

    for pkt, np_pt in zip(packets, norm_pts):
        rr.set_time("session_time_s", duration=pkt.session_time_s)
        rr.log("plots/speed",     rr.Scalars([pkt.speed_kph]))
        rr.log("plots/norm_dist", rr.Scalars([np_pt.norm_dist]))


def make_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(name="Track", origin="/"),
            rrb.Vertical(
                rrb.TimeSeriesView(
                    name="Speed (kph)",
                    origin="/plots/speed",
                    axis_y=rrb.ScalarAxis(zoom_lock=True),
                ),
                rrb.TimeSeriesView(
                    name="Lap progress (0→1)",
                    origin="/plots/norm_dist",
                    axis_y=rrb.ScalarAxis(zoom_lock=True),
                ),
            ),
            column_shares=[1, 1],
        ),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    print("Loading 2025 Bahrain Q …")
    session = fastf1.get_session(2025, "Bahrain", "Q")
    session.load()

    # ── Build TrackShape from all Q3 laps ─────────────────────────────────
    print("Collecting Q3 GPS fixes for track shape …")
    q3_drivers = session.results[session.results["Q3"].notna()]["DriverNumber"].tolist()
    all_fixes: list[list[GpsFix]] = []
    for drv in q3_drivers:
        try:
            lap = session.laps.pick_drivers(drv).pick_fastest()
            fixes = extract_gps_fixes(lap)
            if fixes:
                all_fixes.append(fixes)
                print(f"  {session.get_driver(drv)['Abbreviation']}: {len(fixes)} fixes")
        except Exception as e:
            print(f"  {drv}: skipped ({e})")
    print(f"  {len(all_fixes)} laps collected")
    track = build_track_shape(all_fixes)
    print(f"  TrackShape: {len(track.x)} points")

    # ── Build driver position for fastest lap ─────────────────────────────
    lap = session.laps.pick_fastest()
    abbr = session.get_driver(lap["DriverNumber"])["Abbreviation"]
    print(f"Building position for {abbr} fastest lap …")

    packets      = extract_car_packets(lap)
    odometry     = build_odometry(packets)
    total_dist_m = odometry[-1].distance_m
    norm_pts     = resample(packets, total_dist_m, rate_hz=30.0)
    positions    = compute_positions(norm_pts, track)
    print(f"  {len(packets)} raw packets → {len(norm_pts)} resampled @ 30 Hz")

    # ── Send to Rerun ──────────────────────────────────────────────────────
    viewer = Path(sys.executable).parent / "rerun"
    rr.init("f1_position_pipeline")
    rr.spawn(executable_path=str(viewer))

    circuit_info = session.get_circuit_info()
    log_track(track, circuit_info)
    log_driver(positions, packets, norm_pts, [255, 136, 0], abbr)
    rr.send_blueprint(make_blueprint())
    print("Done — Rerun viewer should be open.")


if __name__ == "__main__":
    main()
