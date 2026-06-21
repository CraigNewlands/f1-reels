from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from .models import (
    CarPacket, CarPositionAtTime, DriverFrames,
    GpsFix, NormalisedPoint, OdometryPoint, TrackShape,
)


def extract_gps_fixes(lap) -> list[GpsFix]:
    """Extract true GPS position updates (Source=='pos' rows only, no forward-fill)."""
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
    """Extract car telemetry. session_time_s is lap-relative (0 = lap start)."""
    car = lap.get_car_data().reset_index(drop=True)
    t0  = float(car["SessionTime"].iloc[0].total_seconds())
    return [
        CarPacket(
            speed_kph=float(row.Speed),
            session_time_s=float(row.SessionTime.total_seconds()) - t0,
        )
        for _, row in car.iterrows()
    ]


def build_odometry(packets: list[CarPacket]) -> list[OdometryPoint]:
    """Integrate wheel speed over time → cumulative lap distance."""
    points: list[OdometryPoint] = []
    dist = 0.0
    for i, p in enumerate(packets):
        if i > 0:
            dt = p.session_time_s - packets[i - 1].session_time_s
            dist += (p.speed_kph / 3.6) * max(dt, 0.0)
        points.append(OdometryPoint(distance_m=dist, session_time_s=p.session_time_s))
    return points


def normalise_distance(points: list[OdometryPoint]) -> list[NormalisedPoint]:
    """Scale odometry to 0→1 fraction of total lap distance."""
    total = points[-1].distance_m
    return [
        NormalisedPoint(norm_dist=p.distance_m / total, session_time_s=p.session_time_s)
        for p in points
    ]


def resample(
    packets: list[CarPacket],
    total_dist_m: float,
    rate_hz: float,
) -> list[NormalisedPoint]:
    """
    Resample onto a uniform time grid by PCHIP-interpolating speed then integrating.
    Avoids the norm_dist humps that occur when interpolating an integral signal
    directly over large gaps (up to 1240ms between car packets).
    """
    t_in   = np.array([p.session_time_s for p in packets])
    spd_in = np.array([p.speed_kph      for p in packets]) / 3.6

    spd_spline = PchipInterpolator(t_in, spd_in)
    t_out      = np.arange(t_in[0], t_in[-1], 1.0 / rate_hz)
    spd_out    = np.clip(spd_spline(t_out), 0.0, None)

    dt_fine  = np.diff(t_out)
    dist_inc = (spd_out[:-1] + spd_out[1:]) / 2.0 * dt_fine
    dist_out = np.concatenate([[0.0], np.cumsum(dist_inc)])

    nd_out = np.clip(dist_out / total_dist_m, 0.0, 1.0)

    # Append an explicit endpoint so the car always reaches norm_dist=1.0.
    # np.arange stops ~33ms short of t_in[-1] and PCHIP integration gives
    # slightly less distance than raw odometry, leaving the car 5-8m short
    # of the start/finish without this.
    t_out  = np.append(t_out,  t_in[-1])
    nd_out = np.append(nd_out, 1.0)

    return [
        NormalisedPoint(norm_dist=float(nd), session_time_s=float(t))
        for nd, t in zip(nd_out, t_out)
    ]


def compute_positions(
    norm_points: list[NormalisedPoint],
    track: TrackShape,
) -> list[CarPositionAtTime]:
    """Map normalised lap positions onto the track shape → X,Y,Z at each moment."""
    return [
        CarPositionAtTime(
            *track.lookup(p.norm_dist),
            norm_dist=p.norm_dist,
            session_time_s=p.session_time_s,
        )
        for p in norm_points
    ]


def build_track_shape(
    all_fixes: list[list[GpsFix]],
    n_bins: int = 500,
    n_out: int = 2000,
) -> TrackShape:
    """
    Build the circuit outline from multi-lap GPS fixes in two steps:

    1. Median — bin fixes into n_bins buckets (~6-7 per bin with 10 Q3 laps)
       and take the median X,Y,Z.  The median rejects freeze-snap GPS artefacts.

    2. Arc-length re-parameterisation — upsample to n_out points that are
       uniformly spaced in physical distance so the dot moves at visually
       consistent speed.  Without this, sparse GPS coverage creates 5× speed
       variation purely from uneven bin spacing.
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

    t_bins = np.arange(n_bins)
    for arr in (x_med, y_med, z_med):
        nans = np.isnan(arr)
        if nans.any():
            arr[nans] = np.interp(t_bins[nans], t_bins[~nans], arr[~nans])

    t_fine = np.linspace(0, n_bins - 1, n_out)
    x_out  = np.interp(t_fine, t_bins, x_med)
    y_out  = np.interp(t_fine, t_bins, y_med)
    z_out  = np.interp(t_fine, t_bins, z_med)

    arc      = np.concatenate([[0.0], np.cumsum(np.sqrt(np.diff(x_out)**2 + np.diff(y_out)**2))])
    arc_norm = arc / arc[-1]
    t_uni    = np.linspace(0.0, 1.0, n_out)

    x_final = np.interp(t_uni, arc_norm, x_out)
    y_final = np.interp(t_uni, arc_norm, y_out)
    z_final = np.interp(t_uni, arc_norm, z_out)

    # Force the last point to equal the first so that lookup(1.0) returns
    # the same position as lookup(0.0).  GPS laps start/end just before/after
    # the timing beacon so the raw first and last points differ by ~10m.
    x_final[-1], y_final[-1], z_final[-1] = x_final[0], y_final[0], z_final[0]

    return TrackShape(x=x_final, y=y_final, z=z_final)


def build_driver_frames(
    lap,
    track: TrackShape,
    color: str,
    abbr: str,
    rate_hz: float = 30.0,
) -> DriverFrames:
    """Full pipeline for one driver: packets → resample → positions → DriverFrames."""
    packets      = extract_car_packets(lap)
    odometry     = build_odometry(packets)
    total_dist_m = odometry[-1].distance_m
    norm_pts     = resample(packets, total_dist_m, rate_hz=rate_hz)
    positions    = compute_positions(norm_pts, track)
    return DriverFrames(abbr=abbr, color=color, positions=positions, packets=packets)
