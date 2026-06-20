"""
Compare GPS start-position alignment across three data sources:
  A) FastF1 get_telemetry() — current pipeline
  B) FastF1 session.pos_data — full-session stream, interpolated at LapStartTime
  C) OpenF1 API — independent data source

Plots start area zoomed in so we can see how close the two drivers' first
points are for each approach.
"""
import sys
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from f1reels.data.session import load_session
from f1reels.data.telemetry import get_pole_laps, _smooth1d, N_POINTS
from f1reels.colors import driver_color

YEAR  = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
ROUND = sys.argv[2]       if len(sys.argv) > 2 else "Bahrain"
OUT   = sys.argv[3]       if len(sys.argv) > 3 else "output/compare_sources.png"

print(f"Loading {YEAR} {ROUND} Q...")
session = load_session(YEAR, ROUND, "Q")
pairs   = get_pole_laps(session, n=2)

drivers = []
for row, lap in pairs:
    abbr  = row["Abbreviation"]
    color = driver_color(abbr, row.get("TeamName", ""))
    lt    = lap["LapTime"].total_seconds()
    drivers.append({"row": row, "lap": lap, "abbr": abbr, "color": color, "lt": lt})

# ── A: FastF1 get_telemetry() ──────────────────────────────────────────────
def build_a(lap):
    tel   = lap.get_telemetry().dropna(subset=["X","Y"])
    raw_t = tel["Time"].dt.total_seconds().values
    t     = raw_t - raw_t[0]
    x     = _smooth1d(tel["X"].values, window=5)
    y     = _smooth1d(tel["Y"].values, window=5)
    grid  = np.linspace(0, t[-1], N_POINTS)
    return (np.interp(grid, t, x), np.interp(grid, t, y), grid)

# ── B: FastF1 session.pos_data (full stream, interpolated at beacon) ───────
def build_b(lap, session):
    drv_num = str(int(lap["DriverNumber"]))
    if drv_num not in session.pos_data:
        print(f"  pos_data not available for driver {drv_num}")
        return None
    pos    = session.pos_data[drv_num]
    pos_t  = pos["Time"].dt.total_seconds().values
    pos_x  = _smooth1d(pos["X"].values, window=5)
    pos_y  = _smooth1d(pos["Y"].values, window=5)

    lap_start = lap["LapStartTime"].total_seconds()
    lap_end   = (lap["LapStartTime"] + lap["LapTime"]).total_seconds()

    # Interpolate exact position at beacon crossing
    x0 = float(np.interp(lap_start, pos_t, pos_x))
    y0 = float(np.interp(lap_start, pos_t, pos_y))

    # Slice lap
    mask  = (pos_t >= lap_start) & (pos_t <= lap_end)
    t_lap = np.concatenate([[0.0], pos_t[mask] - lap_start])
    x_lap = np.concatenate([[x0],  pos_x[mask]])
    y_lap = np.concatenate([[y0],  pos_y[mask]])

    grid = np.linspace(0, t_lap[-1], N_POINTS)
    return (np.interp(grid, t_lap, x_lap), np.interp(grid, t_lap, y_lap), grid)

# ── C: OpenF1 API ──────────────────────────────────────────────────────────
def fetch_openf1(session):
    """Find the matching OpenF1 session using the FastF1 session date."""
    base = "https://api.openf1.org/v1"
    # Use the FastF1 session date to find the right OpenF1 session key
    sess_date = session.date  # datetime of session start
    r = requests.get(f"{base}/sessions", params={
        "session_type": "Qualifying",
        "year": sess_date.year,
    }, timeout=15)
    r.raise_for_status()
    # Match by date proximity
    from datetime import timezone
    if sess_date.tzinfo is None:
        sess_date = sess_date.replace(tzinfo=timezone.utc)
    best, best_dt = None, float("inf")
    for s in r.json():
        try:
            from datetime import datetime
            sd = datetime.fromisoformat(s["date_start"].replace("Z", "+00:00"))
            dt = abs((sd - sess_date).total_seconds())
            if dt < best_dt:
                best, best_dt = s, dt
        except Exception:
            continue
    if best is None or best_dt > 7200:
        raise ValueError(f"No OpenF1 session matched within 2h of {sess_date}")
    sk = best["session_key"]
    print(f"  OpenF1 session_key={sk}  ({best['circuit_short_name']} {best['date_start']})")
    return sk, best

def build_c(lap, session_key, fastf1_session):
    drv_num = int(lap["DriverNumber"])
    lt_fastf1 = lap["LapTime"].total_seconds()

    from datetime import datetime, timezone
    def parse_iso(s):
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

    base = "https://api.openf1.org/v1"

    # Use OpenF1's own lap data to find this driver's fastest lap start time
    # (avoids FastF1 <-> OpenF1 clock alignment issues)
    r = requests.get(f"{base}/laps", params={
        "session_key": session_key, "driver_number": drv_num,
    }, timeout=15)
    r.raise_for_status()
    laps_of1 = [l for l in r.json() if l.get("lap_duration")]
    if not laps_of1:
        print(f"  No OpenF1 lap data for driver {drv_num}")
        return None
    best_lap = min(laps_of1, key=lambda l: l["lap_duration"])
    lt_s       = best_lap["lap_duration"]
    lap_start_abs = parse_iso(best_lap["date_start"])
    lap_end_abs   = lap_start_abs + lt_s
    print(f"  driver {drv_num}: OpenF1 best lap {lt_s:.3f}s (FastF1: {lt_fastf1:.3f}s)")

    # Fetch location data
    r = requests.get(f"{base}/location", params={
        "session_key": session_key, "driver_number": drv_num,
    }, timeout=30)
    r.raise_for_status()
    data = [d for d in r.json() if d["x"] != 0 or d["y"] != 0]

    ts = np.array([parse_iso(d["date"]) for d in data])
    xs = np.array([d["x"] for d in data], dtype=float)
    ys = np.array([d["y"] for d in data], dtype=float)

    mask = (ts >= lap_start_abs - 0.5) & (ts <= lap_end_abs + 0.5)
    if mask.sum() < 10:
        print(f"  Only {mask.sum()} samples in lap window")
        return None

    t_lap = ts[mask] - lap_start_abs
    x_lap = _smooth1d(xs[mask], window=3)
    y_lap = _smooth1d(ys[mask], window=3)

    x0 = float(np.interp(0.0, t_lap, x_lap))
    y0 = float(np.interp(0.0, t_lap, y_lap))
    if t_lap[0] > 0:
        t_lap = np.concatenate([[0.0], t_lap])
        x_lap = np.concatenate([[x0], x_lap])
        y_lap = np.concatenate([[y0], y_lap])

    grid = np.linspace(0, lt_s, N_POINTS)
    return (np.interp(grid, t_lap, x_lap), np.interp(grid, t_lap, y_lap), grid)

# ── Build all sources ────────────────────────────────────────────────────────
print("Building source A (FastF1 get_telemetry)...")
tracks_a = [build_a(d["lap"]) for d in drivers]

print("Building source B (FastF1 session.pos_data)...")
tracks_b = [build_b(d["lap"], session) for d in drivers]

print("Fetching OpenF1 session info...")
try:
    sk, of1_sess = fetch_openf1(session)
    print("Building source C (OpenF1)...")
    tracks_c = [build_c(d["lap"], sk, session) for d in drivers]
except Exception as e:
    print(f"  OpenF1 failed: {e}")
    tracks_c = [None, None]

# ── Plot ─────────────────────────────────────────────────────────────────────
SOURCES = [
    ("A  FastF1 get_telemetry()", tracks_a),
    ("B  FastF1 session.pos_data", tracks_b),
    ("C  OpenF1 API", tracks_c),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 8), facecolor="#0d0d0d")
fig.suptitle(f"{ROUND} {YEAR} Q — GPS start alignment comparison", color="white", fontsize=13)

for ax, (title, tracks) in zip(axes, SOURCES):
    ax.set_facecolor("#0d0d0d")
    ax.set_title(title, color="white", fontsize=10, pad=8)
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    if all(t is None for t in tracks):
        ax.text(0.5, 0.5, "No data", color="#888888", ha="center", va="center",
                transform=ax.transAxes)
        continue

    x0s, y0s = [], []
    for d, track in zip(drivers, tracks):
        if track is None:
            continue
        x, y, _ = track
        ax.plot(x, y, color=d["color"], linewidth=1.2, alpha=0.8)
        ax.scatter([x[0]], [y[0]], color=d["color"], s=80, zorder=5)
        x0s.append(x[0]); y0s.append(y[0])
        ax.annotate(f"{d['abbr']} ({x[0]:.0f}, {y[0]:.0f})",
                    (x[0], y[0]), textcoords="offset points", xytext=(6, 4),
                    color=d["color"], fontsize=7)

    if len(x0s) == 2:
        gap = np.sqrt((x0s[0]-x0s[1])**2 + (y0s[0]-y0s[1])**2)
        ax.set_xlabel(f"start gap = {gap:.1f} m", color="#aaaaaa", fontsize=9)

    # Zoom into start area
    if x0s:
        mx, my = np.mean(x0s), np.mean(y0s)
        ax.set_xlim(mx - 400, mx + 400)
        ax.set_ylim(my - 400, my + 400)

    ax.set_ylabel("Y (m)", color="#888888")

import pathlib
pathlib.Path("output").mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(OUT, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {OUT}")
