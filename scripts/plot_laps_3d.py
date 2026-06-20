"""Quick diagnostic: plot the two fastest qualifying laps in 3D GPS space."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from f1reels.data.session import load_session
from f1reels.data.telemetry import build_telemetry, get_pole_laps
from f1reels.colors import driver_color

YEAR  = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
ROUND = sys.argv[2]       if len(sys.argv) > 2 else "Bahrain"
OUT   = sys.argv[3]       if len(sys.argv) > 3 else "output/laps_3d.png"

print(f"Loading {YEAR} {ROUND} Q...")
session = load_session(YEAR, ROUND, "Q")
pairs   = get_pole_laps(session, n=2)

fig = plt.figure(figsize=(18, 8), facecolor="#0d0d0d")
ax  = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122)
ax.set_facecolor("#0d0d0d")
ax.tick_params(colors="white")
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor("#333333")

laps_data = []
# Build both tracks first so we can compute the centroid offset
raw_tracks = []
for row, lap in pairs:
    tel = build_telemetry(lap)
    raw_tracks.append((row, lap, tel))

# Centroid alignment: both drivers drove the same circuit so their GPS
# centroids should match. Any difference is a systematic offset between
# the two cars' GPS receivers.
# GPS drift correction — longitudinal component only (vector projection).
# Both cars crossed the same timing loop; any coordinate gap at Distance=0
# is receiver drift.  We project the drift onto the track direction vector
# and shift d2 only along-track, preserving the lateral racing-line offset.
t1_ref, t2_ref = raw_tracks[0][2], raw_tracks[1][2]
norm1, norm2 = t1_ref["NormDist"].values, t2_ref["NormDist"].values
d1_x0  = float(np.interp(0,     norm1, t1_ref["X"].values))
d1_y0  = float(np.interp(0,     norm1, t1_ref["Y"].values))
d2_x0  = float(np.interp(0,     norm2, t2_ref["X"].values))
d2_y0  = float(np.interp(0,     norm2, t2_ref["Y"].values))
d1_xah = float(np.interp(0.002, norm1, t1_ref["X"].values))
d1_yah = float(np.interp(0.002, norm1, t1_ref["Y"].values))
tvx, tvy = d1_xah - d1_x0, d1_yah - d1_y0
mag = np.hypot(tvx, tvy)
if mag > 0: tvx /= mag; tvy /= mag
diff_x, diff_y = d1_x0 - d2_x0, d1_y0 - d2_y0
lon = diff_x * tvx + diff_y * tvy
ox, oy = lon * tvx, lon * tvy
print(f"GPS drift correction (longitudinal only) → ({ox:.1f}, {oy:.1f}) m")
raw_tracks[1][2]["X"] += ox
raw_tracks[1][2]["Y"] += oy

for i, (row, lap, tel) in enumerate(raw_tracks):
    abbr  = row["Abbreviation"]
    color = driver_color(abbr, row.get("TeamName", ""))
    lt    = lap["LapTime"]
    lw    = 1.5 if i == 0 else 1.0
    label = f"{abbr}  {lt}"
    z     = [0] * len(tel)

    ax.plot(tel["X"].values, tel["Y"].values, z,
            color=color, linewidth=lw, alpha=0.9, label=label)
    ax.scatter([tel["X"].iloc[0]], [tel["Y"].iloc[0]], [0],
               color=color, s=80, zorder=5)

    ax2.plot(tel["X"].values, tel["Y"].values,
             color=color, linewidth=lw, alpha=0.9, label=label)
    # Start dot at Distance=0 via interpolation (not iloc[0] which may be past the beacon)
    norm = tel["NormDist"].values
    sx = float(np.interp(0, norm, tel["X"].values))
    sy = float(np.interp(0, norm, tel["Y"].values))
    ax2.scatter([sx], [sy], color=color, s=80, zorder=5,
                label=f"{abbr} start ({sx:.0f}, {sy:.0f})")
    laps_data.append((abbr, color, tel))

ax.set_title(f"{ROUND} {YEAR} Q — 3D GPS", color="white", pad=12)
ax.legend(facecolor="#1a1a1a", labelcolor="white", framealpha=0.8, fontsize=8)
ax.set_xlabel("X (m)", color="#888888", labelpad=6)
ax.set_ylabel("Y (m)", color="#888888", labelpad=6)
ax.set_zlabel("Z (m)", color="#888888", labelpad=6)

ax2.set_facecolor("#0d0d0d")
ax2.set_title(f"{ROUND} {YEAR} Q — top-down (X/Y)", color="white", pad=12)
ax2.tick_params(colors="#888888")
ax2.set_xlabel("X (m)", color="#888888")
ax2.set_ylabel("Y (m)", color="#888888")
for spine in ax2.spines.values():
    spine.set_edgecolor("#333333")
ax2.legend(facecolor="#1a1a1a", labelcolor="white", framealpha=0.8, fontsize=7)

# Zoom inset showing start-line area
if laps_data:
    x0s = [t["X"].iloc[0] for _, _, t in laps_data]
    y0s = [t["Y"].iloc[0] for _, _, t in laps_data]
    mx, my = sum(x0s)/len(x0s), sum(y0s)/len(y0s)
    r = 300
    axins = ax2.inset_axes([0.6, 0.0, 0.4, 0.35])
    axins.set_facecolor("#111111")
    axins.set_title("start area", color="#888888", fontsize=7, pad=3)
    for abbr, color, tel in laps_data:
        axins.plot(tel["X"].values, tel["Y"].values, color=color, linewidth=1.5)
        norm = tel["NormDist"].values
        sx = float(np.interp(0, norm, tel["X"].values))
        sy = float(np.interp(0, norm, tel["Y"].values))
        axins.scatter([sx], [sy], color=color, s=50, zorder=5)
    axins.set_xlim(mx - r, mx + r)
    axins.set_ylim(my - r, my + r)
    axins.tick_params(colors="#666666", labelsize=6)
    for spine in axins.spines.values():
        spine.set_edgecolor("#333333")

import pathlib
pathlib.Path("output").mkdir(exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {OUT}")
