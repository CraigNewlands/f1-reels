"""Quick diagnostic: plot the two fastest qualifying laps in 3D GPS space."""
import sys
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
# Start-point alignment: shift d2 so it starts at d1's first GPS sample
t1, t2 = raw_tracks[0][2], raw_tracks[1][2]
dx = t1["X"].iloc[0] - t2["X"].iloc[0]
dy = t1["Y"].iloc[0] - t2["Y"].iloc[0]
print(f"Start-point alignment → d2 shifted by ({dx:.1f}, {dy:.1f}) m")
raw_tracks[1][2]["X"] += dx
raw_tracks[1][2]["Y"] += dy

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
    ax2.scatter([tel["X"].iloc[0]], [tel["Y"].iloc[0]],
                color=color, s=80, zorder=5,
                label=f"{abbr} start ({tel['X'].iloc[0]:.0f}, {tel['Y'].iloc[0]:.0f})")
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
        axins.scatter([tel["X"].iloc[0]], [tel["Y"].iloc[0]], color=color, s=50, zorder=5)
    axins.set_xlim(mx - r, mx + r)
    axins.set_ylim(my - r, my + r)
    axins.tick_params(colors="#666666", labelsize=6)
    for spine in axins.spines.values():
        spine.set_edgecolor("#333333")

import pathlib
pathlib.Path("output").mkdir(exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {OUT}")
