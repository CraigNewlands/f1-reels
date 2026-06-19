import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from f1reels.colors import driver_color
from f1reels.data.telemetry import build_telemetry, get_pole_laps
from f1reels.visualizations.base import Visualization

_BG = "#0d0d0d"
_TRACK_BASE = "#2a2a2a"
_TEXT_DIM = "#888888"
_TEXT_BRIGHT = "#ffffff"


def _fmt_laptime(td) -> str:
    total = td.total_seconds()
    m = int(total // 60)
    s = total % 60
    return f"{m}:{s:06.3f}"


class QualifyingMap(Visualization):
    name = "qualifying-map"

    def __init__(self, session):
        self.session = session
        self._prepare()

    def title(self) -> str:
        event = self.session.event
        return f"{event['EventName']} {event['EventDate'].year}"

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare(self):
        pairs = get_pole_laps(self.session, n=2)
        if len(pairs) < 2:
            raise ValueError("Could not find two qualifying laps in this session.")

        (r1, lap1), (r2, lap2) = pairs[0], pairs[1]

        self.d1 = {
            "abbr": r1["Abbreviation"],
            "laptime": r1["Q3"] if not str(r1.get("Q3")) == "NaT" else lap1["LapTime"],
            "color": driver_color(r1["Abbreviation"], r1.get("TeamName", "")),
        }
        self.d2 = {
            "abbr": r2["Abbreviation"],
            "laptime": r2["Q3"] if not str(r2.get("Q3")) == "NaT" else lap2["LapTime"],
            "color": driver_color(r2["Abbreviation"], r2.get("TeamName", "")),
        }

        self.tel1 = build_telemetry(lap1)
        self.tel2 = build_telemetry(lap2)

        # Speed heatmap segments along the track (based on pole lap)
        x, y = self.tel1["X"].values, self.tel1["Y"].values
        speed = self.tel1["Speed"].values
        pts = np.array([x, y]).T.reshape(-1, 1, 2)
        self._track_segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        self._track_speed = speed[:-1]
        self._speed_norm = Normalize(vmin=speed.min(), vmax=speed.max())

        # Axis bounds with padding
        pad = (x.max() - x.min()) * 0.08
        self._xlim = (x.min() - pad, x.max() + pad)
        self._ylim = (y.min() - pad, y.max() + pad)

        # Label offset: 2% of the Y range so names sit above the dot
        self._label_dy = (self._ylim[1] - self._ylim[0]) * 0.025

        # Delta at each telemetry point (positive → d1 is faster, d2 is behind)
        self._delta = self.tel2["TimeS"].values - self.tel1["TimeS"].values

        # Max speed across both laps (for normalising the speed bars)
        self._vmax = max(self.tel1["Speed"].max(), self.tel2["Speed"].max())

    # ------------------------------------------------------------------
    # Figure setup (static elements drawn once)
    # ------------------------------------------------------------------

    def setup_figure(self, fig: plt.Figure) -> None:
        gs = gridspec.GridSpec(
            3, 1, figure=fig, height_ratios=[1.4, 6.5, 1.6], hspace=0
        )
        self._ax_top = fig.add_subplot(gs[0])
        self._ax_map = fig.add_subplot(gs[1])
        self._ax_bot = fig.add_subplot(gs[2])

        for ax in (self._ax_top, self._ax_map, self._ax_bot):
            ax.set_facecolor(_BG)
            ax.axis("off")

        # Map axis: equal aspect, fixed bounds
        self._ax_map.set_aspect("equal")
        self._ax_map.set_xlim(*self._xlim)
        self._ax_map.set_ylim(*self._ylim)

        # Static: track base + speed heatmap
        self._ax_map.plot(
            self.tel1["X"], self.tel1["Y"],
            color=_TRACK_BASE, linewidth=14, solid_capstyle="round", zorder=0,
        )
        lc = LineCollection(
            self._track_segs, cmap="RdYlGn", norm=self._speed_norm,
            linewidth=10, alpha=0.55, zorder=1,
        )
        lc.set_array(self._track_speed)
        self._ax_map.add_collection(lc)

        # Car dots: glow + solid (stored as handles for per-frame updates)
        _lbl_kw = dict(fontsize=11, fontweight="bold", ha="center", va="bottom",
                       zorder=6, fontfamily="monospace")
        _dot_kw = dict(markersize=14, zorder=5)
        _glow_kw = dict(markersize=28, alpha=0.25, zorder=4)

        (self._glow1,) = self._ax_map.plot([], [], "o", color=self.d1["color"], **_glow_kw)
        (self._dot1,) = self._ax_map.plot([], [], "o", color=self.d1["color"], **_dot_kw)
        self._lbl1 = self._ax_map.text(0, 0, self.d1["abbr"], color=self.d1["color"], **_lbl_kw)

        (self._glow2,) = self._ax_map.plot([], [], "o", color=self.d2["color"], **_glow_kw)
        (self._dot2,) = self._ax_map.plot([], [], "o", color=self.d2["color"], **_dot_kw)
        self._lbl2 = self._ax_map.text(0, 0, self.d2["abbr"], color=self.d2["color"], **_lbl_kw)

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def draw_frame(self, fig: plt.Figure, frame: int, total_frames: int) -> None:
        progress = min(frame / max(total_frames - 1, 1), 1.0)
        idx = int(progress * (len(self.tel1) - 1))

        # Move car dots
        x1, y1 = self.tel1["X"].iloc[idx], self.tel1["Y"].iloc[idx]
        x2, y2 = self.tel2["X"].iloc[idx], self.tel2["Y"].iloc[idx]

        for dot, glow, lbl, x, y in (
            (self._dot1, self._glow1, self._lbl1, x1, y1),
            (self._dot2, self._glow2, self._lbl2, x2, y2),
        ):
            dot.set_data([x], [y])
            glow.set_data([x], [y])
            lbl.set_position((x, y + self._label_dy))

        self._draw_top(idx)
        self._draw_bottom(idx, progress)

    def _draw_top(self, idx: int) -> None:
        ax = self._ax_top
        ax.cla()
        ax.set_facecolor(_BG)
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Round + session label centred
        ax.text(0.5, 0.88, self.title(), color=_TEXT_BRIGHT, fontsize=13,
                ha="center", va="center", fontfamily="monospace", fontweight="bold")
        ax.text(0.5, 0.55, "QUALIFYING", color=_TEXT_DIM, fontsize=10,
                ha="center", va="center", fontfamily="monospace", letter_spacing=2)

        # D1 — left
        ax.text(0.02, 0.75, self.d1["abbr"], color=self.d1["color"],
                fontsize=22, fontweight="bold", va="center", fontfamily="monospace")
        ax.text(0.02, 0.28, _fmt_laptime(self.d1["laptime"]), color="#cccccc",
                fontsize=13, va="center", fontfamily="monospace")

        # D2 — right
        ax.text(0.98, 0.75, self.d2["abbr"], color=self.d2["color"],
                fontsize=22, fontweight="bold", va="center", ha="right", fontfamily="monospace")
        ax.text(0.98, 0.28, _fmt_laptime(self.d2["laptime"]), color="#cccccc",
                fontsize=13, va="center", ha="right", fontfamily="monospace")

    def _draw_bottom(self, idx: int, progress: float) -> None:
        ax = self._ax_bot
        ax.cla()
        ax.set_facecolor(_BG)
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        s1 = self.tel1["Speed"].iloc[idx]
        s2 = self.tel2["Speed"].iloc[idx]
        delta = self._delta[idx]

        # Speed bars
        self._speed_bar(ax, y=0.82, speed=s1, color=self.d1["color"], label=self.d1["abbr"])
        self._speed_bar(ax, y=0.58, speed=s2, color=self.d2["color"], label=self.d2["abbr"])

        # Delta
        if delta >= 0:
            delta_txt = f"+{delta:.3f}s  {self.d1['abbr']}"
            delta_col = self.d1["color"]
        else:
            delta_txt = f"+{abs(delta):.3f}s  {self.d2['abbr']}"
            delta_col = self.d2["color"]

        ax.text(0.5, 0.35, delta_txt, color=delta_col, fontsize=14, fontweight="bold",
                ha="center", va="center", fontfamily="monospace")

        # Lap progress bar
        ax.barh(0.12, 1.0, height=0.1, color="#1e1e1e", left=0)
        ax.barh(0.12, progress, height=0.1, color="#555555", left=0)
        ax.text(0.5, 0.12, f"{progress * 100:.0f}%", color="#aaaaaa", fontsize=9,
                ha="center", va="center", fontfamily="monospace")

    def _speed_bar(self, ax, y: float, speed: float, color: str, label: str) -> None:
        fill = speed / self._vmax
        ax.barh(y, 1.0, height=0.13, color="#1e1e1e", left=0)
        ax.barh(y, fill, height=0.13, color=color, left=0, alpha=0.85)
        ax.text(0.015, y, f"{label}  {speed:.0f} km/h", color=_TEXT_BRIGHT,
                fontsize=10, va="center", fontfamily="monospace")
