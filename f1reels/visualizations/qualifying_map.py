import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from f1reels.colors import driver_color
from f1reels.data.telemetry import build_telemetry, get_pole_laps
from f1reels.visualizations.base import Visualization

_BG = "#0d0d0d"
_TRACK_EDGE = "#ffffff"
_TRACK_ROAD = "#1c1c1c"
_TEXT_DIM = "#555555"
_TEXT_MID = "#999999"
_TEXT_BRIGHT = "#ffffff"

_TILT = 0.42
_VIEWPORT_FRAC = 0.22
_MAP_ASPECT = 1.20
_CAM_SMOOTH_WINDOW = 40
_SMOOTH_W = 9       # GPS smoothing window for raw position series
_N_CAM = 500        # pre-computed camera path resolution


def _fmt_laptime(td) -> str:
    total = td.total_seconds()
    m = int(total // 60)
    s = total % 60
    return f"{m}:{s:06.3f}"


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    padded = np.pad(arr, half, mode="reflect")
    return np.convolve(padded, np.ones(window) / window, mode="valid")[: len(arr)]


def _smooth(arr: np.ndarray, w: int = _SMOOTH_W) -> np.ndarray:
    return _rolling_mean(arr, w)


def _pca_rotation_angle(x: np.ndarray, y: np.ndarray) -> float:
    coords = np.stack([x - x.mean(), y - y.mean()], axis=1)
    _, _, vt = np.linalg.svd(coords, full_matrices=False)
    principal = vt[0]
    angle = float(np.degrees(np.arctan2(principal[0], principal[1])))
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    return angle


def _perspective(x: np.ndarray, y: np.ndarray, rotate_deg: float, tilt: float):
    theta = np.radians(rotate_deg)
    xr = x * np.cos(theta) - y * np.sin(theta)
    yr = x * np.sin(theta) + y * np.cos(theta)
    return xr, yr * tilt


def _raw_time_pos(lap):
    """
    Return (t, x, y) from raw telemetry — lap-relative time, smoothed GPS.
    Uses np.interp-friendly monotonic time array.
    """
    tel = lap.get_telemetry().dropna(subset=["X", "Y"]).reset_index(drop=True)
    t = tel["Time"].dt.total_seconds().values
    t = t - t[0]                          # lap-relative: starts at 0
    t = np.maximum.accumulate(t)          # guarantee monotonic
    x = _smooth(tel["X"].values)
    y = _smooth(tel["Y"].values)
    return t, x, y


class QualifyingMap(Visualization):
    name = "qualifying-map"

    def __init__(self, session):
        self.session = session
        self._prepare()

    def title(self) -> str:
        event = self.session.event
        return f"{event['EventName'].upper()} · {event['EventDate'].year}"

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
            "laptime": r1["Q3"] if str(r1.get("Q3")) != "NaT" else lap1["LapTime"],
            "color": driver_color(r1["Abbreviation"], r1.get("TeamName", "")),
        }
        self.d2 = {
            "abbr": r2["Abbreviation"],
            "laptime": r2["Q3"] if str(r2.get("Q3")) != "NaT" else lap2["LapTime"],
            "color": driver_color(r2["Abbreviation"], r2.get("TeamName", "")),
        }

        # arc-length telemetry — used for track outline, speed, delta
        self.tel1 = build_telemetry(lap1)
        self.tel2 = build_telemetry(lap2)

        # Raw time series for smooth position interpolation.
        # np.interp(T, t_raw, pos_raw) gives each car's exact position at
        # any elapsed lap time T without any index mapping or searchsorted.
        t1, x1, y1 = _raw_time_pos(lap1)
        t2, x2, y2 = _raw_time_pos(lap2)

        # Compute perspective rotation from d1's track shape (pre-transform)
        rotate_deg = _pca_rotation_angle(x1, y1)

        # Project raw positions
        x1p, y1p = _perspective(x1, y1, rotate_deg, _TILT)
        x2p, y2p = _perspective(x2, y2, rotate_deg, _TILT)

        self._t1, self._x1p, self._y1p = t1, x1p, y1p   # projected, for main map
        self._t2, self._x2p, self._y2p = t2, x2p, y2p
        self._x1r, self._y1r = x1, y1                    # raw, for mini-map
        self._x2r, self._y2r = x2, y2

        self._t_max = float(t1[-1])   # animation runs for d1's lap duration

        # Also project the arc-length telemetry (used for track drawing)
        for attr in ("tel1", "tel2"):
            tel = getattr(self, attr).copy()
            px, py = _perspective(tel["X"].values, tel["Y"].values, rotate_deg, _TILT)
            tel["X"], tel["Y"] = px, py
            setattr(self, attr, tel)

        # Pre-compute smoothed camera path at _N_CAM time steps
        T_grid = np.linspace(0, self._t_max, _N_CAM)
        cam_x = (np.interp(T_grid, t1, x1p) + np.interp(T_grid, t2, x2p)) / 2
        cam_y = (np.interp(T_grid, t1, y1p) + np.interp(T_grid, t2, y2p)) / 2
        self._cam_x = _rolling_mean(cam_x, _CAM_SMOOTH_WINDOW)
        self._cam_y = _rolling_mean(cam_y, _CAM_SMOOTH_WINDOW)

        # Delta trace: at each arc-length step, how many seconds is d2 behind?
        self._delta = self.tel2["TimeS"].values - self.tel1["TimeS"].values

        x = self.tel1["X"].values
        y = self.tel1["Y"].values
        track_scale = max(x.max() - x.min(), y.max() - y.min())
        self._viewport_r = track_scale * _VIEWPORT_FRAC
        self._vmax = max(self.tel1["Speed"].max(), self.tel2["Speed"].max())

    # ------------------------------------------------------------------
    # Figure setup — static elements drawn once
    # ------------------------------------------------------------------

    def setup_figure(self, fig: plt.Figure) -> None:
        gs = gridspec.GridSpec(3, 1, figure=fig,
                               height_ratios=[1.1, 7.2, 1.7], hspace=0)
        self._ax_top = fig.add_subplot(gs[0])
        self._ax_map = fig.add_subplot(gs[1])
        self._ax_bot = fig.add_subplot(gs[2])

        for ax in (self._ax_top, self._ax_map, self._ax_bot):
            ax.set_facecolor(_BG)
            ax.axis("off")

        tx, ty = self.tel1["X"], self.tel1["Y"]
        self._ax_map.plot(tx, ty, color=_TRACK_EDGE, linewidth=22,
                          solid_capstyle="round", zorder=0)
        self._ax_map.plot(tx, ty, color=_TRACK_ROAD, linewidth=16,
                          solid_capstyle="round", zorder=1)

        _lbl_kw = dict(fontsize=9, fontweight="bold", ha="center",
                       va="bottom", fontfamily="monospace")

        (self._halo2,) = self._ax_map.plot([], [], "o", color=self.d2["color"],
                                            markersize=19, zorder=4)
        (self._dot2,)  = self._ax_map.plot([], [], "o", color=self.d2["color"],
                                            markersize=13, zorder=5)
        self._lbl2 = self._ax_map.text(0, 0, self.d2["abbr"],
                                        color=self.d2["color"], zorder=6, **_lbl_kw)

        (self._halo1,) = self._ax_map.plot([], [], "o", color=self.d1["color"],
                                            markersize=19, zorder=7)
        (self._dot1,)  = self._ax_map.plot([], [], "o", color=self.d1["color"],
                                            markersize=13, zorder=8)
        self._lbl1 = self._ax_map.text(0, 0, self.d1["abbr"],
                                        color=self.d1["color"], zorder=9, **_lbl_kw)

        # Mini-map inset
        self._ax_mini = self._ax_bot.inset_axes([0.67, 0.04, 0.31, 0.92])
        self._ax_mini.set_facecolor(_BG)
        self._ax_mini.axis("off")
        mx, my = self._x1r, self._y1r
        self._ax_mini.plot(mx, my, color="#2a2a2a", linewidth=4, solid_capstyle="round")
        self._ax_mini.plot(mx, my, color="#555555", linewidth=2, solid_capstyle="round")
        self._ax_mini.set_aspect("equal")
        pad = (mx.max() - mx.min()) * 0.1
        self._ax_mini.set_xlim(mx.min() - pad, mx.max() + pad)
        self._ax_mini.set_ylim(my.min() - pad, my.max() + pad)
        self._ax_mini.set_autoscale_on(False)

        (self._mini_dot1,) = self._ax_mini.plot([], [], "o",
                                                  color=self.d1["color"], markersize=5, zorder=5)
        (self._mini_dot2,) = self._ax_mini.plot([], [], "o",
                                                  color=self.d2["color"], markersize=4, zorder=4)

        self._ax_map.set_autoscale_on(False)
        self._pan_to(self._cam_x[0], self._cam_y[0])

    def _pan_to(self, cx: float, cy: float) -> None:
        r = self._viewport_r
        self._ax_map.set_xlim(cx - r, cx + r)
        self._ax_map.set_ylim(cy - r * _MAP_ASPECT, cy + r * _MAP_ASPECT)

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def draw_frame(self, fig: plt.Figure, frame: int, total_frames: int) -> None:
        progress = min(frame / max(total_frames - 1, 1), 1.0)
        T = progress * self._t_max

        # Direct interpolation — no index mapping, no searchsorted
        x1 = float(np.interp(T, self._t1, self._x1p))
        y1 = float(np.interp(T, self._t1, self._y1p))
        x2 = float(np.interp(T, self._t2, self._x2p))
        y2 = float(np.interp(T, self._t2, self._y2p))

        cam_idx = int(progress * (_N_CAM - 1))
        self._pan_to(self._cam_x[cam_idx], self._cam_y[cam_idx])

        label_dy = self._viewport_r * 0.10

        for halo, dot, lbl, x, y in (
            (self._halo1, self._dot1, self._lbl1, x1, y1),
            (self._halo2, self._dot2, self._lbl2, x2, y2),
        ):
            halo.set_data([x], [y])
            dot.set_data([x], [y])
            lbl.set_position((x, y + label_dy))

        # Mini-map in original (pre-perspective) coordinates
        self._mini_dot1.set_data(
            [float(np.interp(T, self._t1, self._x1r))],
            [float(np.interp(T, self._t1, self._y1r))],
        )
        self._mini_dot2.set_data(
            [float(np.interp(T, self._t2, self._x2r))],
            [float(np.interp(T, self._t2, self._y2r))],
        )

        # Delta at equivalent arc-length position
        idx = int(progress * (len(self._delta) - 1))
        delta = self._delta[idx]

        self._draw_top()
        self._draw_bottom(progress, delta, T)

    def _draw_top(self) -> None:
        ax = self._ax_top
        ax.cla()
        ax.set_facecolor(_BG)
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(0.5, 0.78, self.title(), color=_TEXT_BRIGHT, fontsize=13,
                ha="center", va="center", fontfamily="monospace", fontweight="bold")
        ax.text(0.5, 0.28, "TOP 2  ·  Q LAP COMPARISON", color=_TEXT_DIM,
                fontsize=9, ha="center", va="center", fontfamily="monospace")

    def _draw_bottom(self, progress: float, delta: float, T: float) -> None:
        ax = self._ax_bot
        ax.cla()
        ax.set_facecolor(_BG)
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        secs = abs(delta)
        speed_ms = self._vmax / 3.6 * 0.7   # rough midfield speed for metre estimate
        metres = secs * speed_ms

        leader, trailer = (self.d1, self.d2) if delta >= 0 else (self.d2, self.d1)

        self._gap_row(ax, y=0.73, driver=leader, label="LEADER")
        self._gap_row(ax, y=0.38, driver=trailer,
                      label=f"+{metres:.0f}m  (+{secs:.3f}s)")

        ax.barh(0.07, 0.64, height=0.09, color="#1e1e1e", left=0)
        ax.barh(0.07, 0.64 * progress, height=0.09, color="#3a3a3a", left=0)

    def _gap_row(self, ax, y: float, driver: dict, label: str) -> None:
        ax.plot([0.035], [y], "o", color=driver["color"], markersize=13, zorder=5)
        ax.text(0.09, y + 0.11, driver["abbr"], color=driver["color"],
                fontsize=13, fontweight="bold", va="center", fontfamily="monospace")
        ax.text(0.09, y - 0.15, label, color=_TEXT_MID,
                fontsize=9, va="center", fontfamily="monospace")
