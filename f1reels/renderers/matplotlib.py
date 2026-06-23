"""Matplotlib + FFmpeg renderer — portrait 9:16 MP4 with circuit theming."""

from __future__ import annotations

import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import LineCollection

from f1reels.pipeline.models import DriverFrames, TrackShape
from f1reels.themes import get_theme

matplotlib.use("Agg")

_W, _H  = 9, 16
_DPI    = 120
_WHITE  = "#ffffff"
_DIM    = "#555555"
_MID    = "#999999"

# Timing constants
_ZOOM_ANIM_S = 3.0
_HOLD_S      = 2.0

# Colour for unvisited track segments
_GREY_SEG = (0.20, 0.20, 0.20, 0.5)


# ── Font ──────────────────────────────────────────────────────────────────────

def _ensure_font() -> str:
    """Download Titillium Web Bold if not cached; return the family name."""
    from matplotlib import font_manager
    font_dir  = Path.home() / ".cache" / "f1reels" / "fonts"
    font_dir.mkdir(parents=True, exist_ok=True)
    font_path = font_dir / "TitilliumWeb-Bold.ttf"
    if not font_path.exists():
        url = ("https://github.com/google/fonts/raw/main/ofl/titilliumweb/"
               "TitilliumWeb-Bold.ttf")
        try:
            urllib.request.urlretrieve(url, font_path)
        except Exception:
            return "sans-serif"
    font_manager.fontManager.addfont(str(font_path))
    return "Titillium Web"


def _hex_to_rgba(h: str, alpha: float = 1.0) -> tuple:
    h = h.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    return r, g, b, alpha


def _fmt_laptime(s: float) -> str:
    return f"{int(s // 60)}:{s % 60:06.3f}"


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


# ── Renderer ──────────────────────────────────────────────────────────────────

class MatplotlibRenderer:
    name = "matplotlib"

    def render(
        self,
        track: TrackShape,
        drivers: list[DriverFrames],
        output_path: Path,
        fps: int = 30,
        duration_s: float = 30.0,
        progress_cb: Callable[[int, int], None] | None = None,
        event_name: str = "",
    ) -> Path:
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found — brew install ffmpeg")

        font_family = _ensure_font()
        theme       = get_theme(event_name)
        bg          = theme["bg"]
        accent      = theme["accent"]
        accent_rgba = _hex_to_rgba(accent)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        total_frames     = fps * int(duration_s)
        zoom_anim_frames = int(_ZOOM_ANIM_S * fps)
        hold_frames      = int(_HOLD_S * fps)
        race_frames      = total_frames - zoom_anim_frames - hold_frames
        max_lap_s        = max(d.lap_time_s for d in drivers)
        leader           = min(drivers, key=lambda d: d.official_laptime_s)

        # ── Track smooth points & colour segments ──────────────────────────
        pts    = track.all_points()
        sx, sy = pts[:, 0], pts[:, 1]
        n_segs = len(sx) - 1
        nd_mids = (np.arange(n_segs) + 0.5) / n_segs
        final_rgba = np.array([
            _hex_to_rgba(min(drivers, key=lambda d: d.time_at_norm_dist(float(nd))).color)
            for nd in nd_mids
        ])

        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
        x_rng, y_rng = x_max - x_min, y_max - y_min
        pad    = 0.06
        full_cx = (x_min + x_max) / 2
        full_cy = (y_min + y_max) / 2
        full_r  = max(x_rng, y_rng) / 2 * (1 + pad)
        zoom_r  = max(x_rng, y_rng) * 0.08

        # ── Layout: header (title + leaderboard) / track ──────────────────
        fig = plt.figure(figsize=(_W, _H), facecolor=bg, dpi=_DPI)
        gs  = fig.add_gridspec(2, 1, height_ratios=[0.22, 0.78], hspace=0)
        ax_head  = fig.add_subplot(gs[0])
        ax_track = fig.add_subplot(gs[1])
        for ax in (ax_head, ax_track):
            ax.set_facecolor(bg)
            ax.axis("off")

        # ── Static track — asphalt band style ─────────────────────────────
        segs = [[[sx[i], sy[i]], [sx[i+1], sy[i+1]]] for i in range(n_segs)]

        # Layer 1: outer kerb hint (wider, muted grey)
        ax_track.add_collection(LineCollection(
            segs, colors=[(0.35, 0.35, 0.35, 0.35)] * n_segs,
            linewidths=28, capstyle="round", zorder=0))
        # Layer 2: asphalt surface (grey, clearly lighter than bg)
        ax_track.add_collection(LineCollection(
            segs, colors=[(0.13, 0.13, 0.13, 1.0)] * n_segs,
            linewidths=22, capstyle="round", zorder=1))
        # Layer 3: coloured racing-line segments (progressive reveal)
        colour_arr = np.tile(_GREY_SEG, (n_segs, 1)).astype(float)
        colour_lc  = LineCollection(segs, colors=colour_arr,
                                    linewidths=7, capstyle="round", zorder=2)
        ax_track.add_collection(colour_lc)
        # Layer 4: faint centre marking
        ax_track.plot(sx, sy, color=_WHITE, lw=0.6,
                      solid_capstyle="round", alpha=0.18, zorder=3)

        ax_track.set_aspect("equal")
        ax_track.set_xlim(full_cx - full_r, full_cx + full_r)
        ax_track.set_ylim(full_cy - full_r, full_cy + full_r)
        ax_track.set_autoscale_on(False)

        # ── Start/finish line ──────────────────────────────────────────────
        sf_x, sf_y = track.x[0], track.y[0]
        tdx = float(track.x[1] - track.x[-2])
        tdy = float(track.y[1] - track.y[-2])
        tn  = np.sqrt(tdx**2 + tdy**2)
        perp_x, perp_y = -tdy / tn, tdx / tn
        sf_frac = 0.04
        sf_line, = ax_track.plot([], [], color=_WHITE, lw=2.5, zorder=4)

        # ── Driver dots ───────────────────────────────────────────────────
        dot_artists = {}
        _lbl_dy = [zoom_r * 0.03]
        for drv in drivers:
            halo, = ax_track.plot([], [], "o", color=drv.color,
                                  markersize=22, alpha=0.20, zorder=5)
            dot,  = ax_track.plot([], [], "o", color=drv.color,
                                  markersize=16, markeredgecolor=_WHITE,
                                  markeredgewidth=1.5, zorder=6)
            lbl   = ax_track.text(0, 0, drv.abbr, color=drv.color,
                                  fontsize=11, fontweight="bold",
                                  ha="center", va="bottom",
                                  fontfamily=font_family, zorder=7)
            dot_artists[drv.abbr] = (halo, dot, lbl)

        # ── Static header: title ──────────────────────────────────────────
        ax_head.set_xlim(0, 1); ax_head.set_ylim(0, 1)
        title = event_name.upper() if event_name else "GRAND PRIX"
        ax_head.text(0.5, 0.88, title, color=_WHITE,
                     fontsize=24, fontweight="bold", ha="center", va="top",
                     fontfamily=font_family)
        ax_head.text(0.5, 0.70, f"QUALIFYING  ·  TOP {len(drivers)}",
                     color=_MID, fontsize=12, ha="center", va="top",
                     fontfamily=font_family)
        # Accent divider line under title
        ax_head.axhline(0.62, color=accent, lw=1.5, alpha=0.8)

        # ── animate ───────────────────────────────────────────────────────
        def animate(frame: int) -> None:
            if frame < race_frames:
                t             = (frame / max(race_frames - 1, 1)) * max_lap_s
                zoom_progress = 0.0
            elif frame < race_frames + zoom_anim_frames:
                t             = max_lap_s
                zoom_progress = (frame - race_frames) / max(zoom_anim_frames - 1, 1)
            else:
                t             = max_lap_s
                zoom_progress = 1.0

            z_s = _smoothstep(zoom_progress)
            r   = zoom_r + (full_r - zoom_r) * z_s

            # Reveal coloured segments
            nd_now     = leader.at(t)[2]
            reveal_idx = min(int(nd_now * n_segs), n_segs)
            colour_arr[:reveal_idx] = final_rgba[:reveal_idx]
            colour_lc.set_color(colour_arr)

            # Move dots (P1 on top)
            ordered = sorted(drivers,
                             key=lambda d: (d.at(t)[2], -d.official_laptime_s))
            leader_x = leader_y = 0.0
            for rank, drv in enumerate(ordered):
                x, y, _ = drv.at(t)
                halo, dot, lbl = dot_artists[drv.abbr]
                z = 10 + rank * 3
                halo.set_data([x], [y]); halo.set_zorder(z)
                dot.set_data([x], [y]);  dot.set_zorder(z + 1)
                lbl.set_position((x, y + _lbl_dy[0])); lbl.set_zorder(z + 2)
                if drv is leader:
                    leader_x, leader_y = x, y

            # Camera
            cx = leader_x + (full_cx - leader_x) * z_s
            cy = leader_y + (full_cy - leader_y) * z_s
            ax_track.set_xlim(cx - r, cx + r)
            ax_track.set_ylim(cy - r, cy + r)
            _lbl_dy[0] = r * 0.03

            sf_half = r * sf_frac
            sf_line.set_data(
                [sf_x - perp_x * sf_half, sf_x + perp_x * sf_half],
                [sf_y - perp_y * sf_half, sf_y + perp_y * sf_half],
            )

            # ── Dynamic header: leaderboard (3 columns) ───────────────────
            ax_head.patches and [p.remove() for p in ax_head.patches[:]]
            # Clear dynamic text (keep static title/subtitle/divider)
            for txt in ax_head.texts[3:]:
                txt.remove()

            finished = sorted([d for d in drivers if d.at(t)[2] >= 1.0],
                              key=lambda d: d.official_laptime_s)
            racing   = sorted([d for d in drivers if d.at(t)[2] < 1.0],
                              key=lambda d: d.at(t)[2], reverse=True)
            ranked   = finished + racing
            all_fin  = len(racing) == 0
            leader_lt = finished[0].official_laptime_s if finished else None

            n = len(ranked)
            col_w = 1.0 / n
            for rank, drv in enumerate(ranked):
                cx_col = col_w * rank + col_w / 2

                # Accent divider between columns
                if rank > 0:
                    ax_head.axvline(col_w * rank, ymin=0, ymax=0.55,
                                    color=accent, lw=0.8, alpha=0.4)

                # Position badge
                ax_head.text(cx_col - col_w * 0.35, 0.50,
                             f"P{rank + 1}", color=accent,
                             fontsize=11, fontweight="bold", va="center",
                             fontfamily=font_family)
                # Coloured dot
                ax_head.plot([cx_col - col_w * 0.08], [0.50], "o",
                             color=drv.color, markersize=10, zorder=5)
                # Abbreviation
                ax_head.text(cx_col + col_w * 0.00, 0.50, drv.abbr,
                             color=drv.color, fontsize=16, fontweight="bold",
                             va="center", fontfamily=font_family)
                # Gap / laptime
                if rank == 0 and all_fin:
                    label = _fmt_laptime(leader_lt)
                    col   = _WHITE
                elif rank == 0:
                    label = "LEADER"
                    col   = accent
                else:
                    nd_ldr = ranked[0].at(t)[2]
                    gap    = max(drv.time_at_norm_dist(nd_ldr)
                                 - ranked[0].time_at_norm_dist(nd_ldr), 0.0)
                    label  = f"+{gap:.3f}s"
                    col    = _MID
                ax_head.text(cx_col, 0.24, label, color=col,
                             fontsize=11, ha="center", va="center",
                             fontfamily=font_family)

            if progress_cb is not None:
                progress_cb(frame + 1, total_frames)

        # ── Render ────────────────────────────────────────────────────────
        anim   = FuncAnimation(fig, animate, frames=total_frames,
                               interval=1000 / fps)
        writer = FFMpegWriter(
            fps=fps, bitrate=8000,
            extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                        "-r", str(fps), "-level:v", "5.1"],
        )
        anim.save(str(output_path), writer=writer, dpi=_DPI)
        plt.close(fig)
        return output_path
