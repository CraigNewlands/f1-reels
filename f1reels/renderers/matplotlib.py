"""Matplotlib + FFmpeg renderer — produces a portrait 9:16 MP4."""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from f1reels.pipeline.models import DriverFrames, TrackShape

matplotlib.use("Agg")

_W, _H = 9, 16          # inches  →  1080 × 1920 px at 120 dpi
_DPI   = 120
_BG    = "#0d0d0d"
_WHITE = "#ffffff"
_DIM   = "#444444"
_MID   = "#888888"

# Layout: top label band / track / leaderboard
_RATIOS = [0.08, 0.72, 0.20]


def _hex_to_rgba(h: str, alpha: float = 1.0) -> tuple:
    h = h.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    return r, g, b, alpha


class MatplotlibRenderer:
    name = "matplotlib"

    def render(
        self,
        track: TrackShape,
        drivers: list[DriverFrames],
        output_path: Path,
        fps: int = 30,
        duration_s: float = 45.0,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> Path:
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found in PATH — brew install ffmpeg")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        total_frames = fps * int(duration_s)
        max_lap_s    = max(d.lap_time_s for d in drivers)

        fig = plt.figure(figsize=(_W, _H), facecolor=_BG, dpi=_DPI)
        gs  = fig.add_gridspec(3, 1, height_ratios=_RATIOS, hspace=0)

        ax_top   = fig.add_subplot(gs[0])
        ax_track = fig.add_subplot(gs[1])
        ax_board = fig.add_subplot(gs[2])
        for ax in (ax_top, ax_track, ax_board):
            ax.set_facecolor(_BG)
            ax.axis("off")

        # ── Static track ──────────────────────────────────────────────────
        pts = track.all_points()
        ax_track.plot(pts[:, 0], pts[:, 1], color="#0a0a0a",  lw=22, solid_capstyle="round", zorder=0)
        ax_track.plot(pts[:, 0], pts[:, 1], color=_WHITE,     lw=12, solid_capstyle="round", alpha=0.15, zorder=1)
        ax_track.plot(pts[:, 0], pts[:, 1], color=_WHITE,     lw=3,  solid_capstyle="round", alpha=0.9,  zorder=2)
        ax_track.set_aspect("equal")
        ax_track.autoscale_view()
        ax_track.set_autoscale_on(False)

        # ── Driver dots (one per driver) ──────────────────────────────────
        dot_artists = {}
        for drv in drivers:
            rgba  = _hex_to_rgba(drv.color)
            halo, = ax_track.plot([], [], "o", color=drv.color, markersize=28, alpha=0.18, zorder=3)
            dot,  = ax_track.plot([], [], "o", color=drv.color, markersize=13,
                                  markeredgecolor=_WHITE, markeredgewidth=1.0, zorder=4)
            lbl   = ax_track.text(0, 0, drv.abbr, color=drv.color,
                                  fontsize=7, fontweight="bold", ha="center", va="bottom",
                                  fontfamily="monospace", zorder=5)
            dot_artists[drv.abbr] = (halo, dot, lbl)

        # y-offset for labels relative to track scale
        y_rng = pts[:, 1].max() - pts[:, 1].min()
        lbl_dy = y_rng * 0.015

        # ── Progress bar (drawn once, filled each frame) ──────────────────
        ax_board.set_xlim(0, 1)
        ax_board.set_ylim(0, 1)
        bar_bg = ax_board.barh(0.08, 0.90, height=0.10, left=0.05, color="#1e1e1e")[0]
        bar_fg = ax_board.barh(0.08, 0.00, height=0.10, left=0.05, color="#3a3a3a")[0]

        def animate(frame: int) -> None:
            t = (frame / max(total_frames - 1, 1)) * max_lap_s

            # Move each driver's dot
            ordered = sorted(drivers, key=lambda d: d.at(t)[2])  # norm_dist ascending
            for drv in drivers:
                x, y, nd = drv.at(t)
                halo, dot, lbl = dot_artists[drv.abbr]
                halo.set_data([x], [y])
                dot.set_data([x], [y])
                lbl.set_position((x, y + lbl_dy))

            # Leaderboard — redraw each frame
            ax_board.cla()
            ax_board.set_facecolor(_BG)
            ax_board.axis("off")
            ax_board.set_xlim(0, 1)
            ax_board.set_ylim(0, 1)

            # Time rescaling means nd-based order is correct throughout the lap.
            # When all finish (all nd=1.0), nd sort is unstable on equal values so
            # we fall back to official_laptime_s.  We always need all_finished for
            # the gap switch: nd-based gap collapses to 0 once everyone is at 1.0.
            all_finished = all(d.at(t)[2] >= 1.0 for d in drivers)
            if all_finished:
                ranked        = sorted(drivers, key=lambda d: d.official_laptime_s)
                leader_laptime = ranked[0].official_laptime_s
            else:
                ranked         = list(reversed(ordered))  # nd-ascending → leader first
                leader_laptime = None

            n = len(ranked)
            for rank, drv in enumerate(ranked):
                _, _, nd = drv.at(t)
                row_y = 0.88 - rank * (0.70 / max(n - 1, 1))
                ax_board.text(0.04, row_y, str(rank + 1),
                              color=_DIM, fontsize=9, fontweight="bold",
                              va="center", fontfamily="monospace")
                ax_board.plot([0.10], [row_y], "o", color=drv.color, markersize=10, zorder=5)
                ax_board.text(0.15, row_y, drv.abbr,
                              color=drv.color, fontsize=11, fontweight="bold",
                              va="center", fontfamily="monospace")
                if rank > 0:
                    if all_finished:
                        gap_s = drv.official_laptime_s - leader_laptime
                    else:
                        leader_nd = ranked[0].at(t)[2]
                        gap_s = (leader_nd - nd) * max_lap_s
                    ax_board.text(0.42, row_y, f"+{gap_s:.3f}s",
                                  color=_MID, fontsize=8, va="center", fontfamily="monospace")

            # Progress bar
            progress = frame / max(total_frames - 1, 1)
            ax_board.barh(0.08, 0.90, height=0.10, left=0.05, color="#1e1e1e")
            ax_board.barh(0.08, 0.90 * progress, height=0.10, left=0.05, color="#3a3a3a")

            if progress_cb is not None:
                progress_cb(frame + 1, total_frames)

        # ── Top label ─────────────────────────────────────────────────────
        ax_top.set_xlim(0, 1)
        ax_top.set_ylim(0, 1)
        ax_top.text(0.5, 0.65, "Q LAP COMPARISON",
                    color=_WHITE, fontsize=12, fontweight="bold",
                    ha="center", va="center", fontfamily="monospace")
        ax_top.text(0.5, 0.25, "TOP 10  ·  Q3",
                    color=_DIM, fontsize=8, ha="center", va="center", fontfamily="monospace")

        # ── Render ────────────────────────────────────────────────────────
        anim   = FuncAnimation(fig, animate, frames=total_frames, interval=1000 / fps)
        writer = FFMpegWriter(
            fps=fps,
            bitrate=8000,
            extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                        "-r", str(fps), "-level:v", "5.1"],
        )
        anim.save(str(output_path), writer=writer, dpi=_DPI)
        plt.close(fig)
        return output_path
