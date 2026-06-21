"""Matplotlib + FFmpeg renderer — produces a portrait 9:16 MP4."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import LineCollection

from f1reels.pipeline.models import DriverFrames, TrackShape

matplotlib.use("Agg")

_W, _H   = 9, 16
_DPI     = 120
_BG      = "#0d0d0d"
_WHITE   = "#ffffff"
_DIM     = "#444444"
_MID     = "#888888"
_GREY_SEG = (0.25, 0.25, 0.25, 0.6)   # unvisited track segment colour

# Panels: title / track / leaderboard
_RATIOS = [0.07, 0.73, 0.20]

# 55 % of frames for racing, 45 % for the zoom-out + hold
_RACE_FRAC = 0.55
_ZOOM_FRAC = 0.45

# Zoom animation completes in this fraction of the non-racing frames,
# then holds the full-track view for the rest
_ZOOM_ANIM_FRAC = 0.35


def _hex_to_rgba(h: str, alpha: float = 1.0) -> tuple:
    h = h.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    return r, g, b, alpha


def _fmt_laptime(s: float) -> str:
    return f"{int(s // 60)}:{s % 60:06.3f}"


def _smoothstep(t: float) -> float:
    """Smooth ease in/out: 0→0, 1→1, zero derivative at endpoints."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


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
        race_frames  = int(total_frames * _RACE_FRAC)
        zoom_frames  = total_frames - race_frames
        max_lap_s    = max(d.lap_time_s for d in drivers)
        leader       = min(drivers, key=lambda d: d.official_laptime_s)

        # ── Pre-compute per-segment "fastest driver" colours ──────────────
        n_segs     = len(track.x) - 1
        nd_mids    = (np.arange(n_segs) + 0.5) / n_segs
        final_rgba = np.array([
            _hex_to_rgba(min(drivers, key=lambda d: d.time_at_norm_dist(float(nd))).color)
            for nd in nd_mids
        ])  # shape (n_segs, 4)

        # ── Track geometry ─────────────────────────────────────────────────
        pts     = track.all_points()
        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
        x_rng, y_rng = x_max - x_min, y_max - y_min
        pad = 0.06

        # Full-track viewport (used for zoom-out)
        full_cx = (x_min + x_max) / 2
        full_cy = (y_min + y_max) / 2
        full_r  = max(x_rng, y_rng) / 2 * (1 + pad)

        # Zoomed viewport radius (≈ 18 % of max track dimension)
        zoom_r = max(x_rng, y_rng) * 0.18

        # ── Figure ─────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(_W, _H), facecolor=_BG, dpi=_DPI)
        gs  = fig.add_gridspec(3, 1, height_ratios=_RATIOS, hspace=0)

        ax_top   = fig.add_subplot(gs[0])
        ax_track = fig.add_subplot(gs[1])
        ax_board = fig.add_subplot(gs[2])
        for ax in (ax_top, ax_track, ax_board):
            ax.set_facecolor(_BG)
            ax.axis("off")

        # ── Static track base (dark road) ─────────────────────────────────
        segs = [[[track.x[i], track.y[i]], [track.x[i+1], track.y[i+1]]]
                for i in range(n_segs)]

        ax_track.add_collection(
            LineCollection(segs, colors=["#111111"] * n_segs, linewidths=22, zorder=0))

        # Progressive colour layer — starts all grey, fills in during the lap
        colour_arr = np.tile(_GREY_SEG, (n_segs, 1)).astype(float)
        colour_lc  = LineCollection(segs, colors=colour_arr, linewidths=8, zorder=1)
        ax_track.add_collection(colour_lc)

        # Thin white centre line on top
        ax_track.plot(pts[:, 0], pts[:, 1], color=_WHITE, lw=1.5,
                      solid_capstyle="round", alpha=0.5, zorder=2)

        # ── Start/finish line ──────────────────────────────────────────────
        sf_x, sf_y = track.x[0], track.y[0]
        tdx = float(track.x[1] - track.x[-2])
        tdy = float(track.y[1] - track.y[-2])
        tn  = np.sqrt(tdx**2 + tdy**2)
        perp_x, perp_y = -tdy / tn, tdx / tn
        sf_half = x_rng * 0.025
        ax_track.plot(
            [sf_x - perp_x * sf_half, sf_x + perp_x * sf_half],
            [sf_y - perp_y * sf_half, sf_y + perp_y * sf_half],
            color=_WHITE, lw=2.5, zorder=2,
        )

        ax_track.set_aspect("equal")
        ax_track.set_xlim(full_cx - full_r, full_cx + full_r)
        ax_track.set_ylim(full_cy - full_r, full_cy + full_r)
        ax_track.set_autoscale_on(False)

        # ── Driver dots ───────────────────────────────────────────────────
        dot_artists = {}
        for drv in drivers:
            halo, = ax_track.plot([], [], "o", color=drv.color, markersize=28, alpha=0.18, zorder=3)
            dot,  = ax_track.plot([], [], "o", color=drv.color, markersize=13,
                                  markeredgecolor=_WHITE, markeredgewidth=1.0, zorder=4)
            lbl   = ax_track.text(0, 0, drv.abbr, color=drv.color,
                                  fontsize=7, fontweight="bold",
                                  ha="center", va="bottom", fontfamily="monospace", zorder=5)
            dot_artists[drv.abbr] = (halo, dot, lbl)

        # label offset (updated each frame based on viewport)
        _lbl_dy = [y_rng * 0.015]  # mutable via list

        # Cache last leader position for smooth zoom-out start
        _last_leader_pos = [full_cx, full_cy]

        # ── animate ───────────────────────────────────────────────────────
        def animate(frame: int) -> None:
            # ── Timing ────────────────────────────────────────────────────
            if frame < race_frames:
                t             = (frame / max(race_frames - 1, 1)) * max_lap_s
                zoom_progress = 0.0
            else:
                t             = max_lap_s
                # Zoom completes in the first _ZOOM_ANIM_FRAC of the non-racing
                # frames, then holds the full-track view for the remainder
                anim_frames   = int(zoom_frames * _ZOOM_ANIM_FRAC)
                zoom_progress = min((frame - race_frames) / max(anim_frames - 1, 1), 1.0)

            # ── Reveal coloured segments up to leader's current position ──
            nd_now     = leader.at(t)[2]
            reveal_idx = min(int(nd_now * n_segs), n_segs)
            colour_arr[:reveal_idx] = final_rgba[:reveal_idx]
            colour_lc.set_color(colour_arr)

            # ── Move dots ─────────────────────────────────────────────────
            leader_x = leader_y = 0.0
            for drv in drivers:
                x, y, _ = drv.at(t)
                halo, dot, lbl = dot_artists[drv.abbr]
                halo.set_data([x], [y])
                dot.set_data([x], [y])
                lbl.set_position((x, y + _lbl_dy[0]))
                if drv is leader:
                    leader_x, leader_y = x, y

            _last_leader_pos[0] = leader_x
            _last_leader_pos[1] = leader_y

            # ── Camera ────────────────────────────────────────────────────
            z = _smoothstep(zoom_progress)
            cx = leader_x + (full_cx - leader_x) * z
            cy = leader_y + (full_cy - leader_y) * z
            r  = zoom_r   + (full_r  - zoom_r)   * z

            ax_track.set_xlim(cx - r, cx + r)
            ax_track.set_ylim(cy - r, cy + r)

            # Update label offset to match current viewport scale
            current_y_span = 2 * r
            _lbl_dy[0] = current_y_span * 0.012

            # ── Leaderboard ───────────────────────────────────────────────
            ax_board.cla()
            ax_board.set_facecolor(_BG)
            ax_board.axis("off")
            ax_board.set_xlim(0, 1)
            ax_board.set_ylim(0, 1)

            finished = sorted(
                [d for d in drivers if d.at(t)[2] >= 1.0],
                key=lambda d: d.official_laptime_s,
            )
            racing = sorted(
                [d for d in drivers if d.at(t)[2] < 1.0],
                key=lambda d: d.at(t)[2], reverse=True,
            )
            ranked       = finished + racing
            all_finished = len(racing) == 0
            leader_lt    = finished[0].official_laptime_s if finished else None

            n       = len(ranked)
            row_gap = min(0.18, 0.65 / max(n - 1, 1))
            for rank, drv in enumerate(ranked):
                row_y = 0.88 - rank * row_gap
                ax_board.text(0.04, row_y, str(rank + 1),
                              color=_DIM, fontsize=9, fontweight="bold",
                              va="center", fontfamily="monospace")
                ax_board.plot([0.10], [row_y], "o", color=drv.color, markersize=10, zorder=5)
                ax_board.text(0.15, row_y, drv.abbr,
                              color=drv.color, fontsize=11, fontweight="bold",
                              va="center", fontfamily="monospace")
                if rank == 0 and all_finished:
                    ax_board.text(0.42, row_y, _fmt_laptime(leader_lt),
                                  color=_MID, fontsize=8, va="center", fontfamily="monospace")
                elif rank > 0:
                    nd_ldr = ranked[0].at(t)[2]
                    gap_s  = max(drv.time_at_norm_dist(nd_ldr)
                                 - ranked[0].time_at_norm_dist(nd_ldr), 0.0)
                    ax_board.text(0.42, row_y, f"+{gap_s:.3f}s",
                                  color=_MID, fontsize=8, va="center", fontfamily="monospace")

            if progress_cb is not None:
                progress_cb(frame + 1, total_frames)

        # ── Top label ─────────────────────────────────────────────────────
        ax_top.set_xlim(0, 1)
        ax_top.set_ylim(0, 1)
        ax_top.text(0.5, 0.65, "Q LAP COMPARISON",
                    color=_WHITE, fontsize=12, fontweight="bold",
                    ha="center", va="center", fontfamily="monospace")
        ax_top.text(0.5, 0.25, f"TOP {len(drivers)}  ·  Q3",
                    color=_DIM, fontsize=8, ha="center", va="center", fontfamily="monospace")

        # ── Render ────────────────────────────────────────────────────────
        anim   = FuncAnimation(fig, animate, frames=total_frames, interval=1000 / fps)
        writer = FFMpegWriter(
            fps=fps, bitrate=8000,
            extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                        "-r", str(fps), "-level:v", "5.1"],
        )
        anim.save(str(output_path), writer=writer, dpi=_DPI)
        plt.close(fig)
        return output_path
