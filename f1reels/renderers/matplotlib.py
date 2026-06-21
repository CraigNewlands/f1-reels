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

_W, _H = 9, 16
_DPI   = 120
_BG    = "#0d0d0d"
_WHITE = "#ffffff"
_DIM   = "#444444"
_MID   = "#888888"

# 4 panels: title / track / delta graph / leaderboard
_RATIOS = [0.06, 0.60, 0.15, 0.19]


def _hex_to_rgba(h: str, alpha: float = 1.0) -> tuple:
    h = h.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    return r, g, b, alpha


def _fmt_laptime(s: float) -> str:
    m = int(s // 60)
    return f"{m}:{s % 60:06.3f}"


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
        leader       = min(drivers, key=lambda d: d.official_laptime_s)

        fig = plt.figure(figsize=(_W, _H), facecolor=_BG, dpi=_DPI)
        gs  = fig.add_gridspec(4, 1, height_ratios=_RATIOS, hspace=0)

        ax_top   = fig.add_subplot(gs[0])
        ax_track = fig.add_subplot(gs[1])
        ax_delta = fig.add_subplot(gs[2])
        ax_board = fig.add_subplot(gs[3])
        for ax in (ax_top, ax_track, ax_delta, ax_board):
            ax.set_facecolor(_BG)
            ax.axis("off")

        # ── Coloured track segments — coloured by who is fastest at each point ──
        pts    = track.all_points()
        n_segs = len(track.x) - 1
        nd_mids = (np.arange(n_segs) + 0.5) / n_segs
        seg_colors = [
            _hex_to_rgba(min(drivers, key=lambda d: d.time_at_norm_dist(float(nd))).color)
            for nd in nd_mids
        ]
        segments = [
            [[track.x[i], track.y[i]], [track.x[i + 1], track.y[i + 1]]]
            for i in range(n_segs)
        ]
        ax_track.add_collection(LineCollection(segments, colors=seg_colors, linewidths=8, zorder=1))

        # ── Track outline on top (thin white centre line) ─────────────────
        ax_track.plot(pts[:, 0], pts[:, 1], color="#0a0a0a", lw=18, solid_capstyle="round", zorder=0)
        ax_track.plot(pts[:, 0], pts[:, 1], color=_WHITE,    lw=1.5, solid_capstyle="round", alpha=0.6, zorder=2)
        ax_track.set_aspect("equal")
        ax_track.autoscale_view()
        ax_track.set_autoscale_on(False)

        # ── Start/finish line ──────────────────────────────────────────────
        sf_x, sf_y = track.x[0], track.y[0]
        tdx = float(track.x[1] - track.x[-2])
        tdy = float(track.y[1] - track.y[-2])
        tn  = np.sqrt(tdx**2 + tdy**2)
        tdx, tdy = tdx / tn, tdy / tn
        perp_x, perp_y = -tdy, tdx
        half = (pts[:, 0].max() - pts[:, 0].min()) * 0.025
        ax_track.plot(
            [sf_x - perp_x * half, sf_x + perp_x * half],
            [sf_y - perp_y * half, sf_y + perp_y * half],
            color=_WHITE, lw=2.5, zorder=2,
        )

        # ── Driver dots ───────────────────────────────────────────────────
        dot_artists = {}
        y_rng  = pts[:, 1].max() - pts[:, 1].min()
        lbl_dy = y_rng * 0.015
        for drv in drivers:
            halo, = ax_track.plot([], [], "o", color=drv.color, markersize=28, alpha=0.18, zorder=3)
            dot,  = ax_track.plot([], [], "o", color=drv.color, markersize=13,
                                  markeredgecolor=_WHITE, markeredgewidth=1.0, zorder=4)
            lbl   = ax_track.text(0, 0, drv.abbr, color=drv.color,
                                  fontsize=7, fontweight="bold",
                                  ha="center", va="bottom", fontfamily="monospace", zorder=5)
            dot_artists[drv.abbr] = (halo, dot, lbl)

        # ── Delta time graph — pre-compute full gap traces ─────────────────
        nd_grid   = np.linspace(0, 1, 600)
        trailers  = [d for d in drivers if d is not leader]
        gap_traces = {
            d.abbr: np.array([d.time_at_norm_dist(nd) - leader.time_at_norm_dist(nd)
                              for nd in nd_grid])
            for d in trailers
        }

        ax_delta.set_xlim(0, 1)
        all_gaps  = np.concatenate(list(gap_traces.values()))
        gap_max   = max(abs(all_gaps).max() * 1.3, 0.05)
        ax_delta.set_ylim(-gap_max, gap_max)
        ax_delta.set_autoscale_on(False)

        # Zero line
        ax_delta.axhline(0, color=_DIM, lw=0.8, zorder=0)

        # Tiny axis label
        ax_delta.text(0.01, 0.92, "Δ gap (s)", color=_DIM, fontsize=6,
                      transform=ax_delta.transAxes, va="top", fontfamily="monospace")

        # Animated gap lines — created once, updated each frame
        delta_lines = {}
        for drv in trailers:
            line, = ax_delta.plot([], [], color=drv.color, lw=1.5, zorder=2)
            delta_lines[drv.abbr] = line

        # ── animate ───────────────────────────────────────────────────────
        def animate(frame: int) -> None:
            t = (frame / max(total_frames - 1, 1)) * max_lap_s

            # Move dots
            ordered = sorted(drivers, key=lambda d: d.at(t)[2])
            for drv in drivers:
                x, y, _ = drv.at(t)
                halo, dot, lbl = dot_artists[drv.abbr]
                halo.set_data([x], [y])
                dot.set_data([x], [y])
                lbl.set_position((x, y + lbl_dy))

            # Fill delta graph up to leader's current position
            nd_now = leader.at(t)[2]
            mask   = nd_grid <= nd_now
            for drv in trailers:
                delta_lines[drv.abbr].set_data(nd_grid[mask], gap_traces[drv.abbr][mask])

            # Leaderboard
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
                    nd_ldr  = ranked[0].at(t)[2]
                    gap_s   = max(drv.time_at_norm_dist(nd_ldr) - ranked[0].time_at_norm_dist(nd_ldr), 0.0)
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
