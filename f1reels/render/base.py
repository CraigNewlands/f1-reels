import shutil
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation

from f1reels.visualizations.base import Visualization

# Use non-interactive backend — safe in CI and headless environments
matplotlib.use("Agg")

_FIGURE_WIDTH = 9    # inches → 1080px at 120 dpi
_FIGURE_HEIGHT = 16  # inches → 1920px at 120 dpi
_DPI = 120


class Renderer:
    def __init__(self, fps: int = 30, duration: int = 45, dpi: int = _DPI):
        self.fps = fps
        self.duration = duration
        self.dpi = dpi
        self.total_frames = fps * duration

    def render(self, viz: Visualization, output_path: Path) -> None:
        _check_ffmpeg()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(
            figsize=(_FIGURE_WIDTH, _FIGURE_HEIGHT),
            facecolor="#0d0d0d",
            dpi=self.dpi,
        )
        viz.setup_figure(fig)

        rendered = [0]

        def animate(frame: int):
            try:
                viz.draw_frame(fig, frame, self.total_frames)
            except Exception as exc:
                print(f"\n[frame {frame}] draw_frame error: {exc}", file=sys.stderr)
                raise
            rendered[0] += 1
            pct = rendered[0] / self.total_frames * 100
            msg = f"\r  Rendering {pct:5.1f}%  ({rendered[0]}/{self.total_frames} frames)"
            sys.stdout.write(msg)
            sys.stdout.flush()

        anim = FuncAnimation(fig, animate, frames=self.total_frames, interval=1000 / self.fps)
        writer = FFMpegWriter(
            fps=self.fps,
            bitrate=8000,
            extra_args=[
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-r", str(self.fps),   # force output fps — without this ffmpeg guesses
                "-level:v", "5.1",     # H.264 level 5.1 supports 1080×1920 up to ~120fps
            ],
        )
        anim.save(str(output_path), writer=writer, dpi=self.dpi)
        plt.close(fig)
        print(f"\n  Saved → {output_path}")


def _check_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found in PATH.\n"
            "  macOS:  brew install ffmpeg\n"
            "  Ubuntu: sudo apt install ffmpeg"
        )
