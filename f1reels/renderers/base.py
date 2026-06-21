from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

from f1reels.pipeline.models import DriverFrames, TrackShape

# Callback signature: (frames_done, total_frames) → None
ProgressCallback = Callable[[int, int], None]


@runtime_checkable
class Renderer(Protocol):
    name: str

    def render(
        self,
        track: TrackShape,
        drivers: list[DriverFrames],
        output_path: Path,
        fps: int,
        duration_s: float,
        progress_cb: ProgressCallback | None,
    ) -> Path: ...
