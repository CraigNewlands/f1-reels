"""In-memory job store with background thread per render job."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import fastf1

from f1reels.colors import driver_color
from f1reels.config import CACHE_DIR, OUTPUT_DIR
from f1reels.pipeline.data import (
    build_driver_frames,
    build_track_shape,
    extract_gps_fixes,
)
from f1reels.renderers import get_renderer

from .schemas import RenderRequest


@dataclass
class Job:
    id: str
    status: str = "pending"   # pending | rendering | done | failed
    progress: float = 0.0
    error: str | None = None
    output_path: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)


_store: dict[str, Job] = {}
_store_lock = threading.Lock()


def create_job() -> str:
    job_id = str(uuid.uuid4())
    with _store_lock:
        _store[job_id] = Job(id=job_id)
    return job_id


def get_job(job_id: str) -> Job | None:
    with _store_lock:
        return _store.get(job_id)


def start_render(job_id: str, request: RenderRequest) -> None:
    """Launch the render pipeline in a background thread."""
    thread = threading.Thread(
        target=_run_render,
        args=(job_id, request),
        daemon=True,
    )
    thread.start()


def _run_render(job_id: str, req: RenderRequest) -> None:
    job = _store[job_id]
    job.update(status="rendering")

    try:
        fastf1.Cache.enable_cache(str(CACHE_DIR))
        session = fastf1.get_session(req.year, req.round_name, req.session_type)
        session.load()

        # Collect GPS fixes for track shape
        q3_results = session.results[session.results["Q3"].notna()]
        q3_drivers = q3_results["DriverNumber"].tolist()

        all_fixes = []
        for drv_num in q3_drivers:
            try:
                lap   = session.laps.pick_drivers(drv_num).pick_fastest()
                fixes = extract_gps_fixes(lap)
                if fixes:
                    all_fixes.append(fixes)
            except Exception:
                pass

        track = build_track_shape(all_fixes)

        # Filter to requested drivers (or all Q3)
        target_abbrs = {a.upper() for a in req.drivers} if req.drivers else None
        drv_pool = q3_drivers[: req.top_n] if req.top_n else q3_drivers

        drivers = []
        for drv_num in drv_pool:
            try:
                info  = session.get_driver(drv_num)
                abbr  = info["Abbreviation"]
                if target_abbrs and abbr not in target_abbrs:
                    continue
                team  = info.get("TeamName", "")
                color = driver_color(abbr, team)
                lap   = session.laps.pick_drivers(drv_num).pick_fastest()
                df    = build_driver_frames(lap, track, color, abbr)
                drivers.append(df)
            except Exception:
                pass

        if not drivers:
            raise ValueError("No driver data could be loaded.")

        renderer = get_renderer(req.renderer)

        slug   = req.round_name.lower().replace(" ", "_")
        output = OUTPUT_DIR / f"{slug}_{req.year}_Q_{req.renderer}.mp4"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        total_frames = req.fps * int(req.duration_s)

        def progress_cb(done: int, total: int) -> None:
            job.update(progress=done / total)

        renderer.render(
            track=track,
            drivers=drivers,
            output_path=output,
            fps=req.fps,
            duration_s=req.duration_s,
            progress_cb=progress_cb,
        )

        job.update(status="done", progress=1.0, output_path=str(output))

    except Exception as exc:
        job.update(status="failed", error=str(exc))
