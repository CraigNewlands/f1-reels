from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse

from .jobs import create_job, get_job, start_render
from .schemas import JobStatus, RenderRequest, RenderResponse

app = FastAPI(
    title="F1 Reels API",
    description="Render F1 qualifying lap comparison videos.",
    version="0.1.0",
)


@app.post("/render", response_model=RenderResponse, status_code=202)
def submit_render(request: RenderRequest, background_tasks: BackgroundTasks) -> RenderResponse:
    """Submit a render job.  Returns immediately with a job_id to poll."""
    job_id = create_job()
    background_tasks.add_task(start_render, job_id, request)
    return RenderResponse(job_id=job_id)


@app.get("/render/{job_id}", response_model=JobStatus)
def render_status(job_id: str) -> JobStatus:
    """Poll render progress.  status: pending | rendering | done | failed."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        error=job.error,
        output_path=job.output_path,
    )


@app.get("/render/{job_id}/video")
def download_video(job_id: str) -> FileResponse:
    """Download the rendered MP4 once status is 'done'."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done":
        raise HTTPException(status_code=409, detail=f"Job status is '{job.status}', not done")
    path = Path(job.output_path)
    if not path.exists():
        raise HTTPException(status_code=500, detail="Output file missing")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.get("/renderers")
def list_renderers() -> dict:
    """List available renderer backends."""
    from f1reels.renderers import REGISTRY
    return {"renderers": list(REGISTRY.keys())}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
