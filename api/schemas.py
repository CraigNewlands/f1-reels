from pydantic import BaseModel, Field


class RenderRequest(BaseModel):
    year: int = Field(..., example=2025)
    round_name: str = Field(..., example="Bahrain")
    session_type: str = Field("Q", example="Q")
    renderer: str = Field("matplotlib", example="matplotlib")
    fps: int = Field(30, ge=1, le=120)
    duration_s: float = Field(45.0, gt=0, le=300)
    drivers: list[str] | None = Field(
        None,
        description="Driver abbreviations to include. None = all Q3 drivers.",
        example=["PIA", "NOR"],
    )


class JobStatus(BaseModel):
    job_id: str
    status: str           # pending | rendering | done | failed
    progress: float       # 0.0 → 1.0
    error: str | None = None
    output_path: str | None = None


class RenderResponse(BaseModel):
    job_id: str
