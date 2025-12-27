"""
Pipeline execution endpoints.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from api.schemas.pipeline import PipelineStartRequest, PipelineStatusResponse
from api.database.repository import LocalJobRepository
from api.services.pipeline_service import PipelineService

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


def get_pipeline_service() -> PipelineService:
    """Dependency to get pipeline service."""
    from api.main import get_app_state
    state = get_app_state()
    if state.pipeline_service is None:
        raise RuntimeError("Pipeline service not initialized")
    return state.pipeline_service


def get_job_repo() -> LocalJobRepository:
    """Dependency to get job repository."""
    from api.main import get_app_state
    state = get_app_state()
    if state.job_repo is None:
        raise RuntimeError("Job repository not initialized")
    return state.job_repo


@router.post("/start", response_model=PipelineStatusResponse, status_code=202)
async def start_pipeline(
    request: PipelineStartRequest,
    background_tasks: BackgroundTasks,
):
    """Start processing a document through the pipeline."""
    service = get_pipeline_service()
    
    # Start processing in background
    job = await service.process_document(
        document_id=request.document_id,
        steps=request.steps,
    )
    
    return PipelineStatusResponse(**job.to_dict())


@router.get("/status/{job_id}", response_model=PipelineStatusResponse)
async def get_pipeline_status(job_id: str):
    """Get the status of a processing job."""
    repo = get_job_repo()
    job = repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return PipelineStatusResponse(**job.to_dict())


@router.get("/document/{document_id}/status", response_model=PipelineStatusResponse)
async def get_document_pipeline_status(document_id: str):
    """Get the latest pipeline status for a document."""
    repo = get_job_repo()
    jobs = repo.list(filters={"document_id": document_id}, limit=1)
    if not jobs:
        raise HTTPException(status_code=404, detail="No job found for document")
    return PipelineStatusResponse(**jobs[0].to_dict())

