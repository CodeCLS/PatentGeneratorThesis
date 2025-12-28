"""
Pipeline-related API schemas.
"""
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


class PipelineStartRequest(BaseModel):
    """Schema for starting a pipeline."""
    document_id: str = Field(..., description="Document ID to process")
    steps: Optional[list[str]] = Field(
        default=None,
        description="Optional list of steps to run. If None, runs all steps."
    )


class PipelineStatusResponse(BaseModel):
    """Schema for pipeline status response."""
    job_id: str
    document_id: str
    status: str  # pending, running, completed, failed
    progress: float = Field(..., ge=0.0, le=1.0)
    stage: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    
    class Config:
        from_attributes = True




