"""
Document-related API schemas.
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class DocumentCreate(BaseModel):
    """Schema for creating a document."""
    title: Optional[str] = None
    text: str = Field(..., min_length=1, description="Document text content")
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentUpdate(BaseModel):
    """Schema for updating a document."""
    title: Optional[str] = None
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentResponse(BaseModel):
    """Schema for document response."""
    id: str
    title: Optional[str]
    text: str
    source: Optional[str]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]
    status: str
    processing_error: Optional[str] = None
    
    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """Schema for document list response."""
    items: list[DocumentResponse]
    total: int
    skip: int
    limit: int

