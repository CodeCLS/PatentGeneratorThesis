"""
Pydantic schemas for Chat Documents API (different from pipeline documents).
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


class DocumentCreate(BaseModel):
    """Schema for creating/updating a document."""
    content: Optional[str] = None
    title: str = Field(..., min_length=1)
    kind: Literal["text", "code", "image", "sheet"] = "text"


class DocumentResponse(BaseModel):
    """Schema for document response."""
    id: str
    title: str
    content: Optional[str] = None
    kind: Literal["text", "code", "image", "sheet"]
    userId: str
    createdAt: datetime
    
    class Config:
        from_attributes = True


class DocumentDeleteResponse(BaseModel):
    """Schema for deleted documents."""
    deleted: List[DocumentResponse] = Field(default_factory=list)

