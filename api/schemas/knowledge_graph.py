"""
Pydantic schemas for Knowledge Graph API.
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class TripleCreate(BaseModel):
    """Schema for creating a triple."""
    subject: str = Field(..., min_length=1)
    predicate: str = Field(..., min_length=1)
    object: str = Field(..., min_length=1)


class TripleUpdate(BaseModel):
    """Schema for updating a triple."""
    id: str
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None


class TripleResponse(BaseModel):
    """Schema for triple response."""
    id: str
    subject: str
    predicate: str
    object: str
    userId: str
    createdAt: datetime
    updatedAt: datetime
    
    class Config:
        from_attributes = True

