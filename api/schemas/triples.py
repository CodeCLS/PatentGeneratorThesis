"""
Triple-related API schemas.
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class TripleCreate(BaseModel):
    """Schema for creating a triple."""
    document_id: str
    sentence_id: Optional[str] = None
    head_id: str = Field(..., min_length=1)
    head_name: str = Field(..., min_length=1)
    head_type: Optional[str] = None
    relation: str = Field(..., min_length=1)
    tail_id: str = Field(..., min_length=1)
    tail_name: str = Field(..., min_length=1)
    tail_type: Optional[str] = None
    cluster_id: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TripleUpdate(BaseModel):
    """Schema for updating a triple."""
    head_id: Optional[str] = None
    head_name: Optional[str] = None
    head_type: Optional[str] = None
    relation: Optional[str] = None
    tail_id: Optional[str] = None
    tail_name: Optional[str] = None
    tail_type: Optional[str] = None
    cluster_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class TripleResponse(BaseModel):
    """Schema for triple response."""
    id: str
    document_id: str
    sentence_id: Optional[str]
    head_id: str
    head_name: str
    head_type: Optional[str]
    relation: str
    tail_id: str
    tail_name: str
    tail_type: Optional[str]
    cluster_id: Optional[int]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]
    
    class Config:
        from_attributes = True


class TripleListResponse(BaseModel):
    """Schema for triple list response."""
    items: list[TripleResponse]
    total: int
    skip: int
    limit: int


class TripleBatchCreate(BaseModel):
    """Schema for creating multiple triples."""
    triples: list[TripleCreate]

