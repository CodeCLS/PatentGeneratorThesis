"""
Pydantic schemas for Voting API.
"""
from pydantic import BaseModel, Field
from typing import Literal


class VoteCreate(BaseModel):
    """Schema for creating/updating a vote."""
    chatId: str
    messageId: str
    type: Literal["up", "down"]


class VoteResponse(BaseModel):
    """Schema for vote response."""
    chatId: str
    messageId: str
    isUpvoted: bool
    
    class Config:
        from_attributes = True

