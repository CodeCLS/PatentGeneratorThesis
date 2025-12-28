"""
Pydantic schemas for User API.
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


class UserCreate(BaseModel):
    """Schema for creating a user."""
    email: EmailStr
    password: Optional[str] = None  # Already hashed by Next.js, or None for OAuth


class UserResponse(BaseModel):
    """Schema for user response (without password)."""
    id: str
    email: str
    
    class Config:
        from_attributes = True


class UserWithPassword(BaseModel):
    """Schema for user with password (for Next.js verification only)."""
    id: str
    email: str
    password: Optional[str] = None  # Only returned to Next.js for password verification
    
    class Config:
        from_attributes = True


class UserEnsure(BaseModel):
    """Schema for ensuring user exists (OAuth flow)."""
    email: EmailStr
    password: Optional[str] = None

