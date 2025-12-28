"""
Pydantic schemas for File Upload API.
"""
from pydantic import BaseModel


class FileUploadResponse(BaseModel):
    """Schema for file upload response."""
    url: str
    pathname: str
    contentType: str

