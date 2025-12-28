"""
File upload utilities.
"""
import os
import uuid
from typing import Optional
from fastapi import UploadFile, HTTPException, status
from api.config import settings
import mimetypes


# Allowed MIME types
ALLOWED_IMAGE_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/gif", 
    "image/webp", "image/svg+xml"
}

ALLOWED_DOCUMENT_TYPES = {
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

ALLOWED_TEXT_TYPES = {
    "text/plain", "text/csv", "text/markdown", "text/html"
}

ALLOWED_OTHER_TYPES = {
    "application/json", "application/rtf", "application/zip"
}

ALLOWED_TYPES = (
    ALLOWED_IMAGE_TYPES | 
    ALLOWED_DOCUMENT_TYPES | 
    ALLOWED_TEXT_TYPES | 
    ALLOWED_OTHER_TYPES
)


def validate_file(file: UploadFile) -> None:
    """
    Validate uploaded file.
    Raises HTTPException if validation fails.
    """
    # Check file size
    if hasattr(file, 'size') and file.size:
        size_mb = file.size / (1024 * 1024)
        if size_mb > settings.max_file_size_mb:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": f"File size exceeds maximum of {settings.max_file_size_mb}MB",
                    "code": "bad_request:file"
                }
            )
    
    # Check content type
    content_type = file.content_type
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": f"File type {content_type} not allowed",
                "code": "bad_request:file"
            }
        )


async def save_file_local(file: UploadFile, user_id: str) -> dict:
    """
    Save file to local storage.
    Returns file info dict with url, pathname, contentType.
    """
    # Create uploads directory if it doesn't exist
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    # Generate unique filename
    file_ext = os.path.splitext(file.filename)[1] or mimetypes.guess_extension(file.content_type) or ""
    filename = f"{uuid.uuid4()}{file_ext}"
    filepath = os.path.join(upload_dir, filename)
    
    # Save file
    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)
    
    # Generate URL (assuming files are served statically)
    # In production, this would be a proper storage URL
    url = f"/uploads/{filename}"
    
    return {
        "url": url,
        "pathname": filename,
        "contentType": file.content_type
    }


async def upload_file(file: UploadFile, user_id: str) -> dict:
    """
    Upload file based on storage type configured.
    """
    validate_file(file)
    
    if settings.storage_type == "local":
        return await save_file_local(file, user_id)
    elif settings.storage_type == "s3":
        # TODO: Implement S3 upload
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="S3 storage not yet implemented"
        )
    elif settings.storage_type == "vercel":
        # TODO: Implement Vercel Blob Storage
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Vercel Blob Storage not yet implemented"
        )
    else:
        # Default to local
        return await save_file_local(file, user_id)

