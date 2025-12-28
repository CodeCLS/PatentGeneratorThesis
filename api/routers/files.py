"""
File upload endpoints.
"""
from fastapi import APIRouter, UploadFile, File, Depends
from api.schemas.files import FileUploadResponse
from api.utils.file_upload import upload_file
from api.auth.jwt import get_current_user

router = APIRouter(prefix="/api/files", tags=["files"])


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file_endpoint(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload a file."""
    result = await upload_file(file, current_user["id"])
    return FileUploadResponse(**result)

