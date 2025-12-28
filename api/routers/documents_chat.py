"""
Chat Documents endpoints (different from pipeline documents).
"""
from fastapi import APIRouter, HTTPException, Query, Depends, Request
from typing import List
from datetime import datetime
from api.schemas.documents_chat import DocumentCreate, DocumentResponse, DocumentDeleteResponse
from api.database.repositories_chat import ChatDocumentRepository
from api.database.dependencies import get_db_session
from api.auth.jwt import get_current_user
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/document", tags=["documents-chat"])


def get_document_repository(session: AsyncSession = Depends(get_db_session)) -> ChatDocumentRepository:
    """Get chat document repository."""
    return ChatDocumentRepository(session)


@router.get("", response_model=List[DocumentResponse])
async def get_document(
    id: str = Query(..., alias="id"),
    repo: ChatDocumentRepository = Depends(get_document_repository),
    current_user: dict = Depends(get_current_user)
):
    """Get document by ID (returns all versions)."""
    docs = await repo.get_by_id(id)
    
    # Check ownership
    if docs and not await repo.check_ownership(id, current_user["id"]):
        raise HTTPException(
            status_code=403,
            detail={"error": "Not authorized to access this document", "code": "forbidden:document"}
        )
    
    return [DocumentResponse(**d) for d in docs]


@router.get("/all", response_model=List[DocumentResponse])
async def get_all_documents(
    repo: ChatDocumentRepository = Depends(get_document_repository),
    current_user: dict = Depends(get_current_user)
):
    """Get all documents for the authenticated user."""
    docs = await repo.list_by_user(current_user["id"])
    return [DocumentResponse(**d) for d in docs]


@router.post("", response_model=DocumentResponse)
async def create_or_update_document(
    document_data: DocumentCreate,
    id: str = Query(..., alias="id"),
    repo: ChatDocumentRepository = Depends(get_document_repository),
    current_user: dict = Depends(get_current_user)
):
    """Create or update a document (creates new version)."""
    doc = await repo.create(
        document_id=id,
        user_id=current_user["id"],
        title=document_data.title,
        content=document_data.content,
        kind=document_data.kind
    )
    return DocumentResponse(**doc)


@router.delete("")
async def delete_document_versions(
    id: str = Query(..., alias="id"),
    timestamp: str = Query(..., alias="timestamp"),
    repo: ChatDocumentRepository = Depends(get_document_repository),
    current_user: dict = Depends(get_current_user)
):
    """Delete document versions created after the specified timestamp."""
    # Check ownership
    if not await repo.check_ownership(id, current_user["id"]):
        raise HTTPException(
            status_code=403,
            detail={"error": "Not authorized to delete this document", "code": "forbidden:document"}
        )
    
    # Parse timestamp
    try:
        ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid timestamp format", "code": "bad_request:document"}
        )
    
    deleted = await repo.delete_versions_after(id, ts)
    return DocumentDeleteResponse(deleted=[DocumentResponse(**d) for d in deleted])

