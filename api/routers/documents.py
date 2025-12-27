"""
Document management endpoints.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from api.schemas.documents import DocumentCreate, DocumentUpdate, DocumentResponse, DocumentListResponse
from api.database.repository import LocalDocumentRepository
from api.database.models import Document

router = APIRouter(prefix="/documents", tags=["documents"])


def get_document_repo() -> LocalDocumentRepository:
    """Dependency to get document repository."""
    from api.main import get_app_state
    state = get_app_state()
    if state.document_repo is None:
        raise RuntimeError("Document repository not initialized")
    return state.document_repo


@router.post("", response_model=DocumentResponse, status_code=201)
async def create_document(document: DocumentCreate):
    """Create a new document."""
    repo = get_document_repo()
    db_document = Document(
        title=document.title,
        text=document.text,
        source=document.source,
        metadata=document.metadata,
    )
    db_document = repo.create(db_document)
    return DocumentResponse(**db_document.to_dict())


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = None,
    source: Optional[str] = None,
):
    """List documents with pagination and filters."""
    repo = get_document_repo()
    filters = {}
    if status:
        filters["status"] = status
    if source:
        filters["source"] = source
    
    documents = repo.list(skip=skip, limit=limit, filters=filters)
    total = len(repo.list(filters=filters))  # Get total count
    
    return DocumentListResponse(
        items=[DocumentResponse(**doc.to_dict()) for doc in documents],
        total=total,
        skip=skip,
        limit=limit,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """Get a document by ID."""
    repo = get_document_repo()
    document = repo.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse(**document.to_dict())


@router.patch("/{document_id}", response_model=DocumentResponse)
async def update_document(document_id: str, document: DocumentUpdate):
    """Update a document."""
    repo = get_document_repo()
    updates = document.dict(exclude_unset=True)
    updated = repo.update(document_id, updates)
    if not updated:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse(**updated.to_dict())


@router.delete("/{document_id}", status_code=204)
async def delete_document(document_id: str):
    """Delete a document."""
    repo = get_document_repo()
    # Also delete related sentences and triples
    from api.main import get_app_state
    state = get_app_state()
    state.sentence_repo.delete_by_document(document_id)
    state.triple_repo.delete_by_document(document_id)
    
    deleted = repo.delete(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")

