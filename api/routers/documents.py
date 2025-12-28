"""
Document management endpoints.
"""
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional
from api.schemas.documents import DocumentCreate, DocumentUpdate, DocumentResponse, DocumentListResponse
from api.database.models import Document
from api.database.dependencies import get_document_repository, get_sentence_repository, get_triple_repository
from api.database.dependencies import DocumentRepo, SentenceRepo, TripleRepo

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentResponse, status_code=201)
async def create_document(
    document: DocumentCreate,
    repo: DocumentRepo = Depends(get_document_repository),
):
    """Create a new document."""
    db_document = Document(
        title=document.title,
        text=document.text,
        source=document.source,
        metadata=document.metadata,
    )
    db_document = await repo.create(db_document)
    return DocumentResponse(**db_document.to_dict())


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = None,
    source: Optional[str] = None,
    repo: DocumentRepo = Depends(get_document_repository),
):
    """List documents with pagination and filters."""
    filters = {}
    if status:
        filters["status"] = status
    if source:
        filters["source"] = source
    
    documents = await repo.list(skip=skip, limit=limit, filters=filters)
    all_documents = await repo.list(filters=filters)  # Get total count
    total = len(all_documents)
    
    return DocumentListResponse(
        items=[DocumentResponse(**doc.to_dict()) for doc in documents],
        total=total,
        skip=skip,
        limit=limit,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    repo: DocumentRepo = Depends(get_document_repository),
):
    """Get a document by ID."""
    document = await repo.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse(**document.to_dict())


@router.patch("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: str,
    document: DocumentUpdate,
    repo: DocumentRepo = Depends(get_document_repository),
):
    """Update a document."""
    updates = document.dict(exclude_unset=True)
    updated = await repo.update(document_id, updates)
    if not updated:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse(**updated.to_dict())


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    repo: DocumentRepo = Depends(get_document_repository),
    sentence_repo: SentenceRepo = Depends(get_sentence_repository),
    triple_repo: TripleRepo = Depends(get_triple_repository),
):
    """Delete a document."""
    # Also delete related sentences and triples
    await sentence_repo.delete_by_document(document_id)
    await triple_repo.delete_by_document(document_id)
    
    deleted = await repo.delete(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")

