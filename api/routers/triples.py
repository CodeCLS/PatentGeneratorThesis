"""
Triple management endpoints.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from api.schemas.triples import (
    TripleCreate,
    TripleUpdate,
    TripleResponse,
    TripleListResponse,
    TripleBatchCreate,
)
from api.database.repository import LocalTripleRepository
from api.database.models import Triple

router = APIRouter(prefix="/triples", tags=["triples"])


def get_triple_repo() -> LocalTripleRepository:
    """Dependency to get triple repository."""
    from api.main import get_app_state
    state = get_app_state()
    if state.triple_repo is None:
        raise RuntimeError("Triple repository not initialized")
    return state.triple_repo


@router.post("", response_model=TripleResponse, status_code=201)
async def create_triple(triple: TripleCreate):
    """Create a new triple."""
    repo = get_triple_repo()
    db_triple = Triple(
        document_id=triple.document_id,
        sentence_id=triple.sentence_id,
        head_id=triple.head_id,
        head_name=triple.head_name,
        head_type=triple.head_type,
        relation=triple.relation,
        tail_id=triple.tail_id,
        tail_name=triple.tail_name,
        tail_type=triple.tail_type,
        cluster_id=triple.cluster_id,
        metadata=triple.metadata,
    )
    db_triple = repo.create(db_triple)
    return TripleResponse(**db_triple.to_dict())


@router.post("/batch", response_model=TripleListResponse, status_code=201)
async def create_triples_batch(batch: TripleBatchCreate):
    """Create multiple triples at once."""
    repo = get_triple_repo()
    db_triples = []
    for triple in batch.triples:
        db_triple = Triple(
            document_id=triple.document_id,
            sentence_id=triple.sentence_id,
            head_id=triple.head_id,
            head_name=triple.head_name,
            head_type=triple.head_type,
            relation=triple.relation,
            tail_id=triple.tail_id,
            tail_name=triple.tail_name,
            tail_type=triple.tail_type,
            cluster_id=triple.cluster_id,
            metadata=triple.metadata,
        )
        db_triples.append(db_triple)
    
    created = repo.create_batch(db_triples)
    return TripleListResponse(
        items=[TripleResponse(**t.to_dict()) for t in created],
        total=len(created),
        skip=0,
        limit=len(created),
    )


@router.get("", response_model=TripleListResponse)
async def list_triples(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    document_id: Optional[str] = None,
    cluster_id: Optional[int] = None,
    head_id: Optional[str] = None,
    tail_id: Optional[str] = None,
):
    """List triples with pagination and filters."""
    repo = get_triple_repo()
    filters = {}
    if document_id:
        filters["document_id"] = document_id
    if cluster_id is not None:
        filters["cluster_id"] = cluster_id
    if head_id:
        filters["head_id"] = head_id
    if tail_id:
        filters["tail_id"] = tail_id
    
    triples = repo.list(skip=skip, limit=limit, filters=filters)
    total = len(repo.list(filters=filters))
    
    return TripleListResponse(
        items=[TripleResponse(**t.to_dict()) for t in triples],
        total=total,
        skip=skip,
        limit=limit,
    )


@router.get("/{triple_id}", response_model=TripleResponse)
async def get_triple(triple_id: str):
    """Get a triple by ID."""
    repo = get_triple_repo()
    triple = repo.get(triple_id)
    if not triple:
        raise HTTPException(status_code=404, detail="Triple not found")
    return TripleResponse(**triple.to_dict())


@router.patch("/{triple_id}", response_model=TripleResponse)
async def update_triple(triple_id: str, triple: TripleUpdate):
    """Update a triple."""
    repo = get_triple_repo()
    updates = triple.dict(exclude_unset=True)
    updated = repo.update(triple_id, updates)
    if not updated:
        raise HTTPException(status_code=404, detail="Triple not found")
    return TripleResponse(**updated.to_dict())


@router.delete("/{triple_id}", status_code=204)
async def delete_triple(triple_id: str):
    """Delete a triple."""
    repo = get_triple_repo()
    deleted = repo.delete(triple_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Triple not found")


@router.delete("/document/{document_id}", status_code=200)
async def delete_triples_by_document(document_id: str):
    """Delete all triples for a document."""
    repo = get_triple_repo()
    count = repo.delete_by_document(document_id)
    return {"deleted": count}

