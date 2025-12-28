"""
Knowledge Graph endpoints.
"""
from fastapi import APIRouter, HTTPException, Query, Depends, Request
from typing import List
from api.schemas.knowledge_graph import TripleCreate, TripleUpdate, TripleResponse
from api.database.repositories_chat import KnowledgeGraphTripleRepository
from api.database.dependencies import get_db_session
from api.auth.jwt import get_current_user
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/knowledge-graph", tags=["knowledge-graph"])


def get_triple_repository(session: AsyncSession = Depends(get_db_session)) -> KnowledgeGraphTripleRepository:
    """Get knowledge graph triple repository."""
    return KnowledgeGraphTripleRepository(session)


@router.get("", response_model=List[TripleResponse])
async def get_triples(
    repo: KnowledgeGraphTripleRepository = Depends(get_triple_repository),
    current_user: dict = Depends(get_current_user)
):
    """Get all triples for the authenticated user."""
    triples = await repo.list_by_user(current_user["id"])
    return [TripleResponse(**t) for t in triples]


@router.post("", response_model=TripleResponse, status_code=201)
async def create_triple(
    triple_data: TripleCreate,
    repo: KnowledgeGraphTripleRepository = Depends(get_triple_repository),
    current_user: dict = Depends(get_current_user)
):
    """Create a new triple."""
    triple = await repo.create(
        user_id=current_user["id"],
        subject=triple_data.subject,
        predicate=triple_data.predicate,
        object=triple_data.object
    )
    return TripleResponse(**triple)


@router.patch("", response_model=TripleResponse)
async def update_triple(
    triple_data: TripleUpdate,
    repo: KnowledgeGraphTripleRepository = Depends(get_triple_repository),
    current_user: dict = Depends(get_current_user)
):
    """Update a triple."""
    # Check ownership
    if not await repo.check_ownership(triple_data.id, current_user["id"]):
        raise HTTPException(
            status_code=403,
            detail={"error": "Not authorized to update this triple", "code": "forbidden:triple"}
        )
    
    triple = await repo.update(
        triple_id=triple_data.id,
        subject=triple_data.subject,
        predicate=triple_data.predicate,
        object=triple_data.object
    )
    
    if not triple:
        raise HTTPException(
            status_code=404,
            detail={"error": "Triple not found", "code": "not_found:triple"}
        )
    
    return TripleResponse(**triple)


@router.delete("")
async def delete_triple(
    id: str = Query(..., alias="id"),
    repo: KnowledgeGraphTripleRepository = Depends(get_triple_repository),
    current_user: dict = Depends(get_current_user)
):
    """Delete a triple."""
    # Check ownership
    if not await repo.check_ownership(id, current_user["id"]):
        raise HTTPException(
            status_code=403,
            detail={"error": "Not authorized to delete this triple", "code": "forbidden:triple"}
        )
    
    deleted = await repo.delete(id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail={"error": "Triple not found", "code": "not_found:triple"}
        )
    
    return {"success": True}

