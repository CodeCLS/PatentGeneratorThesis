"""
Voting endpoints.
"""
from fastapi import APIRouter, HTTPException, Query, Depends, Request
from typing import List
from api.schemas.votes import VoteCreate, VoteResponse
from api.database.repositories_chat import VoteRepository, ChatRepository
from api.database.dependencies import get_db_session
from api.auth.jwt import get_current_user
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/vote", tags=["votes"])


def get_vote_repository(session: AsyncSession = Depends(get_db_session)) -> VoteRepository:
    """Get vote repository."""
    return VoteRepository(session)


def get_chat_repository(session: AsyncSession = Depends(get_db_session)) -> ChatRepository:
    """Get chat repository."""
    return ChatRepository(session)


@router.get("", response_model=List[VoteResponse])
async def get_votes(
    chatId: str = Query(..., alias="chatId"),
    vote_repo: VoteRepository = Depends(get_vote_repository),
    chat_repo: ChatRepository = Depends(get_chat_repository),
    current_user: dict = Depends(get_current_user)
):
    """Get all votes for a chat."""
    # Check ownership
    if not await chat_repo.check_ownership(chatId, current_user["id"]):
        raise HTTPException(
            status_code=403,
            detail={"error": "Not authorized to access this chat", "code": "forbidden:vote"}
        )
    
    votes = await vote_repo.get_by_chat(chatId)
    return [VoteResponse(**v) for v in votes]


@router.patch("", status_code=200)
async def vote_on_message(
    vote_data: VoteCreate,
    vote_repo: VoteRepository = Depends(get_vote_repository),
    chat_repo: ChatRepository = Depends(get_chat_repository),
    current_user: dict = Depends(get_current_user)
):
    """Vote on a message (up or down)."""
    # Check ownership
    if not await chat_repo.check_ownership(vote_data.chatId, current_user["id"]):
        raise HTTPException(
            status_code=403,
            detail={"error": "Not authorized to vote on this chat", "code": "forbidden:vote"}
        )
    
    is_upvoted = vote_data.type == "up"
    await vote_repo.upsert(vote_data.chatId, vote_data.messageId, is_upvoted)
    return {"success": True}

