"""
Chat history endpoints.
"""
from fastapi import APIRouter, HTTPException, Query, Depends, Request
from typing import Optional
from api.schemas.chat import ChatHistoryResponse, ChatResponse
from api.database.repositories_chat import ChatRepository, VoteRepository, MessageRepository, StreamRepository
from api.database.dependencies import get_db_session
from api.auth.jwt import get_current_user
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

router = APIRouter(prefix="/api/history", tags=["history"])


def get_chat_repository(session: AsyncSession = Depends(get_db_session)) -> ChatRepository:
    """Get chat repository."""
    return ChatRepository(session)


@router.get("", response_model=ChatHistoryResponse)
async def get_chat_history(
    limit: int = Query(10, ge=1, le=100),
    starting_after: Optional[str] = Query(None),
    ending_before: Optional[str] = Query(None),
    chat_repo: ChatRepository = Depends(get_chat_repository),
    current_user: dict = Depends(get_current_user)
):
    """Get paginated chat history for the authenticated user."""
    chats, has_more = await chat_repo.list_by_user(
        user_id=current_user["id"],
        limit=limit,
        starting_after=starting_after,
        ending_before=ending_before
    )
    
    return ChatHistoryResponse(
        chats=[ChatResponse(**c) for c in chats],
        hasMore=has_more
    )


@router.delete("")
async def delete_all_chats(
    session: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(get_current_user)
):
    """Delete all chats for the authenticated user."""
    chat_repo = ChatRepository(session)
    vote_repo = VoteRepository(session)
    message_repo = MessageRepository(session)
    stream_repo = StreamRepository(session)
    
    # Get all user chats
    chats, _ = await chat_repo.list_by_user(current_user["id"], limit=10000)
    chat_ids = [c["id"] for c in chats]
    
    # Delete votes, messages, streams for all chats
    for chat_id in chat_ids:
        await vote_repo.delete_by_chat(chat_id)
        await message_repo.delete_by_chat(chat_id)
        await stream_repo.delete_by_chat(chat_id)
    
    # Delete all chats
    deleted_count = await chat_repo.delete_all_by_user(current_user["id"])
    
    return {"deletedCount": deleted_count}

