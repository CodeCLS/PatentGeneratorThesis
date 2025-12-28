"""
Chat endpoints with streaming support.
"""
from fastapi import APIRouter, HTTPException, Depends, Request, Header
from fastapi.responses import StreamingResponse
from typing import Optional
import json
from api.schemas.chat import ChatCreate, ChatResponse, Message
from api.database.repositories_chat import (
    ChatRepository,
    MessageRepository,
    VoteRepository,
    StreamRepository
)
from api.database.dependencies import get_db_session
from api.auth.jwt import get_current_user
from api.services.ai_service import AIService
from api.utils.rate_limit import check_rate_limit
from sqlalchemy.ext.asyncio import AsyncSession
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

ai_service = AIService()


def get_chat_repository(session: AsyncSession = Depends(get_db_session)) -> ChatRepository:
    """Get chat repository."""
    return ChatRepository(session)


def get_message_repository(session: AsyncSession = Depends(get_db_session)) -> MessageRepository:
    """Get message repository."""
    return MessageRepository(session)


def get_vote_repository(session: AsyncSession = Depends(get_db_session)) -> VoteRepository:
    """Get vote repository."""
    return VoteRepository(session)


def get_stream_repository(session: AsyncSession = Depends(get_db_session)) -> StreamRepository:
    """Get stream repository."""
    return StreamRepository(session)


@router.post("")
async def create_or_continue_chat(
    chat_data: ChatCreate,
    request: Request,
    chat_repo: ChatRepository = Depends(get_chat_repository),
    message_repo: MessageRepository = Depends(get_message_repository),
    stream_repo: StreamRepository = Depends(get_stream_repository),
    current_user: dict = Depends(get_current_user),
    longitude: Optional[float] = Header(None, alias="x-vercel-ip-longitude"),
    latitude: Optional[float] = Header(None, alias="x-vercel-ip-latitude"),
    city: Optional[str] = Header(None, alias="x-vercel-ip-city"),
    country: Optional[str] = Header(None, alias="x-vercel-ip-country")
):
    """
    Create or continue a chat with streaming response.
    Returns Server-Sent Events (SSE) stream.
    """
    # Check rate limit
    await check_rate_limit(current_user["id"], current_user.get("type", "regular"), message_repo)
    
    # Get or create chat
    chat = await chat_repo.get_by_id(chat_data.id)
    if not chat:
        chat_dict = await chat_repo.create(
            user_id=current_user["id"],
            title="New Chat",
            visibility=chat_data.selectedVisibilityType
        )
        chat_id = chat_dict["id"]
    else:
        # Check ownership
        if chat["userId"] != current_user["id"]:
            raise HTTPException(
                status_code=403,
                detail={"error": "Not authorized to access this chat", "code": "forbidden:chat"}
            )
        chat_id = chat["id"]
    
    # Save user message
    user_message = chat_data.message
    await message_repo.create(
        chat_id=chat_id,
        role=user_message.role,
        parts=[p.dict() for p in user_message.parts],
        attachments=user_message.attachments or []
    )
    
    # Get all messages for context (if provided, otherwise fetch from DB)
    if chat_data.messages:
        # Tool approval flow - use provided messages
        messages = chat_data.messages
    else:
        # Normal flow - get messages from DB
        db_messages = await message_repo.list_by_chat(chat_id)
        messages = [
            Message(
                id=m["id"],
                role=m["role"],
                parts=[MessagePart(**p) for p in m["parts"]],
                attachments=m.get("attachments", []),
                createdAt=m["createdAt"]
            )
            for m in db_messages
        ]
    
    # Create stream record
    stream = await stream_repo.create(chat_id)
    
    # Prepare geolocation hints
    geolocation = None
    if longitude and latitude:
        geolocation = {
            "longitude": longitude,
            "latitude": latitude,
            "city": city,
            "country": country
        }
    
    # Stream response
    async def generate_stream():
        assistant_message_id = None
        assistant_parts = []
        
        try:
            async for event in ai_service.stream_chat_response(
                messages=messages,
                model=chat_data.selectedChatModel,
                geolocation=geolocation
            ):
                yield event
                
                # Parse event to collect assistant message
                if event.startswith("data-appendMessage:"):
                    data = json.loads(event.split(":", 1)[1].strip())
                    assistant_parts.append(data)
                elif event.startswith("data-finishMessage:"):
                    # Save assistant message
                    if not assistant_message_id:
                        assistant_message = await message_repo.create(
                            chat_id=chat_id,
                            role="assistant",
                            parts=assistant_parts,
                            attachments=[]
                        )
                        assistant_message_id = assistant_message["id"]
                
                # Handle title updates (async)
                elif event.startswith("data-chat-title:"):
                    title_data = json.loads(event.split(":", 1)[1].strip())
                    # Update title in background (could use FastAPI BackgroundTasks)
                    await chat_repo.update_title(chat_id, title_data.get("title", "New Chat"))
        
        except Exception as e:
            logger.error(f"Error in chat stream: {e}")
            yield f"data-error: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.delete("")
async def delete_chat(
    id: str,
    chat_repo: ChatRepository = Depends(get_chat_repository),
    vote_repo: VoteRepository = Depends(get_vote_repository),
    message_repo: MessageRepository = Depends(get_message_repository),
    stream_repo: StreamRepository = Depends(get_stream_repository),
    current_user: dict = Depends(get_current_user)
):
    """Delete a chat and all associated data."""
    # Check ownership
    if not await chat_repo.check_ownership(id, current_user["id"]):
        raise HTTPException(
            status_code=403,
            detail={"error": "Not authorized to delete this chat", "code": "forbidden:chat"}
        )
    
    # Delete votes, messages, streams
    await vote_repo.delete_by_chat(id)
    await message_repo.delete_by_chat(id)
    await stream_repo.delete_by_chat(id)
    
    # Delete chat
    deleted = await chat_repo.delete(id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail={"error": "Chat not found", "code": "not_found:chat"}
        )
    
    return {"id": id, "deleted": True}


@router.get("/{chat_id}/messages")
async def get_chat_messages(
    chat_id: str,
    message_repo: MessageRepository = Depends(get_message_repository),
    chat_repo: ChatRepository = Depends(get_chat_repository),
    current_user: dict = Depends(get_current_user)
):
    """Get all messages for a chat."""
    # Check ownership
    if not await chat_repo.check_ownership(chat_id, current_user["id"]):
        raise HTTPException(
            status_code=403,
            detail={"error": "Not authorized to access this chat", "code": "forbidden:chat"}
        )
    
    messages = await message_repo.list_by_chat(chat_id)
    return messages

