"""
Rate limiting utilities.
"""
from typing import Optional
from fastapi import HTTPException, status
from api.config import settings
from api.database.repositories_chat import MessageRepository
from sqlalchemy.ext.asyncio import AsyncSession


async def check_rate_limit(
    user_id: str,
    user_type: str,
    message_repo: MessageRepository
) -> None:
    """
    Check if user has exceeded rate limit.
    Raises HTTPException if limit exceeded.
    """
    # Get message count in last 24 hours
    count = await message_repo.count_user_messages_24h(user_id)
    
    # Determine limit based on user type
    if user_type == "guest":
        limit = settings.rate_limit_messages_guest
    else:
        limit = settings.rate_limit_messages_regular
    
    if count >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": f"Rate limit exceeded. Maximum {limit} messages per 24 hours.",
                "code": "rate_limit:chat"
            }
        )

