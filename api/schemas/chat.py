"""
Pydantic schemas for Chat API.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime


class MessagePart(BaseModel):
    """Message part structure."""
    type: Literal["text", "file"]
    text: Optional[str] = None
    mediaType: Optional[str] = None
    name: Optional[str] = None
    url: Optional[str] = None


class Message(BaseModel):
    """Message structure."""
    id: str
    role: Literal["user", "assistant"]
    parts: List[MessagePart]
    attachments: List[Dict[str, Any]] = Field(default_factory=list)
    createdAt: Optional[datetime] = None


class ChatCreate(BaseModel):
    """Schema for creating/continuing a chat."""
    id: str
    message: Message
    messages: Optional[List[Message]] = None  # For tool approval flows
    selectedChatModel: str
    selectedVisibilityType: Literal["private", "public"] = "private"


class ChatResponse(BaseModel):
    """Schema for chat response."""
    id: str
    title: str
    createdAt: datetime
    visibility: Literal["private", "public"]
    userId: str
    
    class Config:
        from_attributes = True


class ChatHistoryResponse(BaseModel):
    """Schema for paginated chat history."""
    chats: List[ChatResponse]
    hasMore: bool


class ChatDeleteResponse(BaseModel):
    """Schema for deleted chat."""
    id: str
    deleted: bool = True

