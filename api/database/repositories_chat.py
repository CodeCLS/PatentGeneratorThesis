"""
Repositories for chat application entities (User, Chat, Message, Vote, etc.).
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_, func, desc
from datetime import datetime, timedelta
import uuid

from api.database.sql_models import (
    UserModel,
    ChatModel,
    MessageModel,
    VoteModel,
    ChatDocumentModel,
    KnowledgeGraphTripleModel,
    StreamModel,
    SuggestionModel,
)


# User Repository
class UserRepository:
    """Repository for User operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        result = await self.session.execute(
            select(UserModel).where(UserModel.id == user_id)
        )
        model = result.scalar_one_or_none()
        if model:
            return {"id": model.id, "email": model.email, "password": model.password}
        return None
    
    async def get_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email."""
        result = await self.session.execute(
            select(UserModel).where(UserModel.email == email)
        )
        model = result.scalar_one_or_none()
        if model:
            return {"id": model.id, "email": model.email, "password": model.password}
        return None
    
    async def create(self, email: str, password: Optional[str] = None) -> Dict[str, Any]:
        """Create a new user."""
        # Check if user exists
        existing = await self.get_by_email(email)
        if existing:
            raise ValueError("User with this email already exists")
        
        user = UserModel(
            id=str(uuid.uuid4()),
            email=email,
            password=password
        )
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return {"id": user.id, "email": user.email}
    
    async def ensure_exists(self, email: str, password: Optional[str] = None) -> Dict[str, Any]:
        """Ensure user exists (create if not, return if exists)."""
        existing = await self.get_by_email(email)
        if existing:
            return {"id": existing["id"], "email": existing["email"]}
        
        return await self.create(email, password)
    
    async def list_all(self) -> List[Dict[str, Any]]:
        """Get all users."""
        result = await self.session.execute(select(UserModel))
        models = result.scalars().all()
        return [{"id": m.id, "email": m.email} for m in models]


# Chat Repository
class ChatRepository:
    """Repository for Chat operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_by_id(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Get chat by ID."""
        result = await self.session.execute(
            select(ChatModel).where(ChatModel.id == chat_id)
        )
        model = result.scalar_one_or_none()
        if model:
            return {
                "id": model.id,
                "title": model.title,
                "createdAt": model.created_at,
                "visibility": model.visibility,
                "userId": model.user_id
            }
        return None
    
    async def create(self, user_id: str, title: str = "New Chat", visibility: str = "private") -> Dict[str, Any]:
        """Create a new chat."""
        chat = ChatModel(
            id=str(uuid.uuid4()),
            user_id=user_id,
            title=title,
            visibility=visibility,
            created_at=datetime.utcnow()
        )
        self.session.add(chat)
        await self.session.commit()
        await self.session.refresh(chat)
        return {
            "id": chat.id,
            "title": chat.title,
            "createdAt": chat.created_at,
            "visibility": chat.visibility,
            "userId": chat.user_id
        }
    
    async def update_title(self, chat_id: str, title: str) -> bool:
        """Update chat title."""
        stmt = (
            update(ChatModel)
            .where(ChatModel.id == chat_id)
            .values(title=title)
        )
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0
    
    async def list_by_user(
        self,
        user_id: str,
        limit: int = 10,
        starting_after: Optional[str] = None,
        ending_before: Optional[str] = None
    ) -> tuple[List[Dict[str, Any]], bool]:
        """List chats for a user with pagination."""
        query = select(ChatModel).where(ChatModel.user_id == user_id)
        
        if starting_after:
            # Get chats created after the specified chat
            after_chat = await self.get_by_id(starting_after)
            if after_chat:
                query = query.where(ChatModel.created_at < after_chat["createdAt"])
        
        if ending_before:
            # Get chats created before the specified chat
            before_chat = await self.get_by_id(ending_before)
            if before_chat:
                query = query.where(ChatModel.created_at > before_chat["createdAt"])
        
        query = query.order_by(desc(ChatModel.created_at)).limit(limit + 1)
        
        result = await self.session.execute(query)
        chats = result.scalars().all()
        
        has_more = len(chats) > limit
        chats = chats[:limit]
        
        chat_list = [{
            "id": c.id,
            "title": c.title,
            "createdAt": c.created_at,
            "visibility": c.visibility,
            "userId": c.user_id
        } for c in chats]
        
        return chat_list, has_more
    
    async def delete(self, chat_id: str) -> bool:
        """Delete a chat (cascade deletes messages, votes, streams)."""
        stmt = delete(ChatModel).where(ChatModel.id == chat_id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0
    
    async def delete_all_by_user(self, user_id: str) -> int:
        """Delete all chats for a user."""
        stmt = delete(ChatModel).where(ChatModel.user_id == user_id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount
    
    async def check_ownership(self, chat_id: str, user_id: str) -> bool:
        """Check if user owns the chat."""
        chat = await self.get_by_id(chat_id)
        return chat is not None and chat["userId"] == user_id


# Message Repository
class MessageRepository:
    """Repository for Message operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(
        self,
        chat_id: str,
        role: str,
        parts: List[Dict[str, Any]],
        attachments: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new message."""
        message = MessageModel(
            id=str(uuid.uuid4()),
            chat_id=chat_id,
            role=role,
            parts=parts,
            attachments=attachments or [],
            created_at=datetime.utcnow()
        )
        self.session.add(message)
        await self.session.commit()
        await self.session.refresh(message)
        return {
            "id": message.id,
            "chatId": message.chat_id,
            "role": message.role,
            "parts": message.parts,
            "attachments": message.attachments,
            "createdAt": message.created_at
        }
    
    async def list_by_chat(self, chat_id: str) -> List[Dict[str, Any]]:
        """List all messages for a chat, ordered by createdAt ascending."""
        result = await self.session.execute(
            select(MessageModel)
            .where(MessageModel.chat_id == chat_id)
            .order_by(MessageModel.created_at)
        )
        messages = result.scalars().all()
        return [{
            "id": m.id,
            "chatId": m.chat_id,
            "role": m.role,
            "parts": m.parts,
            "attachments": m.attachments,
            "createdAt": m.created_at
        } for m in messages]
    
    async def delete_by_chat(self, chat_id: str) -> int:
        """Delete all messages for a chat."""
        stmt = delete(MessageModel).where(MessageModel.chat_id == chat_id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount
    
    async def count_user_messages_24h(self, user_id: str) -> int:
        """Count messages for user's chats in last 24 hours."""
        # Get all chat IDs for user
        chat_repo = ChatRepository(self.session)
        chats, _ = await chat_repo.list_by_user(user_id, limit=10000)
        chat_ids = [c["id"] for c in chats]
        
        if not chat_ids:
            return 0
        
        # Count messages in last 24 hours
        cutoff = datetime.utcnow() - timedelta(hours=24)
        result = await self.session.execute(
            select(func.count(MessageModel.id))
            .where(
                and_(
                    MessageModel.chat_id.in_(chat_ids),
                    MessageModel.role == "user",
                    MessageModel.created_at >= cutoff
                )
            )
        )
        return result.scalar() or 0


# Vote Repository
class VoteRepository:
    """Repository for Vote operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_by_chat(self, chat_id: str) -> List[Dict[str, Any]]:
        """Get all votes for a chat."""
        result = await self.session.execute(
            select(VoteModel).where(VoteModel.chat_id == chat_id)
        )
        votes = result.scalars().all()
        return [{
            "chatId": v.chat_id,
            "messageId": v.message_id,
            "isUpvoted": v.is_upvoted
        } for v in votes]
    
    async def upsert(self, chat_id: str, message_id: str, is_upvoted: bool) -> Dict[str, Any]:
        """Create or update a vote."""
        # Check if vote exists
        result = await self.session.execute(
            select(VoteModel).where(
                and_(
                    VoteModel.chat_id == chat_id,
                    VoteModel.message_id == message_id
                )
            )
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            # Update
            stmt = (
                update(VoteModel)
                .where(
                    and_(
                        VoteModel.chat_id == chat_id,
                        VoteModel.message_id == message_id
                    )
                )
                .values(is_upvoted=is_upvoted)
            )
            await self.session.execute(stmt)
        else:
            # Create
            vote = VoteModel(
                chat_id=chat_id,
                message_id=message_id,
                is_upvoted=is_upvoted
            )
            self.session.add(vote)
        
        await self.session.commit()
        return {
            "chatId": chat_id,
            "messageId": message_id,
            "isUpvoted": is_upvoted
        }
    
    async def delete_by_chat(self, chat_id: str) -> int:
        """Delete all votes for a chat."""
        stmt = delete(VoteModel).where(VoteModel.chat_id == chat_id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount


# Chat Document Repository
class ChatDocumentRepository:
    """Repository for Chat Document operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(
        self,
        document_id: str,
        user_id: str,
        title: str,
        content: Optional[str],
        kind: str = "text"
    ) -> Dict[str, Any]:
        """Create a new document version."""
        doc = ChatDocumentModel(
            id=document_id,
            user_id=user_id,
            title=title,
            content=content,
            kind=kind,
            created_at=datetime.utcnow()
        )
        self.session.add(doc)
        await self.session.commit()
        await self.session.refresh(doc)
        return {
            "id": doc.id,
            "title": doc.title,
            "content": doc.content,
            "kind": doc.kind,
            "userId": doc.user_id,
            "createdAt": doc.created_at
        }
    
    async def get_by_id(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all versions of a document."""
        result = await self.session.execute(
            select(ChatDocumentModel)
            .where(ChatDocumentModel.id == document_id)
            .order_by(desc(ChatDocumentModel.created_at))
        )
        docs = result.scalars().all()
        return [{
            "id": d.id,
            "title": d.title,
            "content": d.content,
            "kind": d.kind,
            "userId": d.user_id,
            "createdAt": d.created_at
        } for d in docs]
    
    async def list_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """List all documents for a user."""
        # Get latest version of each document
        subquery = (
            select(
                ChatDocumentModel.id,
                func.max(ChatDocumentModel.created_at).label("max_created_at")
            )
            .where(ChatDocumentModel.user_id == user_id)
            .group_by(ChatDocumentModel.id)
            .subquery()
        )
        
        result = await self.session.execute(
            select(ChatDocumentModel)
            .join(
                subquery,
                and_(
                    ChatDocumentModel.id == subquery.c.id,
                    ChatDocumentModel.created_at == subquery.c.max_created_at
                )
            )
            .order_by(desc(ChatDocumentModel.created_at))
        )
        docs = result.scalars().all()
        return [{
            "id": d.id,
            "title": d.title,
            "content": d.content,
            "kind": d.kind,
            "userId": d.user_id,
            "createdAt": d.created_at
        } for d in docs]
    
    async def delete_versions_after(self, document_id: str, timestamp: datetime) -> List[Dict[str, Any]]:
        """Delete document versions created after timestamp."""
        # Get versions to delete
        result = await self.session.execute(
            select(ChatDocumentModel).where(
                and_(
                    ChatDocumentModel.id == document_id,
                    ChatDocumentModel.created_at > timestamp
                )
            )
        )
        docs_to_delete = result.scalars().all()
        deleted_data = [{
            "id": d.id,
            "title": d.title,
            "content": d.content,
            "kind": d.kind,
            "userId": d.user_id,
            "createdAt": d.created_at
        } for d in docs_to_delete]
        
        # Delete from database
        stmt = delete(ChatDocumentModel).where(
            and_(
                ChatDocumentModel.id == document_id,
                ChatDocumentModel.created_at > timestamp
            )
        )
        await self.session.execute(stmt)
        await self.session.commit()
        
        return deleted_data
    
    async def check_ownership(self, document_id: str, user_id: str) -> bool:
        """Check if user owns the document."""
        result = await self.session.execute(
            select(ChatDocumentModel).where(
                and_(
                    ChatDocumentModel.id == document_id,
                    ChatDocumentModel.user_id == user_id
                )
            ).limit(1)
        )
        return result.scalar_one_or_none() is not None


# Knowledge Graph Triple Repository
class KnowledgeGraphTripleRepository:
    """Repository for Knowledge Graph Triple operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, user_id: str, subject: str, predicate: str, object: str) -> Dict[str, Any]:
        """Create a new triple."""
        triple = KnowledgeGraphTripleModel(
            id=str(uuid.uuid4()),
            user_id=user_id,
            subject=subject,
            predicate=predicate,
            object=object,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        self.session.add(triple)
        await self.session.commit()
        await self.session.refresh(triple)
        return {
            "id": triple.id,
            "subject": triple.subject,
            "predicate": triple.predicate,
            "object": triple.object,
            "userId": triple.user_id,
            "createdAt": triple.created_at,
            "updatedAt": triple.updated_at
        }
    
    async def list_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """List all triples for a user."""
        result = await self.session.execute(
            select(KnowledgeGraphTripleModel)
            .where(KnowledgeGraphTripleModel.user_id == user_id)
            .order_by(desc(KnowledgeGraphTripleModel.created_at))
        )
        triples = result.scalars().all()
        return [{
            "id": t.id,
            "subject": t.subject,
            "predicate": t.predicate,
            "object": t.object,
            "userId": t.user_id,
            "createdAt": t.created_at,
            "updatedAt": t.updated_at
        } for t in triples]
    
    async def get_by_id(self, triple_id: str) -> Optional[Dict[str, Any]]:
        """Get triple by ID."""
        result = await self.session.execute(
            select(KnowledgeGraphTripleModel).where(KnowledgeGraphTripleModel.id == triple_id)
        )
        triple = result.scalar_one_or_none()
        if triple:
            return {
                "id": triple.id,
                "subject": triple.subject,
                "predicate": triple.predicate,
                "object": triple.object,
                "userId": triple.user_id,
                "createdAt": triple.created_at,
                "updatedAt": triple.updated_at
            }
        return None
    
    async def update(
        self,
        triple_id: str,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Update a triple."""
        update_dict = {}
        if subject is not None:
            update_dict["subject"] = subject
        if predicate is not None:
            update_dict["predicate"] = predicate
        if object is not None:
            update_dict["object"] = object
        
        if not update_dict:
            return await self.get_by_id(triple_id)
        
        update_dict["updated_at"] = datetime.utcnow()
        
        stmt = (
            update(KnowledgeGraphTripleModel)
            .where(KnowledgeGraphTripleModel.id == triple_id)
            .values(**update_dict)
        )
        await self.session.execute(stmt)
        await self.session.commit()
        
        return await self.get_by_id(triple_id)
    
    async def delete(self, triple_id: str) -> bool:
        """Delete a triple."""
        stmt = delete(KnowledgeGraphTripleModel).where(KnowledgeGraphTripleModel.id == triple_id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0
    
    async def check_ownership(self, triple_id: str, user_id: str) -> bool:
        """Check if user owns the triple."""
        triple = await self.get_by_id(triple_id)
        return triple is not None and triple["userId"] == user_id


# Stream Repository
class StreamRepository:
    """Repository for Stream operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, chat_id: str) -> Dict[str, Any]:
        """Create a new stream."""
        stream = StreamModel(
            id=str(uuid.uuid4()),
            chat_id=chat_id,
            created_at=datetime.utcnow()
        )
        self.session.add(stream)
        await self.session.commit()
        await self.session.refresh(stream)
        return {
            "id": stream.id,
            "chatId": stream.chat_id,
            "createdAt": stream.created_at
        }
    
    async def delete_by_chat(self, chat_id: str) -> int:
        """Delete all streams for a chat."""
        stmt = delete(StreamModel).where(StreamModel.chat_id == chat_id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount

