"""
SQLAlchemy models for database tables.
"""
from sqlalchemy import Column, String, Text, Integer, Float, DateTime, JSON, ForeignKey, Index, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from api.database.connection import Base
import uuid


class DocumentModel(Base):
    """Document table model."""
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=True)
    text = Column(Text, nullable=False, default="")
    source = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    metadata = Column(JSON, nullable=False, default=dict)
    status = Column(String, nullable=False, default="pending")
    processing_error = Column(Text, nullable=True)
    
    __table_args__ = (
        Index("idx_documents_status", "status"),
        Index("idx_documents_source", "source"),
        Index("idx_documents_created_at", "created_at"),
    )


class SentenceModel(Base):
    """Sentence table model."""
    __tablename__ = "sentences"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    text = Column(Text, nullable=False)
    index = Column(Integer, nullable=False, default=0)
    entities = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    __table_args__ = (
        Index("idx_sentences_document_id", "document_id"),
        Index("idx_sentences_document_index", "document_id", "index"),
    )


class TripleModel(Base):
    """Triple table model."""
    __tablename__ = "triples"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    sentence_id = Column(String, ForeignKey("sentences.id", ondelete="SET NULL"), nullable=True)
    head_id = Column(String, nullable=False, index=True)
    head_name = Column(String, nullable=False)
    head_type = Column(String, nullable=True)
    relation = Column(String, nullable=False, index=True)
    tail_id = Column(String, nullable=False, index=True)
    tail_name = Column(String, nullable=False)
    tail_type = Column(String, nullable=True)
    cluster_id = Column(Integer, nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    metadata = Column(JSON, nullable=False, default=dict)
    
    __table_args__ = (
        Index("idx_triples_document_id", "document_id"),
        Index("idx_triples_cluster_id", "cluster_id"),
        Index("idx_triples_head_id", "head_id"),
        Index("idx_triples_tail_id", "tail_id"),
        Index("idx_triples_relation", "relation"),
    )


class ProcessingJobModel(Base):
    """Processing job table model."""
    __tablename__ = "processing_jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(String, nullable=False, default="pending")
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error = Column(Text, nullable=True)
    progress = Column(Float, nullable=False, default=0.0)
    stage = Column(String, nullable=True)
    
    __table_args__ = (
        Index("idx_jobs_document_id", "document_id"),
        Index("idx_jobs_status", "status"),
        Index("idx_jobs_started_at", "started_at"),
    )


# New models for chat application
class UserModel(Base):
    """User table model."""
    __tablename__ = "User"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(64), nullable=False, unique=True, index=True)
    password = Column(String(64), nullable=True)  # NULL for OAuth users
    
    __table_args__ = (
        Index("idx_user_email", "email", unique=True),
    )


class ChatModel(Base):
    """Chat table model."""
    __tablename__ = "Chat"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    title = Column(Text, nullable=False)
    user_id = Column(String, ForeignKey("User.id", ondelete="CASCADE"), nullable=False, index=True)
    visibility = Column(String(10), nullable=False, default="private")
    
    __table_args__ = (
        Index("idx_chat_user_id", "user_id"),
        Index("idx_chat_created_at", "created_at"),
        Index("idx_chat_visibility", "visibility"),
    )


class MessageModel(Base):
    """Message table model (Message_v2)."""
    __tablename__ = "Message_v2"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    chat_id = Column(String, ForeignKey("Chat.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    parts = Column(JSON, nullable=False)
    attachments = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    __table_args__ = (
        Index("idx_message_chat_id", "chat_id"),
        Index("idx_message_created_at", "created_at"),
        Index("idx_message_role", "role"),
    )


class VoteModel(Base):
    """Vote table model (Vote_v2)."""
    __tablename__ = "Vote_v2"
    
    chat_id = Column(String, ForeignKey("Chat.id", ondelete="CASCADE"), primary_key=True)
    message_id = Column(String, ForeignKey("Message_v2.id", ondelete="CASCADE"), primary_key=True)
    is_upvoted = Column(Boolean, nullable=False)
    
    __table_args__ = (
        Index("idx_vote_chat_id", "chat_id"),
        Index("idx_vote_message_id", "message_id"),
    )


class ChatDocumentModel(Base):
    """Document table model (for chat artifacts, versioned)."""
    __tablename__ = "Document"
    
    id = Column(String, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), primary_key=True)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=True)
    kind = Column(String(10), nullable=False, default="text")
    user_id = Column(String, ForeignKey("User.id", ondelete="CASCADE"), nullable=False, index=True)
    
    __table_args__ = (
        Index("idx_document_user_id", "user_id"),
        Index("idx_document_created_at", "created_at"),
        Index("idx_document_kind", "kind"),
    )


class KnowledgeGraphTripleModel(Base):
    """Knowledge graph triple table model."""
    __tablename__ = "KnowledgeGraphTriple"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    subject = Column(Text, nullable=False)
    predicate = Column(Text, nullable=False)
    object = Column(Text, nullable=False)  # 'object' is a reserved word, but required by schema
    user_id = Column(String, ForeignKey("User.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    __table_args__ = (
        Index("idx_kg_triple_user_id", "user_id"),
        Index("idx_kg_triple_created_at", "created_at"),
    )


class StreamModel(Base):
    """Stream table model."""
    __tablename__ = "Stream"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    chat_id = Column(String, ForeignKey("Chat.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    __table_args__ = (
        Index("idx_stream_chat_id", "chat_id"),
        Index("idx_stream_created_at", "created_at"),
    )


class SuggestionModel(Base):
    """Suggestion table model."""
    __tablename__ = "Suggestion"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, nullable=False)
    document_created_at = Column(DateTime(timezone=True), nullable=False)
    original_text = Column(Text, nullable=False)
    suggested_text = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    is_resolved = Column(Boolean, nullable=False, default=False)
    user_id = Column(String, ForeignKey("User.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    __table_args__ = (
        Index("idx_suggestion_document", "document_id", "document_created_at"),
        Index("idx_suggestion_user_id", "user_id"),
        Index("idx_suggestion_created_at", "created_at"),
    )



