"""
Dependency injection for database repositories.
"""
from typing import AsyncGenerator, Union
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.database.connection import get_session, init_database
from api.database.repository import (
    PostgresDocumentRepository,
    PostgresSentenceRepository,
    PostgresTripleRepository,
    PostgresJobRepository,
)
from api.database.async_repository_wrapper import (
    AsyncLocalDocumentRepository,
    AsyncLocalSentenceRepository,
    AsyncLocalTripleRepository,
    AsyncLocalJobRepository,
)
from api.config import settings


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session dependency."""
    if settings.database_type in ["postgres", "supabase"]:
        async for session in get_session():
            yield session
    else:
        # For local storage, we don't need a session
        yield None


# Type aliases for repository unions
DocumentRepo = Union[AsyncLocalDocumentRepository, PostgresDocumentRepository]
SentenceRepo = Union[AsyncLocalSentenceRepository, PostgresSentenceRepository]
TripleRepo = Union[AsyncLocalTripleRepository, PostgresTripleRepository]
JobRepo = Union[AsyncLocalJobRepository, PostgresJobRepository]


def get_document_repository(session: AsyncSession = Depends(get_db_session)) -> DocumentRepo:
    """Get document repository based on database type."""
    if settings.database_type in ["postgres", "supabase"]:
        if session is None:
            raise RuntimeError("Database session required for PostgreSQL/Supabase")
        return PostgresDocumentRepository(session)
    else:
        # Use app state for local repositories
        from api.main import get_app_state
        state = get_app_state()
        if state.document_repo is None:
            raise RuntimeError("Document repository not initialized")
        return AsyncLocalDocumentRepository(state.document_repo)


def get_sentence_repository(session: AsyncSession = Depends(get_db_session)) -> SentenceRepo:
    """Get sentence repository based on database type."""
    if settings.database_type in ["postgres", "supabase"]:
        if session is None:
            raise RuntimeError("Database session required for PostgreSQL/Supabase")
        return PostgresSentenceRepository(session)
    else:
        from api.main import get_app_state
        state = get_app_state()
        if state.sentence_repo is None:
            raise RuntimeError("Sentence repository not initialized")
        return AsyncLocalSentenceRepository(state.sentence_repo)


def get_triple_repository(session: AsyncSession = Depends(get_db_session)) -> TripleRepo:
    """Get triple repository based on database type."""
    if settings.database_type in ["postgres", "supabase"]:
        if session is None:
            raise RuntimeError("Database session required for PostgreSQL/Supabase")
        return PostgresTripleRepository(session)
    else:
        from api.main import get_app_state
        state = get_app_state()
        if state.triple_repo is None:
            raise RuntimeError("Triple repository not initialized")
        return AsyncLocalTripleRepository(state.triple_repo)


def get_job_repository(session: AsyncSession = Depends(get_db_session)) -> JobRepo:
    """Get job repository based on database type."""
    if settings.database_type in ["postgres", "supabase"]:
        if session is None:
            raise RuntimeError("Database session required for PostgreSQL/Supabase")
        return PostgresJobRepository(session)
    else:
        from api.main import get_app_state
        state = get_app_state()
        if state.job_repo is None:
            raise RuntimeError("Job repository not initialized")
        return AsyncLocalJobRepository(state.job_repo)

