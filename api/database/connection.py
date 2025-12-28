"""
Database connection management for PostgreSQL/Supabase.
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from api.config import settings
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

# Global engine and session factory
_engine = None
_session_factory = None


def get_database_url() -> str:
    """Get database connection URL based on settings."""
    if settings.database_type == "supabase":
        # Supabase uses PostgreSQL, construct connection string
        # Supabase connection string format:
        # postgresql+asyncpg://user:password@host:port/dbname
        if settings.supabase_url:
            # If supabase_url is provided, extract connection info
            # Supabase URL format: https://project-ref.supabase.co
            # We need the connection pooler URL or direct connection
            # For Supabase, you typically use the connection pooler on port 6543
            # or direct connection on port 5432
            # You'll need to set POSTGRES_HOST to your Supabase host
            pass
        
        # Use postgres settings for Supabase (it's PostgreSQL under the hood)
        return (
            f"postgresql+asyncpg://{settings.postgres_user}:{settings.postgres_password}"
            f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
        )
    elif settings.database_type == "postgres":
        return (
            f"postgresql+asyncpg://{settings.postgres_user}:{settings.postgres_password}"
            f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
        )
    else:
        raise ValueError(f"Unsupported database type: {settings.database_type}")


def init_database():
    """Initialize database connection."""
    global _engine, _session_factory
    
    if settings.database_type in ["postgres", "supabase"]:
        database_url = get_database_url()
        _engine = create_async_engine(
            database_url,
            echo=settings.debug,
            pool_pre_ping=True,  # Verify connections before using
            pool_size=5,
            max_overflow=10,
        )
        _session_factory = async_sessionmaker(
            _engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        logger.info(f"Database initialized: {settings.database_type}")
    else:
        logger.info("Using local in-memory storage")


async def get_session():
    """Get database session (async generator)."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    async with _session_factory() as session:
        yield session


async def close_database():
    """Close database connections."""
    global _engine
    if _engine:
        await _engine.dispose()
        logger.info("Database connections closed")


def get_engine():
    """Get the database engine."""
    return _engine

