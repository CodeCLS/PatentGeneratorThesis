"""
Database initialization script.
Creates all tables in the database.
"""
import asyncio
from sqlalchemy import text
from api.database.connection import init_database, get_engine, close_database
from api.database.sql_models import Base
from api.config import settings
import logging

logger = logging.getLogger(__name__)


async def create_tables():
    """Create all database tables."""
    if settings.database_type not in ["postgres", "supabase"]:
        logger.info("Skipping table creation for local storage")
        return
    
    init_database()
    engine = get_engine()
    
    if engine is None:
        logger.error("Database engine not initialized")
        return
    
    try:
        async with engine.begin() as conn:
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        raise
    finally:
        await close_database()


async def drop_tables():
    """Drop all database tables (use with caution!)."""
    if settings.database_type not in ["postgres", "supabase"]:
        logger.info("Skipping table dropping for local storage")
        return
    
    init_database()
    engine = get_engine()
    
    if engine is None:
        logger.error("Database engine not initialized")
        return
    
    try:
        async with engine.begin() as conn:
            # Drop all tables
            await conn.run_sync(Base.metadata.drop_all)
            logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error(f"Error dropping tables: {e}")
        raise
    finally:
        await close_database()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "drop":
        asyncio.run(drop_tables())
    else:
        asyncio.run(create_tables())



