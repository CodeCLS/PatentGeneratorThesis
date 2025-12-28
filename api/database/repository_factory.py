"""
Repository factory for creating the appropriate repository based on configuration.
"""
from typing import Union
from api.config import settings
from api.database.repository import (
    LocalDocumentRepository,
    LocalSentenceRepository,
    LocalTripleRepository,
    LocalJobRepository,
    PostgresDocumentRepository,
    PostgresSentenceRepository,
    PostgresTripleRepository,
    PostgresJobRepository,
)
from api.database.connection import init_database, get_session


class RepositoryFactory:
    """Factory for creating repositories."""
    
    @staticmethod
    def create_document_repository():
        """Create document repository."""
        if settings.database_type in ["postgres", "supabase"]:
            # For PostgreSQL, we'll use a session-per-request pattern
            # This will be handled in the dependency injection
            return None  # Will be created per request
        else:
            return LocalDocumentRepository()
    
    @staticmethod
    def create_sentence_repository():
        """Create sentence repository."""
        if settings.database_type in ["postgres", "supabase"]:
            return None  # Will be created per request
        else:
            return LocalSentenceRepository()
    
    @staticmethod
    def create_triple_repository():
        """Create triple repository."""
        if settings.database_type in ["postgres", "supabase"]:
            return None  # Will be created per request
        else:
            return LocalTripleRepository()
    
    @staticmethod
    def create_job_repository():
        """Create job repository."""
        if settings.database_type in ["postgres", "supabase"]:
            return None  # Will be created per request
        else:
            return LocalJobRepository()
    
    @staticmethod
    def initialize_database():
        """Initialize database connection if needed."""
        if settings.database_type in ["postgres", "supabase"]:
            init_database()



