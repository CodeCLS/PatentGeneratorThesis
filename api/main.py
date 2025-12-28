"""
Main FastAPI application.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional

from api.routers import (
    documents, triples, pipeline, graph,
    users, chat, knowledge_graph, documents_chat, files, votes, history
)
from api.database.repository import (
    LocalDocumentRepository,
    LocalSentenceRepository,
    LocalTripleRepository,
    LocalJobRepository,
)
from api.database.repository_factory import RepositoryFactory
from api.database.connection import init_database, close_database
from api.services.pipeline_service import PipelineService
from api.config import settings


class AppState:
    """Application state container."""
    def __init__(self):
        self.document_repo: Optional[LocalDocumentRepository] = None
        self.sentence_repo: Optional[LocalSentenceRepository] = None
        self.triple_repo: Optional[LocalTripleRepository] = None
        self.job_repo: Optional[LocalJobRepository] = None
        self.pipeline_service: Optional[PipelineService] = None


_app_state = AppState()


def get_app_state() -> AppState:
    """Get the application state."""
    return _app_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    # Initialize database connection if using PostgreSQL/Supabase
    if settings.database_type in ["postgres", "supabase"]:
        RepositoryFactory.initialize_database()
        # For PostgreSQL, repositories are created per-request via dependency injection
        # So we don't set them in app state
        _app_state.document_repo = None
        _app_state.sentence_repo = None
        _app_state.triple_repo = None
        _app_state.job_repo = None
        _app_state.pipeline_service = None  # Will need to be created per-request or refactored
    else:
        # Local in-memory storage
        _app_state.document_repo = LocalDocumentRepository()
        _app_state.sentence_repo = LocalSentenceRepository()
        _app_state.triple_repo = LocalTripleRepository()
        _app_state.job_repo = LocalJobRepository()
        
        _app_state.pipeline_service = PipelineService(
            document_repo=_app_state.document_repo,
            sentence_repo=_app_state.sentence_repo,
            triple_repo=_app_state.triple_repo,
            job_repo=_app_state.job_repo,
        )
    
    yield
    
    # Shutdown
    if settings.database_type in ["postgres", "supabase"]:
        await close_database()


# Create FastAPI app
app = FastAPI(
    title="LLM Patent Claim Generator API",
    description="API for processing documents and generating knowledge graphs",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
# Original pipeline routers
app.include_router(documents.router)
app.include_router(triples.router)
app.include_router(pipeline.router)
app.include_router(graph.router)

# New chat application routers
app.include_router(users.router)
app.include_router(chat.router)
app.include_router(knowledge_graph.router)
app.include_router(documents_chat.router)
app.include_router(files.router)
app.include_router(votes.router)
app.include_router(history.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "LLM Patent Claim Generator API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


