"""
Main FastAPI application.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional

from api.routers import documents, triples, pipeline, graph
from api.database.repository import (
    LocalDocumentRepository,
    LocalSentenceRepository,
    LocalTripleRepository,
    LocalJobRepository,
)
from api.services.pipeline_service import PipelineService


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
    # Cleanup if needed


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
app.include_router(documents.router)
app.include_router(triples.router)
app.include_router(pipeline.router)
app.include_router(graph.router)


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

