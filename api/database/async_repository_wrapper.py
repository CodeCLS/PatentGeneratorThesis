"""
Async wrappers for local repositories to make them compatible with async PostgreSQL repositories.
"""
from typing import List, Optional, Dict, Any
from api.database.repository import (
    LocalDocumentRepository,
    LocalSentenceRepository,
    LocalTripleRepository,
    LocalJobRepository,
)
from api.database.models import Document, Sentence, Triple, ProcessingJob


class AsyncLocalDocumentRepository:
    """Async wrapper for LocalDocumentRepository."""
    
    def __init__(self, repo: LocalDocumentRepository):
        self._repo = repo
    
    async def get(self, id: str) -> Optional[Document]:
        return self._repo.get(id)
    
    async def list(self, skip: int = 0, limit: int = 100, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        return self._repo.list(skip=skip, limit=limit, filters=filters)
    
    async def create(self, document: Document) -> Document:
        return self._repo.create(document)
    
    async def update(self, id: str, updates: Dict[str, Any]) -> Optional[Document]:
        return self._repo.update(id, updates)
    
    async def delete(self, id: str) -> bool:
        return self._repo.delete(id)


class AsyncLocalSentenceRepository:
    """Async wrapper for LocalSentenceRepository."""
    
    def __init__(self, repo: LocalSentenceRepository):
        self._repo = repo
    
    async def get(self, id: str) -> Optional[Sentence]:
        return self._repo.get(id)
    
    async def list(self, skip: int = 0, limit: int = 100, filters: Optional[Dict[str, Any]] = None) -> List[Sentence]:
        return self._repo.list(skip=skip, limit=limit, filters=filters)
    
    async def create(self, sentence: Sentence) -> Sentence:
        return self._repo.create(sentence)
    
    async def create_batch(self, sentences: List[Sentence]) -> List[Sentence]:
        return self._repo.create_batch(sentences)
    
    async def update(self, id: str, updates: Dict[str, Any]) -> Optional[Sentence]:
        return self._repo.update(id, updates)
    
    async def delete(self, id: str) -> bool:
        return self._repo.delete(id)
    
    async def delete_by_document(self, document_id: str) -> int:
        return self._repo.delete_by_document(document_id)


class AsyncLocalTripleRepository:
    """Async wrapper for LocalTripleRepository."""
    
    def __init__(self, repo: LocalTripleRepository):
        self._repo = repo
    
    async def get(self, id: str) -> Optional[Triple]:
        return self._repo.get(id)
    
    async def list(self, skip: int = 0, limit: int = 100, filters: Optional[Dict[str, Any]] = None) -> List[Triple]:
        return self._repo.list(skip=skip, limit=limit, filters=filters)
    
    async def create(self, triple: Triple) -> Triple:
        return self._repo.create(triple)
    
    async def create_batch(self, triples: List[Triple]) -> List[Triple]:
        return self._repo.create_batch(triples)
    
    async def update(self, id: str, updates: Dict[str, Any]) -> Optional[Triple]:
        return self._repo.update(id, updates)
    
    async def delete(self, id: str) -> bool:
        return self._repo.delete(id)
    
    async def delete_by_document(self, document_id: str) -> int:
        return self._repo.delete_by_document(document_id)


class AsyncLocalJobRepository:
    """Async wrapper for LocalJobRepository."""
    
    def __init__(self, repo: LocalJobRepository):
        self._repo = repo
    
    async def get(self, id: str) -> Optional[ProcessingJob]:
        return self._repo.get(id)
    
    async def list(self, skip: int = 0, limit: int = 100, filters: Optional[Dict[str, Any]] = None) -> List[ProcessingJob]:
        return self._repo.list(skip=skip, limit=limit, filters=filters)
    
    async def create(self, job: ProcessingJob) -> ProcessingJob:
        return self._repo.create(job)
    
    async def update(self, id: str, updates: Dict[str, Any]) -> Optional[ProcessingJob]:
        return self._repo.update(id, updates)
    
    async def delete(self, id: str) -> bool:
        return self._repo.delete(id)



