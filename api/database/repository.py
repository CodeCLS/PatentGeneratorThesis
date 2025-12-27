"""
Repository pattern for database operations.
Supports both local in-memory storage and PostgreSQL/Supabase.
"""
from __future__ import annotations

from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
from api.database.models import Document, Sentence, Triple, ProcessingJob


class BaseRepository(ABC):
    """Base repository interface."""
    
    @abstractmethod
    def get(self, id: str):
        """Get entity by ID."""
        pass
    
    @abstractmethod
    def list(self, skip: int = 0, limit: int = 100, filters: Optional[Dict[str, Any]] = None) -> List:
        """List entities with pagination and filters."""
        pass
    
    @abstractmethod
    def create(self, entity) -> Any:
        """Create new entity."""
        pass
    
    @abstractmethod
    def update(self, id: str, updates: Dict[str, Any]) -> Any:
        """Update entity."""
        pass
    
    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete entity."""
        pass


class LocalDocumentRepository(BaseRepository):
    """In-memory document repository."""
    
    def __init__(self):
        self._documents: Dict[str, Document] = {}
    
    def get(self, id: str) -> Optional[Document]:
        return self._documents.get(id)
    
    def list(self, skip: int = 0, limit: int = 100, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        docs = list(self._documents.values())
        
        # Apply filters
        if filters:
            if "status" in filters:
                docs = [d for d in docs if d.status == filters["status"]]
            if "source" in filters:
                docs = [d for d in docs if d.source == filters["source"]]
        
        # Sort by created_at descending
        docs.sort(key=lambda d: d.created_at, reverse=True)
        
        return docs[skip:skip + limit]
    
    def create(self, document: Document) -> Document:
        self._documents[document.id] = document
        return document
    
    def update(self, id: str, updates: Dict[str, Any]) -> Optional[Document]:
        if id not in self._documents:
            return None
        doc = self._documents[id]
        for key, value in updates.items():
            if hasattr(doc, key):
                setattr(doc, key, value)
        doc.updated_at = datetime.utcnow()
        return doc
    
    def delete(self, id: str) -> bool:
        if id in self._documents:
            del self._documents[id]
            return True
        return False


class LocalSentenceRepository(BaseRepository):
    """In-memory sentence repository."""
    
    def __init__(self):
        self._sentences: Dict[str, Sentence] = {}
        self._document_sentences: Dict[str, List[str]] = {}  # document_id -> sentence_ids
    
    def get(self, id: str) -> Optional[Sentence]:
        return self._sentences.get(id)
    
    def list(self, skip: int = 0, limit: int = 100, filters: Optional[Dict[str, Any]] = None) -> List[Sentence]:
        sentences = list(self._sentences.values())
        
        if filters:
            if "document_id" in filters:
                sentence_ids = self._document_sentences.get(filters["document_id"], [])
                sentences = [s for s in sentences if s.id in sentence_ids]
        
        sentences.sort(key=lambda s: (s.document_id, s.index))
        return sentences[skip:skip + limit]
    
    def create(self, sentence: Sentence) -> Sentence:
        self._sentences[sentence.id] = sentence
        if sentence.document_id not in self._document_sentences:
            self._document_sentences[sentence.document_id] = []
        if sentence.id not in self._document_sentences[sentence.document_id]:
            self._document_sentences[sentence.document_id].append(sentence.id)
        return sentence
    
    def create_batch(self, sentences: List[Sentence]) -> List[Sentence]:
        """Create multiple sentences at once."""
        for sentence in sentences:
            self.create(sentence)
        return sentences
    
    def update(self, id: str, updates: Dict[str, Any]) -> Optional[Sentence]:
        if id not in self._sentences:
            return None
        sentence = self._sentences[id]
        for key, value in updates.items():
            if hasattr(sentence, key):
                setattr(sentence, key, value)
        return sentence
    
    def delete(self, id: str) -> bool:
        if id in self._sentences:
            sentence = self._sentences[id]
            if sentence.document_id in self._document_sentences:
                self._document_sentences[sentence.document_id].remove(id)
            del self._sentences[id]
            return True
        return False
    
    def delete_by_document(self, document_id: str) -> int:
        """Delete all sentences for a document."""
        sentence_ids = self._document_sentences.get(document_id, [])
        count = 0
        for sentence_id in sentence_ids:
            if self.delete(sentence_id):
                count += 1
        if document_id in self._document_sentences:
            del self._document_sentences[document_id]
        return count


class LocalTripleRepository(BaseRepository):
    """In-memory triple repository."""
    
    def __init__(self):
        self._triples: Dict[str, Triple] = {}
        self._document_triples: Dict[str, List[str]] = {}  # document_id -> triple_ids
    
    def get(self, id: str) -> Optional[Triple]:
        return self._triples.get(id)
    
    def list(self, skip: int = 0, limit: int = 100, filters: Optional[Dict[str, Any]] = None) -> List[Triple]:
        triples = list(self._triples.values())
        
        if filters:
            if "document_id" in filters:
                triple_ids = self._document_triples.get(filters["document_id"], [])
                triples = [t for t in triples if t.id in triple_ids]
            if "cluster_id" in filters:
                triples = [t for t in triples if t.cluster_id == filters["cluster_id"]]
            if "head_id" in filters:
                triples = [t for t in triples if t.head_id == filters["head_id"]]
            if "tail_id" in filters:
                triples = [t for t in triples if t.tail_id == filters["tail_id"]]
        
        triples.sort(key=lambda t: t.created_at, reverse=True)
        return triples[skip:skip + limit]
    
    def create(self, triple: Triple) -> Triple:
        self._triples[triple.id] = triple
        if triple.document_id not in self._document_triples:
            self._document_triples[triple.document_id] = []
        if triple.id not in self._document_triples[triple.document_id]:
            self._document_triples[triple.document_id].append(triple.id)
        return triple
    
    def create_batch(self, triples: List[Triple]) -> List[Triple]:
        """Create multiple triples at once."""
        for triple in triples:
            self.create(triple)
        return triples
    
    def update(self, id: str, updates: Dict[str, Any]) -> Optional[Triple]:
        if id not in self._triples:
            return None
        triple = self._triples[id]
        for key, value in updates.items():
            if hasattr(triple, key):
                setattr(triple, key, value)
        triple.updated_at = datetime.utcnow()
        return triple
    
    def delete(self, id: str) -> bool:
        if id in self._triples:
            triple = self._triples[id]
            if triple.document_id in self._document_triples:
                self._document_triples[triple.document_id].remove(id)
            del self._triples[id]
            return True
        return False
    
    def delete_by_document(self, document_id: str) -> int:
        """Delete all triples for a document."""
        triple_ids = self._document_triples.get(document_id, [])
        count = 0
        for triple_id in triple_ids:
            if self.delete(triple_id):
                count += 1
        if document_id in self._document_triples:
            del self._document_triples[document_id]
        return count


class LocalJobRepository(BaseRepository):
    """In-memory job repository."""
    
    def __init__(self):
        self._jobs: Dict[str, ProcessingJob] = {}
        self._document_jobs: Dict[str, List[str]] = {}  # document_id -> job_ids
    
    def get(self, id: str) -> Optional[ProcessingJob]:
        return self._jobs.get(id)
    
    def list(self, skip: int = 0, limit: int = 100, filters: Optional[Dict[str, Any]] = None) -> List[ProcessingJob]:
        jobs = list(self._jobs.values())
        
        if filters:
            if "document_id" in filters:
                job_ids = self._document_jobs.get(filters["document_id"], [])
                jobs = [j for j in jobs if j.id in job_ids]
            if "status" in filters:
                jobs = [j for j in jobs if j.status == filters["status"]]
        
        jobs.sort(key=lambda j: j.started_at or datetime.min, reverse=True)
        return jobs[skip:skip + limit]
    
    def create(self, job: ProcessingJob) -> ProcessingJob:
        self._jobs[job.id] = job
        if job.document_id not in self._document_jobs:
            self._document_jobs[job.document_id] = []
        if job.id not in self._document_jobs[job.document_id]:
            self._document_jobs[job.document_id].append(job.id)
        return job
    
    def update(self, id: str, updates: Dict[str, Any]) -> Optional[ProcessingJob]:
        if id not in self._jobs:
            return None
        job = self._jobs[id]
        for key, value in updates.items():
            if hasattr(job, key):
                setattr(job, key, value)
        return job
    
    def delete(self, id: str) -> bool:
        if id in self._jobs:
            job = self._jobs[id]
            if job.document_id in self._document_jobs:
                self._document_jobs[job.document_id].remove(id)
            del self._jobs[id]
            return True
        return False


# Import datetime for the repositories
from datetime import datetime

