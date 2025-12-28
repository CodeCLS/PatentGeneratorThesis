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


# PostgreSQL/Supabase repositories
try:
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy import select, update, delete, and_, or_
    from sqlalchemy.orm import selectinload
    from api.database.sql_models import (
        DocumentModel,
        SentenceModel,
        TripleModel,
        ProcessingJobModel,
    )
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


class PostgresDocumentRepository(BaseRepository):
    """PostgreSQL/Supabase document repository."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get(self, id: str):
        """Get document by ID."""
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy not available")
        result = await self.session.execute(
            select(DocumentModel).where(DocumentModel.id == id)
        )
        model = result.scalar_one_or_none()
        if model:
            return self._model_to_document(model)
        return None
    
    async def list(self, skip: int = 0, limit: int = 100, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """List documents with pagination and filters."""
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy not available")
        query = select(DocumentModel)
        
        if filters:
            if "status" in filters:
                query = query.where(DocumentModel.status == filters["status"])
            if "source" in filters:
                query = query.where(DocumentModel.source == filters["source"])
        
        query = query.order_by(DocumentModel.created_at.desc()).offset(skip).limit(limit)
        result = await self.session.execute(query)
        models = result.scalars().all()
        return [self._model_to_document(m) for m in models]
    
    async def create(self, document: Document) -> Document:
        """Create new document."""
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy not available")
        model = DocumentModel(
            id=document.id,
            title=document.title,
            text=document.text,
            source=document.source,
            created_at=document.created_at,
            updated_at=document.updated_at,
            metadata=document.metadata,
            status=document.status,
            processing_error=document.processing_error,
        )
        self.session.add(model)
        await self.session.commit()
        await self.session.refresh(model)
        return document
    
    async def update(self, id: str, updates: Dict[str, Any]) -> Optional[Document]:
        """Update document."""
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy not available")
        # Build update dict
        update_dict = {}
        for key, value in updates.items():
            if hasattr(DocumentModel, key):
                update_dict[key] = value
        
        if not update_dict:
            return None
        
        update_dict["updated_at"] = datetime.utcnow()
        
        stmt = (
            update(DocumentModel)
            .where(DocumentModel.id == id)
            .values(**update_dict)
            .execution_options(synchronize_session="fetch")
        )
        await self.session.execute(stmt)
        await self.session.commit()
        
        return await self.get(id)
    
    async def delete(self, id: str) -> bool:
        """Delete document."""
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy not available")
        stmt = delete(DocumentModel).where(DocumentModel.id == id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0
    
    def _model_to_document(self, model: DocumentModel) -> Document:
        """Convert SQLAlchemy model to Document."""
        return Document(
            id=model.id,
            title=model.title,
            text=model.text,
            source=model.source,
            created_at=model.created_at,
            updated_at=model.updated_at,
            metadata=model.metadata or {},
            status=model.status,
            processing_error=model.processing_error,
        )


class PostgresSentenceRepository(BaseRepository):
    """PostgreSQL/Supabase sentence repository."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get(self, id: str):
        """Get sentence by ID."""
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy not available")
        result = await self.session.execute(
            select(SentenceModel).where(SentenceModel.id == id)
        )
        model = result.scalar_one_or_none()
        if model:
            return self._model_to_sentence(model)
        return None
    
    async def list(self, skip: int = 0, limit: int = 100, filters: Optional[Dict[str, Any]] = None) -> List[Sentence]:
        """List sentences with pagination and filters."""
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy not available")
        query = select(SentenceModel)
        
        if filters and "document_id" in filters:
            query = query.where(SentenceModel.document_id == filters["document_id"])
        
        query = query.order_by(SentenceModel.document_id, SentenceModel.index).offset(skip).limit(limit)
        result = await self.session.execute(query)
        models = result.scalars().all()
        return [self._model_to_sentence(m) for m in models]
    
    async def create(self, sentence: Sentence) -> Sentence:
        """Create new sentence."""
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy not available")
        model = SentenceModel(
            id=sentence.id,
            document_id=sentence.document_id,
            text=sentence.text,
            index=sentence.index,
            entities=sentence.entities,
            created_at=sentence.created_at,
        )
        self.session.add(model)
        await self.session.commit()
        await self.session.refresh(model)
        return sentence
    
    async def create_batch(self, sentences: List[Sentence]) -> List[Sentence]:
        """Create multiple sentences at once."""
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy not available")
        models = [
            SentenceModel(
                id=s.id,
                document_id=s.document_id,
                text=s.text,
                index=s.index,
                entities=s.entities,
                created_at=s.created_at,
            )
            for s in sentences
        ]
        self.session.add_all(models)
        await self.session.commit()
        return sentences
    
    async def update(self, id: str, updates: Dict[str, Any]) -> Optional[Sentence]:
        """Update sentence."""
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy not available")
        update_dict = {}
        for key, value in updates.items():
            if hasattr(SentenceModel, key):
                update_dict[key] = value
        
        if not update_dict:
            return None
        
        stmt = (
            update(SentenceModel)
            .where(SentenceModel.id == id)
            .values(**update_dict)
            .execution_options(synchronize_session="fetch")
        )
        await self.session.execute(stmt)
        await self.session.commit()
        
        return await self.get(id)
    
    async def delete(self, id: str) -> bool:
        """Delete sentence."""
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy not available")
        stmt = delete(SentenceModel).where(SentenceModel.id == id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0
    
    async def delete_by_document(self, document_id: str) -> int:
        """Delete all sentences for a document."""
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy not available")
        stmt = delete(SentenceModel).where(SentenceModel.document_id == document_id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount
    
    def _model_to_sentence(self, model: SentenceModel) -> Sentence:
        """Convert SQLAlchemy model to Sentence."""
        return Sentence(
            id=model.id,
            document_id=model.document_id,
            text=model.text,
            index=model.index,
            entities=model.entities or [],
            created_at=model.created_at,
        )


class PostgresTripleRepository(BaseRepository):
    """PostgreSQL/Supabase triple repository."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get(self, id: str):
        """Get triple by ID."""
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy not available")
        result = await self.session.execute(
            select(TripleModel).where(TripleModel.id == id)
        )
        model = result.scalar_one_or_none()
        if model:
            return self._model_to_triple(model)
        return None
    
    async def list(self, skip: int = 0, limit: int = 100, filters: Optional[Dict[str, Any]] = None) -> List[Triple]:
        """List triples with pagination and filters."""
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy not available")
        query = select(TripleModel)
        
        if filters:
            if "document_id" in filters:
                query = query.where(TripleModel.document_id == filters["document_id"])
            if "cluster_id" in filters:
                query = query.where(TripleModel.cluster_id == filters["cluster_id"])
            if "head_id" in filters:
                query = query.where(TripleModel.head_id == filters["head_id"])
            if "tail_id" in filters:
                query = query.where(TripleModel.tail_id == filters["tail_id"])
        
        query = query.order_by(TripleModel.created_at.desc()).offset(skip).limit(limit)
        result = await self.session.execute(query)
        models = result.scalars().all()
        return [self._model_to_triple(m) for m in models]
    
    async def create(self, triple: Triple) -> Triple:
        """Create new triple."""
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy not available")
        model = TripleModel(
            id=triple.id,
            document_id=triple.document_id,
            sentence_id=triple.sentence_id,
            head_id=triple.head_id,
            head_name=triple.head_name,
            head_type=triple.head_type,
            relation=triple.relation,
            tail_id=triple.tail_id,
            tail_name=triple.tail_name,
            tail_type=triple.tail_type,
            cluster_id=triple.cluster_id,
            created_at=triple.created_at,
            updated_at=triple.updated_at,
            metadata=triple.metadata,
        )
        self.session.add(model)
        await self.session.commit()
        await self.session.refresh(model)
        return triple
    
    async def create_batch(self, triples: List[Triple]) -> List[Triple]:
        """Create multiple triples at once."""
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy not available")
        models = [
            TripleModel(
                id=t.id,
                document_id=t.document_id,
                sentence_id=t.sentence_id,
                head_id=t.head_id,
                head_name=t.head_name,
                head_type=t.head_type,
                relation=t.relation,
                tail_id=t.tail_id,
                tail_name=t.tail_name,
                tail_type=t.tail_type,
                cluster_id=t.cluster_id,
                created_at=t.created_at,
                updated_at=t.updated_at,
                metadata=t.metadata,
            )
            for t in triples
        ]
        self.session.add_all(models)
        await self.session.commit()
        return triples
    
    async def update(self, id: str, updates: Dict[str, Any]) -> Optional[Triple]:
        """Update triple."""
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy not available")
        update_dict = {}
        for key, value in updates.items():
            if hasattr(TripleModel, key):
                update_dict[key] = value
        
        if not update_dict:
            return None
        
        update_dict["updated_at"] = datetime.utcnow()
        
        stmt = (
            update(TripleModel)
            .where(TripleModel.id == id)
            .values(**update_dict)
            .execution_options(synchronize_session="fetch")
        )
        await self.session.execute(stmt)
        await self.session.commit()
        
        return await self.get(id)
    
    async def delete(self, id: str) -> bool:
        """Delete triple."""
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy not available")
        stmt = delete(TripleModel).where(TripleModel.id == id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0
    
    async def delete_by_document(self, document_id: str) -> int:
        """Delete all triples for a document."""
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy not available")
        stmt = delete(TripleModel).where(TripleModel.document_id == document_id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount
    
    def _model_to_triple(self, model: TripleModel) -> Triple:
        """Convert SQLAlchemy model to Triple."""
        return Triple(
            id=model.id,
            document_id=model.document_id,
            sentence_id=model.sentence_id,
            head_id=model.head_id,
            head_name=model.head_name,
            head_type=model.head_type,
            relation=model.relation,
            tail_id=model.tail_id,
            tail_name=model.tail_name,
            tail_type=model.tail_type,
            cluster_id=model.cluster_id,
            created_at=model.created_at,
            updated_at=model.updated_at,
            metadata=model.metadata or {},
        )


class PostgresJobRepository(BaseRepository):
    """PostgreSQL/Supabase job repository."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get(self, id: str):
        """Get job by ID."""
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy not available")
        result = await self.session.execute(
            select(ProcessingJobModel).where(ProcessingJobModel.id == id)
        )
        model = result.scalar_one_or_none()
        if model:
            return self._model_to_job(model)
        return None
    
    async def list(self, skip: int = 0, limit: int = 100, filters: Optional[Dict[str, Any]] = None) -> List[ProcessingJob]:
        """List jobs with pagination and filters."""
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy not available")
        query = select(ProcessingJobModel)
        
        if filters:
            if "document_id" in filters:
                query = query.where(ProcessingJobModel.document_id == filters["document_id"])
            if "status" in filters:
                query = query.where(ProcessingJobModel.status == filters["status"])
        
        query = query.order_by(ProcessingJobModel.started_at.desc().nulls_last()).offset(skip).limit(limit)
        result = await self.session.execute(query)
        models = result.scalars().all()
        return [self._model_to_job(m) for m in models]
    
    async def create(self, job: ProcessingJob) -> ProcessingJob:
        """Create new job."""
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy not available")
        model = ProcessingJobModel(
            id=job.id,
            document_id=job.document_id,
            status=job.status,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error=job.error,
            progress=job.progress,
            stage=job.stage,
        )
        self.session.add(model)
        await self.session.commit()
        await self.session.refresh(model)
        return job
    
    async def update(self, id: str, updates: Dict[str, Any]) -> Optional[ProcessingJob]:
        """Update job."""
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy not available")
        update_dict = {}
        for key, value in updates.items():
            if hasattr(ProcessingJobModel, key):
                update_dict[key] = value
        
        if not update_dict:
            return None
        
        stmt = (
            update(ProcessingJobModel)
            .where(ProcessingJobModel.id == id)
            .values(**update_dict)
            .execution_options(synchronize_session="fetch")
        )
        await self.session.execute(stmt)
        await self.session.commit()
        
        return await self.get(id)
    
    async def delete(self, id: str) -> bool:
        """Delete job."""
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy not available")
        stmt = delete(ProcessingJobModel).where(ProcessingJobModel.id == id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0
    
    def _model_to_job(self, model: ProcessingJobModel) -> ProcessingJob:
        """Convert SQLAlchemy model to ProcessingJob."""
        return ProcessingJob(
            id=model.id,
            document_id=model.document_id,
            status=model.status,
            started_at=model.started_at,
            completed_at=model.completed_at,
            error=model.error,
            progress=model.progress,
            stage=model.stage,
        )


