"""
Service for executing the knowledge graph generation pipeline.
"""
from __future__ import annotations

import asyncio
from typing import Optional, List
from datetime import datetime
import spacy
from concurrent.futures import ThreadPoolExecutor

from api.database.models import Document, Sentence, Triple, ProcessingJob
from api.database.repository import (
    LocalDocumentRepository,
    LocalSentenceRepository,
    LocalTripleRepository,
    LocalJobRepository,
)
from kg.cleaning.referencing import PipelineBuilder, EntityMapper
from kg.formatting.formatting_manager import FormattingManager
from kg.generating_kg.generating.NodeGenerator import NodeGenerator
from tools.sentence.entity import InMemoryEntityRepository
# GraphTriple imported when needed
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class PipelineService:
    """Service for executing the knowledge graph generation pipeline."""
    
    def __init__(
        self,
        document_repo: LocalDocumentRepository,
        sentence_repo: LocalSentenceRepository,
        triple_repo: LocalTripleRepository,
        job_repo: LocalJobRepository,
    ):
        self.document_repo = document_repo
        self.sentence_repo = sentence_repo
        self.triple_repo = triple_repo
        self.job_repo = job_repo
        
        # Initialize components (lazy loading)
        self._nlp = None
        self._pipeline_builder = None
        self._formatter = None
        self._sentence_classifier = None
        self._node_generator = None
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    @property
    def nlp(self):
        """Lazy load spaCy pipeline."""
        if self._nlp is None:
            self._nlp = spacy.load("en_core_web_trf")
        return self._nlp
    
    @property
    def pipeline_builder(self):
        """Lazy load pipeline builder."""
        if self._pipeline_builder is None:
            self._pipeline_builder = PipelineBuilder()
        return self._pipeline_builder
    
    @property
    def formatter(self):
        """Lazy load formatter."""
        if self._formatter is None:
            self._formatter = FormattingManager()
        return self._formatter
    
    @property
    def sentence_classifier(self):
        """Lazy load sentence classifier."""
        if self._sentence_classifier is None:
            model_path = "training/info/done/hf/sentence_classifier_model"
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self._sentence_classifier = {
                "tokenizer": tokenizer,
                "model": model,
                "id2label": model.config.id2label,
            }
        return self._sentence_classifier
    
    @property
    def node_generator(self):
        """Lazy load node generator."""
        if self._node_generator is None:
            self._node_generator = NodeGenerator()
        return self._node_generator
    
    async def process_document(
        self,
        document_id: str,
        steps: Optional[List[str]] = None,
    ) -> ProcessingJob:
        """
        Process a document through the pipeline.
        
        Steps:
        1. splitting - Split text into sentences
        2. filtering - Filter informative sentences
        3. ner_coref - Run NER and coreference resolution
        4. triple_generation - Generate knowledge graph triples
        
        Args:
            document_id: Document ID to process
            steps: Optional list of steps to run. If None, runs all steps.
        
        Returns:
            ProcessingJob instance
        """
        # Get document
        document = self.document_repo.get(document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")
        
        # Create job
        job = ProcessingJob(
            document_id=document_id,
            status="running",
            started_at=datetime.utcnow(),
            progress=0.0,
        )
        job = self.job_repo.create(job)
        
        # Update document status
        self.document_repo.update(document_id, {"status": "processing"})
        
        try:
            # Run pipeline steps
            if steps is None:
                steps = ["splitting", "filtering", "ner_coref", "triple_generation"]
            
            if "splitting" in steps:
                await self._run_splitting(document, job)
            
            if "filtering" in steps:
                await self._run_filtering(document, job)
            
            if "ner_coref" in steps:
                await self._run_ner_coref(document, job)
            
            if "triple_generation" in steps:
                await self._run_triple_generation(document, job)
            
            # Mark as completed
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.progress = 1.0
            self.job_repo.update(job.id, {
                "status": "completed",
                "completed_at": datetime.utcnow(),
                "progress": 1.0,
            })
            
            self.document_repo.update(document_id, {"status": "completed"})
            
        except Exception as e:
            # Mark as failed
            error_msg = str(e)
            job.status = "failed"
            job.completed_at = datetime.utcnow()
            job.error = error_msg
            self.job_repo.update(job.id, {
                "status": "failed",
                "completed_at": datetime.utcnow(),
                "error": error_msg,
            })
            
            self.document_repo.update(document_id, {
                "status": "failed",
                "processing_error": error_msg,
            })
            raise
        
        return job
    
    async def _run_splitting(self, document: Document, job: ProcessingJob):
        """Run sentence splitting step."""
        job.stage = "splitting"
        self.job_repo.update(job.id, {"stage": "splitting", "progress": 0.1})
        
        # Split into sentences using spaCy
        doc = self.nlp(document.text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Use formatter for additional splitting
        sentence_objects = []
        for sent_text in sentences:
            formatted = self.formatter.split(sent_text)
            sentence_objects.extend(formatted)
        
        # Save sentences
        db_sentences = []
        for idx, sent_obj in enumerate(sentence_objects):
            db_sentence = Sentence(
                document_id=document.id,
                text=sent_obj.text,
                index=idx,
                entities=[],  # Will be populated in ner_coref step
            )
            db_sentences.append(db_sentence)
        
        self.sentence_repo.create_batch(db_sentences)
        
        self.job_repo.update(job.id, {"progress": 0.2})
    
    async def _run_filtering(self, document: Document, job: ProcessingJob):
        """Run sentence filtering step."""
        job.stage = "filtering"
        self.job_repo.update(job.id, {"stage": "filtering", "progress": 0.3})
        
        # Get sentences
        sentences = self.sentence_repo.list(filters={"document_id": document.id})
        
        # Classify sentences
        classifier = self.sentence_classifier
        informative_sentences = []
        
        for sentence in sentences:
            inputs = classifier["tokenizer"](
                sentence.text,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )
            with torch.no_grad():
                outputs = classifier["model"](**inputs)
            probs = outputs.logits.softmax(dim=-1)[0]
            pred_id = int(torch.argmax(probs))
            label = classifier["id2label"][pred_id]
            
            if label == "INFORMATIVE":
                informative_sentences.append(sentence)
            else:
                # Delete non-informative sentence
                self.sentence_repo.delete(sentence.id)
        
        self.job_repo.update(job.id, {"progress": 0.4})
    
    async def _run_ner_coref(self, document: Document, job: ProcessingJob):
        """Run NER and coreference resolution step."""
        job.stage = "ner_coref"
        self.job_repo.update(job.id, {"stage": "ner_coref", "progress": 0.5})
        
        # Get sentences
        sentences = self.sentence_repo.list(filters={"document_id": document.id})
        if not sentences:
            return
        
        # Build pipeline
        nlp = self.pipeline_builder.build()
        
        # Process document
        doc = nlp(document.text)
        
        # Map entities to sentences
        entity_mapper = EntityMapper()
        from kg.cleaning.referencing.entity_mapper import join_sentences, JoinedText
        
        # Convert to Sentence objects for entity mapping
        from tools.sentence.sentence import Sentence as ToolSentence
        tool_sentences = [
            ToolSentence(text=s.text, index=s.index, id=s.id)
            for s in sentences
        ]
        
        joined = join_sentences([s.text for s in tool_sentences])
        entity_mapper.map_to_sentences(doc, tool_sentences, joined)
        
        # Update sentences with entities
        entity_repo = InMemoryEntityRepository()
        for tool_sent in tool_sentences:
            db_sentence = self.sentence_repo.get(tool_sent.id)
            if db_sentence:
                entities_data = [
                    {
                        "id": ent.id,
                        "name": ent.name,
                        "label": ent.label,
                        "ref_short": ent.ref_short,
                        "start": ent.start,
                        "end": ent.end,
                        "entity_type": ent.entity_type,
                    }
                    for ent in tool_sent.entities
                ]
                self.sentence_repo.update(db_sentence.id, {"entities": entities_data})
        
        self.job_repo.update(job.id, {"progress": 0.7})
    
    async def _run_triple_generation(self, document: Document, job: ProcessingJob):
        """Run triple generation step."""
        job.stage = "triple_generation"
        self.job_repo.update(job.id, {"stage": "triple_generation", "progress": 0.8})
        
        # Get sentences with entities
        sentences = self.sentence_repo.list(filters={"document_id": document.id})
        
        # Create entity repository
        entity_repo = InMemoryEntityRepository()
        for sentence in sentences:
            for ent_data in sentence.entities:
                from tools.sentence.entity import Entity
                entity = Entity(
                    name=ent_data["name"],
                    label=ent_data["label"],
                    ref_short=ent_data["ref_short"],
                    start=ent_data["start"],
                    end=ent_data["end"],
                    entity_type=ent_data.get("entity_type"),
                    id=ent_data["id"],
                )
                entity_repo.save(entity)
        
        # Generate triples
        db_triples = []
        node_gen = self.node_generator
        
        for sentence in sentences:
            # Convert entities to LLM inventory format
            entities = []
            for ent_data in sentence.entities:
                entities.append({
                    "id": ent_data["id"],
                    "label": ent_data["label"],
                    "span": [ent_data["start"], ent_data["end"]],
                    "text": ent_data["name"],
                    "ref_short": ent_data["ref_short"],
                })
            
            # Generate triples
            graph_triples = node_gen.run(sentence.text, entities, entity_repo)
            
            # Convert to database triples
            for triple in graph_triples:
                db_triple = Triple(
                    document_id=document.id,
                    sentence_id=sentence.id,
                    head_id=triple.head.ref_short or triple.head.id,
                    head_name=triple.head.name,
                    head_type=triple.head.label,
                    relation=triple.relation,
                    tail_id=triple.tail.ref_short or triple.tail.id,
                    tail_name=triple.tail.name,
                    tail_type=triple.tail.label,
                )
                db_triples.append(db_triple)
        
        # Save triples
        self.triple_repo.create_batch(db_triples)
        
        self.job_repo.update(job.id, {"progress": 0.95})

