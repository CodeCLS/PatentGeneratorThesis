"""
Pipeline manager that reproduces the Main.ipynb pipeline in an OOP form.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

from kg.pipeline_builder import PipelineBuilder
from kg.entity_mapper import EntityMapper, join_sentences
from kg.formatting_manager import FormattingManager
from kg.ParallelTripleGenerator import ParallelTripleGenerator
from tools.sentence.sentence import Sentence
from tools.sentence.entity import InMemoryEntityRepository
from tools.sentence.sentence_classifier import SentenceClassifier
from tools.graph.faiss_merger import FAISSEdgeMerger
from tools.graph.relation_simplifier import RelationSimplifier
from tools.graph.visualizer import GraphVisualizer


@dataclass
class PipelineResult:
    graph: Any
    triples: List[Any]
    id_to_name: Dict[str, str]
    sentence_split: List[Any]
    merge_stats: Dict[str, Any]


class PipelineManager:
    """
    Runs the same pipeline as Main.ipynb:
    - Retrieve invention-related sentences
    - Split sentences
    - Filter informative sentences
    - Run spaCy pipeline + entity mapping
    - Build entity repo
    - Generate triples
    - Merge similar relations
    - Simplify relations
    - Build graph + id_to_name map
    """

    def __init__(
        self,
        *,
        use_gpu: bool = True,
        cuda_memory_fraction: float = 0.7,
        torch_num_threads: int = 4,
        torch_num_interop_threads: int = 1,
        formatting_workers: int = 8,
        formatting_split_workers: int = 12,
        classifier_model_path: str = "training/info/done/hf/sentence_classifier_model",
        classifier_batch_size: int = 32,
        triple_max_workers: int = 10,
        triple_rate_limit_per_minute: int = 900,
        merge_sim_threshold: float = 0.8,
        merge_embed_dim: int = 256,
        merge_ngram: int = 3,
        merge_keep: str = "shortest",
        simplify_max_relation_length: int = 4,
        simplify_workers: int = 8,
        split_chunk_size: int = 1000,
        keep_labels: Optional[List[str]] = None,
    ) -> None:
        if use_gpu and torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(cuda_memory_fraction)
            try:
                torch.set_num_threads(torch_num_threads)
            except RuntimeError as e:
                print(f"[Pipeline] Warning: unable to set torch num threads: {e}")
            try:
                torch.set_num_interop_threads(torch_num_interop_threads)
            except RuntimeError as e:
                print(f"[Pipeline] Warning: unable to set torch interop threads: {e}")

        base_dir = Path(__file__).resolve().parents[1]
        if not Path(classifier_model_path).is_absolute():
            classifier_model_path = str((base_dir / classifier_model_path).resolve())

        self.pipeline_builder = PipelineBuilder(use_gpu=use_gpu)
        self.entity_mapper = EntityMapper(sentence_cls=Sentence)
        self.formatting_manager = FormattingManager(
            num_workers=formatting_workers,
            split_workers=formatting_split_workers,
        )
        self.classifier = SentenceClassifier(
            model_path=classifier_model_path,
            batch_size=classifier_batch_size,
            use_gpu=use_gpu,
        )
        self.triple_max_workers = triple_max_workers
        self.triple_rate_limit_per_minute = triple_rate_limit_per_minute
        self.merger = FAISSEdgeMerger(
            sim_threshold=merge_sim_threshold,
            embed_dim=merge_embed_dim,
            ngram=merge_ngram,
            keep=merge_keep,
        )
        self.simplifier = RelationSimplifier(
            max_relation_length=simplify_max_relation_length,
            verbose=True,
            num_workers=simplify_workers,
        )
        self.visualizer = GraphVisualizer()
        self.split_chunk_size = split_chunk_size
        self.keep_labels = keep_labels or ["INFORMATIVE"]

    def run(
        self,
        text: str,
        *,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> PipelineResult:
        if not text or not text.strip():
            raise ValueError("Pipeline requires non-empty text input.")

        def emit(stage: str, message: str, progress: int) -> None:
            print(f"[Pipeline] {stage}: {message} ({progress}%)")
            if progress_callback:
                progress_callback({
                    "stage": stage,
                    "message": message,
                    "progress": progress,
                })

        emit("ingest", "Preparing input text", 5)
        sentences = self.formatting_manager.retrieveContent(text, chunk_size=self.split_chunk_size)
        emit("sentence_split", "Splitting sentences", 15)
        split_sentences = self.formatting_manager.split(sentences)

        emit("informative_filter", "Filtering informative sentences", 25)
        sentence_split = self.classifier.filter_informative(
            split_sentences,
            keep_labels=self.keep_labels,
        )

        emit("ner", "Running NER pipeline", 40)
        joined = join_sentences(sentence_split, sep=" ")
        doc = self.pipeline_builder.nlp(joined.text)
        emit("coref", "Mapping entities / coreference", 50)
        self.entity_mapper.map_to_sentences(doc, sentence_split, joined)

        all_entities: List[Any] = []
        for sentence in sentence_split:
            all_entities.extend(sentence.entities)

        repo = InMemoryEntityRepository()
        for entity in all_entities:
            repo.save(entity)

        generator = ParallelTripleGenerator(
            repo=repo,
            max_workers=self.triple_max_workers,
            rate_limit_per_minute=self.triple_rate_limit_per_minute,
            verbose=True,
        )
        emit("triple_generation", "Generating triples", 65)
        triples = generator.generate(sentence_split)

        emit("merge_relations", "Cleaning graph: merging relations", 80)
        triples, merge_stats = self.merger.merge_relations(triples)

        emit("simplify_relations", "Cleaning graph: simplifying relations", 88)
        triples = self.simplifier.simplify(triples)

        emit("graph_build", "Building graph", 95)
        id_to_name = self.visualizer.build_id_to_name_map(sentence_split)
        graph = self.visualizer.build_graph(triples)

        emit("complete", "Pipeline complete", 100)
        return PipelineResult(
            graph=graph,
            triples=triples,
            id_to_name=id_to_name,
            sentence_split=sentence_split,
            merge_stats=merge_stats,
        )
