
import argparse
import json
import sys
from pathlib import Path
from typing import Optional


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from kg.formatting.formatting_manager import FormattingManager
from tools.sentence.sentence_classifier import SentenceClassifier
from kg.cleaning.referencing import PipelineBuilder, EntityMapper
from tools.sentence.entity import Entity, InMemoryEntityRepository
from kg.generating_kg.generating.ParallelTripleGenerator import ParallelTripleGenerator
from tools.graph.faiss_merger import FAISSEdgeMerger
from tools.graph.relation_simplifier import RelationSimplifier
from tools.graph.visualizer import GraphVisualizer
from tools.graph.assertion_agent import AssertionAgent
from tools.graph.claim_concept_agent import ClaimConceptAgent
from tools.graph.claim_extractor import ClaimExtractor
from tools.graph.claim_drafting_agent import ClaimDraftingAgent
from tools.graph.kg_gen_converter import build_id_to_name_map
from PatentProvider import PatentProvider


def join_sentences(sentences, sep=" "):
    from dataclasses import dataclass
    from typing import List
    
    @dataclass(frozen=True)
    class JoinedText:
        text: str
        starts: List[int]
    
    parts = []
    starts = []
    cur = 0

    for i, s in enumerate(sentences):
        starts.append(cur)
        parts.append(s.text)
        cur += len(s.text)

        if i < len(sentences) - 1:
            parts.append(sep)
            cur += len(sep)

    return JoinedText("".join(parts), starts)


class PatentClaimPipeline:
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        

        self.formatter = FormattingManager(
            num_workers=self.config.get('num_workers', 8),
            split_workers=self.config.get('split_workers', 12)
        )
        
        self.classifier = SentenceClassifier(
            model_path=self.config.get('classifier_model_path', "training/info/done/hf/sentence_classifier_model"),
            batch_size=self.config.get('batch_size', 32),
            use_gpu=self.config.get('use_gpu', True)
        )
        
        self.pipeline_builder = PipelineBuilder()
        self.entity_mapper = EntityMapper()
        
        self.triple_generator = ParallelTripleGenerator(
            max_workers=self.config.get('triple_workers', 10),
            rate_limit_per_minute=self.config.get('rate_limit', 900),
            verbose=self.config.get('verbose', True)
        )
        
        self.merger = FAISSEdgeMerger(
            sim_threshold=self.config.get('sim_threshold', 0.8),
            embed_dim=self.config.get('embed_dim', 256),
            ngram=self.config.get('ngram', 3),
            keep=self.config.get('keep', "shortest")
        )
        
        self.simplifier = RelationSimplifier(
            max_relation_length=self.config.get('max_relation_length', 4),
            verbose=self.config.get('verbose', True)
        )
        
        self.visualizer = GraphVisualizer()
        
    def run(self, patent_text: str, patent_id: Optional[str] = None) -> dict:
        sentences = self.formatter.retrieveContent(patent_text, chunk_size=1000)
        split_sentences = self.formatter.split(sentences)
        
        sentence_split = self.classifier.filter_informative(
            split_sentences, 
            keep_labels=["INFORMATIVE"]
        )
        
        joined = join_sentences(sentence_split, sep=" ")
        doc = self.pipeline_builder.nlp(joined.text)
        clusters = self.entity_mapper.map_to_sentences(doc, sentence_split, joined)
        

        all_entities = []
        for sentence in sentence_split:
            all_entities.extend(sentence.entities)
        

        repo = InMemoryEntityRepository()
        for entity in all_entities:
            repo.save(entity)
        
        triples = self.triple_generator.generate(sentence_split)
        
        graph = self.visualizer.build_graph(triples, deduplicate=True)
        

        triples, merge_stats = self.merger.merge_relations(triples)
        

        triples = self.simplifier.simplify(triples)
        

        graph = self.visualizer.build_graph(triples, deduplicate=True)
        

        id_to_name = build_id_to_name_map(triples)
        

        assertion_agent = AssertionAgent()
        graph = assertion_agent.run(graph, triples=triples)
        

        claim_concept_agent = ClaimConceptAgent()
        graph = claim_concept_agent.run(
            graph,
            num_independent=self.config.get('num_independent', 3),
            num_dependent_per_independent=self.config.get('num_dependent_per_independent', 4)
        )
        

        extractor = ClaimExtractor(id_to_name=id_to_name)
        claim_bundles = extractor.extract(graph)
        

        drafting_agent = ClaimDraftingAgent()
        claims = drafting_agent.draft(claim_bundles, patent_description=patent_text)
        
        return {
            "patent_id": patent_id,
            "sentences": sentence_split,
            "entities": all_entities,
            "triples": triples,
            "graph": graph,
            "claims": claims,
            "id_to_name": id_to_name
        }


def main():
    parser = argparse.ArgumentParser(description="Generate patent claims from patent text")
    parser.add_argument("--patent-id", type=str, help="Patent ID to fetch")
    parser.add_argument("--text-file", type=str, help="Path to text file")
    parser.add_argument("--text", type=str, help="Patent text directly")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--num-independent", type=int, default=3, help="Number of independent claims")
    parser.add_argument("--num-dependent", type=int, default=4, help="Number of dependent claims per independent")
    
    args = parser.parse_args()
    

    if args.patent_id:
        provider = PatentProvider()
        patent_text = provider.getDescription(args.patent_id)
        patent_id = args.patent_id
    elif args.text_file:
        patent_text = Path(args.text_file).read_text()
        patent_id = Path(args.text_file).stem
    elif args.text:
        patent_text = args.text
        patent_id = None
    else:
        parser.error("Must provide --patent-id, --text-file, or --text")
    

    config = {
        'num_independent': args.num_independent,
        'num_dependent_per_independent': args.num_dependent
    }
    pipeline = PatentClaimPipeline(config=config)
    results = pipeline.run(patent_text, patent_id=patent_id)
    

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    

    claims_file = output_dir / "claims.json"
    claims_data = [
        {
            "claim_number": c.claim_number,
            "claim_text": c.claim_text,
            "claim_type": getattr(c, 'claim_type', 'unknown'),
            "parent_claim_number": getattr(c, 'parent_claim_number', None)
        }
        for c in results["claims"]
    ]
    claims_file.write_text(json.dumps(claims_data, indent=2))
    

    graph_file = output_dir / "graph.html"
    results["graph"] = pipeline.visualizer.build_graph(results["triples"], deduplicate=True)
    pipeline.visualizer.visualize_pyvis(
        results["graph"],
        out_file=str(graph_file),
        id_to_name=results["id_to_name"]
    )
    
    print(f"\n✅ Pipeline complete! Results saved to {output_dir}")
    print(f"📄 Generated {len(results['claims'])} claims")
    print(f"📊 Graph has {results['graph'].number_of_nodes()} nodes and {results['graph'].number_of_edges()} edges")
    print(f"📁 Claims saved to: {claims_file}")
    print(f"📁 Graph visualization saved to: {graph_file}")


if __name__ == "__main__":
    main()

