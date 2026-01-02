"""
Example usage of the claim generation pipeline:
1. AssertionAgent - adds assertion nodes
2. ClaimConceptAgent - bundles assertions into claim concepts
3. ClaimExtractor - extracts claim bundles
4. ClaimDraftingAgent - drafts patent claims
"""
from tools.graph.visualizer import GraphVisualizer
from tools.graph.assertion_agent import AssertionAgent
from tools.graph.claim_concept_agent import ClaimConceptAgent
from tools.graph.claim_extractor import ClaimExtractor
from tools.graph.claim_drafting_agent import ClaimDraftingAgent
from tools.graph.Triple import Triple
import networkx as nx


def run_claim_pipeline(
    triples: list[Triple],
    id_to_name: dict[str, str],
    num_independent: int = 3,
    num_dependent_per_independent: int = 2,
) -> list:
    """
    Run the complete claim generation pipeline.
    
    Args:
        triples: List of Triple objects from KG generation
        id_to_name: Mapping from entity ID to display name
        num_independent: Number of independent claims to create
        num_dependent_per_independent: Number of dependent claims per independent
        
    Returns:
        List of DraftedClaim objects
    """
    # Step 1: Build graph from triples
    print("📊 Building graph from triples...")
    visualizer = GraphVisualizer()
    G = visualizer.build_graph(triples, deduplicate=True)
    print(f"✅ Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Step 2: Add assertion nodes
    print("\n🔍 Adding assertion nodes...")
    assertion_agent = AssertionAgent()
    G = assertion_agent.run(G, triples=triples)
    
    # Step 3: Create claim concepts
    print("\n📋 Creating claim concepts...")
    claim_concept_agent = ClaimConceptAgent()
    G = claim_concept_agent.run(
        G,
        status_filter="CANDIDATE",
        num_independent=num_independent,
        num_dependent_per_independent=num_dependent_per_independent,
    )
    
    # Step 4: Extract claim bundles
    print("\n📦 Extracting claim bundles...")
    extractor = ClaimExtractor(id_to_name=id_to_name)
    claim_bundles = extractor.extract(G, status_filter="CANDIDATE")
    
    # Step 5: Draft claims
    print("\n✍️  Drafting patent claims...")
    drafting_agent = ClaimDraftingAgent()
    drafted_claims = drafting_agent.draft(claim_bundles)
    
    # Print results
    print("\n" + "="*80)
    print("FINAL DRAFTED CLAIMS:")
    print("="*80)
    for claim in drafted_claims:
        dep_text = f" (depends on claim {claim.parent_claim_number})" if claim.parent_claim_number else ""
        print(f"\n{claim.claim_number}. {claim.claim_text}{dep_text}")
    
    return drafted_claims

