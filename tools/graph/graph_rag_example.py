"""
Example usage of GraphRAG.

This demonstrates how to use the GraphRAG system to retrieve relevant context
from knowledge graphs.

Note: ClaimDraftingAgent, ClaimExtractor, and related classes have been removed.
You can rebuild these from scratch as needed.
"""
from tools.graph.graph_rag import GraphRAG
from tools.graph.visualizer import GraphVisualizer
from tools.graph.Triple import Triple
import networkx as nx


def example_rag_retrieval(
    G: nx.MultiDiGraph,
    triples: list[Triple],
    id_to_name: dict[str, str],
    query: str = "water tank and bubble generation",
):
    """
    Example: Use GraphRAG to retrieve context for a specific query.
    
    Args:
        G: NetworkX graph
        triples: Original triples
        id_to_name: Mapping from entity ID to display name
        query: Text query to retrieve context for
    """
    print(f"🔍 Retrieving graph context for query: '{query}'...")
    
    # Initialize GraphRAG
    graph_rag = GraphRAG(
        G=G,
        triples=triples,
        id_to_name=id_to_name,
    )
    
    # Retrieve context by query
    context = graph_rag.retrieve_by_query(
        query=query,
        max_entities=15,
        max_triples=20,
        use_semantic_search=True,
    )
    
    print(f"✅ Retrieved context:")
    print(f"   - {len(context.relevant_entities)} relevant entities")
    print(f"   - {len(context.relevant_triples)} relevant triples")
    print(f"   - {len(context.key_relationships)} key relationships")
    print(f"   - {len(context.entity_paths)} entity paths")
    
    # Display formatted context
    print("\n📄 Retrieved Context:")
    print(graph_rag.format_context_for_prompt(context))
    
    return context


def example_rag_for_bundle(
    G: nx.MultiDiGraph,
    triples: list[Triple],
    id_to_name: dict[str, str],
    assertion_ids: list[str],
):
    """
    Example: Use GraphRAG to retrieve context for a specific claim bundle.
    
    Args:
        G: NetworkX graph
        triples: Original triples
        id_to_name: Mapping from entity ID to display name
        assertion_ids: List of assertion IDs in the bundle
    """
    print(f"🔍 Retrieving graph context for claim bundle with {len(assertion_ids)} assertions...")
    
    # Initialize GraphRAG
    graph_rag = GraphRAG(
        G=G,
        triples=triples,
        id_to_name=id_to_name,
    )
    
    # Retrieve context for the bundle
    context = graph_rag.retrieve_for_claim_bundle(
        assertion_ids=assertion_ids,
        max_entities=20,
        max_triples=30,
        max_depth=2,
    )
    
    print(f"✅ Retrieved context:")
    print(f"   - {len(context.relevant_entities)} relevant entities")
    print(f"   - {len(context.relevant_triples)} relevant triples")
    print(f"   - {len(context.key_relationships)} key relationships")
    print(f"   - {len(context.entity_paths)} entity paths")
    
    # Display formatted context
    print("\n📄 Retrieved Context:")
    print(graph_rag.format_context_for_prompt(context))
    
    return context


if __name__ == "__main__":
    # Example usage would go here
    # This file is meant to be imported and used in notebooks
    print("GraphRAG example module loaded.")
    print("Import this module and use the example functions in your notebook.")

