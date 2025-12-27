"""
Graph visualization and clustering endpoints.
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
from api.schemas.triples import TripleListResponse
from api.database.repository import LocalTripleRepository, LocalDocumentRepository
from tools.graph.visualizer import GraphVisualizer
from tools.graph.faiss_merger import FAISSEdgeMerger
from tools.graph.cluster_manager import ClusterManager
import networkx as nx

router = APIRouter(prefix="/graph", tags=["graph"])


def get_triple_repo() -> LocalTripleRepository:
    """Dependency to get triple repository."""
    from api.main import get_app_state
    state = get_app_state()
    if state.triple_repo is None:
        raise RuntimeError("Triple repository not initialized")
    return state.triple_repo


def get_document_repo() -> LocalDocumentRepository:
    """Dependency to get document repository."""
    from api.main import get_app_state
    state = get_app_state()
    if state.document_repo is None:
        raise RuntimeError("Document repository not initialized")
    return state.document_repo


@router.post("/visualize/{document_id}")
async def visualize_graph(
    document_id: str,
    output_file: Optional[str] = "graph.html",
    merge_relations: bool = False,
    sim_threshold: float = 0.85,
):
    """Generate a graph visualization for a document."""
    triple_repo = get_triple_repo()
    document_repo = get_document_repo()
    
    # Verify document exists
    document = document_repo.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get triples
    triples = triple_repo.list(filters={"document_id": document_id})
    if not triples:
        raise HTTPException(status_code=404, detail="No triples found for document")
    
    # Convert to graph triples
    from tools.graph.Triple import Triple as GraphTriple
    from tools.sentence.entity import Entity
    graph_triples = []
    for t in triples:
        head_entity = Entity(
            name=t.head_name,
            label=t.head_type or "UNKNOWN",
            ref_short=t.head_id,
            id=t.head_id,
        )
        tail_entity = Entity(
            name=t.tail_name,
            label=t.tail_type or "UNKNOWN",
            ref_short=t.tail_id,
            id=t.tail_id,
        )
        graph_triple = GraphTriple(
            head=head_entity,
            relation=t.relation,
            tail=tail_entity,
        )
        graph_triples.append(graph_triple)
    
    # Merge relations if requested
    if merge_relations:
        merger = FAISSEdgeMerger(sim_threshold=sim_threshold)
        graph_triples, stats = merger.merge_relations(graph_triples)
    
    # Build and visualize graph
    visualizer = GraphVisualizer()
    G = visualizer.build_graph(graph_triples)
    visualizer.visualize_pyvis(G, out_file=output_file)
    
    return {
        "message": f"Graph visualization saved to {output_file}",
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
    }


@router.post("/cluster/{document_id}")
async def cluster_graph(
    document_id: str,
    method: str = "rule_based",  # "rule_based" or "semantic"
    min_delete_edges: int = 10,
    min_merge_edges: int = 30,
):
    """Cluster the graph for a document."""
    triple_repo = get_triple_repo()
    
    # Get triples
    triples = triple_repo.list(filters={"document_id": document_id})
    if not triples:
        raise HTTPException(status_code=404, detail="No triples found for document")
    
    # Convert to graph triples and build graph
    from tools.graph.Triple import Triple as GraphTriple
    from tools.sentence.entity import Entity
    graph_triples = []
    for t in triples:
        head_entity = Entity(
            name=t.head_name,
            label=t.head_type or "UNKNOWN",
            ref_short=t.head_id,
            id=t.head_id,
        )
        tail_entity = Entity(
            name=t.tail_name,
            label=t.tail_type or "UNKNOWN",
            ref_short=t.tail_id,
            id=t.tail_id,
        )
        graph_triple = GraphTriple(
            head=head_entity,
            relation=t.relation,
            tail=tail_entity,
        )
        graph_triples.append(graph_triple)
    
    visualizer = GraphVisualizer()
    G = visualizer.build_graph(graph_triples)
    
    # Cluster
    cluster_manager = ClusterManager(G)
    if method == "rule_based":
        clusters = cluster_manager.create_rule_based_clusters()
        clusters = cluster_manager.postprocess_clusters(
            min_delete_edges=min_delete_edges,
            min_merge_edges=min_merge_edges,
        )
        cid_to_seedtype = cluster_manager.assign_edges_to_clusters(clusters)
    else:
        # Semantic clustering would require encoder
        raise HTTPException(
            status_code=400,
            detail="Semantic clustering not yet implemented via API"
        )
    
    # Update triples with cluster IDs
    for u, v, k, d in G.edges(keys=True, data=True):
        cluster_id = d.get("cluster_id")
        if cluster_id is not None:
            # Find corresponding triple and update
            for triple in triples:
                if (triple.head_id == u and triple.tail_id == v and
                    triple.relation == d.get("label", "")):
                    triple_repo.update(triple.id, {"cluster_id": cluster_id})
                    break
    
    return {
        "message": "Graph clustered successfully",
        "num_clusters": len(clusters),
        "clusters": [
            {
                "cluster_id": c["cluster_id"],
                "seed": c["seed"],
                "seed_type": c["seed_type"],
                "num_nodes": len(c["nodes"]),
            }
            for c in clusters
        ],
    }

