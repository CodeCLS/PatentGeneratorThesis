# Claim Clustering Guide

This guide explains how to use the claim clustering system to organize knowledge graph triples into patent claims.

## Overview

The claim clustering system identifies **independent claims** (fundamental, core features) and **dependent claims** (secondary features) from a knowledge graph. This is useful for patent analysis where:

- **Claim 1** (and Claims 2-3 if multiple independent claims exist) = Independent claims containing fundamental triples
- **Claim 4+** = Dependent claims containing secondary features

## Architecture

### Base Class: `BaseClaimClusterer`

All clusterers inherit from `BaseClaimClusterer`, which defines the interface:

```python
class BaseClaimClusterer(ABC):
    @abstractmethod
    def cluster(self, G: nx.MultiDiGraph) -> List[ClaimCluster]:
        """Cluster the graph into independent and dependent claims."""
        pass
```

### Output: `ClaimCluster`

Each cluster is represented as a `ClaimCluster` object:

```python
@dataclass
class ClaimCluster:
    cluster_id: int                    # Unique identifier
    nodes: Set[str]                     # Node IDs in this cluster
    edges: Set[Tuple[str, str, str]]    # Edges (head, tail, relation)
    claim_type: str                     # "independent" or "dependent"
    priority: int                       # Lower = more important
    metadata: Dict[str, Any]            # Additional info
```

## Available Clusterers

### 1. InDegreeClaimClusterer

**Strategy**: Nodes with high in-degree (many incoming edges) are fundamental.

**How it works**:
- Calculates in-degree for each node
- Identifies top nodes by in-degree (configurable threshold)
- Builds clusters around these fundamental nodes
- Remaining edges form dependent claims

**Use case**: When you want to identify nodes that are referenced by many other nodes (central concepts).

```python
from tools.graph import InDegreeClaimClusterer, GraphVisualizer

# Build graph from triples
visualizer = GraphVisualizer()
G = visualizer.build_graph(triples)

# Cluster using in-degree
clusterer = InDegreeClaimClusterer(
    in_degree_threshold=0.7,  # Top 70% by in-degree are fundamental
    cluster_radius=2,          # Include nodes within 2 hops
    min_cluster_size=1,        # Minimum edges per cluster
)
clusters = clusterer.cluster(G)

# Separate independent and dependent claims
independent = [c for c in clusters if c.claim_type == "independent"]
dependent = [c for c in clusters if c.claim_type == "dependent"]

print(f"Found {len(independent)} independent claims and {len(dependent)} dependent claims")
```

### 2. EntityTypeClaimClusterer

**Strategy**: Nodes labeled "INVENTION" (or other specified types) are fundamental.

**How it works**:
- Identifies nodes with fundamental entity types (default: "INVENTION")
- Builds clusters around these nodes
- Other clusters are dependent claims

**Use case**: When entity types already indicate importance (e.g., INVENTION nodes are always fundamental).

```python
from tools.graph import EntityTypeClaimClusterer

clusterer = EntityTypeClaimClusterer(
    fundamental_types={"INVENTION", "SUBSYSTEM"},  # Types considered fundamental
    cluster_radius=2,
)
clusters = clusterer.cluster(G)
```

### 3. CentralityClaimClusterer

**Strategy**: Uses network centrality measures (betweenness, closeness, eigenvector) to identify important nodes.

**How it works**:
- Calculates centrality for each node
- High centrality nodes are fundamental
- Builds clusters around these nodes

**Use case**: When you want to use graph theory metrics to identify important nodes.

```python
from tools.graph import CentralityClaimClusterer

clusterer = CentralityClaimClusterer(
    centrality_type="betweenness",  # or "closeness", "eigenvector"
    centrality_threshold=0.7,
    cluster_radius=2,
)
clusters = clusterer.cluster(G)
```

### 4. HierarchicalClaimClusterer

**Strategy**: Builds clusters hierarchically from root nodes (nodes with no incoming edges).

**How it works**:
- Identifies root nodes (no incoming edges) or fundamental type nodes
- Builds clusters hierarchically from roots
- First level = independent claims, deeper levels = dependent claims

**Use case**: When the graph has a clear hierarchical structure.

```python
from tools.graph import HierarchicalClaimClusterer

clusterer = HierarchicalClaimClusterer(
    max_depth=3,  # Maximum depth to traverse
    fundamental_types={"INVENTION"},
)
clusters = clusterer.cluster(G)
```

### 5. PageRankClaimClusterer

**Strategy**: Uses PageRank algorithm to identify important nodes.

**How it works**:
- Calculates PageRank for each node
- High PageRank nodes are fundamental
- Builds clusters around these nodes

**Use case**: When you want to use PageRank (similar to Google's algorithm) to identify important nodes.

```python
from tools.graph import PageRankClaimClusterer

clusterer = PageRankClaimClusterer(
    pagerank_threshold=0.7,
    cluster_radius=2,
    damping=0.85,  # PageRank damping factor
)
clusters = clusterer.cluster(G)
```

### 6. HybridClaimClusterer

**Strategy**: Combines multiple clusterers using voting/weighting.

**How it works**:
- Runs multiple clusterers
- Uses weighted voting to determine which edges are in independent claims
- Edges identified as independent by multiple methods are more likely independent

**Use case**: When you want the most robust clustering by combining multiple approaches.

```python
from tools.graph import (
    HybridClaimClusterer,
    InDegreeClaimClusterer,
    EntityTypeClaimClusterer,
    PageRankClaimClusterer,
)

clusterer = HybridClaimClusterer(
    clusterers=[
        InDegreeClaimClusterer(),
        EntityTypeClaimClusterer(),
        PageRankClaimClusterer(),
    ],
    weights=[1.0, 1.5, 1.0],  # EntityType gets higher weight
    min_votes=2,  # At least 2 clusterers must agree
)
clusters = clusterer.cluster(G)
```

## Complete Example

```python
from tools.graph import (
    GraphVisualizer,
    InDegreeClaimClusterer,
    EntityTypeClaimClusterer,
    HybridClaimClusterer,
)

# 1. Build graph from triples
visualizer = GraphVisualizer()
G = visualizer.build_graph(triples)

# 2. Choose a clusterer
# Option A: Simple in-degree based
clusterer = InDegreeClaimClusterer(
    in_degree_threshold=0.7,
    cluster_radius=2,
)

# Option B: Entity type based
clusterer = EntityTypeClaimClusterer(
    fundamental_types={"INVENTION"},
    cluster_radius=2,
)

# Option C: Hybrid (recommended for best results)
clusterer = HybridClaimClusterer(
    clusterers=[
        InDegreeClaimClusterer(),
        EntityTypeClaimClusterer(),
    ],
    min_votes=2,
)

# 3. Cluster the graph
clusters = clusterer.cluster(G)

# 4. Analyze results
independent_claims = [c for c in clusters if c.claim_type == "independent"]
dependent_claims = [c for c in clusters if c.claim_type == "dependent"]

print(f"Independent Claims: {len(independent_claims)}")
for i, claim in enumerate(independent_claims, 1):
    print(f"  Claim {i}: {claim.size()} edges, {claim.node_count()} nodes")
    print(f"    Priority: {claim.priority}, Metadata: {claim.metadata}")

print(f"\nDependent Claims: {len(dependent_claims)}")
for i, claim in enumerate(dependent_claims, 1):
    print(f"  Dependent Claim {i}: {claim.size()} edges, {claim.node_count()} nodes")

# 5. Visualize clusters (optional)
# Add cluster_id to edges for visualization
for cluster in clusters:
    for u, v, rel in cluster.edges:
        if G.has_edge(u, v):
            # Update edge attributes with cluster info
            for key in G[u][v]:
                G.edges[u, v, key]["cluster_id"] = cluster.cluster_id
                G.edges[u, v, key]["claim_type"] = cluster.claim_type

# Visualize
visualizer.visualize_pyvis(G, out_file="claims_graph.html", id_to_name=id_to_name)
```

## Choosing the Right Clusterer

| Clusterer | Best For | Pros | Cons |
|-----------|----------|------|------|
| **InDegreeClaimClusterer** | Graphs where central concepts are referenced often | Simple, fast, intuitive | May miss important isolated nodes |
| **EntityTypeClaimClusterer** | When entity types indicate importance | Uses semantic information | Requires good entity labeling |
| **CentralityClaimClusterer** | Complex graphs with multiple importance metrics | Uses graph theory | Slower, may be overkill for simple graphs |
| **HierarchicalClaimClusterer** | Hierarchical structures (e.g., systems with subsystems) | Preserves hierarchy | May not work well for flat graphs |
| **PageRankClaimClusterer** | Web-like graphs with many connections | Proven algorithm | May favor highly connected nodes too much |
| **HybridClaimClusterer** | Production use, when you want robustness | Most reliable | Slower, more complex |

## Tips

1. **Start with EntityTypeClaimClusterer** if your entities are well-labeled with "INVENTION" types.

2. **Use HybridClaimClusterer** for production to combine multiple strategies.

3. **Adjust `cluster_radius`** to control cluster size:
   - Smaller radius (1-2) = tighter, more focused clusters
   - Larger radius (3-4) = broader clusters

4. **Filter small clusters** using `min_cluster_size` to remove noise.

5. **Limit cluster count** using `max_clusters` if you have too many small clusters.

6. **Check metadata** in `ClaimCluster.metadata` for debugging and analysis.

## Integration with Existing Code

The claim clusterers work seamlessly with the existing graph tools:

```python
from tools.graph import GraphVisualizer, InDegreeClaimClusterer
from tools.graph.cluster_manager import ClusterManager

# Your existing code
visualizer = GraphVisualizer()
G = visualizer.build_graph(triples)

# Option 1: Use existing ClusterManager for rule-based clustering
cluster_manager = ClusterManager(G)
rule_clusters = cluster_manager.create_rule_based_clusters()

# Option 2: Use new ClaimClusterers for claim-based clustering
claim_clusterer = InDegreeClaimClusterer()
claim_clusters = claim_clusterer.cluster(G)

# Both can be used together or separately
```

