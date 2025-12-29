# Example of how to use the forbidden_head_types parameter in HybridClaimClusterer

from tools.graph.claim_clusterers import HybridClaimClusterer, InDegreeClaimClusterer, EntityTypeClaimClusterer, PageRankClaimClusterer

# Example: Filter out triples where the head node is MATERIAL or PRIOR_ART
clusterer5 = HybridClaimClusterer(
    clusterers=[
        InDegreeClaimClusterer(in_degree_threshold=0.95, cluster_radius=3, min_cluster_size=5, max_dependent_cluster_size=30),
        EntityTypeClaimClusterer(fundamental_types={"INVENTION"}, cluster_radius=3, min_cluster_size=5, max_dependent_cluster_size=30),
        PageRankClaimClusterer(pagerank_threshold=0.95, cluster_radius=3, min_cluster_size=5, max_dependent_cluster_size=30),
    ],
    weights=[1.0, 1.5, 1.0],
    min_votes=2,
    min_cluster_size=5,
    max_clusters=3,
    max_dependent_cluster_size=30,
    forbidden_head_types={"MATERIAL", "PRIOR_ART"},  # <-- Add this parameter
)

