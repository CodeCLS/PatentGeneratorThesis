# Updated HybridClaimClusterer with forbidden_node_types parameter
# This prevents nodes with certain entity types from appearing in independent claims

clusterer5 = HybridClaimClusterer(
    clusterers=[
        InDegreeClaimClusterer(
            in_degree_threshold=0.7, 
            cluster_radius=1, 
            min_cluster_size=4, 
            max_dependent_cluster_size=30,
            forbidden_node_types={"UNKNOWN", "PARAMETER", "MEASUREMENT", "UNCLASSIFIED_ENTITY", "MATERIAL", "PRIOR_ART", "CLAIM_ELEMENT", "CONDITION", "FIGURE_REF"},
        ),
        EntityTypeClaimClusterer(
            fundamental_types={"INVENTION"}, 
            cluster_radius=2, 
            min_cluster_size=4, 
            max_dependent_cluster_size=30,
            forbidden_node_types={"UNKNOWN", "PARAMETER", "MEASUREMENT", "UNCLASSIFIED_ENTITY", "MATERIAL", "PRIOR_ART", "CLAIM_ELEMENT", "CONDITION", "FIGURE_REF"},
        ),
        CentralityClaimClusterer(
            centrality_type="betweenness",
            cluster_radius=2,
            min_cluster_size=4,
            max_clusters=3,
            max_dependent_cluster_size=50,
            forbidden_node_types={"UNKNOWN", "PARAMETER", "MEASUREMENT", "UNCLASSIFIED_ENTITY", "MATERIAL", "PRIOR_ART", "CLAIM_ELEMENT", "CONDITION", "FIGURE_REF"},
        ),
    ],
    weights=[1.0, 1.5, 1.0],
    min_votes=2,
    min_cluster_size=5,
    max_clusters=3,
    max_dependent_cluster_size=30,
    forbidden_head_types={"UNKNOWN", "PARAMETER", "MEASUREMENT", "UNCLASSIFIED_ENTITY", "MATERIAL", "PRIOR_ART", "CLAIM_ELEMENT", "CONDITION", "FIGURE_REF"},
    forbidden_node_types={"UNKNOWN", "PARAMETER", "MEASUREMENT", "UNCLASSIFIED_ENTITY", "MATERIAL", "PRIOR_ART", "CLAIM_ELEMENT", "CONDITION", "FIGURE_REF"},
)

