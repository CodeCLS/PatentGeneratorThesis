# ============================================================================
# PATENT CLAIM GENERATION
# ============================================================================
# Generate formal patent lawyer claims from the clustered claims
# Each cluster is processed independently to create a separate claim

from tools.graph.patent_claim_generator import PatentClaimGenerator

# Ensure we have claim clusters
if 'claim_clusters' not in globals():
    if 'clusters5' in globals():
        claim_clusters = clusters5
    else:
        raise ValueError("No claim clusters available. Please run the clustering cell first.")

# Ensure we have id_to_name map
if 'id_to_name' not in globals():
    if 'sentence_split' in globals() and 'visualizer' in globals():
        id_to_name = visualizer.build_id_to_name_map(sentence_split)
    else:
        id_to_name = {}

# Build id_to_label map to identify INVENTION entities
if 'sentence_split' in globals() and 'visualizer' in globals():
    id_to_label = visualizer.build_id_to_label_map(sentence_split)
else:
    id_to_label = {}

# Initialize claim generator
claim_generator = PatentClaimGenerator(style="formal")

# Generate patent claims from clusters (silently - no print statements)
# Pass id_to_label so INVENTION entities are included in context
patent_claims = claim_generator.generate_claims(
    clusters=claim_clusters,
    id_to_name=id_to_name,
    id_to_label=id_to_label,
)

# Display claims in patent format (numbered only, no titles - like a real patent)
for claim_info in patent_claims:
    claim_num = claim_info["claim_number"]
    claim_text = claim_info["claim_text"]
    print(f"{claim_num}. {claim_text}")
    print()
