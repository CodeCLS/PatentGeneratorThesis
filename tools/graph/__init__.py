"""
Graph processing tools for knowledge graphs.
"""

# Fix Jinja2 compatibility - patch before any Flask imports
import jinja2
if not hasattr(jinja2, 'escape'):
    try:
        from markupsafe import escape
        jinja2.escape = escape
    except ImportError:
        def escape(s):
            if s is None:
                return ''
            s = str(s)
            return (s.replace('&', '&amp;')
                    .replace('<', '&lt;')
                    .replace('>', '&gt;')
                    .replace('"', '&quot;')
                    .replace("'", '&#x27;'))
        jinja2.escape = escape

from tools.graph.llm_relation_filter import LLMRelationFilter
from tools.graph.small_cluster_filter import SmallClusterFilter
from tools.graph.patent_claim_generator import PatentClaimGenerator
from tools.graph.triple_printer import print_triples_vertical, print_triples_compact
from tools.graph.assertion_agent import AssertionAgent, Assertion
from tools.graph.claim_concept_agent import ClaimConceptAgent, ClaimConcept
from tools.graph.claim_extractor import ClaimExtractor, ClaimBundle, AssertionInfo
from tools.graph.claim_drafting_agent import ClaimDraftingAgent, DraftedClaim
from tools.graph.claim_printer import print_claims, print_claims_compact, print_claims_grouped
from tools.graph.graph_rag import GraphRAG, RetrievedContext
from tools.graph.kg_gen_converter import kg_gen_graph_to_triples, build_id_to_name_map
from tools.graph.graph_validator import GraphValidator, Question, Response, Action, ActionType

__all__ = [
    "LLMRelationFilter",
    "SmallClusterFilter",
    "PatentClaimGenerator",
    "print_triples_vertical",
    "print_triples_compact",
    "AssertionAgent",
    "Assertion",
    "ClaimConceptAgent",
    "ClaimConcept",
    "ClaimExtractor",
    "ClaimBundle",
    "AssertionInfo",
    "ClaimDraftingAgent",
    "DraftedClaim",
    "print_claims",
    "print_claims_compact",
    "print_claims_grouped",
    "GraphRAG",
    "RetrievedContext",
    "kg_gen_graph_to_triples",
    "build_id_to_name_map",
    "GraphValidator",
    "Question",
    "Response",
    "Action",
    "ActionType",
]

