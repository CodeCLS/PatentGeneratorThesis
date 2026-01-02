"""
ClaimExtractor: Extracts claim bundles from the graph for drafting.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
import networkx as nx

from tools.graph.claim_concept_agent import ClaimConcept
from tools.graph.assertion_agent import Assertion
from tools.graph.visualizer import GraphVisualizer


@dataclass
class AssertionInfo:
    """Information about an assertion for claim drafting."""
    assertion_id: str
    predicate: str
    subject_label: str
    subject_id: str
    object_label: Optional[str] = None
    object_id: Optional[str] = None
    value: Optional[str] = None  # Literal value if object is not an entity
    evidence_snippet: Optional[str] = None
    terminology: Dict[str, str] = field(default_factory=dict)  # Glossary terms
    category: Optional[str] = None  # INVENTIVE_MECHANISM, TECHNICAL_COMPONENT, etc.


@dataclass
class ClaimBundle:
    """A bundle of assertions ready for claim drafting."""
    claim_id: str
    type: str  # "independent" or "dependent"
    parent_claim_id: Optional[str] = None
    assertions: List[AssertionInfo] = field(default_factory=list)
    breadth: str = "MEDIUM"
    title: Optional[str] = None


class ClaimExtractor:
    """
    Extracts claim bundles from the graph for drafting.
    
    For each ClaimConcept node, collects all linked assertions and
    prepares them as ClaimBundle objects ready for LLM drafting.
    """
    
    def __init__(self, id_to_name: Optional[Dict[str, str]] = None):
        """
        Initialize the claim extractor.
        
        Args:
            id_to_name: Optional mapping from entity ID to display name
        """
        self.id_to_name = id_to_name or {}
        self.visualizer = GraphVisualizer()
    
    def extract(
        self,
        G: nx.MultiDiGraph,
        status_filter: str = "CANDIDATE",
        kind_filter: Optional[str] = None,
    ) -> List[ClaimBundle]:
        """
        Extract claim bundles from the graph.
        
        Args:
            G: Graph with ClaimConcept and Assertion nodes
            status_filter: Filter claim concepts by status
            kind_filter: Optional filter by kind (INDEPENDENT, DEPENDENT)
            
        Returns:
            List of ClaimBundle objects ready for drafting
        """
        bundles = []
        
        # Find all claim concept nodes
        claim_nodes = [
            (node_id, data)
            for node_id, data in G.nodes(data=True)
            if data.get("node_type") == "CLAIM_CONCEPT"
            and data.get("status", "CANDIDATE") == status_filter
            and (kind_filter is None or data.get("kind") == kind_filter)
        ]
        
        # Sort: independent first, then dependents
        independent_claims = [n for n in claim_nodes if n[1].get("kind") == "INDEPENDENT"]
        dependent_claims = [n for n in claim_nodes if n[1].get("kind") == "DEPENDENT"]
        
        claim_nodes = independent_claims + dependent_claims
        
        for claim_id, claim_data in claim_nodes:
            assertion_ids = claim_data.get("assertion_ids", [])
            claim_type = claim_data.get("kind", "INDEPENDENT")
            
            # Validate independent claims have complete technical mechanisms
            if claim_type == "INDEPENDENT":
                assertion_categories = []
                for aid in assertion_ids:
                    if G.has_node(aid):
                        cat = G.nodes[aid].get("category", "UNCLASSIFIED")
                        assertion_categories.append(cat)
                
                # Check if independent claim has required components
                has_components = any(cat in ("TECHNICAL_COMPONENT", "INVENTIVE_MECHANISM") for cat in assertion_categories)
                if not has_components:
                    print(f"  ⚠️  Skipping independent claim {claim_id}: missing technical components (categories: {set(assertion_categories)})")
                    continue
            
            # MECHANISM-FOCUSED FILTERING: Filter and prioritize assertions for invention mechanism
            filtered_assertions = self._filter_mechanism_focused_assertions(
                G, assertion_ids, claim_type
            )
            
            if not filtered_assertions:
                print(f"  ⚠️  Skipping claim {claim_id}: no mechanism-focused assertions after filtering")
                continue
            
            # Collect assertion information for filtered assertions
            assertion_infos = []
            for assertion_id in filtered_assertions:
                if not G.has_node(assertion_id):
                    continue
                
                assertion_data = G.nodes[assertion_id]
                if assertion_data.get("node_type") != "ASSERTION":
                    continue
                
                # Find subject and object entities
                subject_id = None
                object_id = None
                subject_label = ""
                object_label = ""
                
                # Find SUBJECT edge
                for target in G.successors(assertion_id):
                    edge_data = G.get_edge_data(assertion_id, target)
                    if edge_data:
                        for key, data in edge_data.items():
                            if data.get("label") == "SUBJECT":
                                subject_id = target
                                subject_label = self._get_entity_label(G, target)
                                break
                
                # Find OBJECT edge
                for target in G.successors(assertion_id):
                    edge_data = G.get_edge_data(assertion_id, target)
                    if edge_data:
                        for key, data in edge_data.items():
                            if data.get("label") == "OBJECT":
                                object_id = target
                                object_label = self._get_entity_label(G, target)
                                break
                
                # Check for literal value
                value = assertion_data.get("value")
                
                assertion_info = AssertionInfo(
                    assertion_id=assertion_id,
                    predicate=assertion_data.get("predicate", ""),
                    subject_label=subject_label or subject_id or "",
                    subject_id=subject_id or "",
                    object_label=object_label,
                    object_id=object_id,
                    value=value,
                    category=assertion_data.get("category", "UNCLASSIFIED"),
                )
                assertion_infos.append(assertion_info)
            
            # Create claim bundle
            bundle = ClaimBundle(
                claim_id=claim_id,
                type=claim_data.get("kind", "INDEPENDENT").lower(),
                parent_claim_id=claim_data.get("parent_claim_id"),
                assertions=assertion_infos,
                breadth=claim_data.get("breadth", "MEDIUM"),
                title=claim_data.get("title"),
            )
            bundles.append(bundle)
        
        print(f"✅ Extracted {len(bundles)} claim bundles ({len(independent_claims)} independent, {len(dependent_claims)} dependent)")
        return bundles
    
    def _filter_mechanism_focused_assertions(
        self,
        G: nx.MultiDiGraph,
        assertion_ids: List[str],
        claim_type: str,
    ) -> List[str]:
        """
        Filter and prioritize assertions to create a single invention framing/mechanism-focused bundle.
        
        For independent claims:
        - Prioritize: INVENTIVE_MECHANISM → TECHNICAL_COMPONENT → TECHNICAL_EFFECT
        - Exclude: BACKGROUND, PROBLEM, PRIOR_ART, AESTHETIC
        - Ensure mechanism coherence (not just a list of features)
        
        For dependent claims:
        - Allow technical refinements only
        - Exclude: BACKGROUND, PROBLEM, PRIOR_ART, AESTHETIC
        """
        if claim_type == "INDEPENDENT":
            # Category priority for independent claims (mechanism-first)
            category_priority = {
                "INVENTIVE_MECHANISM": 1,
                "TECHNICAL_COMPONENT": 2,
                "TECHNICAL_EFFECT": 3,
                "UNCLASSIFIED": 4,
            }
            
            # Forbidden categories
            forbidden_categories = {"BACKGROUND", "PROBLEM", "PRIOR_ART", "AESTHETIC"}
            
            # Collect and categorize assertions
            valid_assertions = []
            for aid in assertion_ids:
                if not G.has_node(aid):
                    continue
                
                assertion_data = G.nodes[aid]
                if assertion_data.get("node_type") != "ASSERTION":
                    continue
                
                category = assertion_data.get("category", "UNCLASSIFIED")
                
                # Skip forbidden categories
                if category in forbidden_categories:
                    continue
                
                # Skip if not claim-eligible
                if not assertion_data.get("claim_eligible", False):
                    continue
                
                priority = category_priority.get(category, 99)
                valid_assertions.append((aid, priority, category))
            
            if not valid_assertions:
                return []
            
            # Sort by priority (INVENTIVE_MECHANISM first)
            valid_assertions.sort(key=lambda x: x[1])
            
            # Ensure we have at least one INVENTIVE_MECHANISM for independent claims
            has_inventive_mechanism = any(cat == "INVENTIVE_MECHANISM" for _, _, cat in valid_assertions)
            if not has_inventive_mechanism:
                print(f"     ⚠️  No INVENTIVE_MECHANISM assertions found - claim may be invalid")
                return []
            
            # Return sorted assertion IDs (prioritized by mechanism-first)
            filtered_ids = [aid for aid, _, _ in valid_assertions]
            
            # Limit to mechanism-focused core (prefer INVENTIVE_MECHANISM + TECHNICAL_COMPONENT + TECHNICAL_EFFECT)
            # Don't include too many UNCLASSIFIED assertions
            mechanism_core = [
                aid for aid, _, cat in valid_assertions
                if cat in ("INVENTIVE_MECHANISM", "TECHNICAL_COMPONENT", "TECHNICAL_EFFECT")
            ]
            
            unclassified = [
                aid for aid, _, cat in valid_assertions
                if cat == "UNCLASSIFIED"
            ]
            
            # Prefer mechanism core, add unclassified only if needed for coherence
            if mechanism_core:
                # Include all mechanism core assertions
                result = mechanism_core.copy()
                # Add a few unclassified only if they're needed (limit to 2-3)
                result.extend(unclassified[:3])
                return result[:15]  # Cap at 15 assertions for focus
            else:
                # Fallback: return all valid assertions (shouldn't happen if filtering works)
                return filtered_ids[:15]
        
        else:  # DEPENDENT
            # For dependent claims, allow technical refinements
            forbidden_categories = {"BACKGROUND", "PROBLEM", "PRIOR_ART", "AESTHETIC"}
            
            valid_assertions = []
            for aid in assertion_ids:
                if not G.has_node(aid):
                    continue
                
                assertion_data = G.nodes[aid]
                if assertion_data.get("node_type") != "ASSERTION":
                    continue
                
                category = assertion_data.get("category", "UNCLASSIFIED")
                
                # Skip forbidden categories
                if category in forbidden_categories:
                    continue
                
                # Skip if not claim-eligible
                if not assertion_data.get("claim_eligible", False):
                    continue
                
                valid_assertions.append(aid)
            
            return valid_assertions[:10]  # Cap at 10 for dependent claims
    
    def _get_entity_label(self, G: nx.MultiDiGraph, entity_id: str) -> str:
        """Get display label for an entity node."""
        if entity_id in self.id_to_name:
            return self.id_to_name[entity_id]
        
        if G.has_node(entity_id):
            node_data = G.nodes[entity_id]
            # Try various name fields
            for key in ["name", "label", "text", "display_name"]:
                if key in node_data:
                    return str(node_data[key])
        
        return entity_id

