"""
ClaimDraftingAgent: Drafts patent claims from claim bundles using LLM.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
import networkx as nx

from tools.graph.claim_extractor import ClaimBundle, AssertionInfo
from tools.graph.graph_rag import GraphRAG
from tools.api.llm_api_repo import LLmApi_Repo
from tools.helper.json_helper import JsonHelper
import json
import ast


@dataclass
class DraftedClaim:
    """A drafted patent claim."""
    claim_number: int
    claim_text: str
    type: str  # "independent" or "dependent"
    parent_claim_number: Optional[int] = None
    model: Optional[str] = None
    created_at: Optional[str] = None


class ClaimDraftingAgent:
    """
    Drafts patent claims from claim bundles using LLM.
    
    For each claim bundle, constructs a prompt with:
    - Glossary terms
    - Assertions (as structured bullets)
    - Evidence snippets (optional)
    - Previous claims (for dependent claims)
    
    Returns numbered patent claims.
    """
    
    def __init__(
        self,
        api_repo: Optional[LLmApi_Repo] = None,
        graph_rag: Optional[GraphRAG] = None,
        use_rag: bool = True,
    ):
        """
        Initialize the claim drafting agent.
        
        Args:
            api_repo: Optional LLM API repository (defaults to LLmApi_Repo())
            graph_rag: Optional GraphRAG instance for retrieving graph context
            use_rag: Whether to use RAG for enhanced context (default: True)
        """
        self.api_repo = api_repo or LLmApi_Repo()
        self.graph_rag = graph_rag
        self.use_rag = use_rag and graph_rag is not None
        self.task = (
            "You are an expert patent attorney drafting patent claims.\n\n"
            "TASK: Draft a formal patent claim that highlights the INVENTION and DIFFERENTIATION from generic versions.\n\n"
            "CRITICAL REQUIREMENTS (MUST FOLLOW):\n\n"
            "1. NOVEL RELATIONSHIPS, NOT JUST COMPONENTS:\n"
            "- Do NOT simply list generic components (container, conduit, inlet, opening)\n"
            "- REQUIRED: Define a NON-CONVENTIONAL ARRANGEMENT or INTERACTION between components\n"
            "- Each component must have a SPECIFIC RELATIONSHIP to other components that creates the inventive step\n"
            "- Show HOW components interact in a way that is different from standard implementations\n"
            "- Example: Instead of 'a container and a conduit', write 'a container having an elevated storage portion, wherein said conduit connects said storage portion to said container in a configuration that enables...'\n\n"
            "2. PHYSICAL CONFIGURATIONS, NOT JUST OUTCOMES:\n"
            "- Do NOT describe functional results without physical constraints (e.g., 'generates convection')\n"
            "- REQUIRED: Tie EVERY functional result to a SPECIFIC PHYSICAL CONFIGURATION\n"
            "- Every effect must be caused by a concrete structural arrangement\n"
            "- Use language like 'configured such that', 'arranged whereby', 'positioned to cause'\n"
            "- Example: Instead of 'generates convection', write 'wherein said opening is positioned at a height and orientation relative to said storage portion such that liquid flow creates convection currents'\n\n"
            "3. EXCLUDE KNOWN IMPLEMENTATIONS (IMPLICITLY):\n"
            "- REQUIRED: Structure the claim to implicitly exclude conventional systems through specific arrangements\n"
            "- DO NOT explicitly mention 'prior art', 'conventional', 'standard', or 'known' in the claim text\n"
            "- Instead, use specific structural limitations that inherently exclude generic implementations\n"
            "- Focus on precise configurations that would not be found in standard systems\n"
            "- Make it clear through STRUCTURE, not through comparison language\n\n"
            "4. UNIQUE ROLES AND CLEAR BOUNDARIES:\n"
            "- REQUIRED: Each claimed element must have a UNIQUE ROLE and CLEAR spatial or functional boundary\n"
            "- Avoid overlapping or self-referential terms (e.g., 'container' vs 'storage' - define which is which)\n"
            "- Each component must be distinguishable from others\n"
            "- Use specific descriptors: 'first container', 'second container', 'elevated storage portion', 'main tank portion'\n"
            "- Define relationships clearly: 'said storage portion being positioned above said main tank portion'\n\n"
            "5. DISTINCT INDEPENDENT CLAIMS:\n"
            "- REQUIRED: Each independent claim must protect a DISTINCT technical concept or mechanism\n"
            "- Do NOT restate the same idea with different wording\n"
            "- Each independent claim should cover a different inventive aspect or alternative embodiment\n"
            "- If claims are too similar, focus on different mechanisms, arrangements, or technical approaches\n\n"
            "MECHANISM-FIRST DRAFTING:\n"
            "- The claim MUST describe HOW THE INVENTION WORKS through specific physical arrangements\n"
            "- Focus on the invention mechanism: components + their NOVEL INTERACTIONS + technical effects\n"
            "- Show the CAUSAL RELATIONSHIPS between components that create the inventive effect\n"
            "- Every functional statement must be backed by a structural limitation\n\n"
            "CRITICAL RULES:\n"
            "- Base claims on the provided assertions, but frame them to highlight innovation through specific structural arrangements\n"
            "- DO NOT mention prior art, conventional systems, or comparisons in the claim text itself\n"
            "- Ensure antecedent basis (terms introduced in independent claims can be referenced in dependents)\n"
            "- Use broad, general language appropriate for patent claims (not too specific, not too vague)\n"
            "- Follow standard patent claim format (numbered, single sentence per claim)\n"
            "- For dependent claims, reference the parent claim number explicitly\n"
            "- Do NOT include any text other than the claim itself\n"
            "- For independent claims: describe the complete invention mechanism with emphasis on what makes it inventive\n\n"
            "OUTPUT FORMAT:\n"
            "- Return ONLY the claim text, nothing else\n"
            "- No claim number prefix (e.g., don't write '1. ' or 'Claim 1:')\n"
            "- No markdown, no commentary, just the claim text\n"
            "- Example: 'A display device comprising: a main tank portion configured to hold liquid; an elevated storage portion positioned above said main tank portion and configured to store liquid; a conduit connecting said storage portion to said main tank portion, said conduit being arranged at an angle and having a first end connected to said storage portion and a second end positioned within said main tank portion below a liquid surface level; and an opening portion provided at a bottom of said storage portion, said opening portion being configured and positioned such that liquid flow from said storage portion through said opening portion creates convection currents in said main tank portion, whereby said specific configuration of elevated storage, angled conduit, and positioned opening enables dynamic movement of objects within said liquid.'\n"
        )
    
    def draft(
        self,
        claim_bundles: Optional[List[ClaimBundle]] = None,
        G: Optional[nx.MultiDiGraph] = None,
        glossary: Optional[Dict[str, str]] = None,
        previous_claims: Optional[List[str]] = None,
        patent_description: Optional[str] = None,
        use_rag: Optional[bool] = None,
        num_independent: int = 3,
        num_dependent: int = 5,
    ) -> List[DraftedClaim]:
        """
        Draft patent claims from claim bundles or directly from graph.
        
        Args:
            claim_bundles: List of ClaimBundle objects to draft (optional)
            G: NetworkX graph (optional, used if claim_bundles is empty)
            glossary: Optional glossary of canonical terms
            previous_claims: Optional list of previously drafted claims (for dependent claims)
            patent_description: Optional full patent description text for context
            use_rag: Override default RAG setting for this call
            num_independent: Number of independent claims to create if no bundles provided
            num_dependent: Number of dependent claims to create if no bundles provided
            
        Returns:
            List of DraftedClaim objects with numbered claims
        """
        glossary = glossary or {}
        previous_claims = previous_claims or []
        use_rag = use_rag if use_rag is not None else self.use_rag
        
        # If no bundles provided, try to create them from graph
        if not claim_bundles:
            if G is not None:
                print("📦 No claim bundles provided. Creating bundles from graph...")
                claim_bundles = self._create_bundles_from_graph(G, num_independent, num_dependent)
            else:
                print("⚠️  WARNING: No claim bundles provided and no graph available!")
                print("   Provide either claim_bundles or G parameter.")
                return []
        
        if not claim_bundles:
            print("⚠️  WARNING: No claim bundles could be created!")
            return []
        
        drafted_claims: List[DraftedClaim] = []
        claim_number = 1
        
        # Process independent claims first
        independent_bundles = [b for b in claim_bundles if b.type == "independent"]
        dependent_bundles = [b for b in claim_bundles if b.type == "dependent"]
        
        print(f"📝 Drafting {len(independent_bundles)} independent and {len(dependent_bundles)} dependent claims...")
        
        # Create mapping from claim_id to claim_number for dependent claims
        claim_id_to_number: Dict[str, int] = {}
        
        # Draft independent claims
        for bundle in independent_bundles:
            claim_text = self._draft_single_claim(
                bundle=bundle,
                glossary=glossary,
                previous_claims=previous_claims,
                is_dependent=False,
                patent_description=patent_description,
                use_rag=use_rag,
            )
            
            if claim_text:
                drafted = DraftedClaim(
                    claim_number=claim_number,
                    claim_text=claim_text,
                    type="independent",
                )
                drafted_claims.append(drafted)
                claim_id_to_number[bundle.claim_id] = claim_number
                previous_claims.append(claim_text)
                claim_number += 1
        
        # Draft dependent claims
        for bundle in dependent_bundles:
            parent_number = None
            if bundle.parent_claim_id and bundle.parent_claim_id in claim_id_to_number:
                parent_number = claim_id_to_number[bundle.parent_claim_id]
            
            claim_text = self._draft_single_claim(
                bundle=bundle,
                glossary=glossary,
                previous_claims=previous_claims,
                is_dependent=True,
                parent_claim_number=parent_number,
                patent_description=patent_description,
                use_rag=use_rag,
            )
            
            if claim_text:
                drafted = DraftedClaim(
                    claim_number=claim_number,
                    claim_text=claim_text,
                    type="dependent",
                    parent_claim_number=parent_number,
                )
                drafted_claims.append(drafted)
                claim_id_to_number[bundle.claim_id] = claim_number
                previous_claims.append(claim_text)
                claim_number += 1
        
        print(f"✅ Drafted {len(drafted_claims)} claims ({len(independent_bundles)} independent, {len(dependent_bundles)} dependent)")
        return drafted_claims
    
    def _create_bundles_from_graph(
        self,
        G: nx.MultiDiGraph,
        num_independent: int = 3,
        num_dependent: int = 5,
    ) -> List[ClaimBundle]:
        """
        Create claim bundles directly from graph assertions or edges.
        
        Args:
            G: NetworkX graph with Assertion nodes or entity edges
            num_independent: Number of independent claims to create
            num_dependent: Number of dependent claims to create
            
        Returns:
            List of ClaimBundle objects
        """
        from tools.graph.claim_extractor import ClaimBundle, AssertionInfo
        
        bundles = []
        
        # Try to find assertions (claim-eligible first, then any)
        assertion_nodes = [
            (node_id, data)
            for node_id, data in G.nodes(data=True)
            if data.get("node_type") == "ASSERTION"
            and data.get("claim_eligible", False)
        ]
        
        if not assertion_nodes:
            # Fallback: any assertions
            assertion_nodes = [
                (node_id, data)
                for node_id, data in G.nodes(data=True)
                if data.get("node_type") == "ASSERTION"
            ]
            print(f"   Found {len(assertion_nodes)} assertions (not all claim-eligible)")
        
        if not assertion_nodes:
            # Last resort: create from graph edges (entity-to-entity relationships)
            print("   No assertions found. Creating bundles from graph edges...")
            bundles = self._create_bundles_from_edges(G, num_independent, num_dependent)
            return bundles
        
        print(f"   Found {len(assertion_nodes)} assertions to work with")
        
        # Group by category
        inventive_mechanisms = [
            (aid, data) for aid, data in assertion_nodes
            if data.get("category") == "INVENTIVE_MECHANISM"
        ]
        technical_components = [
            (aid, data) for aid, data in assertion_nodes
            if data.get("category") == "TECHNICAL_COMPONENT"
        ]
        technical_effects = [
            (aid, data) for aid, data in assertion_nodes
            if data.get("category") == "TECHNICAL_EFFECT"
        ]
        other_eligible = [
            (aid, data) for aid, data in assertion_nodes
            if data.get("category") not in ("INVENTIVE_MECHANISM", "TECHNICAL_COMPONENT", "TECHNICAL_EFFECT")
            and data.get("claim_eligible", False)
        ]
        
        print(f"   Categories: {len(inventive_mechanisms)} INVENTIVE_MECHANISM, "
              f"{len(technical_components)} TECHNICAL_COMPONENT, "
              f"{len(technical_effects)} TECHNICAL_EFFECT, "
              f"{len(other_eligible)} other")
        
        # Create independent claims
        for i in range(min(num_independent, len(inventive_mechanisms) or 1)):
            assertion_infos = []
            
            # Start with an inventive mechanism if available
            if inventive_mechanisms:
                mechanism_id, _ = inventive_mechanisms[i % len(inventive_mechanisms)]
                info = self._get_assertion_info(G, mechanism_id)
                if info:
                    assertion_infos.append(info)
            
            # Add related components (2-3)
            for j in range(min(2, len(technical_components))):
                comp_id, _ = technical_components[j % len(technical_components)]
                info = self._get_assertion_info(G, comp_id)
                if info and info not in assertion_infos:
                    assertion_infos.append(info)
            
            # Add a technical effect if available
            if technical_effects:
                effect_id, _ = technical_effects[0]
                info = self._get_assertion_info(G, effect_id)
                if info and info not in assertion_infos:
                    assertion_infos.append(info)
            
            # If no inventive mechanisms, use any eligible assertions
            if not assertion_infos and other_eligible:
                for j in range(min(3, len(other_eligible))):
                    other_id, _ = other_eligible[j]
                    info = self._get_assertion_info(G, other_id)
                    if info:
                        assertion_infos.append(info)
            
            if assertion_infos:
                bundle = ClaimBundle(
                    claim_id=f"auto_independent_{i+1}",
                    type="independent",
                    assertions=assertion_infos,
                    breadth="MEDIUM",
                )
                bundles.append(bundle)
                print(f"   Created independent bundle {i+1} with {len(assertion_infos)} assertions")
        
        # Create dependent claims
        if bundles and num_dependent > 0:
            parent_bundle = bundles[0]  # Use first independent as parent
            for i in range(min(num_dependent, len(technical_components) or len(other_eligible))):
                assertion_infos = []
                
                # Add one additional limitation
                if technical_components and i < len(technical_components):
                    comp_id, _ = technical_components[i]
                    info = self._get_assertion_info(G, comp_id)
                    if info:
                        assertion_infos.append(info)
                elif other_eligible and i < len(other_eligible):
                    other_id, _ = other_eligible[i]
                    info = self._get_assertion_info(G, other_id)
                    if info:
                        assertion_infos.append(info)
                
                if assertion_infos:
                    bundle = ClaimBundle(
                        claim_id=f"auto_dependent_{i+1}",
                        type="dependent",
                        parent_claim_id=parent_bundle.claim_id,
                        assertions=assertion_infos,
                        breadth="NARROW",
                    )
                    bundles.append(bundle)
                    print(f"   Created dependent bundle {i+1} with {len(assertion_infos)} assertions")
        
        print(f"✅ Created {len(bundles)} claim bundles from graph")
        return bundles
    
    def _create_bundles_from_edges(
        self,
        G: nx.MultiDiGraph,
        num_independent: int = 3,
        num_dependent: int = 5,
    ) -> List[ClaimBundle]:
        """
        Create claim bundles directly from graph edges (entity relationships).
        This is a fallback when no assertions exist.
        
        Args:
            G: NetworkX graph with entity nodes and edges
            num_independent: Number of independent claims to create
            num_dependent: Number of dependent claims to create
            
        Returns:
            List of ClaimBundle objects
        """
        from tools.graph.claim_extractor import ClaimBundle, AssertionInfo
        
        bundles = []
        
        # Get all entity-to-entity edges (skip assertion/claim links)
        entity_edges = []
        for u, v, k, data in G.edges(keys=True, data=True):
            u_type = G.nodes[u].get("node_type", "")
            v_type = G.nodes[v].get("node_type", "")
            edge_type = data.get("edge_type", "")
            
            # Skip assertion links, claim links, and non-entity nodes
            if edge_type in ("ASSERTION_LINK", "CLAIM_LINK"):
                continue
            if u_type in ("ASSERTION", "CLAIM_CONCEPT", "LEGAL_CLAIM_TEXT"):
                continue
            if v_type in ("ASSERTION", "CLAIM_CONCEPT", "LEGAL_CLAIM_TEXT"):
                continue
            
            relation = data.get("label", "")
            if relation:
                entity_edges.append((u, v, relation, data))
        
        print(f"   Found {len(entity_edges)} entity-to-entity edges")
        
        if not entity_edges:
            print("   ⚠️  No entity edges found in graph")
            return []
        
        # Get id_to_name mapping
        id_to_name = {}
        if self.graph_rag:
            id_to_name = self.graph_rag.id_to_name
        
        # Group edges by head entity (subject)
        edges_by_subject = {}
        for u, v, relation, data in entity_edges:
            if u not in edges_by_subject:
                edges_by_subject[u] = []
            edges_by_subject[u].append((u, v, relation))
        
        # Get entity names
        def get_entity_name(entity_id):
            if entity_id in id_to_name:
                return id_to_name[entity_id]
            if G.has_node(entity_id):
                return G.nodes[entity_id].get("name", entity_id)
            return entity_id
        
        # Create independent claims from important entities (high degree)
        entity_importance = {}
        for entity_id in edges_by_subject.keys():
            in_degree = G.in_degree(entity_id)
            out_degree = G.out_degree(entity_id)
            entity_importance[entity_id] = in_degree + out_degree
        
        # Sort by importance
        important_entities = sorted(
            entity_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:num_independent * 2]  # Get more candidates
        
        # Create independent claims
        for i, (entity_id, importance) in enumerate(important_entities[:num_independent]):
            if entity_id not in edges_by_subject:
                continue
            
            # Get edges for this entity
            entity_edges_list = edges_by_subject[entity_id]
            
            # Create assertion infos from edges
            assertion_infos = []
            for u, v, relation in entity_edges_list[:5]:  # Limit to 5 edges per claim
                u_name = get_entity_name(u)
                v_name = get_entity_name(v)
                
                # Create a pseudo-assertion info from edge
                # Note: category is optional and will default to None
                assertion_info = AssertionInfo(
                    assertion_id=f"edge_{u}_{v}_{relation}",
                    predicate=relation,
                    subject_label=u_name,
                    subject_id=u,
                    object_label=v_name,
                    object_id=v,
                )
                assertion_infos.append(assertion_info)
            
            if assertion_infos:
                bundle = ClaimBundle(
                    claim_id=f"edge_independent_{i+1}",
                    type="independent",
                    assertions=assertion_infos,
                    breadth="MEDIUM",
                )
                bundles.append(bundle)
                print(f"   Created independent bundle {i+1} from entity '{get_entity_name(entity_id)}' ({len(assertion_infos)} edges)")
        
        # Create dependent claims from remaining edges
        if bundles and num_dependent > 0:
            parent_bundle = bundles[0]
            remaining_entities = important_entities[num_independent:]
            
            for i, (entity_id, _) in enumerate(remaining_entities[:num_dependent]):
                if entity_id not in edges_by_subject:
                    continue
                
                entity_edges_list = edges_by_subject[entity_id]
                assertion_infos = []
                
                # Take 1-2 edges for dependent claim
                for u, v, relation in entity_edges_list[:2]:
                    u_name = get_entity_name(u)
                    v_name = get_entity_name(v)
                    
                    assertion_info = AssertionInfo(
                        assertion_id=f"edge_{u}_{v}_{relation}",
                        predicate=relation,
                        subject_label=u_name,
                        subject_id=u,
                        object_label=v_name,
                        object_id=v,
                    )
                    assertion_infos.append(assertion_info)
                
                if assertion_infos:
                    bundle = ClaimBundle(
                        claim_id=f"edge_dependent_{i+1}",
                        type="dependent",
                        parent_claim_id=parent_bundle.claim_id,
                        assertions=assertion_infos,
                        breadth="NARROW",
                    )
                    bundles.append(bundle)
                    print(f"   Created dependent bundle {i+1} from entity '{get_entity_name(entity_id)}' ({len(assertion_infos)} edges)")
        
        return bundles
    
    def _get_assertion_info(self, G: nx.MultiDiGraph, assertion_id: str) -> Optional[AssertionInfo]:
        """Get AssertionInfo for a single assertion."""
        if not G.has_node(assertion_id):
            return None
        
        assertion_data = G.nodes[assertion_id]
        if assertion_data.get("node_type") != "ASSERTION":
            return None
        
        # Find subject and object
        subject_id = None
        object_id = None
        subject_label = ""
        object_label = ""
        
        # Get id_to_name from graph_rag if available
        id_to_name = {}
        if self.graph_rag:
            id_to_name = self.graph_rag.id_to_name
        
        # Find SUBJECT edge
        for target in G.successors(assertion_id):
            edge_data = G.get_edge_data(assertion_id, target)
            if edge_data:
                for key, data in edge_data.items():
                    if data.get("label") == "SUBJECT":
                        subject_id = target
                        # Try to get name
                        if target in id_to_name:
                            subject_label = id_to_name[target]
                        elif G.has_node(target):
                            node_data = G.nodes[target]
                            subject_label = node_data.get("name", target)
                        else:
                            subject_label = target
                        break
        
        # Find OBJECT edge
        for target in G.successors(assertion_id):
            edge_data = G.get_edge_data(assertion_id, target)
            if edge_data:
                for key, data in edge_data.items():
                    if data.get("label") == "OBJECT":
                        object_id = target
                        # Try to get name
                        if target in id_to_name:
                            object_label = id_to_name[target]
                        elif G.has_node(target):
                            node_data = G.nodes[target]
                            object_label = node_data.get("name", target)
                        else:
                            object_label = target
                        break
        
        value = assertion_data.get("value")
        
        return AssertionInfo(
            assertion_id=assertion_id,
            predicate=assertion_data.get("predicate", ""),
            subject_label=subject_label or subject_id or "",
            subject_id=subject_id or "",
            object_label=object_label,
            object_id=object_id,
            value=value,
            category=assertion_data.get("category", "UNCLASSIFIED"),
        )
    
    def _draft_single_claim(
        self,
        bundle: ClaimBundle,
        glossary: Dict[str, str],
        previous_claims: List[str],
        is_dependent: bool,
        parent_claim_number: Optional[int] = None,
        patent_description: Optional[str] = None,
        use_rag: bool = False,
    ) -> Optional[str]:
        """Draft a single claim from a bundle."""
        # Build assertions text (prioritized by category)
        assertions_text = []
        for i, assertion in enumerate(bundle.assertions, 1):
            parts = [f"Subject: {assertion.subject_label}"]
            parts.append(f"Predicate: {assertion.predicate}")
            if assertion.object_label:
                parts.append(f"Object: {assertion.object_label}")
            elif assertion.value:
                parts.append(f"Value: {assertion.value}")
            
            # Add category information to help LLM prioritize (if available)
            category = getattr(assertion, 'category', None) or "UNCLASSIFIED"
            priority_marker = "⭐" if category == "INVENTIVE_MECHANISM" else ""
            parts.append(f"Category: {category}{priority_marker}")
            
            assertions_text.append(f"{i}. {' | '.join(parts)}")
        
        assertions_block = "\n".join(assertions_text)
        
        # Build glossary text
        glossary_text = ""
        if glossary:
            glossary_lines = [f"- {term}: {definition}" for term, definition in glossary.items()]
            glossary_text = "Glossary:\n" + "\n".join(glossary_lines) + "\n\n"
        
        # Build previous claims text
        previous_claims_text = ""
        if previous_claims:
            previous_claims_text = "Previous Claims (for reference):\n"
            for i, prev in enumerate(previous_claims, 1):
                previous_claims_text += f"{i}. {prev}\n"
            previous_claims_text += "\n"
        
        # Retrieve graph context using RAG if enabled
        rag_context = ""
        if use_rag and self.graph_rag:
            try:
                # Extract assertion IDs from bundle
                assertion_ids = [a.assertion_id for a in bundle.assertions if hasattr(a, 'assertion_id')]
                
                if assertion_ids:
                    # Retrieve context for this bundle
                    retrieved = self.graph_rag.retrieve_for_claim_bundle(
                        assertion_ids=assertion_ids,
                        max_entities=15,
                        max_triples=20,
                        max_depth=2,
                    )
                    rag_context = self.graph_rag.format_context_for_prompt(retrieved)
            except Exception as e:
                print(f"⚠️  Error retrieving graph context: {e}")
                rag_context = ""
        
        # Build patent description context (truncated if too long)
        patent_context = ""
        if patent_description:
            # Truncate to first 2000 characters to avoid token limits
            truncated_desc = patent_description[:2000] + "..." if len(patent_description) > 2000 else patent_description
            patent_context = (
                "PATENT DESCRIPTION CONTEXT (for reference only - use to understand the invention, but base claims ONLY on assertions):\n"
                f"{truncated_desc}\n\n"
                "IMPORTANT: The description above is for context only. Your claim must be based EXCLUSIVELY on the assertions provided below.\n"
                "Do NOT copy text directly from the description. Use it only to better understand the technical domain and terminology.\n\n"
            )
        
        # Build mechanism-focused instructions
        mechanism_instructions = ""
        if not is_dependent:
            mechanism_instructions = (
                "\nMECHANISM-FOCUSED DRAFTING FOR INDEPENDENT CLAIM:\n"
                "- This claim must describe HOW THE INVENTION WORKS (the complete mechanism)\n"
                "- CRITICAL REQUIREMENTS:\n"
                "  1. Define NON-CONVENTIONAL ARRANGEMENTS - not just component lists\n"
                "  2. Tie EVERY functional result to SPECIFIC PHYSICAL CONFIGURATIONS\n"
                "  3. Include DISTINGUISHING LIMITATIONS that exclude conventional systems\n"
                "  4. Give each component a UNIQUE ROLE with CLEAR BOUNDARIES\n"
                "  5. Ensure this claim is DISTINCT from other independent claims (different technical concept)\n"
                "- Focus on: (1) core components with SPECIFIC ARRANGEMENTS, (2) NOVEL INTERACTIONS between them, (3) technical effect tied to structure\n"
                "- Do NOT just list features - describe the coherent technical system with SPECIFIC CONFIGURATIONS\n"
                "- Use language: 'configured such that', 'arranged whereby', 'positioned to cause', 'wherein said X is positioned relative to Y'\n"
                "- The assertions are prioritized: INVENTIVE_MECHANISM → TECHNICAL_COMPONENT → TECHNICAL_EFFECT\n"
                "- Use the most important assertions (those describing the mechanism) to form the core of the claim\n"
                "- Ensure the claim describes a working invention with INVENTIVE CHARACTERISTICS through specific structural arrangements\n"
                "- DO NOT mention prior art, conventional systems, or comparisons in the claim text\n"
                "- Think: \"What specific arrangement makes this inventive?\" and define that arrangement precisely through structure\n\n"
            )
        else:
            mechanism_instructions = (
                "\nTECHNICAL REFINEMENT FOR DEPENDENT CLAIM:\n"
                "- CRITICAL: This claim can ONLY REFINE or LIMIT features already mentioned in the independent claim\n"
                "- DO NOT introduce new components, features, or inventions not present in the parent claim\n"
                "- You can ONLY:\n"
                "  * Add specific dimensions, materials, or properties to existing components\n"
                "  * Specify relationships or arrangements between components already claimed\n"
                "  * Add limitations to how existing components function or interact\n"
                "  * Specify additional details about components already in the independent claim\n"
                "- You CANNOT:\n"
                "  * Add new components not mentioned in the independent claim\n"
                "  * Introduce new functional features or mechanisms\n"
                "  * Add new relationships between unclaimed elements\n"
                "- Do NOT include background, problems, or aesthetic features\n"
                "- Do NOT mention prior art, conventional systems, or comparisons\n"
                "- Example: If independent claim has 'a container', dependent can say 'wherein said container is cylindrical' but NOT 'further comprising a pump'\n\n"
            )
        
        # Build claim type instructions
        if is_dependent and parent_claim_number:
            claim_type_instructions = (
                f"This is a DEPENDENT claim that depends on claim {parent_claim_number}.\n"
                "Reference the parent claim explicitly (e.g., 'The device of claim {parent_claim_number}...').\n"
                "CRITICAL RULES FOR DEPENDENT CLAIMS:\n"
                "- You can ONLY refine or limit features ALREADY MENTIONED in claim {parent_claim_number}\n"
                "- DO NOT introduce new components, features, or inventions not in the independent claim\n"
                "- You can add: specific dimensions, materials, properties, arrangements, or limitations to existing components\n"
                "- You CANNOT add: new components, new functional features, or new mechanisms\n"
                "- Use assertions that relate to components already in the parent claim\n"
                "- Do NOT mention prior art, conventional systems, or comparisons\n\n"
            ).format(parent_claim_number=parent_claim_number)
        else:
            claim_type_instructions = (
                "This is an INDEPENDENT claim.\n"
                "Draft a complete, standalone claim using the assertions below.\n\n"
            )
        
        prompt = (
            f"{self.task}\n\n"
            f"{mechanism_instructions}"
            f"{rag_context}"  # Add RAG context
            f"{patent_context}"
            f"{glossary_text}"
            f"{previous_claims_text}"
            f"{claim_type_instructions}"
            f"Assertions to use (prioritized by mechanism importance):\n{assertions_block}\n\n"
            "Draft the claim now with a PATENT LAWYER mindset - FOLLOW ALL REQUIREMENTS:\n"
            "1. Define NOVEL RELATIONSHIPS between components (not just component lists)\n"
            "2. Tie EVERY functional result to SPECIFIC PHYSICAL CONFIGURATIONS (not just outcomes)\n"
            "3. Include DISTINGUISHING LIMITATIONS through specific structural arrangements (NOT by mentioning prior art)\n"
            "4. Give each component a UNIQUE ROLE with CLEAR BOUNDARIES (avoid overlapping terms)\n"
            "5. Ensure this claim is DISTINCT from other independent claims (different technical concept)\n"
            "6. Use precise language: 'configured such that', 'arranged whereby', 'positioned relative to', 'wherein'\n"
            "7. DO NOT mention 'prior art', 'conventional', 'standard', or 'known' in the claim text\n"
            "8. For dependent claims: ONLY refine features from the parent claim - DO NOT add new components or features\n"
            "9. Show HOW THE INVENTION WORKS through specific structural arrangements\n"
            "10. Emphasize WHAT MAKES IT DIFFERENT through structure, not by comparison\n\n"
            "Use the retrieved graph context to better understand entity relationships and technical connections.\n"
            "Return ONLY the claim text, nothing else."
        )
        
        try:
            response = self.api_repo.chat(prompt)
            
            # Extract text from response
            if isinstance(response, dict):
                response_text = response.get("content", response.get("text", response.get("message", "")))
                if not response_text and "choices" in response:
                    response_text = response["choices"][0].get("message", {}).get("content", "")
            else:
                response_text = str(response) if response else ""
            
            # Clean up the response
            claim_text = response_text.strip()
            
            # Remove any claim number prefix if present
            claim_text = claim_text.lstrip("0123456789. ")
            if claim_text.lower().startswith("claim "):
                claim_text = claim_text[6:].lstrip("0123456789. ")
            
            return claim_text if claim_text else None
            
        except Exception as e:
            print(f"⚠️  Error drafting claim for bundle {bundle.claim_id}: {e}")
            return None
    
    def write_to_graph(
        self,
        G: nx.MultiDiGraph,
        drafted_claims: List[DraftedClaim],
        claim_id_to_bundle_id: Dict[int, str],
    ) -> nx.MultiDiGraph:
        """
        Write drafted claims back to the graph as LegalClaimText nodes.
        
        Args:
            G: Graph with ClaimConcept nodes
            drafted_claims: List of drafted claims
            claim_id_to_bundle_id: Mapping from claim number to claim concept ID
            
        Returns:
            Modified graph G
        """
        for drafted in drafted_claims:
            bundle_id = claim_id_to_bundle_id.get(drafted.claim_number)
            if not bundle_id or not G.has_node(bundle_id):
                continue
            
            # Create legal claim text node
            claim_text_id = f"legal_claim_{drafted.claim_number}"
            G.add_node(
                claim_text_id,
                node_type="LEGAL_CLAIM_TEXT",
                claim_number=drafted.claim_number,
                claim_text=drafted.claim_text,
                type=drafted.type,
                parent_claim_number=drafted.parent_claim_number,
                model=drafted.model,
                created_at=drafted.created_at,
            )
            
            # Link to claim concept
            G.add_edge(bundle_id, claim_text_id, label="DRAFTED_AS", edge_type="CLAIM_LINK")
        
        return G

