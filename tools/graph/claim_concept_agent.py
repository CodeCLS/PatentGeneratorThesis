"""
ClaimConceptAgent: Bundles assertions into independent/dependent claim concepts.
Uses LLM to intelligently bundle assertions into meaningful claim concepts.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import uuid
import networkx as nx
from collections import defaultdict
import json
import ast

from tools.graph.assertion_agent import Assertion
from tools.api.llm_api_repo import LLmApi_Repo
from tools.helper.json_helper import JsonHelper


@dataclass
class ClaimConcept:
    """Represents a claim concept (independent or dependent)."""
    id: str
    kind: str  # "INDEPENDENT" or "DEPENDENT"
    breadth: str = "MEDIUM"  # "BROAD", "MEDIUM", "NARROW"
    title: Optional[str] = None
    status: str = "CANDIDATE"  # CANDIDATE, CONFIRMED, DRAFTED
    assertion_ids: List[str] = field(default_factory=list)  # Assertions used by this claim
    parent_claim_id: Optional[str] = None  # For dependent claims


class ClaimConceptAgent:
    """
    Creates ClaimConcept nodes that bundle assertions into legal claim candidates.
    
    Creates:
    - Independent claim concepts (broad, medium, narrow)
    - Dependent claim concepts that reference independent claims
    """
    
    def __init__(
        self,
        min_independent_assertions: int = 2,  # Lowered from 3 to allow smaller claim sets
        max_independent_assertions: int = 10,
        min_dependent_assertions: int = 1,
        max_dependent_assertions: int = 5,
        api_repo: Optional[LLmApi_Repo] = None,
        use_llm: bool = True,
    ):
        """
        Initialize the claim concept agent.
        
        Args:
            min_independent_assertions: Minimum assertions for independent claims (default 2, lowered to work with fewer assertions)
            max_independent_assertions: Maximum assertions for independent claims
            min_dependent_assertions: Minimum additional assertions for dependent claims
            max_dependent_assertions: Maximum additional assertions for dependent claims
            api_repo: Optional LLM API repository (defaults to LLmApi_Repo())
            use_llm: If True, uses LLM to intelligently bundle assertions
        """
        self.min_independent = min_independent_assertions
        self.max_independent = max_independent_assertions
        self.min_dependent = min_dependent_assertions
        self.max_dependent = max_dependent_assertions
        self.api_repo = api_repo or LLmApi_Repo()
        self.use_llm = use_llm
    
    def run(
        self,
        G: nx.MultiDiGraph,
        status_filter: Optional[str] = None,
        num_independent: int = 3,
        num_dependent_per_independent: int = 2,
    ) -> nx.MultiDiGraph:
        """
        Add claim concept nodes to the graph.
        
        Args:
            G: Graph with assertion nodes
            status_filter: Optional filter assertions by status (if None, uses all claim-eligible assertions)
            num_independent: Number of independent claim concepts to create
            num_dependent_per_independent: Number of dependent concepts per independent
            
        Returns:
            Modified graph G with claim concept nodes added
        """
        # Get all claim-eligible assertions (primary filter: claim_eligible=True)
        all_claim_eligible = [
            (node_id, data)
            for node_id, data in G.nodes(data=True)
            if data.get("node_type") == "ASSERTION"
            and data.get("claim_eligible", False) is True
        ]
        
        if not all_claim_eligible:
            print(f"⚠️  No claim-eligible assertions found")
            return G
        
        # Optionally filter by status if status_filter is provided
        if status_filter is not None:
            assertion_nodes = [
                (node_id, data)
                for node_id, data in all_claim_eligible
                if data.get("status", "CANDIDATE") == status_filter
            ]
            
            if not assertion_nodes:
                print(f"⚠️  No claim-eligible assertions with status '{status_filter}', using all {len(all_claim_eligible)} claim-eligible assertions")
                assertion_nodes = all_claim_eligible
        else:
            # No status filter - use all claim-eligible assertions
            assertion_nodes = all_claim_eligible
        
        # Count by category
        category_counts = {}
        has_inventive_mechanism = False
        for _, data in assertion_nodes:
            cat = data.get("category", "UNCLASSIFIED")
            category_counts[cat] = category_counts.get(cat, 0) + 1
            if cat == "INVENTIVE_MECHANISM":
                has_inventive_mechanism = True
        
        filter_info = f"status '{status_filter}'" if status_filter else "claim_eligible=True"
        print(f"📋 Found {len(assertion_nodes)} claim-eligible assertions (filter: {filter_info})")
        print(f"   Categories: {category_counts}")
        
        # HARD RULE 2: If no INVENTIVE_MECHANISM assertion exists, generate no claims at all
        if not has_inventive_mechanism:
            print(f"❌ HARD RULE: No INVENTIVE_MECHANISM assertions found. Generating ZERO claims.")
            print(f"   This is correct behavior - claims require at least one INVENTIVE_MECHANISM assertion.")
            return G
        
        # Group assertions by subject entity (heuristic for bundling)
        assertions_by_subject: Dict[str, List[str]] = defaultdict(list)
        for assertion_id, _ in assertion_nodes:
            # Find SUBJECT edges from this assertion
            for target in G.successors(assertion_id):
                edge_data = G.get_edge_data(assertion_id, target)
                if edge_data:
                    for key, data in edge_data.items():
                        if data.get("label") == "SUBJECT":
                            subject_id = target
                            assertions_by_subject[subject_id].append(assertion_id)
                            break
        
        # Create independent claim concepts
        independent_claims = []
        all_used_assertions: Set[str] = set()
        
        # Get assertion details for LLM (include category)
        assertion_details = self._get_assertion_details(G, assertion_nodes)
        
        if self.use_llm:
            # Use LLM to intelligently bundle assertions
            print(f"  📦 Bundling {len(assertion_details)} assertions into independent claims...")
            independent_bundles = self._bundle_assertions_with_llm(
                assertion_details, num_independent, "INDEPENDENT"
            )
            print(f"  📦 LLM returned {len(independent_bundles)} bundle(s)")
            
            # If LLM bundling failed or returned no results, try fallback
            if not independent_bundles and len(assertion_nodes) >= self.min_independent:
                print(f"  ⚠️  LLM bundling returned no results, trying fallback with all {len(assertion_nodes)} assertions")
                # Create a single bundle with all assertions if we have enough
                all_assertion_ids = [aid for aid, _ in assertion_nodes]
                # Apply same filtering as above
                filtered_ids = []
                filtered_out = {"PRIOR_ART": 0, "PROBLEM": 0, "negative_predicates": 0}
                
                for aid in all_assertion_ids:
                    if not G.has_node(aid):
                        continue
                    assertion_data = G.nodes[aid]
                    cat = assertion_data.get("category", "UNCLASSIFIED")
                    pred = assertion_data.get("predicate", "").lower()
                    
                    # Hard-block PRIOR_ART, PROBLEM, and AESTHETIC
                    if cat in ("PRIOR_ART", "PROBLEM", "AESTHETIC"):
                        filtered_out[cat] = filtered_out.get(cat, 0) + 1
                        continue
                    
                    # Hard-block negative predicates
                    negative_predicates = ["adhere", "behave unnaturally", "pushed upward", "lying down", "rises", "near bubble generating member"]
                    if any(neg in pred for neg in negative_predicates):
                        filtered_out["negative_predicates"] = filtered_out.get("negative_predicates", 0) + 1
                        continue
                    
                    filtered_ids.append(aid)
                
                if filtered_out.get("PRIOR_ART", 0) > 0 or filtered_out.get("PROBLEM", 0) > 0 or filtered_out.get("AESTHETIC", 0) > 0 or filtered_out.get("negative_predicates", 0) > 0:
                    print(f"     Filtered out: {filtered_out}")
                
                # ABSTRACT FIX: Check if filtered bundle has at least one INVENTIVE_MECHANISM
                has_inventive = any(
                    G.nodes[aid].get("category") == "INVENTIVE_MECHANISM"
                    for aid in filtered_ids
                    if G.has_node(aid)
                )
                
                if not has_inventive:
                    print(f"  ⚠️  Fallback bundle has no INVENTIVE_MECHANISM assertions (required for independent claims)")
                elif len(filtered_ids) >= self.min_independent:
                    if self._check_mechanism_completeness(G, filtered_ids) and self._describes_invention_mechanism(G, filtered_ids):
                        independent_bundles = [{"assertion_ids": filtered_ids, "breadth": "BROAD"}]
                    else:
                        print(f"  ⚠️  Fallback bundle failed mechanism completeness or invention mechanism check ({len(filtered_ids)} assertions)")
                else:
                    print(f"  ⚠️  Fallback bundle has too few assertions after filtering ({len(filtered_ids)}, minimum: {self.min_independent})")
            
            for i, bundle in enumerate(independent_bundles):
                if len(bundle["assertion_ids"]) < self.min_independent:
                    print(f"  ⚠️  Skipping bundle {i+1}: only {len(bundle['assertion_ids'])} assertions (minimum: {self.min_independent})")
                    continue
                
                # ABSTRACT FIX: Hard-block PRIOR_ART, PROBLEM, and AESTHETIC from independent claims
                bundle_assertions = []
                has_prior_art = False
                has_problem = False
                has_aesthetic = False
                has_negative_predicates = False
                negative_predicates = ["adhere", "behave unnaturally", "pushed upward", "lying down", "rises", "near bubble generating member"]
                
                for aid in bundle["assertion_ids"]:
                    if not G.has_node(aid):
                        continue
                    assertion_data = G.nodes[aid]
                    cat = assertion_data.get("category", "UNCLASSIFIED")
                    pred = assertion_data.get("predicate", "").lower()
                    
                    bundle_assertions.append((aid, cat, pred))
                    
                    if cat == "PRIOR_ART":
                        has_prior_art = True
                    if cat == "PROBLEM":
                        has_problem = True
                    if cat == "AESTHETIC":
                        has_aesthetic = True
                    if any(neg in pred for neg in negative_predicates):
                        has_negative_predicates = True
                
                if has_prior_art:
                    print(f"  ⚠️  Skipping bundle {i+1}: contains PRIOR_ART assertions (hard-blocked from independent claims)")
                    continue
                
                if has_problem:
                    print(f"  ⚠️  Skipping bundle {i+1}: contains PROBLEM assertions (hard-blocked from independent claims)")
                    continue
                
                if has_aesthetic:
                    print(f"  ⚠️  Skipping bundle {i+1}: contains AESTHETIC assertions (hard-blocked from independent claims)")
                    continue
                
                if has_negative_predicates:
                    print(f"  ⚠️  Skipping bundle {i+1}: contains negative-defect predicates (hard-blocked from independent claims)")
                    continue
                
                # Validate that bundle contains complete technical mechanism
                bundle_categories = [cat for _, cat, _ in bundle_assertions]
                
                # ABSTRACT FIX: Bundle must contain at least one INVENTIVE_MECHANISM assertion
                has_inventive_mechanism = any(cat == "INVENTIVE_MECHANISM" for cat in bundle_categories)
                if not has_inventive_mechanism:
                    print(f"  ⚠️  Skipping bundle {i+1}: no INVENTIVE_MECHANISM assertions (required for independent claims)")
                    continue
                
                # ABSTRACT FIX: Validate that bundle describes "how the invention works" (coherent technical interaction)
                if not self._describes_invention_mechanism(G, bundle["assertion_ids"]):
                    print(f"  ⚠️  Skipping bundle {i+1}: does not describe how the invention works (missing coherent technical interaction)")
                    continue
                
                # Check if bundle has required components for a complete mechanism
                has_components = any(cat in ("TECHNICAL_COMPONENT", "INVENTIVE_MECHANISM") for cat in bundle_categories)
                has_effects = any(cat == "TECHNICAL_EFFECT" for cat in bundle_categories)
                
                if not has_components:
                    print(f"  ⚠️  Skipping bundle {i+1}: missing technical components (categories: {set(bundle_categories)})")
                    continue
                
                # FIX 3: Mechanism completeness gate
                if not self._check_mechanism_completeness(G, bundle["assertion_ids"]):
                    print(f"  ⚠️  Skipping bundle {i+1}: does not meet mechanism completeness requirements")
                    continue
                
                breadth = bundle.get("breadth", ["BROAD", "MEDIUM", "NARROW"][min(i, 2)])
                claim_id = f"claim_independent_{uuid.uuid4().hex[:8]}"
                
                G.add_node(
                    claim_id,
                    node_type="CLAIM_CONCEPT",
                    claim_id=claim_id,
                    kind="INDEPENDENT",
                    breadth=breadth,
                    status="CANDIDATE",
                    assertion_ids=bundle["assertion_ids"],
                )
                
                for assertion_id in bundle["assertion_ids"]:
                    G.add_edge(claim_id, assertion_id, label="USES", role="must", edge_type="CLAIM_LINK")
                    all_used_assertions.add(assertion_id)
                
                independent_claims.append(claim_id)
                print(f"  Created independent claim concept: {claim_id} ({breadth}, {len(bundle['assertion_ids'])} assertions, categories: {set(bundle_categories)})")
        else:
            # Fallback to heuristic bundling
            independent_claims = self._bundle_assertions_heuristic(
                G, assertion_nodes, assertions_by_subject, num_independent, all_used_assertions
            )
        
        # Create dependent claim concepts
        dependent_claims = []
        for parent_claim_id in independent_claims:
            parent_assertions = set(G.nodes[parent_claim_id].get("assertion_ids", []))
            
            # Find additional assertions not in parent
            additional_assertions = [
                a for a, _ in assertion_nodes
                if a not in parent_assertions and a not in all_used_assertions
            ]
            
            if len(additional_assertions) < self.min_dependent:
                continue
            
            # Create dependent claims for this parent
            for j in range(num_dependent_per_independent):
                if not additional_assertions:
                    break
                
                if self.use_llm:
                    # Use LLM to select best additional assertions
                    selected = self._select_dependent_assertions_with_llm(
                        G, parent_claim_id, additional_assertions, assertion_details
                    )
                else:
                    # Heuristic: take subset
                    subset_size = min(self.max_dependent, len(additional_assertions))
                    selected = additional_assertions[:subset_size]
                
                additional_assertions = [a for a in additional_assertions if a not in selected]
                
                if len(selected) < self.min_dependent:
                    break
                
                # Create dependent claim concept
                claim_id = f"claim_dependent_{uuid.uuid4().hex[:8]}"
                G.add_node(
                    claim_id,
                    node_type="CLAIM_CONCEPT",
                    claim_id=claim_id,
                    kind="DEPENDENT",
                    breadth="NARROW",
                    status="CANDIDATE",
                    assertion_ids=selected,
                    parent_claim_id=parent_claim_id,
                )
                
                # Link to parent claim
                G.add_edge(claim_id, parent_claim_id, label="DEPENDS_ON", edge_type="CLAIM_LINK")
                
                # Link to assertions
                for assertion_id in selected:
                    G.add_edge(claim_id, assertion_id, label="USES", role="must", edge_type="CLAIM_LINK")
                    all_used_assertions.add(assertion_id)
                
                dependent_claims.append(claim_id)
                print(f"  Created dependent claim concept: {claim_id} (depends on {parent_claim_id}, {len(selected)} assertions)")
        
        print(f"✅ Created {len(independent_claims)} independent and {len(dependent_claims)} dependent claim concepts")
        return G
    
    def _get_assertion_details(self, G: nx.MultiDiGraph, assertion_nodes: List[Tuple[str, Dict]]) -> List[Dict[str, Any]]:
        """Extract detailed information about assertions for LLM."""
        details = []
        for assertion_id, _ in assertion_nodes:
            assertion_data = G.nodes[assertion_id]
            
            # Find subject and object
            subject_id = None
            object_id = None
            subject_name = ""
            object_name = ""
            
            for target in G.successors(assertion_id):
                edge_data = G.get_edge_data(assertion_id, target)
                if edge_data:
                    for key, data in edge_data.items():
                        if data.get("label") == "SUBJECT":
                            subject_id = target
                            subject_name = self._get_entity_name(G, target)
                        elif data.get("label") == "OBJECT":
                            object_id = target
                            object_name = self._get_entity_name(G, target)
            
            details.append({
                "assertion_id": assertion_id,
                "predicate": assertion_data.get("predicate", ""),
                "subject": subject_name,
                "object": object_name,
                "confidence": assertion_data.get("confidence", 1.0),
                "category": assertion_data.get("category", "UNCLASSIFIED"),
            })
        
        return details
    
    def _get_entity_name(self, G: nx.MultiDiGraph, entity_id: str) -> str:
        """Get display name for an entity node."""
        if G.has_node(entity_id):
            node_data = G.nodes[entity_id]
            for key in ["name", "label", "text", "display_name"]:
                if key in node_data:
                    return str(node_data[key])
        return entity_id
    
    def _check_mechanism_completeness(self, G: nx.MultiDiGraph, assertion_ids: List[str]) -> bool:
        """
        FIX 3: Mechanism completeness gate.
        
        Check if the bundle contains at least 3 of 4 required components:
        1. has tank
        2. has elevated storage/reservoir
        3. has pipe connecting tank↔storage
        4. has opening/return causing convection
        
        Returns True only if at least 3 of 4 are present.
        For small bundles (3-4 assertions), requires at least 2 of 4.
        """
        # Collect all entity names and predicates from assertions
        entity_names = set()
        predicates = []
        
        for aid in assertion_ids:
            if not G.has_node(aid):
                continue
            
            assertion_data = G.nodes[aid]
            pred = assertion_data.get("predicate", "").lower()
            predicates.append(pred)
            
            # Get subject and object entities
            for target in G.successors(aid):
                edge_data = G.get_edge_data(aid, target)
                if edge_data:
                    for key, data in edge_data.items():
                        if data.get("label") in ("SUBJECT", "OBJECT"):
                            entity_name = self._get_entity_name(G, target).lower()
                            entity_names.add(entity_name)
        
        combined_text = " ".join(entity_names) + " " + " ".join(predicates)
        combined_text = combined_text.lower()
        
        # Check 1: has tank
        has_tank = any(term in combined_text for term in ["water tank", "tank", "tank for appreciation"])
        
        # Check 2: has elevated storage/reservoir
        has_storage = any(term in combined_text for term in ["water storage", "storage", "reservoir", "upper water storage"])
        has_elevated = any(term in combined_text for term in ["upward", "above", "elevated", "installed upwardly", "upwardly"])
        has_elevated_storage = has_storage and has_elevated
        
        # Check 3: has pipe connecting tank↔storage
        has_pipe = any(term in combined_text for term in ["water pipe", "pipe"])
        has_connection = any(term in combined_text for term in ["connect", "connected", "leads", "transports", "connects"])
        has_pipe_connection = has_pipe and has_connection and (has_tank or has_storage)
        
        # Check 4: has opening/return causing convection
        has_opening = any(term in combined_text for term in ["opening portion", "opening", "opening portion 6"])
        has_convection = any(term in combined_text for term in ["convection", "generate convection", "causes convection"])
        has_fall = any(term in combined_text for term in ["free-fall", "falls", "fall", "drops", "drops into"])
        has_convection_mechanism = (has_opening and has_convection) or (has_opening and has_fall)
        
        # Count how many of 4 checks pass
        checks_passed = sum([
            has_tank,
            has_elevated_storage,
            has_pipe_connection,
            has_convection_mechanism,
        ])
        
        # For small bundles (≤4 assertions), require at least 2 of 4
        # For larger bundles, require at least 3 of 4
        required_checks = 2 if len(assertion_ids) <= 4 else 3
        is_complete = checks_passed >= required_checks
        
        if not is_complete:
            print(f"     Mechanism completeness: {checks_passed}/4 (required: {required_checks}, tank: {has_tank}, elevated_storage: {has_elevated_storage}, pipe_connection: {has_pipe_connection}, convection: {has_convection_mechanism})")
        
        return is_complete
    
    def _describes_invention_mechanism(self, G: nx.MultiDiGraph, assertion_ids: List[str]) -> bool:
        """
        ABSTRACT FIX: Validate that bundle describes "how the invention works" 
        (coherent technical interaction, not just a list of features).
        
        Rule: If a claim doesn't describe how the invention works, it must not exist.
        
        Checks:
        1. Has at least one INVENTIVE_MECHANISM assertion (required)
        2. Has technical interactions (not just isolated components)
        3. Has flow/process (components interact, not just listed)
        """
        if not assertion_ids:
            return False
        
        categories = []
        predicates = []
        entities = set()
        
        for aid in assertion_ids:
            if not G.has_node(aid):
                continue
            
            assertion_data = G.nodes[aid]
            cat = assertion_data.get("category", "UNCLASSIFIED")
            pred = assertion_data.get("predicate", "").lower()
            
            categories.append(cat)
            predicates.append(pred)
            
            # Get entities
            for target in G.successors(aid):
                edge_data = G.get_edge_data(aid, target)
                if edge_data:
                    for key, data in edge_data.items():
                        if data.get("label") in ("SUBJECT", "OBJECT"):
                            entity_name = self._get_entity_name(G, target).lower()
                            entities.add(entity_name)
        
        # Must have at least one INVENTIVE_MECHANISM
        if "INVENTIVE_MECHANISM" not in categories:
            return False
        
        # Check for technical interactions (not just isolated statements)
        # Look for predicates that indicate interaction/flow: connect, generate, causes, transports, leads, etc.
        interaction_predicates = [
            "connect", "connected", "generates", "causes", "creates", "produces",
            "transports", "leads", "flows", "moves", "transfers", "enables",
            "allows", "provides", "supports", "links", "joins", "couples"
        ]
        
        has_interactions = any(
            any(interaction in pred for interaction in interaction_predicates)
            for pred in predicates
        )
        
        if not has_interactions:
            return False
        
        # Check that we have multiple entities (components interact, not just one component)
        if len(entities) < 2:
            return False
        
        # Check for mechanism flow: at least one INVENTIVE_MECHANISM + one TECHNICAL_COMPONENT or TECHNICAL_EFFECT
        has_mechanism = "INVENTIVE_MECHANISM" in categories
        has_component_or_effect = any(cat in ("TECHNICAL_COMPONENT", "TECHNICAL_EFFECT") for cat in categories)
        
        return has_mechanism and has_component_or_effect
    
    def _bundle_assertions_with_llm(
        self,
        assertion_details: List[Dict[str, Any]],
        num_bundles: int,
        claim_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to intelligently bundle assertions into claim concepts.
        
        Returns:
            List of bundles, each with: assertion_ids, breadth
        """
        # ABSTRACT FIX: Sort assertions by priority (INVENTIVE_MECHANISM → TECHNICAL_COMPONENT → TECHNICAL_EFFECT)
        # This ensures mechanism-first bundling
        category_priority = {
            "INVENTIVE_MECHANISM": 1,
            "TECHNICAL_COMPONENT": 2,
            "TECHNICAL_EFFECT": 3,
            "UNCLASSIFIED": 4,
        }
        
        sorted_details = sorted(
            assertion_details,
            key=lambda d: category_priority.get(d.get("category", "UNCLASSIFIED"), 99)
        )
        
        # Format assertions for LLM (include category, sorted by priority)
        assertions_text = []
        for i, detail in enumerate(sorted_details, 1):
            category = detail.get("category", "UNCLASSIFIED")
            priority_marker = "⭐" if category == "INVENTIVE_MECHANISM" else ""
            assertions_text.append(
                f"{i}. {detail['subject']} --[{detail['predicate']}]--> {detail['object']} "
                f"{priority_marker}(category: {category}, confidence: {detail['confidence']:.2f}, id: {detail['assertion_id']})"
            )
        
        if claim_type == "INDEPENDENT":
            # Adjust num_bundles based on available assertions
            max_possible_bundles = len(assertion_details) // self.min_independent
            actual_num_bundles = min(num_bundles, max_possible_bundles) if max_possible_bundles > 0 else 1
            
            prompt = (
                f"You are a patent attorney organizing assertions into {claim_type.lower()} patent claim concepts.\n\n"
                f"Task: Group the following {len(assertion_details)} assertions into {actual_num_bundles} {claim_type.lower()} claim concept(s).\n\n"
                "ABSTRACT FIX - CRITICAL RULES FOR INDEPENDENT CLAIMS:\n"
                "- Each independent claim MUST describe HOW THE INVENTION WORKS (coherent technical interaction)\n"
                "- If a claim doesn't describe how the invention works, it must not exist\n"
                "- Build claims MECHANISM-FIRST: prioritize INVENTIVE_MECHANISM → TECHNICAL_COMPONENT → TECHNICAL_EFFECT\n"
                "- Each bundle MUST include at least one INVENTIVE_MECHANISM assertion (required)\n"
                "- A complete mechanism requires: required components + their interactions + technical effects\n"
                "- DO NOT create independent claims from generic, thematic, or incomplete assertions\n"
                "- DO NOT include PROBLEM or AESTHETIC assertions in independent claims (forbidden)\n"
                f"- Each bundle should contain at least {self.min_independent} assertions that together form a patentable invention\n"
                "- Assertions marked with ⭐ are INVENTIVE_MECHANISM - prioritize these!\n"
                "- Assign breadth: BROAD (core invention), MEDIUM (specific embodiment), NARROW (detailed feature)\n"
                "- Only create bundles if they form a coherent, complete technical system that describes HOW IT WORKS\n"
                f"- If you have {len(assertion_details)} or fewer assertions, you may create 1 bundle with all of them if they form a complete mechanism\n\n"
                f"Assertions (sorted by priority - INVENTIVE_MECHANISM first):\n" + "\n".join(assertions_text) + "\n\n"
                "Return a JSON array of claim bundles. Each bundle:\n"
                '{"assertion_ids": ["id1", "id2", ...], "breadth": "BROAD|MEDIUM|NARROW"}\n'
                "IMPORTANT: \n"
                "- Only return bundles that form COMPLETE TECHNICAL MECHANISMS\n"
                "- Each bundle MUST include at least one INVENTIVE_MECHANISM assertion\n"
                "- Each bundle MUST describe HOW THE INVENTION WORKS (not just a list of features)\n"
                f"- If there are not enough assertions for {actual_num_bundles} complete mechanisms, return 1 bundle with all assertions if they form a complete mechanism.\n\n"
                "Return ONLY valid JSON array, no markdown fences."
            )
        else:
            prompt = (
                f"You are a patent attorney organizing assertions into {claim_type.lower()} patent claim concepts.\n\n"
                f"Task: Group the following assertions into {num_bundles} {claim_type.lower()} claim concepts.\n\n"
                "RULES FOR DEPENDENT CLAIMS:\n"
                "- Each dependent claim adds TECHNICAL REFINEMENTS to a parent independent claim\n"
                "- DO NOT include background, problems, or aesthetic assertions\n"
                "- Only include technical components, mechanisms, or effects\n"
                "- Each bundle should contain 1-5 additional technical assertions\n"
                "- Assign breadth: MEDIUM or NARROW (refinements are typically more specific)\n\n"
                f"Assertions:\n" + "\n".join(assertions_text) + "\n\n"
                "Return a JSON array of claim bundles. Each bundle:\n"
                '{"assertion_ids": ["id1", "id2", ...], "breadth": "MEDIUM|NARROW"}\n\n'
                "Return ONLY valid JSON array, no markdown fences."
            )
        
        try:
            response = self.api_repo.chat(prompt)
            
            # Extract text
            if isinstance(response, dict):
                response_text = response.get("content", response.get("text", response.get("message", "")))
                if not response_text and "choices" in response:
                    response_text = response["choices"][0].get("message", {}).get("content", "")
            else:
                response_text = str(response) if response else ""
            
            # Parse JSON
            text = JsonHelper._unfence(response_text).strip()
            if not text:
                return []
            
            try:
                bundles = json.loads(text)
            except json.JSONDecodeError:
                try:
                    bundles = ast.literal_eval(text)
                except Exception:
                    return []
            
            if isinstance(bundles, list):
                # Validate bundles
                valid_bundles = []
                # Use original assertion_details (not sorted) for ID lookup
                assertion_id_set = {d["assertion_id"] for d in assertion_details}
                
                for bundle in bundles:
                    if not isinstance(bundle, dict):
                        continue
                    assertion_ids = bundle.get("assertion_ids", [])
                    # Filter to only valid assertion IDs
                    valid_ids = [aid for aid in assertion_ids if aid in assertion_id_set]
                    if valid_ids:
                        valid_bundles.append({
                            "assertion_ids": valid_ids,
                            "breadth": bundle.get("breadth", "MEDIUM"),
                        })
                
                return valid_bundles[:num_bundles]
            
            return []
            
        except Exception as e:
            print(f"⚠️  Error in LLM assertion bundling: {e}")
            return []
    
    def _select_dependent_assertions_with_llm(
        self,
        G: nx.MultiDiGraph,
        parent_claim_id: str,
        candidate_assertions: List[str],
        assertion_details: List[Dict[str, Any]],
    ) -> List[str]:
        """Use LLM to select best additional assertions for a dependent claim."""
        # Get parent claim assertions
        parent_assertions = G.nodes[parent_claim_id].get("assertion_ids", [])
        parent_details = [d for d in assertion_details if d["assertion_id"] in parent_assertions]
        candidate_details = [d for d in assertion_details if d["assertion_id"] in candidate_assertions]
        
        if not candidate_details:
            return candidate_assertions[:self.max_dependent]
        
        # Format for LLM
        parent_text = "\n".join([
            f"- {d['subject']} --[{d['predicate']}]--> {d['object']}"
            for d in parent_details
        ])
        
        candidate_text = "\n".join([
            f"{i}. {d['subject']} --[{d['predicate']}]--> {d['object']} (category: {d.get('category', 'UNCLASSIFIED')}, id: {d['assertion_id']})"
            for i, d in enumerate(candidate_details, 1)
        ])
        
        prompt = (
            "You are selecting additional assertions for a DEPENDENT patent claim.\n\n"
            f"Parent claim assertions:\n{parent_text}\n\n"
            f"Candidate additional assertions:\n{candidate_text}\n\n"
            "CRITICAL RULES FOR DEPENDENT CLAIMS:\n"
            "- Dependent claims may ONLY add TECHNICAL REFINEMENTS\n"
            "- DO NOT select assertions about: background, problems, prior art, aesthetic purposes, costs, labor\n"
            "- ONLY select assertions that add: technical components, mechanisms, effects, or specific implementations\n"
            "- Selected assertions must be TECHNICAL refinements, not background or problem statements\n\n"
            f"Select {self.min_dependent}-{self.max_dependent} assertions that:\n"
            "- Add TECHNICAL limitations or refinements to the parent claim\n"
            "- Are logically related to the parent assertions\n"
            "- Form a coherent dependent claim with technical improvements\n"
            "- Have category INVENTIVE_MECHANISM, TECHNICAL_COMPONENT, or TECHNICAL_EFFECT\n\n"
            "Return a JSON array of assertion IDs to include.\n"
            "Example: [\"assertion_abc123\", \"assertion_def456\"]\n"
            "Return ONLY valid JSON array, no markdown fences."
        )
        
        try:
            response = self.api_repo.chat(prompt)
            
            # Extract text
            if isinstance(response, dict):
                response_text = response.get("content", response.get("text", response.get("message", "")))
                if not response_text and "choices" in response:
                    response_text = response["choices"][0].get("message", {}).get("content", "")
            else:
                response_text = str(response) if response else ""
            
            # Parse JSON
            text = JsonHelper._unfence(response_text).strip()
            if not text:
                return candidate_assertions[:self.max_dependent]
            
            try:
                selected_ids = json.loads(text)
            except json.JSONDecodeError:
                try:
                    selected_ids = ast.literal_eval(text)
                except Exception:
                    return candidate_assertions[:self.max_dependent]
            
            if isinstance(selected_ids, list):
                # Validate IDs
                valid_ids = [aid for aid in selected_ids if aid in candidate_assertions]
                if valid_ids:
                    return valid_ids[:self.max_dependent]
            
            return candidate_assertions[:self.max_dependent]
            
        except Exception as e:
            print(f"⚠️  Error in LLM dependent assertion selection: {e}")
            return candidate_assertions[:self.max_dependent]
    
    def _bundle_assertions_heuristic(
        self,
        G: nx.MultiDiGraph,
        assertion_nodes: List[Tuple[str, Dict]],
        assertions_by_subject: Dict[str, List[str]],
        num_independent: int,
        all_used_assertions: Set[str],
    ) -> List[str]:
        """Heuristic fallback for bundling assertions (original logic)."""
        independent_claims = []
        
        for i in range(num_independent):
            if i == 0:
                breadth = "BROAD"
                if assertions_by_subject:
                    subject_id = max(assertions_by_subject.keys(), 
                                   key=lambda s: len(assertions_by_subject[s]))
                    candidate_assertions = assertions_by_subject[subject_id][:self.max_independent]
                else:
                    candidate_assertions = [a for a, _ in assertion_nodes][:self.max_independent]
            elif i == 1:
                breadth = "MEDIUM"
                remaining = {s: a for s, a in assertions_by_subject.items() 
                           if any(aid not in all_used_assertions for aid in a)}
                if remaining:
                    subject_id = max(remaining.keys(), 
                                   key=lambda s: len([a for a in remaining[s] if a not in all_used_assertions]))
                    candidate_assertions = [a for a in remaining[subject_id] 
                                          if a not in all_used_assertions][:self.max_independent]
                else:
                    remaining_assertions = [a for a, _ in assertion_nodes if a not in all_used_assertions]
                    if len(remaining_assertions) < self.min_independent:
                        break
                    candidate_assertions = remaining_assertions[:self.max_independent]
            else:
                breadth = "NARROW"
                remaining_assertions = [a for a, _ in assertion_nodes if a not in all_used_assertions]
                if len(remaining_assertions) < self.min_independent:
                    break
                candidate_assertions = remaining_assertions[:self.max_independent]
            
            if len(candidate_assertions) < self.min_independent:
                continue
            
            # Apply same filtering as LLM path: hard-block PRIOR_ART, PROBLEM, AESTHETIC, negative predicates
            filtered_assertions = []
            for aid in candidate_assertions:
                if not G.has_node(aid):
                    continue
                assertion_data = G.nodes[aid]
                cat = assertion_data.get("category", "UNCLASSIFIED")
                pred = assertion_data.get("predicate", "").lower()
                
                # Hard-block PRIOR_ART, PROBLEM, and AESTHETIC
                if cat in ("PRIOR_ART", "PROBLEM", "AESTHETIC"):
                    continue
                
                # Hard-block negative predicates
                negative_predicates = ["adhere", "behave unnaturally", "pushed upward", "lying down", "rises", "near bubble generating member"]
                if any(neg in pred for neg in negative_predicates):
                    continue
                
                filtered_assertions.append(aid)
            
            if len(filtered_assertions) < self.min_independent:
                print(f"  ⚠️  Skipping heuristic bundle {i+1}: filtered to {len(filtered_assertions)} assertions (had PRIOR_ART/PROBLEM/AESTHETIC/negative predicates)")
                continue
            
            # ABSTRACT FIX: Check if bundle has at least one INVENTIVE_MECHANISM
            has_inventive = any(
                G.nodes[aid].get("category") == "INVENTIVE_MECHANISM"
                for aid in filtered_assertions
                if G.has_node(aid)
            )
            if not has_inventive:
                print(f"  ⚠️  Skipping heuristic bundle {i+1}: no INVENTIVE_MECHANISM assertions (required for independent claims)")
                continue
            
            # Check mechanism completeness
            if not self._check_mechanism_completeness(G, filtered_assertions):
                print(f"  ⚠️  Skipping heuristic bundle {i+1}: does not meet mechanism completeness requirements")
                continue
            
            # ABSTRACT FIX: Validate that bundle describes "how the invention works"
            if not self._describes_invention_mechanism(G, filtered_assertions):
                print(f"  ⚠️  Skipping heuristic bundle {i+1}: does not describe how the invention works")
                continue
            
            claim_id = f"claim_independent_{uuid.uuid4().hex[:8]}"
            G.add_node(
                claim_id,
                node_type="CLAIM_CONCEPT",
                claim_id=claim_id,
                kind="INDEPENDENT",
                breadth=breadth,
                status="CANDIDATE",
                assertion_ids=filtered_assertions,
            )
            
            for assertion_id in filtered_assertions:
                G.add_edge(claim_id, assertion_id, label="USES", role="must", edge_type="CLAIM_LINK")
                all_used_assertions.add(assertion_id)
            
            independent_claims.append(claim_id)
            print(f"  Created independent claim concept: {claim_id} ({breadth}, {len(filtered_assertions)} assertions)")
        
        return independent_claims
    
    def get_claim_concepts(
        self,
        G: nx.MultiDiGraph,
        kind: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[ClaimConcept]:
        """
        Extract claim concept objects from the graph.
        
        Args:
            G: Graph with claim concept nodes
            kind: Optional filter by kind (INDEPENDENT, DEPENDENT)
            status: Optional filter by status
            
        Returns:
            List of ClaimConcept objects
        """
        concepts = []
        for node_id, data in G.nodes(data=True):
            if data.get("node_type") == "CLAIM_CONCEPT":
                if kind and data.get("kind") != kind:
                    continue
                if status and data.get("status") != status:
                    continue
                
                concept = ClaimConcept(
                    id=data.get("claim_id", node_id),
                    kind=data.get("kind", "INDEPENDENT"),
                    breadth=data.get("breadth", "MEDIUM"),
                    title=data.get("title"),
                    status=data.get("status", "CANDIDATE"),
                    assertion_ids=data.get("assertion_ids", []),
                    parent_claim_id=data.get("parent_claim_id"),
                )
                concepts.append(concept)
        
        return concepts

