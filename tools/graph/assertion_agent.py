"""
AssertionAgent: Adds assertion nodes to the graph based on existing relations.
Uses LLM to intelligently determine assertion placement, confidence, and qualifiers.
Assertions represent atomic statements with provenance.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
import uuid
import networkx as nx
import json
import ast

from tools.graph.Triple import Triple
from tools.graph.visualizer import GraphVisualizer
from tools.api.llm_api_repo import LLmApi_Repo
from tools.helper.json_helper import JsonHelper


@dataclass
class Assertion:
    """Represents an atomic assertion statement."""
    id: str
    predicate: str  # The relation/predicate
    value: Optional[str] = None  # Literal value if object is not an entity
    qualifiers: Dict[str, Any] = field(default_factory=dict)  # Additional qualifiers
    confidence: float = 1.0  # Confidence score [0, 1]
    status: str = "CANDIDATE"  # CANDIDATE, CONFIRMED, REJECTED
    category: str = "UNCLASSIFIED"  # INVENTIVE_MECHANISM, TECHNICAL_COMPONENT, TECHNICAL_EFFECT, BACKGROUND, PROBLEM, PRIOR_ART, AESTHETIC
    claim_eligible: bool = False  # Whether this assertion is eligible for patent claims
    evidence_span: Optional[Tuple[int, int]] = None  # Character span in source text
    source_triple_id: Optional[str] = None  # ID of original triple if derived from KG edge


class AssertionAgent:
    """
    Enriches a graph with Assertion nodes based on existing relations.
    
    For each relation edge in the graph, creates an Assertion node that:
    - Links to the subject entity (SUBJECT relation)
    - Links to the object entity (OBJECT relation) OR stores a literal value
    - Optionally links to evidence (SUPPORTED_BY relation)
    - Optionally links back to the original edge (DERIVED_FROM relation)
    """
    
    def __init__(
        self,
        include_evidence: bool = False,
        default_confidence: float = 1.0,
        api_repo: Optional[LLmApi_Repo] = None,
        use_llm: bool = True,
        inventive_description: Optional[str] = None,
    ):
        """
        Initialize the assertion agent.
        
        Args:
            include_evidence: If True, creates evidence nodes and links assertions to them
            default_confidence: Default confidence score for assertions
            api_repo: Optional LLM API repository (defaults to LLmApi_Repo())
            use_llm: If True, uses LLM to determine assertion confidence and qualifiers
            inventive_description: Optional one-sentence description of the inventive part of the product
                                  (helps classify assertions as INVENTIVE_MECHANISM)
        """
        self.include_evidence = include_evidence
        self.default_confidence = default_confidence
        self.visualizer = GraphVisualizer()
        self.api_repo = api_repo or LLmApi_Repo()
        self.use_llm = use_llm
        self.inventive_description = inventive_description
    
    def run(self, G: nx.MultiDiGraph, triples: Optional[List[Triple]] = None) -> nx.MultiDiGraph:
        """
        Add assertion nodes to the graph based on existing edges.
        
        Args:
            G: NetworkX MultiDiGraph with entities and relations
            triples: Optional list of Triple objects for traceability
            
        Returns:
            Modified graph G with assertion nodes added
        """
        # Build mapping from edges to triples if provided
        edge_to_triple: Dict[Tuple[str, str, str], Triple] = {}
        if triples:
            for triple in triples:
                head_id = self.visualizer._entity_key(triple.head)
                tail_id = self.visualizer._entity_key(triple.tail)
                relation = getattr(triple, "relation", "").strip()
                if head_id and tail_id and relation:
                    edge_to_triple[(head_id, tail_id, relation)] = triple
        
        # Track processed edges to avoid duplicates
        processed_edges: Set[Tuple[str, str, str]] = set()
        
        # Cache all node names first to avoid accessing graph during iteration
        # Materialize node list first to avoid iteration issues
        all_nodes = list(G.nodes(data=True))
        node_name_cache: Dict[str, str] = {}
        for node_id, node_data in all_nodes:
            # Extract name directly from node data to avoid graph access
            name = node_id
            for key in ["name", "label", "text", "display_name"]:
                if key in node_data:
                    name = str(node_data[key])
                    break
            node_name_cache[node_id] = name
        
        # Collect all edge data first to avoid "dictionary changed size during iteration" error
        # Force materialization of the edge iterator
        all_edges = list(G.edges(keys=True, data=True))
        
        edges_data = []
        for u, v, k, data in all_edges:
            # Skip assertion links and claim links (we only process entity-to-entity edges)
            if data.get("edge_type") in ("ASSERTION_LINK", "CLAIM_LINK"):
                continue
            
            relation = data.get("label", "").strip()
            if not relation:
                continue
            
            # Skip if we've already processed this edge
            edge_key = (u, v, relation)
            if edge_key in processed_edges:
                continue
            processed_edges.add(edge_key)
            
            # Use cached names instead of accessing graph
            subject_name = node_name_cache.get(u, u)
            object_name = node_name_cache.get(v, v)
            
            edges_data.append({
                "u": u,
                "v": v,
                "k": k,
                "relation": relation,
                "subject_name": subject_name,
                "object_name": object_name,
                "edge_key": edge_key,
            })
        
        # Now process all edges and modify graph (no iteration over G during this phase)
        assertions_created = 0
        for edge_info in edges_data:
            u = edge_info["u"]
            v = edge_info["v"]
            k = edge_info["k"]
            relation = edge_info["relation"]
            subject_name = edge_info["subject_name"]
            object_name = edge_info["object_name"]
            edge_key = edge_info["edge_key"]
            
            # Use LLM to determine assertion properties if enabled
            confidence = self.default_confidence
            qualifiers = {}
            status = "CANDIDATE"
            category = "UNCLASSIFIED"
            claim_eligible = False
            
            if self.use_llm:
                llm_result = self._analyze_assertion_with_llm(
                    subject_name, relation, object_name, u, v
                )
                if llm_result:
                    confidence = llm_result.get("confidence", self.default_confidence)
                    qualifiers = llm_result.get("qualifiers", {})
                    status = llm_result.get("status", "CANDIDATE")
                    category = llm_result.get("category", "UNCLASSIFIED")
                    claim_eligible = llm_result.get("claim_eligible", False)
            
            # HARD RULE 1: Forbid assertions expressing problems, disadvantages, costs, biological requirements, or unnatural behavior
            if self._is_forbidden_assertion(subject_name, relation, object_name):
                claim_eligible = False
                if category not in ("PROBLEM", "BACKGROUND", "PRIOR_ART"):
                    category = "PROBLEM"  # Reclassify as PROBLEM if not already
                status = "REJECTED"
            
            # Create assertion node
            assertion_id = f"assertion_{uuid.uuid4().hex[:8]}"
            assertion = Assertion(
                id=assertion_id,
                predicate=relation,
                confidence=confidence,
                status=status,
                qualifiers=qualifiers,
                category=category,
                claim_eligible=claim_eligible,
            )
            
            # Link to source triple if available
            if edge_key in edge_to_triple:
                triple = edge_to_triple[edge_key]
                assertion.source_triple_id = getattr(triple, "id", None)
            
            # Add assertion node to graph
            G.add_node(
                assertion_id,
                node_type="ASSERTION",
                assertion_id=assertion.id,
                predicate=assertion.predicate,
                value=assertion.value,
                qualifiers=assertion.qualifiers,
                confidence=assertion.confidence,
                status=assertion.status,
                category=assertion.category,
                claim_eligible=assertion.claim_eligible,
                source_triple_id=assertion.source_triple_id,
            )
            
            # Link assertion to subject entity (SUBJECT relation)
            # u should be an entity node (head of the relation)
            # Use cached check - if u is in our name cache, it exists
            if u in node_name_cache:
                G.add_edge(assertion_id, u, label="SUBJECT", edge_type="ASSERTION_LINK")
            
            # Link assertion to object entity (OBJECT relation)
            # v should be an entity node (tail of the relation)
            # Use cached check - if v is in our name cache, it exists
            if v in node_name_cache:
                G.add_edge(assertion_id, v, label="OBJECT", edge_type="ASSERTION_LINK")
            else:
                # If v is not a node, it might be a literal value
                # Store it in the assertion node
                G.nodes[assertion_id]["value"] = str(v)
            
            # Link back to original edge if traceability is desired
            # We store the edge reference in the assertion node
            G.nodes[assertion_id]["derived_from_edge"] = (u, v, k)
            
            assertions_created += 1
        
        # Count claim-eligible vs non-claim-eligible
        claim_eligible_count = sum(
            1 for node_id in G.nodes()
            if G.nodes[node_id].get("node_type") == "ASSERTION"
            and G.nodes[node_id].get("claim_eligible", False) is True
        )
        
        # Count by category
        category_counts = {}
        for node_id in G.nodes():
            if G.nodes[node_id].get("node_type") == "ASSERTION":
                cat = G.nodes[node_id].get("category", "UNCLASSIFIED")
                category_counts[cat] = category_counts.get(cat, 0) + 1
        
        print(f"✅ Created {assertions_created} assertion nodes")
        print(f"   Claim-eligible: {claim_eligible_count} / {assertions_created}")
        print(f"   Categories: {category_counts}")
        return G
    
    def _get_entity_name(self, G: nx.MultiDiGraph, entity_id: str) -> str:
        """Get display name for an entity node."""
        if G.has_node(entity_id):
            node_data = G.nodes[entity_id]
            for key in ["name", "label", "text", "display_name"]:
                if key in node_data:
                    return str(node_data[key])
        return entity_id
    
    def _is_forbidden_assertion(self, subject: str, predicate: str, obj: str) -> bool:
        """
        HARD RULE 1: Check if assertion expresses problems, disadvantages, costs, 
        biological requirements, or unnatural behavior.
        
        These are FORBIDDEN in claims (non-negotiable).
        """
        combined = f"{subject} {predicate} {obj}".lower()
        predicate_lower = predicate.lower()
        
        # Problem/disadvantage indicators
        problem_terms = [
            "problem", "disadvantage", "limitation", "drawback", "issue", "difficulty",
            "fails", "failure", "defect", "error", "fault", "weakness", "shortcoming"
        ]
        
        # Cost/labor indicators
        cost_terms = [
            "cost", "expensive", "cheap", "price", "labor", "maintenance", "maintain",
            "upkeep", "effort", "time-consuming", "requires daily", "must be fed",
            "feeding", "care", "attention"
        ]
        
        # Biological requirement indicators
        biological_terms = [
            "feeding", "food", "temperature", "water temperature", "living conditions",
            "must be maintained", "biological", "survival", "health", "disease",
            "illness", "sick", "die", "death", "life", "living", "alive"
        ]
        
        # Unnatural behavior indicators
        unnatural_terms = [
            "behave unnaturally", "unnatural behavior", "unnatural", "imbalance",
            "imbalanced", "adhere", "adhesion", "pushed upward", "lying down",
            "lies down", "sudden rising", "rises suddenly", "near bubble generating member rises"
        ]
        
        # Check predicate for forbidden terms
        if any(term in predicate_lower for term in problem_terms + cost_terms + biological_terms + unnatural_terms):
            return True
        
        # Check combined text for forbidden patterns
        if any(term in combined for term in problem_terms + cost_terms + biological_terms + unnatural_terms):
            return True
        
        # Specific patterns
        if "behaves" in predicate_lower and "unnaturally" in combined:
            return True
        
        if "adhere" in predicate_lower or "adhesion" in combined:
            return True
        
        return False
    
    def _promote_to_inventive_mechanism(
        self, subject: str, predicate: str, obj: str, current_category: str
    ) -> str:
        """
        Promote assertions to INVENTIVE_MECHANISM based on pattern matching and inventive description.
        
        Uses both hardcoded patterns and the user-provided inventive description to identify
        assertions that describe the core inventive mechanism.
        """
        subject_lower = subject.lower()
        predicate_lower = predicate.lower()
        obj_lower = obj.lower()
        combined = f"{subject_lower} {predicate_lower} {obj_lower}".lower()
        
        # If inventive description is provided, check if assertion matches key terms
        if self.inventive_description:
            inventive_lower = self.inventive_description.lower()
            # Extract key terms from inventive description (simple heuristic)
            inventive_terms = set(inventive_lower.split())
            # Remove common stop words
            stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being", 
                         "have", "has", "had", "do", "does", "did", "will", "would", "could", 
                         "should", "may", "might", "can", "to", "of", "in", "on", "at", "by", 
                         "for", "with", "from", "as", "and", "or", "but", "that", "this", "these", "those"}
            inventive_key_terms = inventive_terms - stop_words
            
            # Check if assertion contains multiple key terms from inventive description
            assertion_terms = set(combined.split())
            matching_terms = inventive_key_terms & assertion_terms
            
            # If assertion contains 2+ key terms from inventive description, likely inventive
            if len(matching_terms) >= 2:
                return "INVENTIVE_MECHANISM"
        
        # Pattern 1: Elevated water storage/reservoir above tank
        if any(term in combined for term in ["water storage", "water reservoir", "reservoir"]) and \
           any(term in predicate_lower for term in ["upward", "above", "elevated", "installed upwardly"]):
            return "INVENTIVE_MECHANISM"
        
        # Pattern 2: Water pipe connecting tank ↔ storage
        if "water pipe" in combined and \
           any(term in predicate_lower for term in ["connect", "connected", "leads", "transports"]):
            if any(term in combined for term in ["water storage", "water tank", "tank", "storage"]):
                return "INVENTIVE_MECHANISM"
        
        # Pattern 3: Opening portion + free-fall/falls
        if any(term in combined for term in ["opening portion", "opening"]) and \
           any(term in predicate_lower for term in ["free-fall", "falls", "fall", "drops", "drops into"]):
            return "INVENTIVE_MECHANISM"
        
        # Pattern 4: Air bubble + led into water pipe (not directly into tank)
        if "air bubble" in combined and \
           any(term in predicate_lower for term in ["led into", "inject", "injected", "into water pipe", "inside water pipe"]):
            if "water pipe" in combined:
                return "INVENTIVE_MECHANISM"
        
        # Pattern 5: Convection generation via liquid flow
        if "convection" in combined and \
           any(term in predicate_lower for term in ["generate", "generates", "causes", "creates", "produces"]):
            if any(term in combined for term in ["liquid flow", "liquid current", "water flow", "opening portion"]):
                return "INVENTIVE_MECHANISM"
        
        return current_category
    
    def _analyze_assertion_with_llm(
        self,
        subject: str,
        predicate: str,
        obj: str,
        subject_id: str,
        object_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to analyze an assertion and determine confidence, qualifiers, category, and claim eligibility.
        
        Returns:
            Dict with keys: confidence, qualifiers, status, category, claim_eligible
        """
        # Build prompt with inventive description if provided
        inventive_context = ""
        if self.inventive_description:
            inventive_context = (
                f"\nINVENTIVE PART OF THE PRODUCT:\n"
                f"{self.inventive_description}\n\n"
                "Use this description to identify assertions that describe the core inventive mechanism.\n"
                "Assertions that match key aspects of this invention should be classified as INVENTIVE_MECHANISM.\n\n"
            )
        
        prompt = (
            "You are analyzing a knowledge graph assertion for a patent claim generation system.\n\n"
            f"Assertion: {subject} --[{predicate}]--> {obj}\n\n"
            f"{inventive_context}"
            "CRITICAL: Classify this assertion and determine if it's claim-eligible.\n\n"
            "ASSERTION CATEGORIES:\n"
            "- INVENTIVE_MECHANISM: Describes a novel technical mechanism, structure, or method (especially if it matches the inventive part described above)\n"
            "- TECHNICAL_COMPONENT: Describes a physical component, device, or system part\n"
            "- TECHNICAL_EFFECT: Describes a technical function, operation, or result\n"
            "- BACKGROUND: General background information, context, or prior state\n"
            "- PROBLEM: Describes a problem, disadvantage, or limitation\n"
            "- PRIOR_ART: Describes existing solutions, prior art, or known methods\n"
            "- AESTHETIC: Describes aesthetic, visual, or non-technical aspects\n\n"
            "CLAIM ELIGIBILITY RULES:\n"
            "An assertion is claim-eligible ONLY if:\n"
            "1. It describes CONCRETE technical structure or mechanism (INVENTIVE_MECHANISM, TECHNICAL_COMPONENT, TECHNICAL_EFFECT)\n"
            "2. It is NOT about: costs/labor, biological necessities, disadvantages, aesthetic purpose, prior art, problems, or background\n"
            "3. It describes something that can be claimed as a patentable feature\n\n"
            "HARD RULE - FORBIDDEN IN CLAIMS (claim_eligible MUST be false):\n"
            "- Problem statements (problem, disadvantage, limitation, drawback, issue, difficulty, fails, failure, defect, error, fault, weakness, shortcoming)\n"
            "- Cost/labor statements (cost, expensive, labor, maintenance, upkeep, effort, time-consuming, requires daily)\n"
            "- Biological requirements (feeding, food, temperature, water temperature, living conditions, must be maintained, biological, survival, health)\n"
            "- Unnatural behavior (behaves unnaturally, unnatural behavior, imbalance, imbalanced, adhere, adhesion, pushed upward, lying down, sudden rising)\n"
            "- Background information\n"
            "- Prior art descriptions\n"
            "- Aesthetic purposes\n"
            "- Generic thematic statements\n\n"
            "If the assertion contains ANY of these forbidden elements, you MUST set claim_eligible=false and category=PROBLEM (or appropriate category).\n\n"
            "Return a JSON object with:\n"
            "- confidence: float between 0.0 and 1.0 (how certain is this assertion?)\n"
            "- qualifiers: dict with optional keys like 'temporal', 'spatial', 'conditional', etc.\n"
            "- status: 'CANDIDATE', 'CONFIRMED', or 'REJECTED' (reject if clearly wrong or non-claimable)\n"
            "- category: One of the categories above (INVENTIVE_MECHANISM, TECHNICAL_COMPONENT, TECHNICAL_EFFECT, BACKGROUND, PROBLEM, PRIOR_ART, AESTHETIC)\n"
            "- claim_eligible: boolean (true ONLY if it describes concrete technical structure/mechanism)\n\n"
            "Return ONLY valid JSON, no markdown fences.\n"
            'Example: {"confidence": 0.95, "qualifiers": {"temporal": "during operation"}, "status": "CONFIRMED", "category": "INVENTIVE_MECHANISM", "claim_eligible": true}'
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
            
            # Parse JSON
            text = JsonHelper._unfence(response_text).strip()
            if not text:
                return None
            
            try:
                result = json.loads(text)
            except json.JSONDecodeError:
                try:
                    result = ast.literal_eval(text)
                except Exception:
                    return None
            
            if isinstance(result, dict):
                # Validate and normalize
                confidence = result.get("confidence", self.default_confidence)
                confidence = max(0.0, min(1.0, float(confidence)))
                
                category = result.get("category", "UNCLASSIFIED")
                claim_eligible = bool(result.get("claim_eligible", False))
                
                # FIX 1: Pattern-based promotion to INVENTIVE_MECHANISM
                category = self._promote_to_inventive_mechanism(subject, predicate, obj, category)
                
                # Auto-reject non-claimable categories
                if category in ("BACKGROUND", "PROBLEM", "PRIOR_ART", "AESTHETIC"):
                    claim_eligible = False
                
                return {
                    "confidence": confidence,
                    "qualifiers": result.get("qualifiers", {}),
                    "status": result.get("status", "CANDIDATE"),
                    "category": category,
                    "claim_eligible": claim_eligible,
                }
            
            return None
            
        except Exception as e:
            print(f"⚠️  Error in LLM assertion analysis: {e}")
            return None
    
    def get_assertions(self, G: nx.MultiDiGraph, status: Optional[str] = None) -> List[Assertion]:
        """
        Extract assertion objects from the graph.
        
        Args:
            G: Graph with assertion nodes
            status: Optional filter by status (CANDIDATE, CONFIRMED, REJECTED)
            
        Returns:
            List of Assertion objects
        """
        assertions = []
        for node_id, data in G.nodes(data=True):
            if data.get("node_type") == "ASSERTION":
                if status and data.get("status") != status:
                    continue
                
                assertion = Assertion(
                    id=data.get("assertion_id", node_id),
                    predicate=data.get("predicate", ""),
                    value=data.get("value"),
                    qualifiers=data.get("qualifiers", {}),
                    confidence=data.get("confidence", 1.0),
                    status=data.get("status", "CANDIDATE"),
                    category=data.get("category", "UNCLASSIFIED"),
                    claim_eligible=data.get("claim_eligible", False),
                    source_triple_id=data.get("source_triple_id"),
                )
                assertions.append(assertion)
        
        return assertions

