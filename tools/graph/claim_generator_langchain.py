"""
LangChain-based Patent Claim Generator.

Uses two agents:
1. Planning Agent: Plans claim structure from patent description
2. Generation Agent: Generates each claim using GraphRAG to ensure triple adherence
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import json
import re

from tools.graph.graph_rag import GraphRAG
from tools.graph.Triple import Triple
from tools.api.llm_api_repo import LLmApi_Repo


@dataclass
class PlannedClaim:
    """A planned claim with its focus and metadata."""
    claim_number: int
    claim_type: str  # "independent" or "dependent"
    focus: str  # What this claim should focus on
    parent_claim_number: Optional[int] = None  # For dependent claims
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)


@dataclass
class GeneratedClaim:
    """A generated patent claim."""
    claim_number: int
    claim_text: str
    claim_type: str  # "independent" or "dependent"
    parent_claim_number: Optional[int] = None
    focus: str = ""
    used_triples: List[Dict[str, Any]] = field(default_factory=list)


class ClaimGeneratorLangChain:
    """
    LangChain-based patent claim generator.
    
    Uses two-stage approach:
    1. Planning: Analyzes patent description and plans claim structure
    2. Generation: Generates each claim using GraphRAG for triple adherence
    """
    
    def __init__(
        self,
        graph_rag: Optional[GraphRAG] = None,
        api_repo: Optional[LLmApi_Repo] = None,
    ):
        """
        Initialize the claim generator.
        
        Args:
            graph_rag: GraphRAG instance for retrieving relevant triples
            api_repo: LLM API repository for agent communication
        """
        self.graph_rag = graph_rag
        self.api_repo = api_repo or LLmApi_Repo()
    
    def plan_claims(
        self,
        patent_description: str,
        num_independent: int = 3,
        num_dependent_per_independent: int = 2,
    ) -> List[PlannedClaim]:
        """
        Plan the claim structure from patent description.
        
        Args:
            patent_description: The patent description text
            num_independent: Number of independent claims to plan
            num_dependent_per_independent: Number of dependent claims per independent
            
        Returns:
            List of PlannedClaim objects
        """
        # Get prioritized entities from GraphRAG if available
        entities_context = ""
        if self.graph_rag:
            try:
                entities_context = self.graph_rag.format_entities_for_prompt(max_entities=30)
                if entities_context:
                    entities_context = "\n\n" + entities_context
            except Exception as e:
                print(f"[DEBUG] plan_claims: Error getting entities from GraphRAG: {e}")
        
        prompt = f"""You are a patent claim planning expert. Analyze the following patent description and plan a structured set of patent claims.

Patent Description:
{patent_description[:3000]}  # Limit to avoid token limits{entities_context}

Based on the patent description, plan an appropriate number of independent and dependent claims. 
- Aim for approximately {num_independent} independent claims (but adjust if the invention requires more or fewer)
- For each independent claim, plan approximately {num_dependent_per_independent} dependent claims (but adjust based on what makes sense for that particular independent claim)
- The total number of claims should be reasonable and comprehensive, covering all major aspects of the invention
- You may have different numbers of dependent claims for different independent claims if that better reflects the invention

For each claim, provide:
- claim_number: Sequential number (1, 2, 3...)
- claim_type: "independent" or "dependent"
- focus: A clear description of what this claim should focus on (e.g., "water tank mechanism", "bubble generation system", "filtration component")
- parent_claim_number: For dependent claims, the number of the parent independent claim (null for independent claims)
- keywords: List of key terms relevant to this claim
- entities: List of entity names/components relevant to this claim

Return ONLY a JSON array of claim plans. Example format:
[
  {{
    "claim_number": 1,
    "claim_type": "independent",
    "focus": "Main water tank system with circulation",
    "parent_claim_number": null,
    "keywords": ["tank", "water", "circulation", "system"],
    "entities": ["Water Tank", "Circulation Pump"]
  }},
  {{
    "claim_number": 2,
    "claim_type": "dependent",
    "focus": "Tank opening mechanism",
    "parent_claim_number": 1,
    "keywords": ["opening", "mechanism", "access"],
    "entities": ["Opening Mechanism"]
  }}
]

Return ONLY the JSON array, no other text."""

        try:
            print(f"[DEBUG] plan_claims: Calling LLM API with prompt length: {len(prompt)}")
            print(f"[DEBUG] plan_claims: Patent description preview: {patent_description[:200] if patent_description else 'EMPTY'}...")
            
            response = self.api_repo.chat(prompt)
            
            print(f"[DEBUG] plan_claims: LLM response type: {type(response)}")
            print(f"[DEBUG] plan_claims: LLM response: {response}")
            
            # Extract JSON from response
            if isinstance(response, dict):
                response_text = response.get("content", response.get("text", ""))
                print(f"[DEBUG] plan_claims: Extracted from dict - keys: {list(response.keys())}")
            else:
                response_text = str(response)
                print(f"[DEBUG] plan_claims: Converted response to string")
            
            print(f"[DEBUG] plan_claims: Raw response_text length: {len(response_text)}")
            print(f"[DEBUG] plan_claims: Response text (first 500 chars): {response_text[:500]}")
            
            # Clean response text
            print(f"[DEBUG] plan_claims: Cleaning response text...")
            response_text = response_text.strip()
            print(f"[DEBUG] plan_claims: After strip, length: {len(response_text)}")
            
            if "```json" in response_text:
                print(f"[DEBUG] plan_claims: Found ```json marker")
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                print(f"[DEBUG] plan_claims: Found ``` marker")
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Try to extract JSON array if response contains other text
            if response_text.startswith("["):
                print(f"[DEBUG] plan_claims: Response starts with [, extracting array...")
                # Find the first [ and last ]
                start_idx = response_text.find("[")
                end_idx = response_text.rfind("]") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    response_text = response_text[start_idx:end_idx]
                    print(f"[DEBUG] plan_claims: Extracted array, new length: {len(response_text)}")
            
            print(f"[DEBUG] plan_claims: Cleaned response (first 500 chars): {response_text[:500]}")
            print(f"[DEBUG] plan_claims: Cleaned response (last 200 chars): {response_text[-200:]}")
            
            # Parse JSON
            print(f"[DEBUG] plan_claims: Attempting JSON parse...")
            plans_data = json.loads(response_text)
            print(f"[DEBUG] plan_claims: JSON parse successful! Type: {type(plans_data)}, Length: {len(plans_data) if isinstance(plans_data, list) else 'N/A'}")
            
            if not isinstance(plans_data, list):
                print(f"⚠️  Response is not a list: {type(plans_data)}")
                return self._default_plan(num_independent, num_dependent_per_independent)
            
            if len(plans_data) == 0:
                print(f"⚠️  Response is an empty list, using default plan")
                return self._default_plan(num_independent, num_dependent_per_independent)
            
            # Convert to PlannedClaim objects
            print(f"[DEBUG] plan_claims: Converting {len(plans_data)} plan items to PlannedClaim objects...")
            planned_claims = []
            for idx, plan_data in enumerate(plans_data):
                print(f"[DEBUG] plan_claims: Processing plan {idx+1}/{len(plans_data)}: {plan_data}")
                if not isinstance(plan_data, dict):
                    print(f"[DEBUG] plan_claims: ⚠️  Skipping invalid plan data (not a dict): {plan_data} (type: {type(plan_data)})")
                    continue
                
                claim_num = plan_data.get("claim_number", 0)
                claim_type = plan_data.get("claim_type", "independent")
                focus = plan_data.get("focus", "")
                print(f"[DEBUG] plan_claims: Creating PlannedClaim: number={claim_num}, type={claim_type}, focus={focus[:50]}...")
                
                planned_claims.append(PlannedClaim(
                    claim_number=claim_num,
                    claim_type=claim_type,
                    focus=focus,
                    parent_claim_number=plan_data.get("parent_claim_number"),
                    keywords=plan_data.get("keywords", []),
                    entities=plan_data.get("entities", []),
                ))
            
            print(f"[DEBUG] plan_claims: Created {len(planned_claims)} PlannedClaim objects")
            
            if len(planned_claims) == 0:
                print(f"[DEBUG] plan_claims: ⚠️  No valid planned claims after parsing, using default plan")
                return self._default_plan(num_independent, num_dependent_per_independent)
            
            print(f"[DEBUG] plan_claims: ✅ Successfully parsed {len(planned_claims)} planned claims")
            for pc in planned_claims:
                print(f"[DEBUG] plan_claims:   - Claim {pc.claim_number}: {pc.claim_type}, focus='{pc.focus[:50]}...'")
            return planned_claims
            
        except json.JSONDecodeError as e:
            print(f"[DEBUG] plan_claims: ⚠️  JSON decode error in claim planning: {type(e).__name__}: {e}")
            print(f"[DEBUG] plan_claims: Error at position: {e.pos if hasattr(e, 'pos') else 'N/A'}")
            print(f"[DEBUG] plan_claims: Response text length: {len(response_text) if 'response_text' in locals() else 'N/A'}")
            print(f"[DEBUG] plan_claims: Response text (first 1000 chars): {response_text[:1000] if 'response_text' in locals() else 'N/A'}")
            print(f"[DEBUG] plan_claims: Response text (last 500 chars): {response_text[-500:] if 'response_text' in locals() and len(response_text) > 500 else 'N/A'}")
            import traceback
            print(f"[DEBUG] plan_claims: Full traceback:")
            traceback.print_exc()
            # Return default plan if planning fails
            print(f"[DEBUG] plan_claims: Falling back to default plan due to JSON error")
            return self._default_plan(num_independent, num_dependent_per_independent)
        except Exception as e:
            print(f"[DEBUG] plan_claims: ⚠️  Error in claim planning: {type(e).__name__}: {e}")
            print(f"[DEBUG] plan_claims: Exception args: {e.args}")
            import traceback
            print(f"[DEBUG] plan_claims: Full traceback:")
            traceback.print_exc()
            # Return default plan if planning fails
            print(f"[DEBUG] plan_claims: Falling back to default plan due to exception")
            return self._default_plan(num_independent, num_dependent_per_independent)
    
    def _default_plan(self, num_independent: int, num_dependent_per_independent: int = 2) -> List[PlannedClaim]:
        """Generate a default plan if planning fails."""
        print(f"[DEBUG] _default_plan: Called with num_independent={num_independent}, num_dependent_per_independent={num_dependent_per_independent}")
        print(f"[DEBUG] _default_plan: Using default plan: {num_independent} independent claims")
        plans = []
        claim_num = 1
        
        # Add independent claims
        print(f"[DEBUG] _default_plan: Creating {num_independent} independent claims...")
        for i in range(1, num_independent + 1):
            print(f"[DEBUG] _default_plan: Creating independent claim {i} (claim_number={claim_num})")
            plans.append(PlannedClaim(
                claim_number=claim_num,
                claim_type="independent",
                focus=f"Main invention component {i}",
            ))
            claim_num += 1
            
            # Add dependent claims for this independent claim
            print(f"[DEBUG] _default_plan: Creating {num_dependent_per_independent} dependent claims for independent {i}...")
            for j in range(1, num_dependent_per_independent + 1):
                print(f"[DEBUG] _default_plan: Creating dependent claim {j} for parent {i} (claim_number={claim_num})")
                plans.append(PlannedClaim(
                    claim_number=claim_num,
                    claim_type="dependent",
                    focus=f"Variation or detail of component {i}",
                    parent_claim_number=i,
                ))
                claim_num += 1
        
        print(f"[DEBUG] _default_plan: ✅ Default plan created: {len(plans)} claims ({num_independent} independent, {len(plans) - num_independent} dependent)")
        print(f"[DEBUG] _default_plan: Returning plans: {plans}")
        return plans
    
    def generate_claim(
        self,
        planned_claim: PlannedClaim,
        patent_description: str,
        previous_claims: List[GeneratedClaim] = None,
        similarity_threshold: float = 0.3,
    ) -> GeneratedClaim:
        """
        Generate a single claim using GraphRAG for triple adherence.
        
        Args:
            planned_claim: The planned claim structure
            patent_description: The patent description
            previous_claims: Previously generated claims (for dependent claims)
            similarity_threshold: Minimum cosine similarity threshold for triples (0.0-1.0)
            
        Returns:
            GeneratedClaim object
        """
        if previous_claims is None:
            previous_claims = []
        
        # Retrieve relevant triples using GraphRAG
        relevant_triples = []
        if self.graph_rag:
            # Build query from planned claim focus and keywords
            query_parts = [planned_claim.focus]
            query_parts.extend(planned_claim.keywords)
            query_parts.extend(planned_claim.entities)
            query = " ".join(query_parts)
            
            # Find similar triples
            similar_triples = self.graph_rag.find_similar_triples(
                query_text=query,
                top_k=20,
                similarity_threshold=similarity_threshold,
            )
            
            # Convert to dict format
            for triple, similarity in similar_triples:
                relevant_triples.append({
                    "head": triple.head.name if hasattr(triple.head, 'name') else str(triple.head),
                    "relation": triple.relation,
                    "tail": triple.tail.name if hasattr(triple.tail, 'name') else str(triple.tail),
                    "similarity": similarity,
                })
        
        # Build context for claim generation
        context_parts = []
        
        if relevant_triples:
            context_parts.append("Relevant Knowledge Graph Triples:")
            for i, triple in enumerate(relevant_triples[:15], 1):  # Limit to top 15
                context_parts.append(
                    f"{i}. {triple['head']} --[{triple['relation']}]--> {triple['tail']}"
                )
        
        if previous_claims and planned_claim.claim_type == "dependent":
            context_parts.append("\nPrevious Claims (for reference):")
            for prev_claim in previous_claims:
                if prev_claim.claim_number == planned_claim.parent_claim_number:
                    context_parts.append(f"Claim {prev_claim.claim_number}: {prev_claim.claim_text}")
        
        context = "\n".join(context_parts)
        
        # Generate claim text
        claim_type_text = "independent" if planned_claim.claim_type == "independent" else "dependent"
        parent_text = ""
        if planned_claim.claim_type == "dependent" and planned_claim.parent_claim_number:
            parent_text = f"\nThis is a dependent claim that depends on claim {planned_claim.parent_claim_number}."
        
        prompt = f"""You are a patent claim drafting expert. Generate a formal patent claim based on the following information.

Patent Description (excerpt):
{patent_description[:2000]}

Claim Plan:
- Claim Number: {planned_claim.claim_number}
- Claim Type: {claim_type_text}
- Focus: {planned_claim.focus}
- Keywords: {', '.join(planned_claim.keywords) if planned_claim.keywords else 'N/A'}
- Relevant Entities: {', '.join(planned_claim.entities) if planned_claim.entities else 'N/A'}
{parent_text}

{context}

Instructions:
1. Draft a formal patent claim in standard patent claim format
2. The claim must accurately reflect the triples provided above
3. Use precise technical language
4. For independent claims: Start with "A [system/method/apparatus] comprising:"
5. For dependent claims: Start with "The [system/method/apparatus] of claim {planned_claim.parent_claim_number}, wherein..."
6. Ensure all mentioned components/features correspond to entities and relations in the knowledge graph
7. Number the claim as "{planned_claim.claim_number}."

Return ONLY the claim text, numbered as "{planned_claim.claim_number}." No additional explanation."""

        try:
            print(f"[DEBUG] generate_claim: Generating claim {planned_claim.claim_number} ({planned_claim.claim_type})")
            print(f"[DEBUG] generate_claim: Focus: {planned_claim.focus}")
            print(f"[DEBUG] generate_claim: Parent claim: {planned_claim.parent_claim_number}")
            print(f"[DEBUG] generate_claim: Relevant triples count: {len(relevant_triples)}")
            print(f"[DEBUG] generate_claim: Previous claims count: {len(previous_claims)}")
            print(f"[DEBUG] generate_claim: Prompt length: {len(prompt)}")
            print(f"[DEBUG] generate_claim: Prompt preview: {prompt[:300]}...")
            
            response = self.api_repo.chat(prompt)
            
            print(f"[DEBUG] generate_claim: LLM response type: {type(response)}")
            print(f"[DEBUG] generate_claim: LLM response: {response}")
            
            # Extract claim text
            if isinstance(response, dict):
                claim_text = response.get("content", response.get("text", ""))
                print(f"[DEBUG] generate_claim: Extracted from dict - keys: {list(response.keys())}")
            else:
                claim_text = str(response)
                print(f"[DEBUG] generate_claim: Converted response to string")
            
            print(f"[DEBUG] generate_claim: Raw claim_text length: {len(claim_text)}")
            print(f"[DEBUG] generate_claim: Raw claim_text: {claim_text[:500]}")
            
            # Clean up claim text
            claim_text = claim_text.strip()
            print(f"[DEBUG] generate_claim: After strip, length: {len(claim_text)}")
            
            # Remove markdown code blocks if present
            if "```" in claim_text:
                print(f"[DEBUG] generate_claim: Found ``` markers, removing...")
                lines = claim_text.split("\n")
                claim_text = "\n".join([l for l in lines if not l.strip().startswith("```")])
                print(f"[DEBUG] generate_claim: After removing ```, length: {len(claim_text)}")
            
            # Ensure claim number is correct
            if not claim_text.startswith(f"{planned_claim.claim_number}."):
                print(f"[DEBUG] generate_claim: Claim doesn't start with '{planned_claim.claim_number}.' - prepending")
                claim_text = f"{planned_claim.claim_number}. {claim_text}"
            
            print(f"[DEBUG] generate_claim: Final claim_text: {claim_text[:200]}...")
            
            return GeneratedClaim(
                claim_number=planned_claim.claim_number,
                claim_text=claim_text,
                claim_type=planned_claim.claim_type,
                parent_claim_number=planned_claim.parent_claim_number,
                focus=planned_claim.focus,
                used_triples=relevant_triples[:10],  # Store top 10 used triples
            )
            
        except Exception as e:
            print(f"[DEBUG] generate_claim: ⚠️  Error generating claim {planned_claim.claim_number}: {type(e).__name__}: {e}")
            print(f"[DEBUG] generate_claim: Exception args: {e.args}")
            import traceback
            print(f"[DEBUG] generate_claim: Full traceback:")
            traceback.print_exc()
            # Return placeholder claim
            error_claim = GeneratedClaim(
                claim_number=planned_claim.claim_number,
                claim_text=f"{planned_claim.claim_number}. [Error generating claim: {str(e)}]",
                claim_type=planned_claim.claim_type,
                parent_claim_number=planned_claim.parent_claim_number,
                focus=planned_claim.focus,
            )
            print(f"[DEBUG] generate_claim: Returning error placeholder claim")
            return error_claim
    
    def generate_all_claims(
        self,
        patent_description: str,
        triples: List[Triple],
        graph: Optional[Any] = None,
        id_to_name: Optional[Dict[str, str]] = None,
        num_independent: int = 3,
        num_dependent_per_independent: int = 2,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        similarity_threshold: float = 0.3,
    ) -> List[GeneratedClaim]:
        """
        Generate all claims: plan structure, then generate each claim.
        
        Args:
            patent_description: The patent description text
            triples: List of Triple objects
            graph: Optional NetworkX graph
            id_to_name: Optional mapping from entity ID to name
            num_independent: Number of independent claims
            num_dependent_per_independent: Number of dependent claims per independent
            
        Returns:
            List of GeneratedClaim objects (guaranteed to have at least 1 claim)
        """
        print(f"🚀 generate_all_claims called")
        print(f"🚀 Patent description: {patent_description[:200] if patent_description else 'EMPTY'}...")
        print(f"🚀 Triples count: {len(triples) if triples else 0}")
        
        # Initialize GraphRAG if not provided
        if self.graph_rag is None:
            print(f"🚀 Initializing GraphRAG...")
            self.graph_rag = GraphRAG(
                G=graph,
                triples=triples,
                id_to_name=id_to_name,
            )
            print(f"🚀 GraphRAG initialized")
        
        if progress_callback:
            progress_callback({"stage": "planning", "message": "Planning claim structure...", "progress": 0})
        
        print(f"📋 Planning claim structure...")
        print(f"📋 Patent description length: {len(patent_description)} chars")
        print(f"📋 Requesting: {num_independent} independent, {num_dependent_per_independent} dependent each")
        
        planned_claims = []
        try:
            print(f"[DEBUG] generate_all_claims: Calling plan_claims()...")
            print(f"[DEBUG] generate_all_claims: Parameters - patent_description length: {len(patent_description) if patent_description else 0}")
            print(f"[DEBUG] generate_all_claims: Parameters - num_independent: {num_independent}")
            print(f"[DEBUG] generate_all_claims: Parameters - num_dependent_per_independent: {num_dependent_per_independent}")
            
            planned_claims = self.plan_claims(
                patent_description=patent_description,
                num_independent=num_independent,
                num_dependent_per_independent=num_dependent_per_independent,
            )
            
            print(f"[DEBUG] generate_all_claims: plan_claims() returned")
            print(f"[DEBUG] generate_all_claims:   - Value: {planned_claims}")
            print(f"[DEBUG] generate_all_claims:   - Type: {type(planned_claims)}")
            print(f"[DEBUG] generate_all_claims:   - Is None: {planned_claims is None}")
            print(f"[DEBUG] generate_all_claims:   - Length: {len(planned_claims) if planned_claims else 'N/A'}")
            if planned_claims and len(planned_claims) > 0:
                print(f"[DEBUG] generate_all_claims:   - First planned claim: {planned_claims[0]}")
            else:
                print(f"[DEBUG] generate_all_claims:   - ⚠️  planned_claims is empty or None!")
        except Exception as e:
            print(f"[DEBUG] generate_all_claims: ⚠️  Exception during planning: {type(e).__name__}: {e}")
            import traceback
            print(f"[DEBUG] generate_all_claims: Full traceback:")
            traceback.print_exc()
            planned_claims = []
            print(f"[DEBUG] generate_all_claims: Set planned_claims to empty list after exception")
        
        print(f"[DEBUG] generate_all_claims: Checking if planned_claims is empty...")
        print(f"[DEBUG] generate_all_claims:   - planned_claims: {planned_claims}")
        print(f"[DEBUG] generate_all_claims:   - planned_claims is None: {planned_claims is None}")
        print(f"[DEBUG] generate_all_claims:   - planned_claims == []: {planned_claims == []}")
        print(f"[DEBUG] generate_all_claims:   - len(planned_claims) if exists: {len(planned_claims) if planned_claims else 'N/A'}")
        print(f"[DEBUG] generate_all_claims:   - bool(planned_claims): {bool(planned_claims)}")
        
        if not planned_claims or len(planned_claims) == 0:
            print(f"[DEBUG] generate_all_claims: ⚠️  WARNING: No claims planned! Using default plan.")
            print(f"[DEBUG] generate_all_claims: planned_claims value: {planned_claims}")
            print(f"[DEBUG] generate_all_claims: planned_claims is None: {planned_claims is None}")
            print(f"[DEBUG] generate_all_claims: len(planned_claims): {len(planned_claims) if planned_claims else 'N/A'}")
            try:
                print(f"[DEBUG] generate_all_claims: Calling _default_plan({num_independent}, {num_dependent_per_independent})...")
                default_result = self._default_plan(num_independent, num_dependent_per_independent)
                print(f"[DEBUG] generate_all_claims: _default_plan returned: {default_result}")
                print(f"[DEBUG] generate_all_claims: _default_plan type: {type(default_result)}")
                print(f"[DEBUG] generate_all_claims: _default_plan length: {len(default_result) if default_result else 'N/A'}")
                planned_claims = default_result
                print(f"[DEBUG] generate_all_claims: Assigned default_result to planned_claims")
                print(f"[DEBUG] generate_all_claims: planned_claims now: {planned_claims}")
                print(f"[DEBUG] generate_all_claims: len(planned_claims): {len(planned_claims) if planned_claims else 'N/A'}")
            except Exception as e:
                print(f"[DEBUG] generate_all_claims: ❌ ERROR: Default plan failed: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                # Force create at least one claim
                print(f"[DEBUG] generate_all_claims: Creating emergency fallback PlannedClaim...")
                planned_claims = [PlannedClaim(
                    claim_number=1,
                    claim_type="independent",
                    focus="Main invention",
                )]
                print(f"[DEBUG] generate_all_claims: Created emergency fallback plan: 1 claim")
                print(f"[DEBUG] generate_all_claims: planned_claims after emergency: {planned_claims}")
        
        print(f"[DEBUG] generate_all_claims: Final check before proceeding...")
        print(f"[DEBUG] generate_all_claims:   - planned_claims: {planned_claims}")
        print(f"[DEBUG] generate_all_claims:   - planned_claims is None: {planned_claims is None}")
        print(f"[DEBUG] generate_all_claims:   - len(planned_claims): {len(planned_claims) if planned_claims else 'N/A'}")
        
        if not planned_claims or len(planned_claims) == 0:
            print(f"[DEBUG] generate_all_claims: ❌ ERROR: Even emergency fallback failed! Creating minimal claim.")
            # Last resort - create a single claim manually
            planned_claims = [PlannedClaim(
                claim_number=1,
                claim_type="independent",
                focus="Main invention",
            )]
            print(f"[DEBUG] generate_all_claims: Created minimal fallback plan")
            print(f"[DEBUG] generate_all_claims: planned_claims after minimal: {planned_claims}")
        
        print(f"[DEBUG] generate_all_claims: ✅ Planned {len(planned_claims)} claims")
        print(f"[DEBUG] generate_all_claims: planned_claims type: {type(planned_claims)}")
        print(f"[DEBUG] generate_all_claims: planned_claims is None: {planned_claims is None}")
        print(f"[DEBUG] generate_all_claims: len(planned_claims): {len(planned_claims) if planned_claims else 'N/A'}")
        if planned_claims:
            print(f"[DEBUG] generate_all_claims: First planned claim: {planned_claims[0]}")
            print(f"[DEBUG] generate_all_claims: All planned claims:")
            for pc in planned_claims:
                print(f"[DEBUG] generate_all_claims:   - {pc}")
        
        if progress_callback:
            print(f"[DEBUG] generate_all_claims: Calling progress_callback with planning_complete, num_claims={len(planned_claims)}")
            progress_callback({
                "stage": "planning_complete",
                "message": f"Planned {len(planned_claims)} claims",
                "progress": 20,
                "num_claims": len(planned_claims),
            })
        
        # Generate claims in order
        generated_claims = []
        total_claims = len(planned_claims)
        
        if total_claims == 0:
            print(f"❌ ERROR: No planned claims to generate!")
            raise ValueError("No planned claims available for generation")
        
        print(f"🔄 Starting generation of {total_claims} claims...")
        
        for idx, planned_claim in enumerate(planned_claims):
            if progress_callback:
                progress = 20 + int((idx / total_claims) * 80)
                progress_callback({
                    "stage": "generating",
                    "message": f"Generating claim {planned_claim.claim_number} ({planned_claim.claim_type})...",
                    "progress": progress,
                    "current_claim": planned_claim.claim_number,
                    "total_claims": total_claims,
                })
            
            print(f"✍️  Generating claim {planned_claim.claim_number} ({planned_claim.claim_type})...")
            
            try:
                generated_claim = self.generate_claim(
                    planned_claim=planned_claim,
                    patent_description=patent_description,
                    previous_claims=generated_claims,
                    similarity_threshold=similarity_threshold,
                )
                
                if generated_claim:
                    generated_claims.append(generated_claim)
                    print(f"✅ Generated claim {planned_claim.claim_number}: {generated_claim.claim_text[:100] if generated_claim.claim_text else 'NO TEXT'}...")
                else:
                    print(f"⚠️  Claim {planned_claim.claim_number} generation returned None")
            except Exception as e:
                print(f"[DEBUG] generate_all_claims: ⚠️  Error generating claim {planned_claim.claim_number}: {type(e).__name__}: {e}")
                print(f"[DEBUG] generate_all_claims: Exception args: {e.args}")
                import traceback
                print(f"[DEBUG] generate_all_claims: Full traceback:")
                traceback.print_exc()
                # Continue with next claim even if one fails
                print(f"[DEBUG] generate_all_claims: Continuing with next claim...")
                continue
        
        print(f"✅ Finished generating {len(generated_claims)} out of {total_claims} planned claims")
        
        # CRITICAL: Ensure we always return at least one claim
        print(f"[DEBUG] generate_all_claims: Final check - generated_claims count: {len(generated_claims)}")
        if len(generated_claims) == 0:
            print(f"[DEBUG] generate_all_claims: ⚠️  WARNING: No claims were generated! Creating emergency fallback claim.")
            print(f"[DEBUG] generate_all_claims: total_claims was: {total_claims}")
            print(f"[DEBUG] generate_all_claims: planned_claims count: {len(planned_claims) if planned_claims else 0}")
            # Create a minimal fallback claim
            fallback_claim = GeneratedClaim(
                claim_number=1,
                claim_text="1. A system comprising components as described in the patent description.",
                claim_type="independent",
                focus="Main invention",
            )
            generated_claims = [fallback_claim]
            print(f"[DEBUG] generate_all_claims: ✅ Created emergency fallback claim")
        
        if progress_callback:
            progress_callback({
                "stage": "complete",
                "message": f"Successfully generated {len(generated_claims)} claims!",
                "progress": 100,
                "num_claims": len(generated_claims),
            })
        
        print(f"🎉 Returning {len(generated_claims)} claims")
        return generated_claims

