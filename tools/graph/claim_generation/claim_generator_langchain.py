"""
LangChain-based Patent Claim Generator.

Uses three agents:
1. Planning Agent: Plans claim structure from patent description
2. Generation Agent: Generates each claim using GraphRAG to ensure triple adherence
3. Judging Agent: Evaluates claims for unity and proposes improvements (with LangGraph refinement loop)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any, Callable, TypedDict
from dataclasses import dataclass, field
import json
import re

from tools.graph.rag.graph_rag import GraphRAG
from tools.graph.data.Triple import Triple
from tools.api.llm_api_repo import LLmApi_Repo

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("[WARNING] LangGraph not available. Refinement loop will be disabled.")


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
    prompt: str = ""  # The prompt used to generate this claim
    refinement_iterations: int = 0  # Number of refinement iterations performed
    final_score: float = 0.0  # Final score from judging agent


class ClaimRefinementState(TypedDict):
    """State for claim refinement loop."""
    claims: List[GeneratedClaim]
    claim_index: int  # Current claim being refined
    iteration: int  # Current iteration (0-indexed)
    scores: List[float]  # Scores for each claim
    criticisms: List[str]  # Criticisms for each claim
    patent_description: str
    planned_claim: PlannedClaim
    previous_claims: List[GeneratedClaim]
    similarity_threshold: float
    max_iterations: int
    min_score: float


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
        
        prompt = f"""You are a patent claim planning expert. Analyze the following patent description and plan a structured set of patent claims that will be defensible, internally consistent, and commercially meaningful.

Patent Description:
{patent_description[:3000]}  # Limit to avoid token limits{entities_context}

Based on the patent description, plan an appropriate number of independent and dependent claims.
- Aim for approximately {num_independent} independent claims (but adjust if the invention requires more or fewer)
- For each independent claim, plan approximately {num_dependent_per_independent} dependent claims (but adjust based on what makes sense for that particular independent claim)
- The total number of claims should be reasonable and comprehensive, covering all major aspects of the invention
- You may have different numbers of dependent claims for different independent claims if that better reflects the invention

Planning principles:
- Plan so that the resulting claims read as proper patent claims (clear legal scope, consistent terminology), not as engineering specifications or implementation details.
- Each independent claim should define a distinct, defensible scope that could be commercially enforced; avoid planning claims that are so narrow they cover only one trivial implementation.
- Ensure the plan supports internal consistency: the same concepts should be referred to with the same terms across claims, and dependents should logically narrow their parent without contradicting it.
- Extract real components and features from the description, but frame the focus in terms of what the claim should protect (inventive concept and scope), not just a list of technical details.

For each claim, provide:
- claim_number: Sequential number (1, 2, 3...)
- claim_type: "independent" or "dependent"
- focus: A clear description (2-4 sentences) of what this claim should focus on, using actual components/features from the patent description. For independent claims: the main inventive concept and key elements that define a defensible scope. For dependent claims: a specific limitation that narrows the parent in a consistent way. Be concrete but avoid encouraging over-specification or spec-style drafting.
- parent_claim_number: For dependent claims, the number of the parent independent claim (null for independent claims)
- keywords: List of key terms relevant to this claim (extract actual terms from the description)
- entities: List of entity names/components relevant to this claim (extract actual component names from the description)

Return ONLY a JSON array of claim plans. Example structure:
[
  {{
    "claim_number": 1,
    "claim_type": "independent",
    "focus": "Focus on [ACTUAL MAIN COMPONENT FROM DESCRIPTION]. This claim should cover [its structure/function as described]. Include details about [specific features mentioned in the patent].",
    "parent_claim_number": null,
    "keywords": ["actual", "terms", "from", "description"],
    "entities": ["Actual Component Name", "Another Component"]
  }},
  {{
    "claim_number": 2,
    "claim_type": "dependent",
    "focus": "This is a dependent claim of claim 1. Focus specifically on [ACTUAL FEATURE/DETAIL FROM DESCRIPTION] - [the specific aspect mentioned in the patent]. Detail [how it works/where it's located/its characteristics as described].",
    "parent_claim_number": 1,
    "keywords": ["actual", "feature", "terms"],
    "entities": ["Actual Feature Name"]
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
            
            # Parse JSON with error handling
            print(f"[DEBUG] plan_claims: Attempting JSON parse...")
            plans_data = None
            try:
                plans_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"[DEBUG] plan_claims: JSON parse error: {e}, attempting to fix...")
                # Try to extract just the JSON array more carefully
                # Find the first complete JSON array by counting brackets
                bracket_count = 0
                start_pos = response_text.find("[")
                if start_pos >= 0:
                    for i in range(start_pos, len(response_text)):
                        if response_text[i] == "[":
                            bracket_count += 1
                        elif response_text[i] == "]":
                            bracket_count -= 1
                            if bracket_count == 0:
                                # Found complete JSON array
                                extracted_text = response_text[start_pos:i+1]
                                print(f"[DEBUG] plan_claims: Extracted complete array (length: {len(extracted_text)})")
                                try:
                                    plans_data = json.loads(extracted_text)
                                    print(f"[DEBUG] plan_claims: Successfully parsed JSON after extraction")
                                except json.JSONDecodeError as e2:
                                    print(f"[DEBUG] plan_claims: Still can't parse extracted JSON: {e2}")
                                    # Try to fix common JSON issues
                                    try:
                                        # Fix trailing commas
                                        fixed_text = re.sub(r',\s*}', '}', extracted_text)
                                        fixed_text = re.sub(r',\s*]', ']', fixed_text)
                                        # Fix unescaped quotes in strings (basic attempt)
                                        plans_data = json.loads(fixed_text)
                                        print(f"[DEBUG] plan_claims: Successfully parsed JSON after fixing trailing commas")
                                    except json.JSONDecodeError as e3:
                                        print(f"[DEBUG] plan_claims: All JSON parsing attempts failed, using fallback")
                                        plans_data = None
                                break
                
                if plans_data is None:
                    print(f"[DEBUG] plan_claims: Could not extract valid JSON, using fallback plan")
                    return self._default_plan(num_independent, num_dependent_per_independent)
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
                focus=f"Focus on a main component or system from the patent description. Describe its structure, function, and key features as mentioned in the description.",
            ))
            claim_num += 1
            
            # Add dependent claims for this independent claim
            print(f"[DEBUG] _default_plan: Creating {num_dependent_per_independent} dependent claims for independent {i}...")
            for j in range(1, num_dependent_per_independent + 1):
                print(f"[DEBUG] _default_plan: Creating dependent claim {j} for parent {i} (claim_number={claim_num})")
                plans.append(PlannedClaim(
                    claim_number=claim_num,
                    claim_type="dependent",
                    focus=f"This is a dependent claim of claim {i}. Focus on a specific feature, detail, or variation of the component from claim {i} as described in the patent. Add concrete details about a particular aspect or implementation.",
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
            # If threshold filtered everything out, get top triples anyway for display
            if not similar_triples:
                similar_triples = self.graph_rag.find_similar_triples(
                    query_text=query,
                    top_k=10,
                    similarity_threshold=0.0,
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
        
        prompt = f"""You are a patent claim drafting expert. Generate a formal patent claim based on the following information. Draft in proper patent claim style: clear, legally defensible scope with consistent terminology—not like an engineering specification.

Patent Description:
{patent_description[:4000]}

Claim Plan:
- Claim Number: {planned_claim.claim_number}
- Claim Type: {claim_type_text}
- Planned Focus (detailed guidance): {planned_claim.focus}
- Keywords: {', '.join(planned_claim.keywords) if planned_claim.keywords else 'N/A'}
- Relevant Entities: {', '.join(planned_claim.entities) if planned_claim.entities else 'N/A'}
{parent_text}

{context}

Instructions:
1. Draft a formal patent claim in standard patent claim format. Use claim-style language (clear scope, defined elements); avoid reading like a technical spec or implementation manual.
2. The claim must accurately reflect the triples provided above. Use consistent terms for the same concepts (align with the patent description and, where applicable, with language used in previous claims).
3. Balance specificity with defensibility: be precise enough to be supported by the description and triples, but do not limit the claim to a single implementation or trivial detail where the invention is broader; aim for scope that is commercially meaningful and enforceable.
4. For independent claims: Start with "A [system/method/apparatus] comprising:" and define elements that together express the inventive concept without unnecessary implementation detail.
5. For dependent claims: Start with "The [system/method/apparatus] of claim {planned_claim.parent_claim_number}, wherein..." and add a clear limitation that is internally consistent with the parent.
6. Ensure all mentioned components/features correspond to entities and relations in the knowledge graph. Keep the claim internally consistent (no contradictory or redundant phrasing).
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
                prompt=prompt,  # Store the prompt used for generation
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
                prompt=prompt,  # Store the prompt even on error
            )
            print(f"[DEBUG] generate_claim: Returning error placeholder claim")
            return error_claim
    
    def judge_claims(
        self,
        claims: List[GeneratedClaim],
        patent_description: str,
    ) -> Dict[str, Any]:
        """
        Judge all claims together for unity and propose improvements for each claim.
        
        Returns:
            Dict with 'unity_score' (float 0-100), 'unity_feedback' (str), 
            and 'claim_criticisms' (List[Dict] with 'claim_number', 'score', 'criticism')
        """
        if not claims:
            return {
                "unity_score": 0.0,
                "unity_feedback": "No claims to judge",
                "claim_criticisms": []
            }
        
        # Format claims for prompt
        claims_text = "\n\n".join([
            f"Claim {c.claim_number} ({c.claim_type}):\n{c.claim_text}"
            for c in claims
        ])
        
        prompt = f"""You are a patent claim evaluation expert. Evaluate the following set of patent claims for unity, quality, defensibility, and consistency.

Patent Description:
{patent_description[:2000]}

Claims:
{claims_text}

Evaluate:
1. Unity: Do all claims relate to a single inventive concept? Score 0-100.
2. Quality and clarity: Are claims well-drafted, clear, and properly structured? Do they read as patent claims rather than engineering specifications? Score 0-100.
3. Defensibility and scope: Is the scope clear and legally defensible? Are claims overly narrow (e.g., tied to one implementation) so that commercial enforceability is weak? Flag if claims read like specs or are over-engineered.
4. Internal consistency: Is terminology consistent across claims? Do dependents align with their parents without contradiction or redundancy?
5. Improvements: For each claim, identify specific improvements needed (e.g., clarify scope, fix inconsistency, reduce over-specification, strengthen defensibility).

Return a JSON object with:
{{
    "unity_score": <float 0-100>,
    "unity_feedback": "<explanation of unity assessment>",
    "claim_criticisms": [
        {{
            "claim_number": <int>,
            "score": <float 0-100>,
            "criticism": "<specific improvements needed for this claim>"
        }},
        ...
    ]
}}

Return ONLY the JSON object, no other text."""
        
        try:
            response = self.api_repo.chat(prompt)
            
            # Extract JSON from response
            if isinstance(response, dict):
                response_text = response.get("content", response.get("text", ""))
            else:
                response_text = str(response)
            
            # Clean response
            response_text = response_text.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Extract JSON object
            if "{" in response_text:
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    response_text = response_text[start_idx:end_idx]
            
            # Try to parse JSON, with fallback for malformed JSON
            result = None
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"[DEBUG] judge_claims: JSON parse error: {e}, attempting to fix...")
                # Try to extract just the JSON part more carefully
                # Find the first complete JSON object by counting braces
                brace_count = 0
                start_pos = response_text.find("{")
                if start_pos >= 0:
                    for i in range(start_pos, len(response_text)):
                        if response_text[i] == "{":
                            brace_count += 1
                        elif response_text[i] == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                # Found complete JSON object
                                extracted_text = response_text[start_pos:i+1]
                                print(f"[DEBUG] judge_claims: Extracted complete object (length: {len(extracted_text)})")
                                try:
                                    result = json.loads(extracted_text)
                                    print(f"[DEBUG] judge_claims: Successfully parsed JSON after extraction")
                                except json.JSONDecodeError as e2:
                                    print(f"[DEBUG] judge_claims: Still can't parse extracted JSON: {e2}")
                                    # Try to fix common JSON issues
                                    try:
                                        # Fix trailing commas
                                        fixed_text = re.sub(r',\s*}', '}', extracted_text)
                                        fixed_text = re.sub(r',\s*]', ']', fixed_text)
                                        result = json.loads(fixed_text)
                                        print(f"[DEBUG] judge_claims: Successfully parsed JSON after fixing trailing commas")
                                    except json.JSONDecodeError as e3:
                                        print(f"[DEBUG] judge_claims: All JSON parsing attempts failed, using default judgment")
                                        result = None
                                break
                
                if result is None:
                    print(f"[DEBUG] judge_claims: Could not parse JSON, returning default judgment")
                    # Return default judgment instead of raising
                    return {
                        "unity_score": 50.0,
                        "unity_feedback": "Unable to parse judgment response. Using default scores.",
                        "claim_criticisms": [
                            {
                                "claim_number": c.claim_number,
                                "score": 50.0,
                                "criticism": "Unable to evaluate due to JSON parsing error."
                            }
                            for c in claims
                        ]
                    }
            
            # Ensure all claims have criticisms
            claim_numbers = {c.claim_number for c in claims}
            criticism_numbers = {c.get("claim_number") for c in result.get("claim_criticisms", [])}
            
            # Add missing criticisms with default values
            for claim_num in claim_numbers:
                if claim_num not in criticism_numbers:
                    result.setdefault("claim_criticisms", []).append({
                        "claim_number": claim_num,
                        "score": result.get("unity_score", 50.0),
                        "criticism": "No specific criticism provided."
                    })
            
            return result
            
        except Exception as e:
            print(f"[DEBUG] judge_claims: Error judging claims: {e}")
            import traceback
            traceback.print_exc()
            # Return default scores
            return {
                "unity_score": 50.0,
                "unity_feedback": f"Error during judgment: {str(e)}",
                "claim_criticisms": [
                    {
                        "claim_number": c.claim_number,
                        "score": 50.0,
                        "criticism": "Unable to evaluate due to error."
                    }
                    for c in claims
                ]
            }
    
    def refine_claim_with_criticism(
        self,
        claim: GeneratedClaim,
        planned_claim: PlannedClaim,
        criticism: str,
        patent_description: str,
        previous_claims: List[GeneratedClaim],
        similarity_threshold: float,
    ) -> GeneratedClaim:
        """
        Refine a claim based on criticism, incorporating the old claim and improvements.
        """
        # Retrieve relevant triples
        relevant_triples = []
        if self.graph_rag:
            query_parts = [planned_claim.focus]
            query_parts.extend(planned_claim.keywords)
            query_parts.extend(planned_claim.entities)
            query = " ".join(query_parts)
            
            similar_triples = self.graph_rag.find_similar_triples(
                query_text=query,
                top_k=20,
                similarity_threshold=similarity_threshold,
            )
            if not similar_triples:
                similar_triples = self.graph_rag.find_similar_triples(
                    query_text=query,
                    top_k=10,
                    similarity_threshold=0.0,
                )
            for triple, similarity in similar_triples:
                relevant_triples.append({
                    "head": triple.head.name if hasattr(triple.head, 'name') else str(triple.head),
                    "relation": triple.relation,
                    "tail": triple.tail.name if hasattr(triple.tail, 'name') else str(triple.tail),
                    "similarity": similarity,
                })
        
        # Build context
        context_parts = []
        if relevant_triples:
            context_parts.append("Relevant Knowledge Graph Triples:")
            for i, triple in enumerate(relevant_triples[:15], 1):
                context_parts.append(
                    f"{i}. {triple['head']} --[{triple['relation']}]--> {triple['tail']}"
                )
        
        if previous_claims and planned_claim.claim_type == "dependent":
            context_parts.append("\nPrevious Claims (for reference):")
            for prev_claim in previous_claims:
                if prev_claim.claim_number == planned_claim.parent_claim_number:
                    context_parts.append(f"Claim {prev_claim.claim_number}: {prev_claim.claim_text}")
        
        context = "\n".join(context_parts)
        
        claim_type_text = "independent" if planned_claim.claim_type == "independent" else "dependent"
        parent_text = ""
        if planned_claim.claim_type == "dependent" and planned_claim.parent_claim_number:
            parent_text = f"\nThis is a dependent claim that depends on claim {planned_claim.parent_claim_number}."
        
        prompt = f"""You are a patent claim drafting expert. Refine the following claim based on the provided criticism. The result should read as a proper patent claim (clear, defensible, consistent)—not as an engineering specification.

Patent Description:
{patent_description[:4000]}

Original Claim:
{claim.claim_text}

Criticism and Required Improvements:
{criticism}

Claim Plan:
- Claim Number: {planned_claim.claim_number}
- Claim Type: {claim_type_text}
- Planned Focus: {planned_claim.focus}
- Keywords: {', '.join(planned_claim.keywords) if planned_claim.keywords else 'N/A'}
- Relevant Entities: {', '.join(planned_claim.entities) if planned_claim.entities else 'N/A'}
{parent_text}

{context}

Instructions:
1. Address the criticism and incorporate the required improvements.
2. Keep the good parts of the original claim. Improve clarity, defensibility, and internal consistency; avoid adding implementation detail that makes the claim read like a spec or overly narrow.
3. Maintain proper patent claim format and claim-style language. Use consistent terminology with the patent description and, where relevant, with other claims.
4. Ensure the refined claim remains supported by the triples and description. Do not introduce contradictions or redundant limitations.
5. Number the claim as "{planned_claim.claim_number}."

Return ONLY the refined claim text, numbered as "{planned_claim.claim_number}." No additional explanation."""
        
        try:
            response = self.api_repo.chat(prompt)
            
            # Extract claim text
            if isinstance(response, dict):
                claim_text = response.get("content", response.get("text", ""))
            else:
                claim_text = str(response)
            
            claim_text = claim_text.strip()
            if "```" in claim_text:
                lines = claim_text.split("\n")
                claim_text = "\n".join([l for l in lines if not l.strip().startswith("```")])
            
            if not claim_text.startswith(f"{planned_claim.claim_number}."):
                claim_text = f"{planned_claim.claim_number}. {claim_text}"
            
            return GeneratedClaim(
                claim_number=planned_claim.claim_number,
                claim_text=claim_text,
                claim_type=planned_claim.claim_type,
                parent_claim_number=planned_claim.parent_claim_number,
                focus=planned_claim.focus,
                used_triples=relevant_triples[:10],
                prompt=prompt,
                refinement_iterations=claim.refinement_iterations + 1,
            )
            
        except Exception as e:
            print(f"[DEBUG] refine_claim_with_criticism: Error refining claim: {e}")
            # Return original claim if refinement fails
            return claim
    
    def refine_claims_with_langgraph(
        self,
        claims: List[GeneratedClaim],
        planned_claims: List[PlannedClaim],
        patent_description: str,
        similarity_threshold: float,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_iterations: int = 2,
        min_score: float = 90.0,
    ) -> List[GeneratedClaim]:
        """
        Refine claims using LangGraph loop until scores >= min_score or max_iterations reached.
        """
        if not LANGGRAPH_AVAILABLE:
            print("[WARNING] LangGraph not available, skipping refinement")
            return claims
        
        # Create a mapping from claim number to planned claim
        planned_map = {pc.claim_number: pc for pc in planned_claims}
        
        def judge_node(state: ClaimRefinementState) -> ClaimRefinementState:
            """Judge all claims and get scores/criticisms."""
            print(f"[Refinement] Iteration {state['iteration']}: Judging all claims...")
            
            if progress_callback:
                progress_callback({
                    "stage": "refining",
                    "message": f"Evaluating claims for unity and quality (iteration {state['iteration'] + 1}/{max_iterations})...",
                    "progress": 80 + (state['iteration'] * 5),
                })
            
            try:
                print(f"[Refinement] Calling judge_claims with {len(state['claims'])} claims...")
                judgment = self.judge_claims(state['claims'], state['patent_description'])
                print(f"[Refinement] judge_claims completed successfully")
            except Exception as e:
                print(f"[Refinement] Error in judge_claims: {e}")
                import traceback
                traceback.print_exc()
                # Return default judgment on error
                judgment = {
                    "unity_score": 50.0,
                    "unity_feedback": f"Error during judgment: {str(e)}",
                    "claim_criticisms": [
                        {
                            "claim_number": c.claim_number,
                            "score": 50.0,
                            "criticism": "Unable to evaluate due to error."
                        }
                        for c in state['claims']
                    ]
                }
            
            # Update scores and criticisms
            new_scores = []
            new_criticisms = []
            
            for claim in state['claims']:
                # Find criticism for this claim
                criticism_data = next(
                    (c for c in judgment['claim_criticisms'] if c['claim_number'] == claim.claim_number),
                    {"score": judgment['unity_score'], "criticism": "No specific criticism."}
                )
                new_scores.append(criticism_data['score'])
                new_criticisms.append(criticism_data['criticism'])
            
            state['scores'] = new_scores
            state['criticisms'] = new_criticisms
            
            print(f"[Refinement] Unity score: {judgment['unity_score']:.1f}")
            print(f"[Refinement] Individual scores: {new_scores}")
            
            if progress_callback:
                avg_score = sum(new_scores) / len(new_scores) if new_scores else 0.0
                progress_callback({
                    "stage": "refining",
                    "message": f"Evaluation complete: avg score {avg_score:.1f}/100 (iteration {state['iteration'] + 1}/{max_iterations})",
                    "progress": 82 + (state['iteration'] * 5),
                })
            
            return state
        
        def refine_node(state: ClaimRefinementState) -> ClaimRefinementState:
            """Refine the current claim based on criticism."""
            # Increment iteration HERE, not in the conditional edge function
            # This ensures the state update persists to the next node
            current_iter = state.get('iteration', 0)
            state['iteration'] = current_iter + 1
            
            claim_idx = state['claim_index']
            claim = state['claims'][claim_idx]
            planned_claim = state['planned_claim']
            criticism = state['criticisms'][claim_idx] if claim_idx < len(state['criticisms']) else "No specific criticism."
            
            print(f"[Refinement] Refining claim {claim.claim_number} (iteration {state['iteration']})...")
            
            if progress_callback:
                progress_callback({
                    "stage": "refining",
                    "message": f"Refining claim {claim.claim_number} based on feedback (iteration {state['iteration']}/{max_iterations})...",
                    "progress": 84 + (state['iteration'] * 5),
                })
            
            try:
                refined_claim = self.refine_claim_with_criticism(
                claim=claim,
                planned_claim=planned_claim,
                criticism=criticism,
                patent_description=state['patent_description'],
                    previous_claims=state['previous_claims'],
                    similarity_threshold=state['similarity_threshold'],
                )
                print(f"[Refinement] Claim {claim.claim_number} refinement completed")
            except Exception as e:
                print(f"[Refinement] Error refining claim {claim.claim_number}: {e}")
                import traceback
                traceback.print_exc()
                # Keep original claim if refinement fails
                refined_claim = claim
            
            # Update the claim in the list
            new_claims = state['claims'].copy()
            new_claims[claim_idx] = refined_claim
            state['claims'] = new_claims
            
            if progress_callback:
                progress_callback({
                    "stage": "refining",
                    "message": f"Claim {claim.claim_number} refined (score: {state['scores'][claim_idx]:.1f}/100)",
                    "progress": 86 + (state['iteration'] * 5),
                })
            
            return state
        
        def should_continue_after_judge(state: ClaimRefinementState) -> str:
            """Decide whether to continue refining after judging."""
            current_iter = state.get('iteration', 0)
            
            print(f"[Refinement] should_continue_after_judge: iteration={current_iter}, max_iterations={max_iterations}")
            
            # Check if we've reached max iterations (check BEFORE incrementing)
            # With max_iterations=2, we allow iterations 0 and 1, so stop at 2
            if current_iter >= max_iterations:
                print(f"[Refinement] Reached max iterations ({max_iterations}), current: {current_iter}, stopping")
                return "end"
            
            # Check if score is above threshold (for single claim refinement)
            if state.get('scores') and len(state['scores']) > 0:
                score = state['scores'][0]
                print(f"[Refinement] should_continue_after_judge: score={score:.1f}, min_score={min_score}")
                if score >= min_score:
                    print(f"[Refinement] Score {score:.1f} >= {min_score}, stopping")
                    return "end"
            
            # Continue to refinement
            print(f"[Refinement] should_continue_after_judge: continuing to refine")
            return "refine"
        
        def should_continue_after_refine(state: ClaimRefinementState) -> str:
            """Decide whether to continue after refining."""
            # Iteration was already incremented in refine_node
            current_iter = state.get('iteration', 0)
            
            print(f"[Refinement] should_continue_after_refine: iteration={current_iter}, max_iterations={max_iterations}")
            
            # Check if we've reached max iterations AFTER incrementing
            # With max_iterations=2, after iteration 1 completes, iteration=2, so stop
            if current_iter >= max_iterations:
                print(f"[Refinement] Reached max iterations ({max_iterations}) after refine, iteration: {current_iter}, stopping")
                return "end"
            
            # Go back to judge for next iteration
            print(f"[Refinement] should_continue_after_refine: continuing to judge")
            return "judge"
        
        # Build LangGraph workflow
        workflow = StateGraph(ClaimRefinementState)
        
        workflow.add_node("judge", judge_node)
        workflow.add_node("refine", refine_node)
        
        workflow.set_entry_point("judge")
        
        workflow.add_conditional_edges(
            "judge",
            should_continue_after_judge,
            {
                "end": END,
                "refine": "refine",
            }
        )
        
        workflow.add_conditional_edges(
            "refine",
            should_continue_after_refine,
            {
                "judge": "judge",
                "end": END,
            }
        )
        
        app = workflow.compile()
        
        # Refine each claim iteratively
        refined_claims = []
        
        for claim_idx, claim in enumerate(claims):
            planned_claim = planned_map.get(claim.claim_number)
            if not planned_claim:
                print(f"[Refinement] No planned claim found for claim {claim.claim_number}, skipping")
                refined_claims.append(claim)
                continue
            
            print(f"[Refinement] Starting refinement for claim {claim.claim_number}...")
            
            # Initial state for this claim
            initial_state: ClaimRefinementState = {
                "claims": [claim],  # Single claim refinement
                "claim_index": 0,
                "iteration": 0,
                "scores": [0.0],
                "criticisms": [""],
                "patent_description": patent_description,
                "planned_claim": planned_claim,
                "previous_claims": refined_claims,
                "similarity_threshold": similarity_threshold,
                "max_iterations": max_iterations,
                "min_score": min_score,
            }
            
            # Run refinement loop with recursion limit
            try:
                print(f"[Refinement] Starting LangGraph workflow for claim {claim.claim_number}...")
                final_state = None
                iteration_count = 0
                
                # Set recursion limit config conservatively
                # Each iteration = judge + refine (2 nodes)
                # For max_iterations=2: judge(1) + refine(2) + judge(3) + refine(4) = 4 nodes maximum
                # Set limit to exactly what we expect: max_iterations * 2
                # If we hit this limit, it means the stop conditions aren't working correctly
                # Add a small buffer to avoid false positives when the graph emits
                # an extra terminal step before END.
                config = {"recursion_limit": (max_iterations * 2) + 2}

                for state_update in app.stream(initial_state, config=config):
                    node_name = list(state_update.keys())[-1]
                    final_state = list(state_update.values())[-1]
                    iteration_count += 1
                    print(f"[Refinement] LangGraph step {iteration_count}: {node_name}, iteration={final_state.get('iteration', 0)}")
                    
                    # Update progress during workflow
                    if progress_callback and final_state:
                        current_iter = final_state.get('iteration', 0)
                        if node_name == 'judge':
                            progress_callback({
                                "stage": "refining",
                                "message": f"Evaluating claim {claim.claim_number} (iteration {current_iter + 1}/{max_iterations})...",
                                "progress": 80 + min(15, current_iter * 5),
                            })
                        elif node_name == 'refine':
                            progress_callback({
                                "stage": "refining",
                                "message": f"Refining claim {claim.claim_number} (iteration {current_iter}/{max_iterations})...",
                                "progress": 84 + min(11, current_iter * 5),
                            })
                
                print(f"[Refinement] LangGraph workflow completed after {iteration_count} steps")
                
                if final_state and final_state['claims']:
                    refined_claim = final_state['claims'][0]
                    refined_claim.final_score = final_state['scores'][0] if final_state.get('scores') and len(final_state['scores']) > 0 else 0.0
                    refined_claims.append(refined_claim)
                    print(f"[Refinement] Claim {claim.claim_number} refined: score={refined_claim.final_score:.1f}, iterations={refined_claim.refinement_iterations}")
                else:
                    print(f"[Refinement] No final state or claims, using original claim")
                    refined_claims.append(claim)
            except Exception as e:
                print(f"[Refinement] Error refining claim {claim.claim_number}: {e}")
                import traceback
                traceback.print_exc()
                refined_claims.append(claim)
        
        return refined_claims
    
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
                prompt="",  # No prompt available for fallback claim
            )
            generated_claims = [fallback_claim]
            print(f"[DEBUG] generate_all_claims: ✅ Created emergency fallback claim")
        
        # Refine claims using judging agent and LangGraph loop
        if LANGGRAPH_AVAILABLE and len(generated_claims) > 0:
            print(f"🔍 Starting claim refinement with judging agent...")
            if progress_callback:
                progress_callback({
                    "stage": "refining",
                    "message": "Evaluating and refining claims...",
                    "progress": 90,
                })
            
            try:
                refined_claims = self.refine_claims_with_langgraph(
                    claims=generated_claims,
                    planned_claims=planned_claims,
                    patent_description=patent_description,
                    similarity_threshold=similarity_threshold,
                    progress_callback=progress_callback,
                    max_iterations=2,
                    min_score=90.0,
                )
                generated_claims = refined_claims
                print(f"✨ Refined {len(generated_claims)} claims")
            except Exception as e:
                print(f"[DEBUG] generate_all_claims: Error during refinement: {e}")
                import traceback
                traceback.print_exc()
                print(f"[DEBUG] generate_all_claims: Continuing with unrefined claims")
        
        if progress_callback:
            progress_callback({
                "stage": "complete",
                "message": f"Successfully generated {len(generated_claims)} claims!",
                "progress": 100,
                "num_claims": len(generated_claims),
            })
        
        print(f"🎉 Returning {len(generated_claims)} claims")
        return generated_claims

