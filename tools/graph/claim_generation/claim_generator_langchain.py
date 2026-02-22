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
from collections import Counter

import numpy as np
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
    
    def _extract_json(self, text: str) -> Optional[Any]:
        """Robustly extract JSON from LLM response."""
        if not text:
            return None
            
        text = text.strip()
        # Remove markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
            
        # Find first { or [ and last } or ]
        start_brace = text.find("{")
        start_bracket = text.find("[")
        
        if start_brace == -1 and start_bracket == -1:
            return None
            
        if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
            # Probably an object
            start_idx = start_brace
            end_idx = text.rfind("}") + 1
        else:
            # Probably an array
            start_idx = start_bracket
            end_idx = text.rfind("]") + 1
            
        if start_idx == -1 or end_idx <= start_idx:
            return None
            
        json_text = text[start_idx:end_idx]
        
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            # Try basic fixes for common LLM JSON errors
            try:
                # Fix trailing commas before } or ]
                fixed_text = re.sub(r',\s*}', '}', json_text)
                fixed_text = re.sub(r',\s*]', ']', fixed_text)
                # Remove unescaped newlines in strings (very common)
                # This is a bit aggressive but often works for LLM output
                return json.loads(fixed_text)
            except Exception:
                # Last resort: try to find a valid JSON object by counting braces
                brace_count = 0
                for i in range(start_idx, len(text)):
                    if text[i] == "{" or text[i] == "[":
                        brace_count += 1
                    elif text[i] == "}" or text[i] == "]":
                        brace_count -= 1
                        if brace_count == 0:
                            try:
                                return json.loads(text[start_idx:i+1])
                            except:
                                continue
                return None

    def plan_claims(
        self,
        patent_description: str,
        num_independent: int = 3,
    ) -> List[PlannedClaim]:
        """
        Plan the independent claim structure from patent description.
        
        Args:
            patent_description: The patent description text
            num_independent: Number of independent claims to plan
            
        Returns:
            List of PlannedClaim objects (independents only)
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
        
        prompt = f"""You are a patent claim planning expert. Analyze the following patent description and plan a set of independent patent claims that will be defensible, internally consistent, and commercially meaningful.

Patent Description:
{patent_description[:3000]}  # Limit to avoid token limits{entities_context}

Based on the patent description, plan a set of independent claims.
- Typically, you should aim for {num_independent} independent claims to cover different embodiments or aspects of the invention, unless there is a strong reason the invention warrants significantly more or fewer.
- Each independent claim should define a distinct, defensible scope that could be commercially enforced.

Planning principles:
- Plan so that the resulting claims read as proper patent claims (clear legal scope, consistent terminology), not as engineering specifications or implementation details.
- Each independent claim should define a distinct, defensible scope; avoid planning claims that are so narrow they cover only one trivial implementation.
- Extract real components and features from the description, but frame the focus in terms of what the claim should protect (inventive concept and scope), not just a list of technical details.

For each claim, provide:
- claim_number: Use integers (1, 2, 3...).
- claim_type: Always "independent"
- focus: A clear description (2-4 sentences) of what this claim should focus on, using actual components/features from the patent description. Define the main inventive concept and key elements that define a defensible scope. Be concrete but avoid encouraging over-specification or spec-style drafting.
- parent_claim_number: Always null for independent claims
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
    "claim_type": "independent",
    "focus": "Focus on [ANOTHER EMBODIMENT OR ASPECT]. Detail [how it works/where it's located/its characteristics as described].",
    "parent_claim_number": null,
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
                    return self._default_plan(num_independent)
            print(f"[DEBUG] plan_claims: JSON parse successful! Type: {type(plans_data)}, Length: {len(plans_data) if isinstance(plans_data, list) else 'N/A'}")
            
            if not isinstance(plans_data, list):
                print(f"⚠️  Response is not a list: {type(plans_data)}")
                return self._default_plan(num_independent)
            
            if len(plans_data) == 0:
                print(f"⚠️  Response is an empty list, using default plan")
                return self._default_plan(num_independent)
            
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
                return self._default_plan(num_independent)
            
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
            return self._default_plan(num_independent)
        except Exception as e:
            print(f"[DEBUG] plan_claims: ⚠️  Error in claim planning: {type(e).__name__}: {e}")
            print(f"[DEBUG] plan_claims: Exception args: {e.args}")
            import traceback
            print(f"[DEBUG] plan_claims: Full traceback:")
            traceback.print_exc()
            # Return default plan if planning fails
            print(f"[DEBUG] plan_claims: Falling back to default plan due to exception")
            return self._default_plan(num_independent)
    
    def _default_plan(self, num_independent: int) -> List[PlannedClaim]:
        """Generate a default plan of independent claims if planning fails."""
        print(f"[DEBUG] _default_plan: Called with num_independent={num_independent}")
        print(f"[DEBUG] _default_plan: Using default plan: {num_independent} independent claims")
        plans = []
        
        # Add independent claims
        print(f"[DEBUG] _default_plan: Creating {num_independent} independent claims...")
        for i in range(1, num_independent + 1):
            print(f"[DEBUG] _default_plan: Creating independent claim {i}")
            plans.append(PlannedClaim(
                claim_number=i,
                claim_type="independent",
                focus=f"Focus on a main component or system from the patent description. Describe its structure, function, and key features as mentioned in the description.",
            ))
        
        print(f"[DEBUG] _default_plan: ✅ Default plan created: {len(plans)} independent claims")
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
7. Number the claim as "{planned_claim.claim_number}.
8"

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
3. Defensibility and scope: Is the scope clear (not obvious) and legally defensible? Are claims overly narrow (e.g., tied to one implementation) so that commercial enforceability is weak? Flag if claims read like specs or are over-engineered.
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
            response_text = response.get("content", response.get("text", str(response))) if isinstance(response, dict) else str(response)
            result = self._extract_json(response_text)
            
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
2. Keep the good parts of the original claim. Improve clarity, obviousness, defensibility, and internal consistency; avoid adding implementation detail that makes the claim read like a spec or overly narrow.
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
            
            # If no planned claim found (e.g. for dependent claims 1.1, 1.2), create a synthetic one
            if not planned_claim:
                print(f"[Refinement] Creating synthetic planned claim for dependent claim {claim.claim_number}")
                parent_num = None
                if "." in str(claim.claim_number):
                    try:
                        parent_num = int(str(claim.claim_number).split(".")[0])
                    except:
                        parent_num = 1
                
                planned_claim = PlannedClaim(
                    claim_number=claim.claim_number,
                    claim_type=claim.claim_type,
                    focus=claim.focus or f"Dependent claim narrowing claim {parent_num or 1}",
                    parent_claim_number=parent_num or claim.parent_claim_number,
                )
            
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
                        # Distributed over 70% to 90% range
                        base_progress = 70 + int((claim_idx / len(claims)) * 20)
                        step_progress = min(2, (iteration_count % 4)) 
                        if node_name == 'judge':
                            progress_callback({
                                "stage": "quality_assessment",
                                "message": f"Quality check: Claim {claim.claim_number} (iteration {current_iter + 1}/{max_iterations})...",
                                "progress": base_progress + step_progress,
                            })
                        elif node_name == 'refine':
                            progress_callback({
                                "stage": "quality_assessment",
                                "message": f"Refining claim {claim.claim_number} (iteration {current_iter}/{max_iterations})...",
                                "progress": base_progress + step_progress + 1,
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
    
    def legal_refinement(
        self,
        claims: List[GeneratedClaim],
        planned_claims: List[PlannedClaim],
        patent_description: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[GeneratedClaim]:
        """
        Perform a legal review of the claims for §101, §102, §103, and §112 hurdles.
        One-pass refinement for speed.
        """
        print(f"⚖️ Starting legal refinement for {len(claims)} claims...")
        refined_claims = []
        planned_map = {pc.claim_number: pc for pc in planned_claims}

        for idx, claim in enumerate(claims):
            if progress_callback:
                # Range: 90% to 95% (leaving room for final polish)
                progress = 90 + int((idx / len(claims)) * 5)
                progress_callback({
                    "stage": "legal_check",
                    "message": f"Legal check: Claim {claim.claim_number} (§101, §102, §103, §112)...",
                    "progress": progress,
                })

            planned_claim = planned_map.get(claim.claim_number)
            
            prompt = f"""You are a patent attorney expert. Review and refine the following patent claim to ensure it overcomes critical legal hurdles. 

### Legal Hurdles & Quality Tests to Address:
1. **§101 – Abstract Detector Test**: Delete hardware nouns temporarily. If the claim still reads like "Collect data, analyze data, output result," it's abstract. Ensure it collapses if physical components are removed. It must describe a technical improvement, not just an abstract process.
2. **§102 – Single Reference Kill Shot Test**: Ask: could one reasonably detailed prior art paper include everything here? If yes, you need a tighter distinguishing feature. (e.g., your protection might be a specific correlation mechanism rather than just "event camera" + "rotation").
3. **§103 – Obvious Lego Test**: Can an examiner describe the claim as "Take known A + add known B + process with known C"? If so, it's too modular. Fix by adding a non-trivial interaction. 
   *Example: Instead of just "rotation + event camera," use "correlating pixel events with angular position."*
4. **§112(a) – Spec Support Test**: Ensure every technical phrase (e.g., "spatiotemporal pattern", "angular correlation", "reflectance characteristic", "geometry data") is clearly supported by the patent description.
5. **§112(b) – Definiteness & Correlation Vagueness Test**: Ensure all terms have proper antecedent basis. If using words like "correlation", "processing", or "determine", tie them to a specific technical mechanism. 
   *Better: "Correlating pixel events with angular positions" instead of a broad "determine characteristic."*
6. **Causation Chain Test**: The claim should clearly show a physical chain: Physical action → physical signal → defined processing → technical output. 
   *Example: Rotation → brightness change → asynchronous events → reflectance.*
7. **"What Actually Makes This New?" Test**: Force a one-sentence answer. If the answer is "Because it uses machine learning," it is weak. If the answer is "Because it correlates asynchronous event-camera brightness changes with angular position to derive reflectance," it is strong. The claim must visibly express this strength.
8. **Litigator Attack Test**: Imagine a defense attorney saying: "This is just using a camera and AI to analyze a rotating object." The claim should make that simplification obviously wrong.

### Patent Description (Context):
{patent_description[:3000]}

### Current Claim {claim.claim_number}:
{claim.claim_text}

### Instructions:
- Refine the claim to be more legally robust while maintaining its core technical focus.
- **Medium changes per claim are encouraged** to address the tests above.
- Do NOT make it an engineering spec; keep it in proper patent claim format.
- Ensure consistent terminology.
- Fix any vague language or modularity issues.

Return ONLY the refined claim text, numbered as "{claim.claim_number}." No commentary."""

            try:
                response = self.api_repo.chat(prompt)
                response_text = str(response.get("content", response)) if isinstance(response, dict) else str(response)
                
                # Clean up response
                response_text = response_text.strip()
                if "```" in response_text:
                    lines = response_text.split("\n")
                    response_text = "\n".join([l for l in lines if not l.strip().startswith("```")])
                
                if not response_text.startswith(f"{claim.claim_number}."):
                    response_text = f"{claim.claim_number}. {response_text}"

                # Update the claim text
                claim.claim_text = response_text
                refined_claims.append(claim)
                print(f"⚖️ Legal review complete for Claim {claim.claim_number}")
            except Exception as e:
                print(f"⚠️ Legal review failed for claim {claim.claim_number}: {e}")
                refined_claims.append(claim)

        return refined_claims

    def final_legal_alignment(
        self,
        claims: List[GeneratedClaim],
        patent_description: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[GeneratedClaim]:
        """
        Final minimal polish of all claims together for legal consistency and redundancy.
        """
        if not claims:
            return claims

        print(f"⚖️ Starting final legal alignment for {len(claims)} claims...")
        if progress_callback:
            progress_callback({
                "stage": "final_alignment",
                "message": "Final legal alignment and consistency check...",
                "progress": 95,
            })

        claims_text = "\n\n".join([f"Claim {c.claim_number}:\n{c.claim_text}" for c in claims])
        
        prompt = f"""You are a patent attorney expert. Perform a final, **very minimal** polish on the following set of patent claims to ensure total output consistency and address final legal requirements.

### Final Legal Checks:
1. **Redundant Dependent Test**: Look at dependent claims. If you can swap their wording and nothing changes legally, they are redundant. 
   *Bad Dependents: "using ML", "using ANN", "trained ML model".*
   *Good Dependents: "Add encoder synchronization", "Add polarity analysis", "Add angular binning".*
   Ensure each dependent claim adds a distinct, technical limitation.
2. **Fallback Ladder Test**: For each independent claim, ensure there is a clear "ladder" of increasingly narrow, defensible dependent claims. If an examiner rejects an independent claim, there should be a clear, strong limitation to add next.
3. **Internal Consistency**: Ensure terminology is perfectly consistent across the entire set.

### Patent Description (Context):
{patent_description[:2000]}

### Current Claims:
{claims_text}

### Instructions:
- Perform **ONLY very minimal changes** to the total output.
- Focus on fixing redundancies in dependent claims and ensuring a consistent fallback ladder.
- Maintain proper patent claim format.
- Do NOT rewrite the claims; only tweak for consistency and to remove redundant dependents.

Return ONLY the final set of claims in a JSON array of strings, where each string is a full claim (e.g. ["1. A system...", "1.1. The system..."]). 
Return ONLY the JSON array, no other text."""

        try:
            response = self.api_repo.chat(prompt)
            response_text = response.get("content", response.get("text", str(response))) if isinstance(response, dict) else str(response)
            
            final_claims_list = self._extract_json(response_text)
            
            if isinstance(final_claims_list, list) and len(final_claims_list) == len(claims):
                for idx, new_text in enumerate(final_claims_list):
                    claims[idx].claim_text = new_text
                print(f"⚖️ Final legal alignment complete")
            else:
                print(f"⚠️ Final legal alignment returned unexpected count: {len(final_claims_list) if isinstance(final_claims_list, list) else 'not a list'}")
        except Exception as e:
            print(f"⚠️ Final legal alignment failed: {e}")
            
        return claims

    def draft_independent_claims(
        self,
        planned_claims: List[PlannedClaim],
        patent_description: str,
        similarity_threshold: float,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[GeneratedClaim]:
        """Phase 1: Draft initial independent claims."""
        independents = [pc for pc in planned_claims if pc.claim_type == "independent"]
        generated_independents = []
        total = len(independents)
        
        print(f"✍️ Drafting {total} Independent Claims...")
        for idx, pc in enumerate(independents):
            if progress_callback:
                progress = 20 + int((idx / total) * 20) # 20% -> 40%
                progress_callback({
                    "stage": "drafting_independents",
                    "message": f"Drafting Independent Claim {pc.claim_number}...",
                    "progress": progress,
                })
            
            claim = self.generate_claim(pc, patent_description, similarity_threshold=similarity_threshold)
            generated_independents.append(claim)
            
        return generated_independents

    def draft_dependent_claims(
        self,
        independent_claims: List[GeneratedClaim],
        patent_description: str,
        num_dependent_per_independent: int,
        similarity_threshold: float,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[GeneratedClaim]:
        """Phase 2: Draft dependent claims (fallbacks) for each independent claim."""
        all_claims = list(independent_claims)
        total_dependents = len(independent_claims) * num_dependent_per_independent
        
        if total_dependents == 0:
            return all_claims

        print(f"✍️ Drafting {total_dependents} Dependent Claims (Fallbacks & Limitations)...")
        
        count = 0
        for parent_claim in independent_claims:
            for j in range(1, num_dependent_per_independent + 1):
                count += 1
                # Use decimal notation: 1.1, 1.2, etc.
                claim_number_str = f"{parent_claim.claim_number}.{j}"
                
                if progress_callback:
                    progress = 40 + int((count / total_dependents) * 30) # 40% -> 70%
                    progress_callback({
                        "stage": "drafting_dependents",
                        "message": f"Drafting Dependent Claim {claim_number_str} (fallback)...",
                        "progress": progress,
                    })
                
                # Create a planned claim for this dependent on-the-fly
                pc = PlannedClaim(
                    claim_number=0,  # Will be set in GeneratedClaim anyway
                    claim_type="dependent",
                    focus=f"This is a dependent claim that narrows claim {parent_claim.claim_number}. Provide a meaningful fallback limitation or variation based on concrete physical embodiments or technical constraints described in the patent.",
                    parent_claim_number=parent_claim.claim_number,
                )
                # Overwrite claim_number with our string representation for generate_claim
                pc.claim_number = claim_number_str
                
                claim = self.generate_claim(
                    pc, 
                    patent_description, 
                    previous_claims=all_claims, 
                    similarity_threshold=similarity_threshold
                )
                
                all_claims.append(claim)
                
        return all_claims

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
        drop_duplicates: bool = False,
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
            progress_callback: Optional callback for progress updates
            similarity_threshold: Cosine similarity threshold for RAG retrieval
            drop_duplicates: Whether to remove near-duplicate claims after generation
            
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
            
            planned_claims = self.plan_claims(
                patent_description=patent_description,
                num_independent=num_independent,
            )
            
            print(f"[DEBUG] generate_all_claims: plan_claims() returned {len(planned_claims) if planned_claims else 0} claims")
        except Exception as e:
            print(f"[DEBUG] generate_all_claims: ⚠️  Exception during planning: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            planned_claims = []
        
        if not planned_claims:
            print(f"[DEBUG] generate_all_claims: ⚠️  No claims planned! Using default plan.")
            planned_claims = self._default_plan(num_independent)
        
        # 1. PLAN RE-NUMBERING & NORMALIZATION (Independents only: 1, 2, 3...)
        independents = [pc for pc in planned_claims if pc.claim_type == "independent"]
        
        new_planned_claims = []
        for idx, pc in enumerate(independents, 1):
            pc.claim_number = idx
            new_planned_claims.append(pc)
            
        planned_claims = new_planned_claims
        print(f"✅ Normalized {len(planned_claims)} independent claims")

        if progress_callback:
            progress_callback({
                "stage": "planning_complete",
                "message": f"Planned {len(planned_claims)} independent claims",
                "progress": 20,
                "num_claims": len(planned_claims),
            })
        
        # 2. DRAFTING PHASE
        # 2a. Independent Claim Drafter
        generated_independents = self.draft_independent_claims(
            planned_claims=planned_claims,
            patent_description=patent_description,
            similarity_threshold=similarity_threshold,
            progress_callback=progress_callback
        )
        
        # 2b. Dependent Claim Drafter (Systematically adds fallbacks for each independent)
        generated_claims = self.draft_dependent_claims(
            independent_claims=generated_independents,
            patent_description=patent_description,
            num_dependent_per_independent=num_dependent_per_independent,
            similarity_threshold=similarity_threshold,
            progress_callback=progress_callback
        )
        
        print(f"✅ Finished drafting {len(generated_claims)} total claims")
        
        # CRITICAL: Ensure we always return at least one claim
        if len(generated_claims) == 0:
            print(f"[DEBUG] generate_all_claims: ⚠️  WARNING: No claims were generated! Creating emergency fallback claim.")
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

        # Final pass before refinement: remove near-duplicate claims (very similar content)
        def _normalize_claim_text(text: str) -> str:
            """Normalize claim text for similarity comparison (strip numbers, lowercase, remove punctuation)."""
            if not text:
                return ""
            # Remove leading claim numbering like '1.', '1.1.', '2 '
            cleaned = re.sub(r"^\s*\d+(\.\d+)?\.?\s*", "", text.strip())
            # Lowercase and keep only word characters and spaces
            cleaned = re.sub(r"[^\w\s]", " ", cleaned.lower())
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            return cleaned

        def _cosine_similarity(a: str, b: str) -> float:
            """Calculate cosine similarity based on word counts."""
            if not a or not b:
                return 0.0
            
            # Tokenize and count
            words_a = a.split()
            words_b = b.split()
            
            if not words_a or not words_b:
                return 0.0

            counts_a = Counter(words_a)
            counts_b = Counter(words_b)
            
            # Get unique words
            all_words = set(counts_a.keys()) | set(counts_b.keys())
            
            # Create frequency vectors
            v_a = np.array([counts_a.get(w, 0) for w in all_words])
            v_b = np.array([counts_b.get(w, 0) for w in all_words])
            
            # Calculate cosine similarity: (A . B) / (||A|| * ||B||)
            dot_product = np.dot(v_a, v_b)
            norm_a = np.linalg.norm(v_a)
            norm_b = np.linalg.norm(v_b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
                return float(dot_product / (norm_a * norm_b))

        if drop_duplicates and generated_claims:
            print(f"[Dedup] Starting duplicate check for {len(generated_claims)} claims using cosine similarity")
            unique_claims: List[GeneratedClaim] = []
            normalized_texts: List[str] = []
            for claim in generated_claims:
                norm = _normalize_claim_text(claim.claim_text or "")
                is_duplicate = False
                for prev_norm in normalized_texts:
                    try:
                        sim = _cosine_similarity(norm, prev_norm)
                        if sim >= 0.8:
                            is_duplicate = True
                            break
                    except Exception as e:
                        print(f"[Dedup] Warning: Error comparing claims: {e}")
                        continue
                if not is_duplicate:
                    unique_claims.append(claim)
                    normalized_texts.append(norm)
                else:
                    # 'sim' is guaranteed to be set if is_duplicate is True
                    print(f"[Dedup] Dropping near-duplicate claim {claim.claim_number} (sim: {sim:.2f})")
            if len(unique_claims) < len(generated_claims):
                print(f"[Dedup] Removed {len(generated_claims) - len(unique_claims)} near-duplicate claims")
            generated_claims = unique_claims
        elif not drop_duplicates:
            print(f"[Dedup] Skipping duplicate check (drop_duplicates=False)")
        
        # Phase 2: Quality Assessment (Unity & Quality)
        if LANGGRAPH_AVAILABLE and len(generated_claims) > 0:
            print(f"🔍 Starting quality assessment...")
            if progress_callback:
                progress_callback({
                    "stage": "quality_assessment",
                    "message": "Performing Quality Assessment (consistency check)...",
                    "progress": 70,
                })
            
            try:
                # Range within LangGraph refinement is adjusted to 70% -> 90%
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
                print(f"✨ Quality assessment complete: {len(generated_claims)} claims")
            except Exception as e:
                print(f"[DEBUG] generate_all_claims: Error during quality assessment: {e}")
                import traceback
                traceback.print_exc()
                print(f"[DEBUG] generate_all_claims: Continuing with unrefined claims")

        # Phase 3: Legal Requirements Check (§101, §102, §103, §112)
        if len(generated_claims) > 0:
            try:
                # Per-claim legal refinement (medium changes)
                generated_claims = self.legal_refinement(
                    claims=generated_claims,
                    planned_claims=planned_claims,
                    patent_description=patent_description,
                    progress_callback=progress_callback
                )
                
                # Final total output alignment (very minimal changes)
                generated_claims = self.final_legal_alignment(
                    claims=generated_claims,
                    patent_description=patent_description,
                    progress_callback=progress_callback
                )
            except Exception as e:
                print(f"[DEBUG] generate_all_claims: Error during legal check: {e}")
        
        if progress_callback:
            progress_callback({
                "stage": "complete",
                "message": f"Successfully generated {len(generated_claims)} claims!",
                "progress": 100,
                "num_claims": len(generated_claims),
            })
        
        print(f"🎉 Returning {len(generated_claims)} claims")
        return generated_claims

