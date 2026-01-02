"""
ClaimDraftingAgent: Drafts patent claims from claim bundles using LLM.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
import networkx as nx

from tools.graph.claim_extractor import ClaimBundle, AssertionInfo
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
    
    def __init__(self, api_repo: Optional[LLmApi_Repo] = None):
        """
        Initialize the claim drafting agent.
        
        Args:
            api_repo: Optional LLM API repository (defaults to LLmApi_Repo())
        """
        self.api_repo = api_repo or LLmApi_Repo()
        self.task = (
            "You are an expert patent attorney drafting patent claims.\n\n"
            "TASK: Draft a formal patent claim based on the provided assertions.\n\n"
            "MECHANISM-FIRST DRAFTING (CRITICAL):\n"
            "- The claim MUST describe HOW THE INVENTION WORKS, not just list features\n"
            "- Focus on the invention mechanism: components + their interactions + technical effects\n"
            "- Prioritize assertions that describe the inventive mechanism and technical components\n"
            "- Ensure the claim describes a coherent technical system, not just isolated parts\n"
            "- If assertions don't form a coherent mechanism, focus on the core inventive mechanism\n\n"
            "CRITICAL RULES:\n"
            "- Use ONLY the provided assertions - do NOT invent new limitations\n"
            "- Ensure antecedent basis (terms introduced in independent claims can be referenced in dependents)\n"
            "- Use broad, general language appropriate for patent claims\n"
            "- Follow standard patent claim format (numbered, single sentence per claim)\n"
            "- For dependent claims, reference the parent claim number explicitly\n"
            "- Do NOT include any text other than the claim itself\n"
            "- For independent claims: describe the complete invention mechanism, not just components\n\n"
            "OUTPUT FORMAT:\n"
            "- Return ONLY the claim text, nothing else\n"
            "- No claim number prefix (e.g., don't write '1. ' or 'Claim 1:')\n"
            "- No markdown, no commentary, just the claim text\n"
            "- Example: 'A display device comprising a water tank and a bubble generator.'\n"
        )
    
    def draft(
        self,
        claim_bundles: List[ClaimBundle],
        glossary: Optional[Dict[str, str]] = None,
        previous_claims: Optional[List[str]] = None,
        patent_description: Optional[str] = None,
    ) -> List[DraftedClaim]:
        """
        Draft patent claims from claim bundles.
        
        Args:
            claim_bundles: List of ClaimBundle objects to draft
            glossary: Optional glossary of canonical terms
            previous_claims: Optional list of previously drafted claims (for dependent claims)
            patent_description: Optional full patent description text for context (not directly used, but helps understand the invention)
            
        Returns:
            List of DraftedClaim objects with numbered claims
        """
        glossary = glossary or {}
        previous_claims = previous_claims or []
        
        drafted_claims: List[DraftedClaim] = []
        claim_number = 1
        
        # Process independent claims first
        independent_bundles = [b for b in claim_bundles if b.type == "independent"]
        dependent_bundles = [b for b in claim_bundles if b.type == "dependent"]
        
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
    
    def _draft_single_claim(
        self,
        bundle: ClaimBundle,
        glossary: Dict[str, str],
        previous_claims: List[str],
        is_dependent: bool,
        parent_claim_number: Optional[int] = None,
        patent_description: Optional[str] = None,
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
            
            # Add category information to help LLM prioritize
            category = assertion.category or "UNCLASSIFIED"
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
                "- Focus on: (1) core components, (2) how they interact, (3) the technical effect\n"
                "- Do NOT just list features - describe the coherent technical system\n"
                "- The assertions are prioritized: INVENTIVE_MECHANISM → TECHNICAL_COMPONENT → TECHNICAL_EFFECT\n"
                "- Use the most important assertions (those describing the mechanism) to form the core of the claim\n"
                "- Ensure the claim describes a working invention, not just a collection of parts\n\n"
            )
        else:
            mechanism_instructions = (
                "\nTECHNICAL REFINEMENT FOR DEPENDENT CLAIM:\n"
                "- This claim adds technical refinements to the parent claim\n"
                "- Focus on specific improvements or additional technical features\n"
                "- Do NOT include background, problems, or aesthetic features\n\n"
            )
        
        # Build claim type instructions
        if is_dependent and parent_claim_number:
            claim_type_instructions = (
                f"This is a DEPENDENT claim that depends on claim {parent_claim_number}.\n"
                "Reference the parent claim explicitly (e.g., 'The device of claim {parent_claim_number}...').\n"
                "Add ONE or more additional limitations from the assertions below.\n\n"
            )
        else:
            claim_type_instructions = (
                "This is an INDEPENDENT claim.\n"
                "Draft a complete, standalone claim using the assertions below.\n\n"
            )
        
        prompt = (
            f"{self.task}\n\n"
            f"{mechanism_instructions}"
            f"{patent_context}"
            f"{glossary_text}"
            f"{previous_claims_text}"
            f"{claim_type_instructions}"
            f"Assertions to use (prioritized by mechanism importance):\n{assertions_block}\n\n"
            "Draft the claim now. Focus on describing HOW THE INVENTION WORKS (the mechanism), not just listing features.\n"
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

