"""
LLM-based relation filtering for knowledge graph triples.
Uses an LLM to review relations for each node and identify unnecessary, duplicate, or uninformative relations.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import re

from tools.graph.data.Triple import Triple
from tools.api.llm_api_repo import LLmApi_Repo
from tools.graph.visualizer import GraphVisualizer


class LLMRelationFilter:
    """
    Filters relations using an LLM to identify unnecessary, duplicate, or uninformative relations.
    
    For each node, collects all its relations and asks an LLM to review them.
    Removes relations that are flagged as unnecessary, duplicate, or uninformative.
    """

    def __init__(
        self,
        review_mode: str = "strict",  # "strict", "moderate", "lenient"
        batch_size: int = 10,  # Number of nodes to process per batch
    ):
        """
        Initialize the LLM relation filter.

        Args:
            review_mode: How strict the filtering should be
                - "strict": Remove all flagged relations (most aggressive)
                - "moderate": Remove only clearly unnecessary/duplicate (balanced)
                - "lenient": Only remove obvious duplicates (conservative)
            batch_size: Number of nodes to process in each LLM call
        """
        self.review_mode = review_mode
        self.batch_size = batch_size
        self.api_repo = LLmApi_Repo()

    @staticmethod
    def _entity_key(e) -> str:
        """Get stable entity key."""
        if e is None:
            return ""
        if isinstance(e, str):
            return e
        for attr in ("ref", "id", "ref_short"):
            if hasattr(e, attr):
                v = getattr(e, attr)
                if v:
                    return str(v)
        return str(e)

    @staticmethod
    def _entity_name(e) -> str:
        """Get entity display name."""
        if e is None:
            return ""
        for attr in ("name", "text", "surface", "value"):
            if hasattr(e, attr):
                v = getattr(e, attr)
                if v:
                    return str(v)
        return str(e)

    def _group_triples_by_node(self, triples: List[Triple]) -> Dict[str, List[Tuple[Triple, str]]]:
        """
        Group triples by node (both as head and tail).
        
        Returns:
            Dict mapping node_id -> list of (triple, role) where role is "head" or "tail"
        """
        node_triples = defaultdict(list)
        
        for triple in triples:
            head_id = self._entity_key(triple.head)
            tail_id = self._entity_key(triple.tail)
            relation = getattr(triple, "relation", "").strip()
            
            if not head_id or not tail_id or not relation:
                continue
            
            # Add triple from head's perspective
            node_triples[head_id].append((triple, "head"))
            # Add triple from tail's perspective
            node_triples[tail_id].append((triple, "tail"))
        
        return node_triples

    def _format_node_relations(self, node_id: str, node_name: str, relations: List[Tuple[Triple, str]]) -> str:
        """Format relations for a node into a readable string for the LLM."""
        lines = []
        for i, (triple, role) in enumerate(relations, 1):
            if role == "head":
                other_entity = self._entity_name(triple.tail)
                other_id = self._entity_key(triple.tail)
                direction = "outgoing"
            else:
                other_entity = self._entity_name(triple.head)
                other_id = self._entity_key(triple.head)
                direction = "incoming"
            
            relation = getattr(triple, "relation", "").strip()
            lines.append(
                f"{i}. [{direction}] {relation} → {other_entity} (id: {other_id})"
            )
        
        return "\n".join(lines)

    def _build_prompt(self, node_id: str, node_name: str, relations: List[Tuple[Triple, str]]) -> str:
        """Build prompt for LLM to review relations."""
        relations_text = self._format_node_relations(node_id, node_name, relations)
        
        mode_instructions = {
            "strict": "Be thorough and remove any relations that are unnecessary, redundant, or add little value.",
            "moderate": "Remove relations that are clearly unnecessary, duplicate, or uninformative. Keep relations that add meaningful information.",
            "lenient": "Only remove relations that are obvious duplicates or completely redundant. Be conservative.",
        }
        
        prompt = f"""You are reviewing relations in a knowledge graph for the entity "{node_name}" (id: {node_id}).

The entity has the following relations:
{relations_text}

Task: Identify which relations should be REMOVED because they are:
1. Unnecessary - don't add meaningful information
2. Duplicate - redundant with other relations
3. Uninformative - too generic or vague to be useful

Review mode: {self.review_mode}
{mode_instructions.get(self.review_mode, mode_instructions["moderate"])}

Return ONLY a JSON array of the relation numbers (1, 2, 3, etc.) that should be REMOVED.
If no relations should be removed, return an empty array: [].

Example response: [2, 5, 7] or []

Return ONLY the JSON array, no other text:"""

        return prompt

    def _parse_llm_response(self, response: str) -> Set[int]:
        """Parse LLM response to extract relation numbers to remove."""
        # Try to extract JSON array from response
        response = response.strip()
        
        # Remove markdown code fences if present
        response = re.sub(r"^```(?:json)?\s*|\s*```$", "", response, flags=re.MULTILINE)
        
        try:
            # Try to parse as JSON
            parsed = json.loads(response)
            if isinstance(parsed, list):
                # Convert to set of integers (1-indexed relation numbers)
                return {int(x) for x in parsed if isinstance(x, (int, str)) and str(x).isdigit()}
        except (json.JSONDecodeError, ValueError):
            # Try to extract numbers from text
            numbers = re.findall(r'\b(\d+)\b', response)
            if numbers:
                return {int(n) for n in numbers}
        
        return set()

    def _filter_node_relations(
        self, 
        node_id: str, 
        node_name: str, 
        relations: List[Tuple[Triple, str]]
    ) -> List[Tuple[Triple, str]]:
        """Use LLM to filter relations for a single node."""
        if len(relations) <= 1:
            # No filtering needed for nodes with 0 or 1 relation
            return relations
        
        prompt = self._build_prompt(node_id, node_name, relations)
        
        try:
            response = self.api_repo.chat(prompt)
            # Extract text from response (handle different response formats)
            # LLmApi_Repo.chat() typically returns a string (the message content)
            if isinstance(response, dict):
                # Try common response formats
                text = response.get("content", response.get("text", response.get("message", "")))
                if not text:
                    # Try nested structures (e.g., OpenAI format)
                    if "choices" in response and len(response["choices"]) > 0:
                        choice = response["choices"][0]
                        if isinstance(choice, dict):
                            text = choice.get("message", {}).get("content", "")
                    if not text:
                        # Try to get first value that's a string
                        for v in response.values():
                            if isinstance(v, str):
                                text = v
                                break
            elif isinstance(response, list):
                # Some models return a list of strings
                text = " ".join(str(x) for x in response) if response else ""
            elif isinstance(response, str):
                text = response
            else:
                text = str(response)
            
            if not text:
                # If we couldn't extract text, keep all relations
                return relations
            
            # Parse which relations to remove (1-indexed)
            to_remove = self._parse_llm_response(text)
            
            # Filter out relations that should be removed
            filtered = [
                rel for i, rel in enumerate(relations, 1) 
                if i not in to_remove
            ]
            
            return filtered
            
        except Exception as e:
            print(f"  ⚠️  Error filtering relations for node {node_id}: {e}")
            # On error, keep all relations
            return relations

    def filter_relations(
        self,
        triples: List[Triple],
        id_to_name: Dict[str, str] = None,
    ) -> Tuple[List[Triple], Dict[str, any]]:
        """
        Filter relations using LLM review.

        Args:
            triples: List of Triple objects to filter
            id_to_name: Optional mapping from node ID to display name

        Returns:
            Tuple of (filtered_triples, stats_dict)
        """
        id_to_name = id_to_name or {}
        
        # Group triples by node
        node_triples = self._group_triples_by_node(triples)
        
        print(f"Reviewing relations for {len(node_triples)} nodes...")
        print(f"Review mode: {self.review_mode}")
        print("=" * 80)
        
        # Track which triples to keep
        triple_to_keep = {id(t): True for t in triples}
        total_removed = 0
        nodes_processed = 0
        
        # Process nodes
        for node_id, relations in node_triples.items():
            node_name = id_to_name.get(node_id, self._entity_name(relations[0][0].head if relations[0][1] == "head" else relations[0][0].tail))
            
            if len(relations) <= 1:
                # Skip nodes with 0 or 1 relation
                continue
            
            nodes_processed += 1
            print(f"\n[{nodes_processed}/{len(node_triples)}] Reviewing node: {node_name} ({len(relations)} relations)")
            
            # Filter relations for this node
            filtered_relations = self._filter_node_relations(node_id, node_name, relations)
            
            # Mark removed triples
            removed_count = len(relations) - len(filtered_relations)
            if removed_count > 0:
                kept_triple_ids = {id(t) for t, _ in filtered_relations}
                for triple, _ in relations:
                    if id(triple) not in kept_triple_ids:
                        triple_to_keep[id(triple)] = False
                        total_removed += 1
                print(f"  ✅ Removed {removed_count} relation(s)")
            else:
                print(f"  ✓ All relations kept")
        
        # Build filtered triples list
        filtered_triples = [t for t in triples if triple_to_keep.get(id(t), True)]
        
        stats = {
            "input_triples": len(triples),
            "output_triples": len(filtered_triples),
            "removed_triples": total_removed,
            "nodes_reviewed": nodes_processed,
            "review_mode": self.review_mode,
        }
        
        print("\n" + "=" * 80)
        print(f"✅ Relation filtering complete!")
        print(f"   Input: {len(triples)} triples")
        print(f"   Output: {len(filtered_triples)} triples")
        print(f"   Removed: {total_removed} triples")
        print("=" * 80)
        
        return filtered_triples, stats

