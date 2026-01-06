"""
Relation decomposer for breaking down complex relations into RHF-compatible simpler relations.
Creates helper entities (RELATION_MODIFIER) to preserve information when decomposing.
"""
from __future__ import annotations

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from tools.graph.Triple import Triple
from tools.sentence.entity import Entity, InMemoryEntityRepository
from tools.api.llm_api_repo import LLmApi_Repo
from tools.helper.json_helper import JsonHelper


class RelationDecomposer:
    """
    Decomposes complex relations into simpler RHF-compatible relations.
    
    For complex relations like "Drives very slowly and interestingly to",
    creates a chain of simpler relations with helper modifier entities:
    - Original: Caleb -> "Drives very slowly and interestingly to" -> Nathan
    - Decomposed: 
      * Caleb -> "Drives" -> [Helper: "very slowly and interestingly"] -> "to" -> Nathan
      * Helper entity is labeled as RELATION_MODIFIER
    """
    
    # Helper entity label - clear and descriptive
    HELPER_ENTITY_LABEL = "RELATION_MODIFIER"
    
    def __init__(
        self,
        max_relation_length: int = 5,
        max_decomposition_depth: int = 2,
        api_repo: Optional[LLmApi_Repo] = None,
        verbose: bool = True
    ):
        """
        Initialize relation decomposer.
        
        Args:
            max_relation_length: Maximum number of words in a relation before considering it complex (default: 5)
            max_decomposition_depth: Maximum depth of decomposition to avoid infinite recursion (default: 2)
            api_repo: LLM API repository for decomposition (default: creates new one)
            verbose: Whether to print progress (default: True)
        """
        self.max_relation_length = max_relation_length
        self.max_decomposition_depth = max_decomposition_depth
        self.api_repo = api_repo or LLmApi_Repo()
        self.verbose = verbose
        
        # Track created helper entities to avoid duplicates
        self._helper_entities: Dict[str, Entity] = {}
    
    def _is_complex_relation(self, relation: str) -> bool:
        """
        Check if a relation is too complex and needs decomposition.
        
        Args:
            relation: The relation string to check
            
        Returns:
            True if relation is complex and should be decomposed
        """
        if not relation or not relation.strip():
            return False
        
        # Count words (simple heuristic)
        words = relation.strip().split()
        if len(words) > self.max_relation_length:
            return True
        
        # Check for complex patterns (multiple verbs, conjunctions, etc.)
        complex_patterns = [
            r'\b(and|or|but|with|while|when|where|that|which)\b',  # Conjunctions
            r'\b(very|extremely|quite|rather|somewhat|highly)\b',  # Intensifiers
            r'\b(more|most|less|least)\b',  # Comparatives
        ]
        
        for pattern in complex_patterns:
            if len(re.findall(pattern, relation, re.IGNORECASE)) > 1:
                return True
        
        return False
    
    def _decompose_relation_with_llm(
        self, 
        relation: str, 
        head_entity: Entity, 
        tail_entity: Entity,
        depth: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to decompose a complex relation into simpler relations.
        
        Args:
            relation: The complex relation to decompose
            head_entity: The head entity of the original triple
            tail_entity: The tail entity of the original triple
            depth: Current decomposition depth (to prevent infinite recursion)
            
        Returns:
            List of decomposition steps, each with:
            - "relation": simplified relation
            - "modifier": modifier text (if any)
            - "next_entity": next entity in chain (if any)
        """
        if depth >= self.max_decomposition_depth:
            if self.verbose:
                print(f"    ⚠️  Max decomposition depth reached for: {relation}")
            return []
        
        prompt = f"""You are a knowledge graph relation decomposer. Your task is to break down complex relations into simpler, RHF-compatible relations.

RHF (Relation Head Format) requires relations to be:
- Short (1-3 words ideally)
- Simple (single verb or preposition)
- Clear and unambiguous

Given a complex relation, decompose it into a chain of simpler relations with helper modifier entities when needed.

Example:
Input: "Drives very slowly and interestingly to"
Output: [
  {{"relation": "Drives", "modifier": "very slowly and interestingly", "next_relation": "to"}}
]

Another example:
Input: "Connects with high precision and accuracy"
Output: [
  {{"relation": "Connects", "modifier": "with high precision and accuracy"}}
]

Rules:
1. Extract the main verb/action as the first relation
2. Extract modifiers (adverbs, adjectives, prepositional phrases) as modifier text
3. If there's a preposition leading to the tail, extract it as a separate relation
4. Modifiers should be preserved as helper entities (RELATION_MODIFIER)
5. Each relation should be 1-3 words maximum
6. Preserve the semantic meaning

Input relation: "{relation}"
Head entity: {head_entity.name} (label: {head_entity.label})
Tail entity: {tail_entity.name} (label: {tail_entity.label})

Return ONLY a JSON array of objects with keys: "relation", "modifier" (optional), "next_relation" (optional).
If the relation cannot be meaningfully decomposed, return an empty array: [].

Example format:
[{{"relation": "Drives", "modifier": "very slowly and interestingly", "next_relation": "to"}}]

Return ONLY the JSON array, no other text:"""

        try:
            response = self.api_repo.chat(prompt)
            
            # Handle different response formats
            if isinstance(response, dict):
                response_text = response.get("content", response.get("text", response.get("message", "")))
                if not response_text and "choices" in response:
                    response_text = response["choices"][0].get("message", {}).get("content", "")
            else:
                response_text = str(response) if response else ""
            
            # Parse JSON
            parsed = JsonHelper.parse_json(response_text)
            
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
            else:
                if self.verbose:
                    print(f"    ⚠️  LLM returned empty or invalid decomposition for: {relation}")
                return []
                
        except Exception as e:
            if self.verbose:
                print(f"    ❌ Error decomposing relation '{relation}': {e}")
            return []
    
    def _create_helper_entity(self, modifier_text: str) -> Entity:
        """
        Create or retrieve a helper entity for a modifier.
        
        Args:
            modifier_text: The modifier text
            
        Returns:
            Entity object labeled as RELATION_MODIFIER
        """
        # Normalize modifier text for deduplication
        normalized = modifier_text.strip().lower()
        
        if normalized in self._helper_entities:
            return self._helper_entities[normalized]
        
        # Create new helper entity
        helper_entity = Entity(
            name=modifier_text.strip(),
            label=self.HELPER_ENTITY_LABEL,
            ref_short=f"MOD_{len(self._helper_entities) + 1}",
            entity_type="RELATION_MODIFIER"
        )
        
        self._helper_entities[normalized] = helper_entity
        return helper_entity
    
    def _decompose_triple(
        self, 
        triple: Triple,
        depth: int = 0
    ) -> List[Triple]:
        """
        Decompose a single triple if its relation is complex.
        
        Args:
            triple: The triple to potentially decompose
            depth: Current decomposition depth
            
        Returns:
            List of new triples (may be empty if no decomposition needed, or contain decomposed triples)
        """
        if not self._is_complex_relation(triple.relation):
            return []  # No decomposition needed
        
        if self.verbose:
            print(f"  Decomposing: {triple.head.name} -> '{triple.relation}' -> {triple.tail.name}")
        
        # Get decomposition from LLM
        decomposition = self._decompose_relation_with_llm(
            triple.relation,
            triple.head,
            triple.tail,
            depth
        )
        
        if not decomposition:
            if self.verbose:
                print(f"    ⚠️  Could not decompose, keeping original")
            return []  # Could not decompose, keep original
        
        # Build chain of triples from decomposition
        # Example: "Drives very slowly and interestingly to"
        # Step: {"relation": "Drives", "modifier": "very slowly and interestingly", "next_relation": "to"}
        # Result: 
        #   1. head -> "Drives" -> [Helper: "very slowly and interestingly"]
        #   2. [Helper: "very slowly and interestingly"] -> "to" -> tail
        
        new_triples = []
        current_entity = triple.head
        
        for step in decomposition:
            relation = step.get("relation", "").strip()
            modifier = step.get("modifier", "").strip()
            next_relation = step.get("next_relation", "").strip()
            
            if not relation:
                continue
            
            # Step 1: Create triple with main relation
            if modifier:
                # Create helper entity for modifier
                helper_entity = self._create_helper_entity(modifier)
                
                # Link: current_entity -> relation -> helper_entity
                new_triple = Triple(
                    head=current_entity,
                    relation=relation,
                    tail=helper_entity
                )
                new_triple.add_tag("decomposed")
                new_triple.add_tag(f"decomposition_depth_{depth}")
                new_triples.append(new_triple)
                
                current_entity = helper_entity
            else:
                # No modifier, but might have next_relation
                if next_relation:
                    # Need intermediate helper for chaining
                    helper_entity = self._create_helper_entity(f"via {relation}")
                    
                    new_triple = Triple(
                        head=current_entity,
                        relation=relation,
                        tail=helper_entity
                    )
                    new_triple.add_tag("decomposed")
                    new_triple.add_tag(f"decomposition_depth_{depth}")
                    new_triples.append(new_triple)
                    
                    current_entity = helper_entity
                else:
                    # No modifier, no next_relation - link directly to tail
                    new_triple = Triple(
                        head=current_entity,
                        relation=relation,
                        tail=triple.tail
                    )
                    new_triple.add_tag("decomposed")
                    new_triple.add_tag(f"decomposition_depth_{depth}")
                    new_triples.append(new_triple)
                    return new_triples  # Done
            
            # Step 2: If there's a next_relation, link to tail
            if next_relation:
                final_triple = Triple(
                    head=current_entity,
                    relation=next_relation,
                    tail=triple.tail
                )
                final_triple.add_tag("decomposed")
                final_triple.add_tag(f"decomposition_depth_{depth}")
                new_triples.append(final_triple)
        
        return new_triples
        
        if self.verbose:
            print(f"    ✅ Created {len(new_triples)} new triples from decomposition")
        
        return new_triples
    
    def decompose(
        self, 
        triples: List[Triple],
        entity_repo: Optional[InMemoryEntityRepository] = None
    ) -> Tuple[List[Triple], InMemoryEntityRepository]:
        """
        Decompose complex relations in a list of triples.
        
        Args:
            triples: List of triples to process
            entity_repo: Optional entity repository (will be created if not provided)
            
        Returns:
            Tuple of (new_triples_list, updated_entity_repo)
            - new_triples_list: All triples (original simple ones + decomposed ones)
            - updated_entity_repo: Entity repository including helper entities
        """
        if entity_repo is None:
            entity_repo = InMemoryEntityRepository()
        
        # Collect all entities from original triples
        for triple in triples:
            entity_repo.save(triple.head)
            entity_repo.save(triple.tail)
        
        # Reset helper entities for this decomposition run
        self._helper_entities.clear()
        
        new_triples = []
        decomposed_count = 0
        kept_count = 0
        
        if self.verbose:
            print(f"Analyzing {len(triples)} triples for complex relations...")
            print("=" * 80)
        
        for triple in triples:
            if self._is_complex_relation(triple.relation):
                decomposed = self._decompose_triple(triple)
                if decomposed:
                    new_triples.extend(decomposed)
                    # Save helper entities to repo
                    for helper_entity in self._helper_entities.values():
                        entity_repo.save(helper_entity)
                    decomposed_count += 1
                else:
                    # Could not decompose, keep original
                    new_triples.append(triple)
                    kept_count += 1
            else:
                # Simple relation, keep as-is
                new_triples.append(triple)
                kept_count += 1
        
        if self.verbose:
            print("\n" + "=" * 80)
            print(f"✅ Decomposition complete!")
            print(f"   Original triples: {len(triples)}")
            print(f"   Decomposed: {decomposed_count}")
            print(f"   Kept as-is: {kept_count}")
            print(f"   New triples: {len(new_triples)}")
            print(f"   Helper entities created: {len(self._helper_entities)}")
            print("=" * 80)
        
        return new_triples, entity_repo

