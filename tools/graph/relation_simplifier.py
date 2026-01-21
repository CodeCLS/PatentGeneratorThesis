"""
Relation simplifier for breaking down complex relations into RHF-compatible simple relations
with properties/qualifiers (hyper-relational approach).

This is the RECOMMENDED approach as it:
- Keeps relations simple and RHF-compatible
- Preserves modifiers as properties (standard knowledge graph practice)
- Avoids creating helper nodes (cleaner graph structure)
- Aligns with hyper-relational knowledge graph standards
"""
from __future__ import annotations

import re
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
from tools.graph.data.Triple import Triple
from tools.sentence.entity import Entity, InMemoryEntityRepository
from tools.api.llm_api_repo import LLmApi_Repo
from tools.helper.json_helper import JsonHelper


class RelationSimplifier:
    """
    Simplifies complex relations into RHF-compatible simple relations with properties.
    
    This uses a hyper-relational approach (Option 2) which is the recommended
    standard in knowledge graphs. Instead of creating helper nodes, modifiers
    are stored as properties/qualifiers on the relation.
    
    Example:
    - Original: Caleb -> "Drives very slowly and interestingly to" -> Nathan
    - Simplified: Caleb -> "Drives" -> Nathan
      Properties: {"manner": "very slowly and interestingly", "direction": "to"}
    
    This approach:
    - Keeps relations simple (1-3 words, RHF-compatible)
    - Preserves all information as properties
    - Follows hyper-relational knowledge graph standards
    - Avoids graph bloat from helper nodes
    """
    
    def __init__(
        self,
        max_relation_length: int = 5,
        api_repo: Optional[LLmApi_Repo] = None,
        verbose: bool = True,
        num_workers: int = 8
    ):
        """
        Initialize relation simplifier.
        
        Args:
            max_relation_length: Maximum number of words in a relation before considering it complex (default: 5)
            api_repo: LLM API repository for simplification (default: creates new one)
            verbose: Whether to print progress (default: True)
            num_workers: Number of parallel workers for processing (default: 8). Set to 1 for sequential processing.
        """
        self.max_relation_length = max_relation_length
        self.api_repo = api_repo
        self.verbose = verbose
        self.num_workers = num_workers
        self.sequential = num_workers == 1
        
        # Thread-local storage for API clients (one per worker thread)
        self._thread_local = threading.local()
        
        # Single API repo for sequential mode
        if self.sequential and self.api_repo is None:
            self.api_repo = LLmApi_Repo()
    
    def _get_api_repo(self) -> LLmApi_Repo:
        """Get or create API repo for current thread."""
        if self.sequential:
            return self.api_repo
        else:
            if not hasattr(self._thread_local, 'api_repo'):
                self._thread_local.api_repo = LLmApi_Repo() if self.api_repo is None else self.api_repo
            return self._thread_local.api_repo
    
    def _is_complex_relation(self, relation: str) -> bool:
        """
        Check if a relation is too complex and needs simplification.
        
        Args:
            relation: The relation string to check
            
        Returns:
            True if relation is complex and should be simplified
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
    
    def _simplify_relation_with_llm(
        self, 
        relation: str, 
        head_entity: Entity, 
        tail_entity: Entity,
        api_repo: Optional[LLmApi_Repo] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to simplify a complex relation and extract properties.
        
        Args:
            relation: The complex relation to simplify
            head_entity: The head entity of the original triple
            tail_entity: The tail entity of the original triple
            api_repo: Optional API repo to use (if None, uses thread-local or instance repo)
            
        Returns:
            Dictionary with:
            - "simple_relation": simplified relation (1-3 words)
            - "properties": dict of properties/qualifiers
            Returns empty dict if simplification fails
        """
        # Use provided api_repo, or get thread-local/instance one
        repo = api_repo or self._get_api_repo()
        
        prompt = f"""You are a knowledge graph relation simplifier. Your task is to break down complex relations into simple, RHF-compatible relations with properties/qualifiers.

RHF (Relation Head Format) requires relations to be:
- Short (1-3 words ideally)
- Simple (single verb or preposition)
- Clear and unambiguous

For complex relations, extract:
1. The core simple relation (main verb/action)
2. Modifiers as properties/qualifiers (manner, direction, intensity, etc.)

Example:
Input: "Drives very slowly and interestingly to"
Output: {{
  "simple_relation": "Drives",
  "properties": {{
    "manner": "very slowly and interestingly",
    "direction": "to"
  }}
}}

Another example:
Input: "Connects with high precision and accuracy"
Output: {{
  "simple_relation": "Connects",
  "properties": {{
    "manner": "with high precision and accuracy"
  }}
}}

Another example:
Input: "Transmits data quickly over network"
Output: {{
  "simple_relation": "Transmits",
  "properties": {{
    "object": "data",
    "manner": "quickly",
    "medium": "over network"
  }}
}}

Rules:
1. Extract the main verb/action as the simple relation (1-3 words max)
2. Extract all modifiers, adverbs, adjectives, prepositional phrases as properties
3. Use standard property keys: "manner", "direction", "intensity", "object", "medium", "condition", "time", "location", etc.
4. Preserve all semantic information in properties
5. If the relation is already simple, return it as-is with empty properties

Input relation: "{relation}"
Head entity: {head_entity.name} (label: {head_entity.label})
Tail entity: {tail_entity.name} (label: {tail_entity.label})

Return ONLY a JSON object with keys: "simple_relation" (string) and "properties" (dict).
If the relation cannot be simplified, return an empty object: {{}}.

Example format:
{{"simple_relation": "Drives", "properties": {{"manner": "very slowly and interestingly", "direction": "to"}}}}

Return ONLY the JSON object, no other text:"""

        try:
            response = repo.chat(prompt)
            
            # Handle different response formats
            if isinstance(response, dict):
                response_text = response.get("content", response.get("text", response.get("message", "")))
                if not response_text and "choices" in response:
                    response_text = response["choices"][0].get("message", {}).get("content", "")
            else:
                response_text = str(response) if response else ""
            
            # Parse JSON
            parsed = JsonHelper.parse_json(response_text)
            
            if isinstance(parsed, dict) and "simple_relation" in parsed:
                return parsed
            else:
                if self.verbose:
                    print(f"    ⚠️  LLM returned invalid simplification for: {relation}")
                return {}
                
        except Exception as e:
            if self.verbose:
                print(f"    ❌ Error simplifying relation '{relation}': {e}")
            return {}
    
    def _simplify_triple(self, triple: Triple, api_repo: Optional[LLmApi_Repo] = None) -> Optional[Triple]:
        """
        Simplify a single triple if its relation is complex.
        
        Args:
            triple: The triple to potentially simplify
            api_repo: Optional API repo to use (for parallel processing)
            
        Returns:
            New simplified triple with properties, or None if simplification failed
        """
        if not self._is_complex_relation(triple.relation):
            return None  # No simplification needed
        
        if self.verbose:
            print(f"  Simplifying: {triple.head.name} -> '{triple.relation}' -> {triple.tail.name}")
        
        # Get simplification from LLM
        simplification = self._simplify_relation_with_llm(
            triple.relation,
            triple.head,
            triple.tail,
            api_repo=api_repo
        )
        
        if not simplification or "simple_relation" not in simplification:
            if self.verbose:
                print(f"    ⚠️  Could not simplify, keeping original")
            return None  # Could not simplify, keep original
        
        simple_relation = simplification.get("simple_relation", "").strip()
        properties = simplification.get("properties", {})
        
        if not simple_relation:
            return None
        
        # Create new simplified triple with properties
        simplified_triple = Triple(
            head=triple.head,
            relation=simple_relation,
            tail=triple.tail
        )
        
        # Copy existing metadata
        simplified_triple.id = triple.id
        simplified_triple.lang = triple.lang
        simplified_triple.importance = triple.importance
        simplified_triple.info_quality = triple.info_quality
        simplified_triple.novelty = triple.novelty
        simplified_triple.embedding = triple.embedding.copy()
        simplified_triple.tags = triple.tags.copy()
        
        # Add properties/qualifiers
        for key, value in properties.items():
            simplified_triple.set_property(key, value)
        
        # Mark as simplified
        simplified_triple.add_tag("simplified")
        simplified_triple.set_property("original_relation", triple.relation)
        
        if self.verbose:
            props_str = ", ".join([f"{k}: {v}" for k, v in properties.items()])
            print(f"    ✅ Simplified to: '{simple_relation}' (properties: {props_str})")
        
        return simplified_triple
    
    def simplify(
        self, 
        triples: List[Triple],
        entity_repo: Optional[InMemoryEntityRepository] = None
    ) -> List[Triple]:
        """
        Simplify complex relations in a list of triples.
        
        Args:
            triples: List of triples to process
            entity_repo: Optional entity repository (not used in this approach, kept for API compatibility)
            
        Returns:
            List of triples (simplified ones replaced, simple ones kept as-is)
        """
        simplified_count = 0
        kept_count = 0
        
        if self.verbose:
            print(f"Analyzing {len(triples)} triples for complex relations...")
            if not self.sequential:
                print(f"Using {self.num_workers} parallel workers")
            print("=" * 80)
        
        # Filter out triples that don't need simplification (fast check)
        triples_to_process = []
        for i, triple in enumerate(triples):
            if self._is_complex_relation(triple.relation):
                triples_to_process.append((i, triple))
        
        if self.sequential or len(triples_to_process) == 0:
            # Sequential processing or nothing to simplify
            new_triples = []
            for i, triple in enumerate(triples):
                if self._is_complex_relation(triple.relation):
                    simplified = self._simplify_triple(triple)
                    if simplified:
                        new_triples.append(simplified)
                        simplified_count += 1
                    else:
                        new_triples.append(triple)
                        kept_count += 1
                else:
                    new_triples.append(triple)
                    kept_count += 1
        else:
            # Parallel processing
            new_triples = [None] * len(triples)
            
            # First, set all simple triples (kept as-is)
            for i, triple in enumerate(triples):
                if not self._is_complex_relation(triple.relation):
                    new_triples[i] = triple
                    kept_count += 1
            
            # Process complex triples in parallel
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all simplification tasks
                future_to_index = {
                    executor.submit(self._simplify_triple, triple, api_repo=None): idx
                    for idx, triple in triples_to_process
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        simplified = future.result()
                        if simplified:
                            new_triples[idx] = simplified
                            simplified_count += 1
                        else:
                            new_triples[idx] = triples[idx]  # Keep original
                            kept_count += 1
                    except Exception as e:
                        if self.verbose:
                            print(f"    ❌ Error processing triple {idx}: {e}")
                        new_triples[idx] = triples[idx]  # Keep original on error
                        kept_count += 1
        
        if self.verbose:
            print("\n" + "=" * 80)
            print(f"✅ Simplification complete!")
            print(f"   Original triples: {len(triples)}")
            print(f"   Simplified: {simplified_count}")
            print(f"   Kept as-is: {kept_count}")
            print(f"   Final triples: {len(new_triples)}")
            print("=" * 80)
        
        return new_triples

