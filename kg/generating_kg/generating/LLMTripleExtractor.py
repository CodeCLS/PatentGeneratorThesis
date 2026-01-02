"""
LLM-based triple extractor that extracts entities and triples in a single API call.
"""
from __future__ import annotations

from typing import List, Dict, Any
from tools.api.llm_api_repo import LLmApi_Repo
from tools.helper.json_helper import JsonHelper
from tools.sentence.entity import Entity
from tools.graph.Triple import Triple
import uuid


class LLMTripleExtractor:
    """
    Extracts entities and triples from text in a single LLM API call.
    """
    
    def __init__(self):
        self.api_repo = LLmApi_Repo()
        self.task = (
            "You are an expert knowledge graph extraction system for patent text.\n\n"
            "Task: Extract all entities and their relationships (triples) from the given patent text.\n\n"
            "INSTRUCTIONS:\n"
            "1. First, identify all technical entities (components, materials, processes, functions, etc.)\n"
            "2. Then, extract relationships between these entities as triples\n"
            "3. Each triple must be EXPLICITLY and UNEQUIVOCALLY stated in the text\n"
            "4. DO NOT infer, assume, or guess relationships\n"
            "5. If uncertain about a relationship, DO NOT include it\n\n"
            "ENTITY TYPES:\n"
            "INVENTION, COMPONENT, MATERIAL, FUNCTION, PROCESS_STEP, METHOD, PARAMETER, "
            "MEASUREMENT, CONDITION, HARDWARE, SOFTWARE, UNCLASSIFIED_ENTITY\n\n"
            "OUTPUT FORMAT:\n"
            "Return a JSON object with two keys:\n"
            "- 'entities': Array of objects with keys: name, label, start, end\n"
            "- 'triples': Array of objects with keys: head, relation, tail\n"
            "  where head and tail are entity names (you will assign IDs)\n\n"
            "Example:\n"
            '{\n'
            '  "entities": [\n'
            '    {"name": "display device", "label": "INVENTION", "start": 4, "end": 18},\n'
            '    {"name": "pseudo space", "label": "CONDITION", "start": 35, "end": 47}\n'
            '  ],\n'
            '  "triples": [\n'
            '    {"head": "display device", "relation": "is for appreciation regarding", "tail": "pseudo space"}\n'
            '  ]\n'
            '}\n\n'
            "Return ONLY valid JSON, no markdown fences, no commentary."
        )
    
    def extract(self, text: str) -> tuple[List[Entity], List[Dict[str, str]]]:
        """
        Extract entities and triples from text in a single LLM call.
        
        Args:
            text: The text to extract from
            
        Returns:
            Tuple of (entities, triples) where:
            - entities: List of Entity objects
            - triples: List of dicts with keys: head, relation, tail (using entity names)
        """
        prompt = (
            f"{self.task}\n\n"
            f"Text:\n{text}\n\n"
            "Return ONLY a valid JSON object with 'entities' and 'triples' keys.\n"
            "No markdown fences, no commentary."
        )
        
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
            data = self._parse_json(response_text)
            
            if not isinstance(data, dict):
                print("⚠️  LLM response was not a JSON object")
                return [], []
            
            # Extract entities
            entities_data = data.get("entities", [])
            entities = []
            entity_name_to_id = {}  # Map entity names to IDs
            
            for ent_data in entities_data:
                if not isinstance(ent_data, dict):
                    continue
                
                name = str(ent_data.get("name", "")).strip()
                label = str(ent_data.get("label", "UNCLASSIFIED_ENTITY")).strip().upper()
                start = ent_data.get("start", 0)
                end = ent_data.get("end", len(text))
                
                if not name or start < 0 or end <= start:
                    continue
                
                # Generate unique ID
                entity_id = str(uuid.uuid4())
                ref_short = entity_id[-4:] if len(entity_id) >= 4 else entity_id
                entity_name_to_id[name] = entity_id
                
                entity = Entity(
                    id=entity_id,
                    name=name,
                    label=label,
                    ref_short=ref_short,
                    ref=entity_id,
                    start=start,
                    end=end,
                    sentence_id="s0",
                    entity_type=label
                )
                entities.append(entity)
            
            # Extract triples (convert entity names to IDs)
            triples_data = data.get("triples", [])
            triples = []
            
            for triple_data in triples_data:
                if not isinstance(triple_data, dict):
                    continue
                
                head_name = str(triple_data.get("head", "")).strip()
                tail_name = str(triple_data.get("tail", "")).strip()
                relation = str(triple_data.get("relation", "")).strip()
                
                if not head_name or not tail_name or not relation:
                    continue
                
                # Look up entity IDs by name
                head_id = entity_name_to_id.get(head_name)
                tail_id = entity_name_to_id.get(tail_name)
                
                if not head_id or not tail_id:
                    # Entity not found - skip this triple
                    continue
                
                triples.append({
                    "head": head_id,
                    "relation": relation,
                    "tail": tail_id
                })
            
            return entities, triples
            
        except Exception as e:
            print(f"⚠️  Error extracting entities and triples: {e}")
            return [], []
    
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        import json
        import ast
        
        # Remove markdown fences
        text = JsonHelper._unfence(text).strip()
        if not text:
            return {}
        
        # Try strict JSON first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try Python literal
            try:
                return ast.literal_eval(text)
            except Exception:
                return {}

