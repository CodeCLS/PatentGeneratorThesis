"""
LLM-based entity extractor for extracting entities from sentences using LLM API calls.
"""
from __future__ import annotations

from typing import List, Dict, Any
from tools.api.llm_api_repo import LLmApi_Repo
from tools.helper.json_helper import JsonHelper
from tools.sentence.entity import Entity
import uuid
import re


class LLMEntityExtractor:
    """
    Extracts entities from text (sentences or entire documents) using LLM API calls.
    No spaCy or other NLP libraries - pure LLM extraction.
    """
    
    def __init__(self):
        self.api_repo = LLmApi_Repo()
        self.task = (
            "You are an expert entity extraction system for patent text.\n\n"
            "Task: Extract all named entities from the given text (which may be a sentence or entire document).\n\n"
            "INCLUDE:\n"
            "- Technical components (devices, systems, parts)\n"
            "- Materials, chemicals, substances\n"
            "- Processes, methods, functions\n"
            "- Parameters, measurements, conditions\n"
            "- Any technical terms that represent concrete concepts\n\n"
            "EXCLUDE:\n"
            "- Common words (the, a, an, is, are, etc.)\n"
            "- Generic verbs (has, does, makes, etc.)\n"
            "- Prepositions and conjunctions\n"
            "- Very short fragments (< 2 characters)\n\n"
            "For each entity, provide:\n"
            "- name: The exact text of the entity\n"
            "- label: Entity type (INVENTION, COMPONENT, MATERIAL, FUNCTION, PROCESS_STEP, METHOD, PARAMETER, MEASUREMENT, CONDITION, HARDWARE, SOFTWARE, UNCLASSIFIED_ENTITY, etc.)\n"
            "- start: Character start position in the text (0-indexed)\n"
            "- end: Character end position in the text (exclusive)\n\n"
            "Return ONLY a valid JSON array of objects with keys: name, label, start, end.\n"
            "No markdown fences, no commentary, just the JSON array.\n"
            'Example: [{"name": "display device", "label": "INVENTION", "start": 4, "end": 18}, {"name": "pseudo space", "label": "CONDITION", "start": 35, "end": 47}]'
        )
    
    def extract(self, text: str) -> List[Entity]:
        """
        Extract entities from text (sentence or entire document) using LLM.
        
        Args:
            text: The text to extract entities from (can be a sentence or entire document)
            
        Returns:
            List of Entity objects
        """
        prompt = (
            f"{self.task}\n\n"
            f"Text: {text}\n\n"
            "Return ONLY a valid JSON array of entity objects.\n"
            "Each object must have: name, label, start, end.\n"
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
            
            # Parse JSON using similar approach to JsonHelper.parse_triple_list
            entities_data = self._parse_entity_json(response_text)
            
            # Convert to Entity objects
            entities = []
            for i, ent_data in enumerate(entities_data):
                if not isinstance(ent_data, dict):
                    continue
                
                name = ent_data.get("name", "").strip()
                label = ent_data.get("label", "UNCLASSIFIED_ENTITY").strip().upper()
                start = ent_data.get("start", 0)
                end = ent_data.get("end", len(text))
                
                if not name or start < 0 or end <= start:
                    continue
                
                # Generate unique ID
                entity_id = str(uuid.uuid4())
                ref_short = entity_id[-4:] if len(entity_id) >= 4 else entity_id
                
                entity = Entity(
                    id=entity_id,
                    name=name,
                    label=label,
                    ref_short=ref_short,
                    ref=entity_id,
                    start=start,
                    end=end,
                    sentence_id=f"s0",  # Will be updated by caller if needed
                    entity_type=label
                )
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            print(f"⚠️  Error extracting entities: {e}")
            return []
    
    def _parse_entity_json(self, text: str) -> List[Dict[str, Any]]:
        """Parse entity JSON from LLM response using similar approach to JsonHelper."""
        import json
        import ast
        
        # Remove markdown fences if present (similar to JsonHelper._unfence)
        text = JsonHelper._unfence(text).strip()
        if not text:
            return []
        
        # Try strict JSON first
        try:
            val = json.loads(text)
        except json.JSONDecodeError:
            # Try Python literal
            try:
                val = ast.literal_eval(text)
            except Exception:
                return []
        
        if not isinstance(val, list):
            return []
        
        # Normalize entities
        entities = []
        for item in val:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            label = str(item.get("label", "UNCLASSIFIED_ENTITY")).strip()
            start = item.get("start", 0)
            end = item.get("end", 0)
            
            if name and isinstance(start, (int, float)) and isinstance(end, (int, float)):
                entities.append({
                    "name": name,
                    "label": label,
                    "start": int(start),
                    "end": int(end)
                })
        
        return entities

