"""
Question creator for checking entity completeness (e.g., Invention entities should be connected to descriptive entities).
"""

from typing import List
from tools.graph.langgraph.question import Question
from tools.graph.langgraph.nodes.question_creators.base import BaseQuestionCreator
from tools.graph.langgraph.helpers import get_triple_head_id, get_triple_tail_id, get_triple_head_name, get_triple_tail_name
from tools.helper.json_helper import JsonHelper


class EntityCompletenessQuestionCreator(BaseQuestionCreator):
    """Checks if entities labeled as Invention (or other tags) are properly connected."""
    
    def generate_questions(self) -> List[Question]:
        """Check if Invention entities are connected to descriptive entities."""
        if not self.triples or not self.graph:
            return []
        
        # Find entities labeled as "INVENTION" or similar
        invention_entities = []
        entity_labels = {}
        
        # Collect entity information from triples
        for triple in self.triples:
            head_id = get_triple_head_id(triple)
            tail_id = get_triple_tail_id(triple)
            head_name = self.id_to_name.get(head_id, get_triple_head_name(triple))
            tail_name = self.id_to_name.get(tail_id, get_triple_tail_name(triple))
            
            # Get labels from entities
            if hasattr(triple.head, 'label'):
                entity_labels[head_id] = triple.head.label
            if hasattr(triple.tail, 'label'):
                entity_labels[tail_id] = triple.tail.label
            
            # Check if head or tail is an Invention
            head_label = entity_labels.get(head_id, "")
            tail_label = entity_labels.get(tail_id, "")
            
            if "INVENTION" in head_label.upper():
                invention_entities.append({
                    "id": head_id,
                    "name": head_name,
                    "label": head_label
                })
            if "INVENTION" in tail_label.upper() and tail_id not in [e["id"] for e in invention_entities]:
                invention_entities.append({
                    "id": tail_id,
                    "name": tail_name,
                    "label": tail_label
                })
        
        if not invention_entities:
            return []
        
        # Check connections for each Invention entity
        incomplete_inventions = []
        for inv_entity in invention_entities:
            entity_id = inv_entity["id"]
            connected_descriptive = False
            
            # Check if connected to any descriptive entity
            for triple in self.triples:
                head_id = get_triple_head_id(triple)
                tail_id = get_triple_tail_id(triple)
                
                if head_id == entity_id:
                    # Get tail label from entity_labels or directly from triple
                    tail_label = entity_labels.get(tail_id, "")
                    if not tail_label and hasattr(triple.tail, 'label'):
                        tail_label = triple.tail.label
                    if "DESCRIPTIVE" in tail_label.upper() or "DESCRIPTION" in tail_label.upper():
                        connected_descriptive = True
                        break
                elif tail_id == entity_id:
                    # Get head label from entity_labels or directly from triple
                    head_label = entity_labels.get(head_id, "")
                    if not head_label and hasattr(triple.head, 'label'):
                        head_label = triple.head.label
                    if "DESCRIPTIVE" in head_label.upper() or "DESCRIPTION" in head_label.upper():
                        connected_descriptive = True
                        break
            
            if not connected_descriptive:
                incomplete_inventions.append(inv_entity)
        
        if not incomplete_inventions:
            return []
        
        # Use LLM to generate questions
        inventions_text = "\n".join([
            f"  - {inv['name']} (label: {inv['label']})"
            for inv in incomplete_inventions[:5]
        ])
        
        prompt = (
            f"Graph: {len(self.triples)} triples, {len(self.id_to_name)} entities\n\n"
            f"Found {len(incomplete_inventions)} Invention entities that may not be properly explained:\n{inventions_text}\n\n"
            "An Invention entity should be connected to at least one descriptive entity that explains it.\n"
            "Generate 2-3 SPECIFIC questions. Each must:\n"
            "- Use actual entity names\n"
            "- Ask if the invention needs more descriptive connections\n"
            "- Be conversational\n\n"
            "- And contain all important inforation about this triple \n\n"
            "Return ONLY JSON array:\n"
            '[{"id": "q1", "text": "Is the invention \'Entity Name\' properly explained? Should it be connected to more descriptive entities?",triples_contained: ["Triple Id A", "Triple Id B"], "entities_contained": ["Entity Id A", "Entity Id B"], "category": "completeness", "priority": 7}]\n'
        )
        
        response = self.api_repo.chat(prompt)
        questions = JsonHelper.parse_json(str(response))
        
        if not questions:
            return []
        
        if not isinstance(questions, list):
            questions = [questions]
        
        all_questions = []
        for i, q in enumerate(questions):
            if isinstance(q, dict):
                q["id"] = f"comp_{len(all_questions) + 1}"
                all_questions.append(Question.from_dict(q))
        
        return all_questions[:5]

