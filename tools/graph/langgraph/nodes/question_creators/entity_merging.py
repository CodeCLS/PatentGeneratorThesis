"""
Question creator for checking if entities can be merged (LLM-based).
"""

from typing import List
from tools.graph.langgraph.question import Question
from tools.graph.langgraph.nodes.question_creators.base import BaseQuestionCreator
from tools.graph.langgraph.helpers import get_triple_head_id, get_triple_head_name, get_triple_tail_id, get_triple_tail_name
from tools.helper.json_helper import JsonHelper


class EntityMergingQuestionCreator(BaseQuestionCreator):
    """Uses LLM to check if entities can be merged."""
    
    def generate_questions(self) -> List[Question]:
        """Check if entities can be merged using LLM."""
        if not self.triples:
            return []
        
        # Collect entity information
        entities_info = []
        seen_entities = set()
        
        for triple in self.triples:
            head_id = get_triple_head_id(triple)
            tail_id = get_triple_tail_id(triple)
            head_name = self.id_to_name.get(head_id, get_triple_head_name(triple))
            tail_name = self.id_to_name.get(tail_id, get_triple_tail_name(triple))
            
            # Collect head entity info
            if head_id not in seen_entities:
                entity_info = {
                    "id": head_id,
                    "name": head_name,
                    "label": triple.head.label if hasattr(triple.head, 'label') else "",
                    "relations": []
                }
                entities_info.append(entity_info)
                seen_entities.add(head_id)
            
            # Collect tail entity info
            if tail_id not in seen_entities:
                entity_info = {
                    "id": tail_id,
                    "name": tail_name,
                    "label": triple.tail.label if hasattr(triple.tail, 'label') else "",
                    "relations": []
                }
                entities_info.append(entity_info)
                seen_entities.add(tail_id)
        
        if len(entities_info) < 2:
            return []
        
        # Build entity summary for LLM
        entities_text = "\n".join([
            f"  - {e['name']} (ID: {e['id'][:8]}..., Label: {e['label']})"
            for e in entities_info[:20]  # Limit to avoid too long prompts
        ])
        
        prompt = (
            f"Graph: {len(self.triples)} triples, {len(entities_info)} entities\n\n"
            f"Entities in the graph:\n{entities_text}\n\n"
            "Analyze if any entities should be merged (they represent the same concept).\n"
            "Generate 2-4 SPECIFIC questions. Each must:\n"
            "- Identify specific entity pairs that might be merged\n"
            "- Use actual entity names\n"
            "- And contain all important inforation about this triple \n\n"
            "- Ask if they represent the same concept\n"
            "- Be conversational\n\n"
            "Return ONLY JSON array:\n"
            '[{"id": "q1", "text": "Do \'Entity A\' and \'Entity B\' represent the same concept? Should they be merged?",triples_contained: ["Triple Id A", "Triple Id B"],entities_contained: ["Entity Id A", "Entity Id B"], "category": "merging", "priority": 6}]\n'
        )
        
        response = self.api_repo.chat(prompt)
        questions = JsonHelper.parse_json(str(response))
        
        if not questions:
            return []
        
        if not isinstance(questions, list):
            questions = [questions]
        
        all_questions = []
        for q in questions:
            if isinstance(q, dict):
                q["id"] = f"merge_ent_{len(all_questions) + 1}"
                all_questions.append(Question.from_dict(q))
        
        return all_questions[:5]

