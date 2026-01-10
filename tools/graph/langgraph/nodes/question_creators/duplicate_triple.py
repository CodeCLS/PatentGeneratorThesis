"""
Question creator for finding duplicate triples (algorithmic + LLM).
"""

from typing import List
from tools.graph.langgraph.question import Question
from tools.graph.langgraph.nodes.question_creators.base import BaseQuestionCreator
from tools.graph.langgraph.helpers import get_triple_head_id, get_triple_head_name, get_triple_tail_id, get_triple_tail_name
from tools.helper.json_helper import JsonHelper


class DuplicateTripleQuestionCreator(BaseQuestionCreator):
    """Finds duplicate triples algorithmically and generates questions via LLM."""
    
    def generate_questions(self) -> List[Question]:
        """Find duplicate triples and generate questions about them."""
        if not self.triples:
            return []
        
        # Find duplicate relations algorithmically
        seen = {}
        duplicate_triples = []
        all_triples_info = []
        
        for i, triple in enumerate(self.triples):
            head_id = get_triple_head_id(triple)
            tail_id = get_triple_tail_id(triple)
            head_name = self.id_to_name.get(head_id, get_triple_head_name(triple))
            tail_name = self.id_to_name.get(tail_id, get_triple_tail_name(triple))
            
            triple_info = {
                "index": i,
                "head": head_name,
                "relation": triple.relation,
                "tail": tail_name
            }
            all_triples_info.append(triple_info)
            
            # Check for duplicates
            key = (head_name, tail_name, triple.relation)
            if key in seen:
                duplicate_triples.append(triple_info)
            else:
                seen[key] = i
        
        if not duplicate_triples:
            return []
        
        # Use LLM to generate questions about duplicates
        duplicates_text = "\n".join([
            f"  Triple {t['index']}: {t['head']} --[{t['relation']}]--> {t['tail']}"
            for t in duplicate_triples[:5]
        ])
        
        prompt = (
            f"Graph: {len(self.triples)} triples, {len(self.id_to_name)} entities\n\n"
            f"Found {len(duplicate_triples)} duplicate relations:\n{duplicates_text}\n\n"
            "Generate 3-4 SPECIFIC questions. Each must:\n"
            "- Include triple index (e.g., 'triple 5')\n"
            "- Use actual entity names\n"
            "- Focus on ONE duplicate triple\n"
            "- Be conversational\n\n"
            "- And contain all important inforation about this triple \n\n"

            "Return ONLY JSON array:\n"
            '[{"id": "q1", "text": "Triple 5: Entity A --[connects]--> Entity B. Should this be removed?", "entities_contained": ["Entity Id A", "Entity Id B"], "category": "mistake", "priority": 8}]\n'
        )
        
        all_questions = []
        for batch_num in range(3):
            response = self.api_repo.chat(prompt)
            questions = JsonHelper.parse_json(str(response))
            
            if not questions:
                break
            
            if not isinstance(questions, list):
                questions = [questions]
            
            for q in questions:
                if isinstance(q, dict):
                    q["id"] = f"dup_{len(all_questions) + 1}"
                    all_questions.append(Question.from_dict(q))
        
        return all_questions[:12]

