"""
Question creator for checking if triples can be merged (LLM-based).
"""

from typing import List
from tools.graph.langgraph.question import Question
from tools.graph.langgraph.nodes.question_creators.base import BaseQuestionCreator
from tools.graph.langgraph.helpers import get_triple_head_id, get_triple_head_name, get_triple_tail_id, get_triple_tail_name
from tools.helper.json_helper import JsonHelper


class TripleMergingQuestionCreator(BaseQuestionCreator):
    """Uses LLM to check if triples can be merged or simplified."""
    
    def generate_questions(self) -> List[Question]:
        """Check if triples can be merged using LLM."""
        if not self.triples:
            return []
        
        # Build triple summary
        triples_info = []
        for i, triple in enumerate(self.triples[:30]):  # Limit to avoid too long prompts
            head_id = get_triple_head_id(triple)
            tail_id = get_triple_tail_id(triple)
            head_name = self.id_to_name.get(head_id, get_triple_head_name(triple))
            tail_name = self.id_to_name.get(tail_id, get_triple_tail_name(triple))
            
            triples_info.append({
                "index": i,
                "head": head_name,
                "relation": triple.relation,
                "tail": tail_name
            })
        
        triples_text = "\n".join([
            f"  Triple {t['index']}: {t['head']} --[{t['relation']}]--> {t['tail']}"
            for t in triples_info
        ])
        
        prompt = (
            f"Graph: {len(self.triples)} triples, {len(self.id_to_name)} entities\n\n"
            f"Triples in the graph:\n{triples_text}\n\n"
            "Analyze if any triples can be merged or simplified.\n"
            "For example, if multiple triples express the same relationship, they might be redundant.\n"
            "Generate 2-4 SPECIFIC questions. Each must:\n"
            "- Identify specific triples that might be merged\n"
            "- Include triple indices (e.g., 'triple 5')\n"
            "- Use actual entity names and relations\n"
            "- Be conversational\n\n"
            "- And contain all important inforation about this triple \n\n"
            "Return ONLY JSON array:\n"
            '[{"id": "q1", "text": "Triples 5 and 7 both connect Entity A to Entity B. Should these be merged into one?",triples_contained: ["Triple Id A", "Triple Id B"], "entities_contained": ["Entity Id A", "Entity Id B"], "category": "merging", "priority": 6}]\n'
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
                q["id"] = f"merge_triple_{len(all_questions) + 1}"
                all_questions.append(Question.from_dict(q))
        
        return all_questions[:5]

