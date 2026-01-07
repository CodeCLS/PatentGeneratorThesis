"""
Analyzer node - analyzes the graph and generates validation questions.
"""

from typing import TYPE_CHECKING, List, Union
from tools.helper.json_helper import JsonHelper
from tools.graph.langgraph.state import GraphValidatorState
from tools.graph.langgraph.helpers import get_triple_head_id, get_triple_tail_id, get_triple_head_name, get_triple_tail_name
from tools.graph.langgraph.question import Question

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def generate_questions(validator: "GraphValidatorLangGraph") -> List[Question]:
    """Analyze the graph and generate validation questions."""
    if not validator.triples:
        return []
    
    # Find duplicate relations and build triple info in one pass
    seen = {}
    duplicate_triples = []
    all_triples_info = []
    
    for i, triple in enumerate(validator.triples):
        head_id = get_triple_head_id(triple)
        tail_id = get_triple_tail_id(triple)
        head_name = validator.id_to_name.get(head_id, get_triple_head_name(triple))
        tail_name = validator.id_to_name.get(tail_id, get_triple_tail_name(triple))
        
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
    
    # Build prompt with duplicate triples
    duplicates_text = "\n".join([
        f"  Triple {t['index']}: {t['head']} --[{t['relation']}]--> {t['tail']}"
        for t in duplicate_triples[:5]
    ])
    
    prompt = (
        f"Graph: {len(validator.triples)} triples, {len(validator.id_to_name)} entities\n\n"
        f"Found {len(duplicate_triples)} duplicate relations:\n{duplicates_text}\n\n"
        "Generate 3-4 SPECIFIC questions. Each must:\n"
        "- Include triple index (e.g., 'triple 5')\n"
        "- Use actual entity names\n"
        "- Focus on ONE duplicate triple\n"
        "- Be conversational\n\n"
        "Return ONLY JSON array:\n"
        '[{"id": "q1", "text": "Triple 5: Entity A --[connects]--> Entity B. Should this be removed?", "category": "mistake", "priority": 8}]\n'
    )
    
    all_questions = []
    for batch_num in range(3):
        response = validator.api_repo.chat(prompt)
        questions = JsonHelper.parse_json(str(response))
        
        if not questions:
            break
        
        if not isinstance(questions, list):
            questions = [questions]
        
        for q in questions:
            if isinstance(q, dict):
                q["id"] = f"q{len(all_questions) + 1}"
                all_questions.append(Question.from_dict(q))
            else:
                all_questions.append(q)
    
    return all_questions[:12]


def analyzer_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState") -> "GraphValidatorState":
    """Analysis agent - generates new questions and analyzes the graph."""
    questions = generate_questions(validator)
    
    messages = state.get("messages", [])
    next_question = None
    question_id = None
    
    if questions:
        first_q = questions[0]
        if isinstance(first_q, Question):
            next_question = first_q.text
            question_id = first_q.id
        else:
            question = Question.from_dict(first_q.to_dict() if hasattr(first_q, 'to_dict') else {"id": "", "text": str(first_q)})
            next_question = question.text
            question_id = question.id
    else:
        messages.append({"role": "bot", "content": "I've analyzed your graph and found no issues. Validation is complete!"})
    
    return {
        **state,
        "messages": messages,
        "questions": questions,
        "current_question_text": next_question,
        "current_question_id": question_id,
        "validation_complete": not questions,
        "next_agent": None if not questions else "communicator",
    }
