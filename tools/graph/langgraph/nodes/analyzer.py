"""
Analyzer node - analyzes the graph and generates validation questions.
"""

from typing import TYPE_CHECKING, List
from tools.graph.langgraph.state import GraphValidatorState
from tools.graph.langgraph.question import Question
from tools.graph.langgraph.nodes.question_creators import (
    DuplicateTripleQuestionCreator,
    EntityCompletenessQuestionCreator,
    EntityMergingQuestionCreator,
    TripleMergingQuestionCreator,
)

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def generate_questions(validator: "GraphValidatorLangGraph") -> List[Question]:
    """Analyze the graph and generate validation questions using multiple question creators."""
    if not validator.triples:
        return []
    
    all_questions = []
    
    # Use all question creators
    creators = [
        DuplicateTripleQuestionCreator(validator),
        EntityCompletenessQuestionCreator(validator),
        EntityMergingQuestionCreator(validator),
        TripleMergingQuestionCreator(validator),
    ]
    
    for creator in creators:
        try:
            questions = creator.generate_questions()
            all_questions.extend(questions)
        except Exception as e:
            # Continue with other creators if one fails
            print(f"Warning: {creator.__class__.__name__} failed: {e}")
            continue
    
    return all_questions


def analyzer_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState") -> "GraphValidatorState":
    """Analysis agent - generates new questions and analyzes the graph."""
    # Check if questions already exist (avoid regenerating)
    existing_questions = state.get("questions", [])
    if existing_questions:
        questions = existing_questions
    else:
        questions = generate_questions(validator)
    
    # Convert Question objects to dictionaries for state
    questions_dict = []
    for q in questions:
        if isinstance(q, Question):
            questions_dict.append(q.to_dict())
        elif isinstance(q, dict):
            questions_dict.append(q)
        else:
            questions_dict.append(q.to_dict() if hasattr(q, 'to_dict') else {"id": "", "text": str(q)})
    
    messages = state.get("messages", [])
    next_question = None
    question_id = None
    
    if questions_dict:
        first_q = questions_dict[0]
        if isinstance(first_q, dict):
            next_question = first_q.get("text")
            question_id = first_q.get("id")
        else:
            question = Question.from_dict(first_q.to_dict() if hasattr(first_q, 'to_dict') else {"id": "", "text": str(first_q)})
            next_question = question.text
            question_id = question.id
    else:
        messages.append({"role": "bot", "content": "I've analyzed your graph and found no issues. Validation is complete!"})
    
    return {
        **state,
        "messages": messages,
        "questions": questions_dict,  # Use converted dicts
        "current_question_text": next_question,
        "current_question_id": question_id,
        "validation_complete": not questions_dict,
        "next_agent": None if not questions_dict else "communicator",
    }
