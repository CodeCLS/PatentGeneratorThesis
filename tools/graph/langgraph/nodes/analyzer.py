"""
Analyzer node - analyzes the graph and generates validation questions.
"""

from typing import TYPE_CHECKING, List
from tools.graph.langgraph.state import GraphValidatorState
from tools.graph.langgraph.question import Question
from tools.graph.langgraph.message import Message, MessageRole
from tools.graph.langgraph.nodes.question_creators import (
    DuplicateTripleQuestionCreator,
    EntityCompletenessQuestionCreator,
    EntityMergingQuestionCreator,
    TripleMergingQuestionCreator,
)
from tools.graph.constants_graph import (
    AGENT_COMMUNICATOR,
    STATE_MESSAGES,
    STATE_QUESTIONS,
    STATE_CURRENT_QUESTION_TEXT,
    STATE_CURRENT_QUESTION_ID,
    STATE_VALIDATION_COMPLETE,
    STATE_NEXT_AGENT,
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
    existing_questions = state.get(STATE_QUESTIONS, [])
    if existing_questions:
        # Ensure all questions are Question objects
        questions = []
        for q in existing_questions:
            if isinstance(q, Question):
                questions.append(q)
    else:
        questions = generate_questions(validator)
    
    messages = state.get(STATE_MESSAGES, [])
    next_question = None
    question_id = None
    
    if questions:
        first_q = questions[0]
        if isinstance(first_q, Question):
            next_question = first_q.text
            question_id = first_q.id
    else:
        messages.append(Message(role=MessageRole.BOT, content="I've analyzed your graph and found no issues. Validation is complete!"))
    
    return {
        **state,
        STATE_MESSAGES: messages,
        STATE_QUESTIONS: questions,  # Keep as Question objects
        STATE_CURRENT_QUESTION_TEXT: next_question,
        STATE_CURRENT_QUESTION_ID: question_id,
        STATE_VALIDATION_COMPLETE: not questions,
        STATE_NEXT_AGENT: None if not questions else AGENT_COMMUNICATOR,
    }
