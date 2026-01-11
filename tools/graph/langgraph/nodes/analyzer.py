"""
Analyzer node - analyzes the graph and generates validation questions.
"""

from typing import TYPE_CHECKING, List
from tools.graph.langgraph.state import GraphValidatorState, consume_agent
from tools.graph.langgraph.question import Question
from tools.graph.langgraph.message import Message, MessageRole
from tools.graph.langgraph.nodes.question_creators import (
    DuplicateTripleQuestionCreator,
    EntityCompletenessQuestionCreator,
    EntityMergingQuestionCreator,
    TripleMergingQuestionCreator,
)
from tools.graph.constants_graph import (
    STATE_QUESTIONS,
    STATE_VALIDATION_COMPLETE,
    AGENT_ANALYZER,
    STATE_CURRENT_QUESTION
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
    existing_questions = state.get(STATE_QUESTIONS, [])
    if existing_questions:
        questions = []
        for q in existing_questions:
            if isinstance(q, Question):
                questions.append(q)
    else:
        questions = generate_questions(validator)
    
    # Consume current agent from queue
    updated_state = consume_agent(state, AGENT_ANALYZER)
    
    # Set current question if none exists and we have questions
    current_q = state.get(STATE_CURRENT_QUESTION)
    if not current_q and questions:
        current_q = questions[0]
    
    return {
        **updated_state,
        STATE_QUESTIONS: questions,
        STATE_CURRENT_QUESTION: current_q,
        STATE_VALIDATION_COMPLETE: not questions,
    }
