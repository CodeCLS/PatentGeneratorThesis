"""
Analyzer node - analyzes the graph and generates validation questions.
"""

from typing import TYPE_CHECKING, List
from tools.graph.langgraph.state import GraphValidatorState,consume_agent
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
    STATE_AGENT_QUEUE,
    AGENT_ANALYZER,
)

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def generate_questions(validator: "GraphValidatorLangGraph") -> List[Question]:
    """Analyze the graph and generate validation questions using multiple question creators."""
    if not validator.triples:
        return []
    
    all_questions = []
    
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
            print(f"Warning: {creator.__class__.__name__} failed: {e}")
            continue
    
    return all_questions


def analyzer_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState") -> "GraphValidatorState":
    existing_questions = state.get(STATE_QUESTIONS, [])
    if existing_questions:
        questions = []
        for q in existing_questions:
            if isinstance(q, Question):
                questions.append(q)
    else:
        questions = generate_questions(validator)
    
    
    consume_agent(state, AGENT_ANALYZER)
    
    return {
        **state,
        STATE_QUESTIONS: questions,  
        STATE_VALIDATION_COMPLETE: not questions,
    }
