"""
Analyzer node - analyzes the graph and generates validation questions.
"""
from langfuse_client import langfuse, LangfuseSummary

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
    with langfuse.start_as_current_observation(
        name="analyzer_node",
        input=LangfuseSummary.from_chat_agent_state(state),
        as_type="span",
    ) as span:
        existing_questions = state.get(STATE_QUESTIONS, [])

        if existing_questions:
            questions = [q for q in existing_questions if isinstance(q, Question)]
            source = "state"
        else:
            questions = generate_questions(validator)
            source = "generated"

        updated_state = consume_agent(state, AGENT_ANALYZER)

        current_q = state.get(STATE_CURRENT_QUESTION)
        if not current_q and questions:
            current_q = questions[0]

        result = {
            **updated_state,
            STATE_QUESTIONS: questions,
            STATE_CURRENT_QUESTION: current_q,
            STATE_VALIDATION_COMPLETE: not questions,
        }

        # Log only safe + useful metadata
        span.update(
            metadata={
                "question_source": source,
                "questions_count": len(questions),
                "has_current_question": bool(current_q),
                # if Question has an ID/title, log only that
                "current_question_id": getattr(current_q, "id", None) if current_q else None,
            }
        )

        return result

