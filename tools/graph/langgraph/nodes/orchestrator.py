"""
Orchestrator node - decides which agents to run and in what order.
"""

from typing import TYPE_CHECKING, Union
from tools.helper.json_helper import JsonHelper
from tools.graph.langgraph.state import GraphValidatorState, get_last_message
from tools.graph.langgraph.message import Message, MessageRole
from tools.graph.langgraph.prompts import get_registry
from tools.graph.constants_graph import (
    AGENT_ORCHESTRATOR,
    AGENT_COMMUNICATOR,
    AGENT_ANALYZER,
    AGENT_RETRIEVER,
    AGENT_MODIFIER,
    AGENT_VISUALIZER,
    STATE_MESSAGES,
    STATE_AGENT_QUEUE,
    STATE_QUESTIONS,
    STATE_VALIDATION_COMPLETE,
    STATE_MODE,
    STATE_PLAN,
    STATE_NEEDS_RETRIEVAL,
    STATE_WRITE,
    STATE_RESPONSE_STYLE,
    MESSAGE_ROLE_USER,
    KEY_ROLE,
    KEY_CONTENT,
    MODE_INITIAL,
    MODE_QA,
)

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph




def orchestrator_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState") -> "GraphValidatorState":
    """Orchestrator agent - decides the flow of agents based on user intent."""
    messages = state.get(STATE_MESSAGES, [])
        
    user_message = get_last_message(messages, MessageRole.USER)
    if not user_message:
        existing_questions = state.get(STATE_QUESTIONS, [])
        validation_complete = state.get(STATE_VALIDATION_COMPLETE, False)
        if not existing_questions and validation_complete:
            return {
                **state,
                STATE_AGENT_QUEUE: [AGENT_COMMUNICATOR],
                STATE_MODE: MODE_INITIAL,
                STATE_PLAN: "No questions found, communicate completion to user",
                STATE_NEEDS_RETRIEVAL: False,
                STATE_WRITE: False,
            }
        
        # Otherwise, analyze graph first
        return {
            **state,
            STATE_AGENT_QUEUE: [AGENT_ANALYZER, AGENT_COMMUNICATOR],
            STATE_MODE: MODE_INITIAL,
            STATE_PLAN: "Initial analysis: analyze graph",
            STATE_NEEDS_RETRIEVAL: False,
            STATE_WRITE: False,
        }
    
    registry = get_registry()
    prompt = registry.build_prompt(
        AGENT_ORCHESTRATOR,
        user_message=user_message
    )
    
    response = validator.api_repo.chat(prompt)
    decision = JsonHelper.parse_json(str(response))
    if not decision:
        # Default to Q&A flow
        decision = {
            STATE_MODE: MODE_QA,
            STATE_NEEDS_RETRIEVAL: False,
            STATE_WRITE: False,
            STATE_RESPONSE_STYLE: "conversational",
            STATE_AGENT_QUEUE: [AGENT_COMMUNICATOR],
            STATE_PLAN: "Please tell the user that something went wrong and ask them to try again"
        }
    
    agent_queue = decision.get(STATE_AGENT_QUEUE, [AGENT_COMMUNICATOR])
    # Ensure communicator is always last
    if AGENT_COMMUNICATOR not in agent_queue:
        agent_queue.append(AGENT_COMMUNICATOR)
    elif agent_queue[-1] != AGENT_COMMUNICATOR:
        agent_queue.remove(AGENT_COMMUNICATOR)
        agent_queue.append(AGENT_COMMUNICATOR)
    
    return {
        **state,
        STATE_AGENT_QUEUE: agent_queue,
        STATE_MODE: decision.get(STATE_MODE, MODE_QA),
        STATE_PLAN: decision.get(STATE_PLAN, "Process user request"),
        STATE_NEEDS_RETRIEVAL: decision.get(STATE_NEEDS_RETRIEVAL, False),
        STATE_WRITE: decision.get(STATE_WRITE, False),
        STATE_RESPONSE_STYLE: decision.get(STATE_RESPONSE_STYLE, "conversational"),
    }

