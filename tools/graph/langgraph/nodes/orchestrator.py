"""
Orchestrator node - decides which agents to run and in what order.
"""

from typing import TYPE_CHECKING, Union, Dict, Any, List
from tools.helper.json_helper import JsonHelper
from tools.graph.langgraph.state import GraphValidatorState, get_last_message, get_message_content, consume_agent
from tools.graph.langgraph.helpers import format_conversation_history
from tools.graph.langgraph.message import Message, MessageRole
from tools.graph.langgraph.prompts import get_registry
from tools.graph.constants_graph import *
if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def orchestrator_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState") -> "GraphValidatorState":
    """Orchestrator agent - decides the flow of agents based on user intent."""
    messages = state.get(STATE_MESSAGES, [])



    # Reset transient turn-based state
    recent_context = state.get(STATE_CHAT_CONTEXT_INFORMATION, [])
    if len(recent_context) > 3:
        recent_context = recent_context[-3:]
    state = {
        **state,
        STATE_DISPLAY_ACTIONS: [],
        STATE_CHAT_CHANGES_INFORMATION: [],
        STATE_CHANGES_SUMMARY: [],
        STATE_CHAT_CONTEXT_INFORMATION: recent_context,
    }
        
    user_msg_obj = get_last_message(messages, MessageRole.USER)
    user_message = get_message_content(user_msg_obj)
    
    # If no user message, this is initial startup - analyze graph first
    if not user_message:
        existing_questions = state.get(STATE_QUESTIONS, [])
        validation_complete = state.get(STATE_VALIDATION_COMPLETE, False)
        
        if not existing_questions and validation_complete:
            return_state = {
                **state,
                STATE_AGENT_QUEUE: [AGENT_COMMUNICATOR],
                STATE_MODE: MODE_INITIAL,
                STATE_PLAN: "No questions found, communicate completion to user",
                STATE_NEEDS_RETRIEVAL: False,
                STATE_WRITE: False,
            }
            return consume_agent(return_state, AGENT_ORCHESTRATOR)
        
        # Otherwise, analyze graph first
        return_state = {
            **state,
            STATE_AGENT_QUEUE: [AGENT_ANALYZER, AGENT_COMMUNICATOR],
            STATE_MODE: MODE_INITIAL,
            STATE_PLAN: "Initial analysis: analyze graph",
            STATE_NEEDS_RETRIEVAL: False,
            STATE_WRITE: False,
        }
        return consume_agent(return_state, AGENT_ORCHESTRATOR)
    
    # Analyze user intent to determine flow
    registry = get_registry()
    
    # Get conversation history for better intent analysis
    conversation_history = format_conversation_history(messages, limit=10)
    
    prompt = registry.build_prompt(
        AGENT_ORCHESTRATOR,
        user_message=user_message,
        conversation_history=conversation_history
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
    
    # Ensure communicator is always last if not already there
    if AGENT_COMMUNICATOR not in agent_queue:
        agent_queue.append(AGENT_COMMUNICATOR)
    elif agent_queue[-1] != AGENT_COMMUNICATOR:
        agent_queue.remove(AGENT_COMMUNICATOR)
        agent_queue.append(AGENT_COMMUNICATOR)
    
    return_state = {
        **state,
        STATE_AGENT_QUEUE: agent_queue,
        STATE_MODE: decision.get(STATE_MODE, MODE_QA),
        STATE_PLAN: decision.get(STATE_PLAN, "Process user request"),
        STATE_NEEDS_RETRIEVAL: decision.get(STATE_NEEDS_RETRIEVAL, False),
        STATE_WRITE: decision.get(STATE_WRITE, False),
        STATE_RESPONSE_STYLE: decision.get(STATE_RESPONSE_STYLE, "conversational"),
    }
    return consume_agent(return_state, AGENT_ORCHESTRATOR)
