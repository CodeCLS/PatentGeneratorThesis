"""
Orchestrator node - decides which agents to run and in what order.
"""

from typing import TYPE_CHECKING, Union
from tools.helper.json_helper import JsonHelper
from tools.graph.langgraph.state import GraphValidatorState
from tools.graph.langgraph.message import Message, MessageRole
from tools.graph.langgraph.prompts import get_registry
from tools.graph.constants_graph import (
    AGENT_COMMUNICATOR,
    AGENT_ANALYZER,
    AGENT_RETRIEVER,
    AGENT_MODIFIER,
    AGENT_VISUALIZER,
    STATE_MESSAGES,
    STATE_AGENT_QUEUE,
    STATE_MODE,
    STATE_PLAN,
    STATE_NEEDS_RETRIEVAL,
    STATE_WRITE,
    STATE_RESPONSE_STYLE,
)

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def _get_message_content(msg: Union[Message, dict]) -> str:
    """Extract content from a Message object or dict."""
    if isinstance(msg, Message):
        return msg.content
    elif isinstance(msg, dict):
        return msg.get("content", "")
    return ""


def orchestrator_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState") -> "GraphValidatorState":
    """Orchestrator agent - decides the flow of agents based on user intent."""
    messages = state.get(STATE_MESSAGES, [])
    
    # Get the last user message
    user_message = None
    for msg in reversed(messages):
        if isinstance(msg, Message):
            if msg.role == MessageRole.USER or (isinstance(msg.role, str) and msg.role == "user"):
                user_message = msg.content
                break
        elif isinstance(msg, dict) and msg.get("role") == "user":
            user_message = msg.get("content", "")
            break
    
    # If no user message, this is initial startup - analyze graph first
    if not user_message:
        return {
            **state,
            STATE_AGENT_QUEUE: [AGENT_ANALYZER, AGENT_COMMUNICATOR],
            STATE_MODE: "INITIAL",
            STATE_PLAN: "Initial analysis: analyze graph, then communicate with user",
            STATE_NEEDS_RETRIEVAL: False,
            STATE_WRITE: False,
        }
    
    # Analyze user intent to determine flow
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
            "mode": "Q&A",
            "needs_retrieval": True,
            "write": False,
            "response_style": "conversational",
            "agent_queue": [AGENT_RETRIEVER, AGENT_ANALYZER, AGENT_COMMUNICATOR],
            "plan": "Standard Q&A flow: retrieve info, analyze, communicate"
        }
    
    agent_queue = decision.get("agent_queue", [AGENT_COMMUNICATOR])
    # Ensure communicator is always last
    if AGENT_COMMUNICATOR not in agent_queue:
        agent_queue.append(AGENT_COMMUNICATOR)
    elif agent_queue[-1] != AGENT_COMMUNICATOR:
        agent_queue.remove(AGENT_COMMUNICATOR)
        agent_queue.append(AGENT_COMMUNICATOR)
    
    return {
        **state,
        STATE_AGENT_QUEUE: agent_queue,
        STATE_MODE: decision.get("mode", "Q&A"),
        STATE_PLAN: decision.get("plan", "Process user request"),
        STATE_NEEDS_RETRIEVAL: decision.get("needs_retrieval", False),
        STATE_WRITE: decision.get("write", False),
        STATE_RESPONSE_STYLE: decision.get("response_style", "conversational"),
    }

