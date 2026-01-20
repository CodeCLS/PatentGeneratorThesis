"""
Communicator node - main communication agent that handles user messages and coordinates other agents.
"""

from typing import TYPE_CHECKING, Optional, Union
import json

try:
    from langgraph.graph import END
except ImportError:
    END = "__end__"

from tools.helper.json_helper import JsonHelper
from tools.graph.langgraph.helpers import (
    extract_text_from_response,
    get_triple_head_name,
    get_triple_tail_name,
    process_retrieved_info_for_widget,
    extract_retrieved_info,
    format_conversation_history,
)
from tools.graph.langgraph.state import GraphValidatorState, consume_agent, get_last_message, get_message_content
from tools.graph.langgraph.question import Question
from tools.graph.langgraph.message import Message, MessageRole
from tools.graph.langgraph.message.widgets import Widget, EdgesWidget
from tools.graph.langgraph.prompts import get_registry
from tools.graph.constants_graph import (
    AGENT_COMMUNICATOR,
    AGENT_ANALYZER,
    AGENT_MODIFIER,
    AGENT_VISUALIZER,
    AGENT_RETRIEVER,
    ACTION_DELETE_TRIPLES,
    ACTION_UPDATE_ENTITY_LABEL,
    ACTION_MERGE_ENTITIES,
    STATE_MESSAGES,
    STATE_CURRENT_QUESTION_TEXT,
    STATE_QUESTIONS,
    STATE_CHANGES_SUMMARY,
    STATE_CURRENT_QUESTION_ID,
    STATE_VALIDATION_COMPLETE,
    STATE_NEXT_AGENT,
    STATE_AGENT_QUEUE,
    STATE_HIDDEN_ACTIONS,
    STATE_SHOW_WIDGET,
    STATE_WIDGET_TYPE,
    STATE_WIDGET_DATA,
    STATE_TEXT,
    STATE_NEXT_QUESTION,
    MESSAGE_ROLE_USER,
    MESSAGE_ROLE_SYSTEM,
    KEY_CONTENT,
    KEY_ROLE,
    KEY_RELATED_TRIPLES,
    KEY_SEARCH_RESULTS,
    KEY_TRIPLES,
    KEY_NAME,
    KEY_ERROR,
    KEY_INDEX,
    KEY_HEAD,
    KEY_TAIL,
    KEY_RELATION,
    KEY_RETRIEVED_INFO_MARKER,
    KEY_REASON,
    WIDGET_TYPE_EDGES,
    WIDGET_TYPE_VISUALIZATION,
    STATE_INTERNAL_RETRIEVED_TRIPLES,
    STATE_INTERNAL_RETRIEVED_INFO_PROCESSED,
    STATE_INTERNAL_NEEDS_WIDGET,
    DEFAULT_ENTITY,
    STATE_PLAN,
    STATE_CURRENT_QUESTION,
    STATE_CHAT_CONTEXT_INFORMATION
)

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def build_communicator_prompt(validator: "GraphValidatorLangGraph", state: "GraphValidatorState", user_message: Optional[str], system_message: Optional[str], current_plan: Optional[str]) -> str:
    """Build prompt for the communicator agent."""
    messages = state.get(STATE_MESSAGES, [])
    questions = state.get(STATE_QUESTIONS, [])
    current_question = state.get(STATE_CURRENT_QUESTION)
    remaining_count = len(questions)
    triples = validator.triples
    id_to_name = validator.id_to_name
    
    # Get recent context from structured information
    context_info_list = state.get(STATE_CHAT_CONTEXT_INFORMATION, [])
    context_text = ""
    if context_info_list:
        last_context = context_info_list[-1]
        entities_str = ', '.join(last_context.entities_in_focus) if last_context.entities_in_focus else "None"
        triples_str = ', '.join(map(str, last_context.relevant_triples)) if last_context.relevant_triples else "None"
        context_text = (
            f"RECENT CONTEXT:\n"
            f"- Intent: {last_context.intent}\n"
            f"- Entities in focus: {entities_str}\n"
            f"- Relevant triples: {triples_str}\n"
            f"- Additional context: {last_context.additional_context}\n"
        )
    
    # Build conversation history
    conv_text = format_conversation_history(messages, limit=15, include_system=True)
    
    # Add changes that were actually made
    changes_summary = state.get(STATE_CHANGES_SUMMARY, [])
    changes_text = ""
    if changes_summary:
        changes_text = "\n".join([f"- {c}" for c in changes_summary[-5:]])
    
    # Build context-specific additions
    context_additions = ""
    if context_text:
        context_additions += f"{context_text}\n\n"
        
    if changes_text:
        context_additions += (
            f"CHANGES MADE TO GRAPH:\n{changes_text}\n\n"
            "IMPORTANT: These are the ACTUAL changes that were applied to the graph.\n\n"
        )
    
    if remaining_count > 0:
        context_additions += f"REMAINING QUESTIONS: {remaining_count} questions left to review.\n\n"
    
    if current_question:
        context_additions += (
            f"CURRENT QUESTION: {current_question.text}\n\n"
            "The user is responding to this question. Engage naturally.\n\n"
        )
    if system_message:
        context_additions += f"SYSTEM MESSAGE/System Retrieved Information: {system_message}\n\n"
    
    if user_message:
        context_additions += f"USER JUST SAID: {user_message}\n\n"
    else:
        context_additions += "This is the start of the conversation so perhaps ask the question in detail if you deem it necessary.\n\n"
    
    # Get base prompt from registry
    registry = get_registry()
    base_prompt = registry.build_prompt(
        AGENT_COMMUNICATOR,
        current_question=current_question.text if current_question else "None",
        remaining_count=remaining_count,
        triples_count=len(triples),
        entities_count=len(id_to_name)
    )
    
    # Add dynamic context
    full_prompt = (
        f"CURRENT PLAN, FOLLOW this Plan strictly: {current_plan}\n"
        f"GRAPH: {len(triples)} triples, {len(id_to_name)} entities\n"
        f"HISTORY:\n{conv_text}\n\n"
        f"{context_additions}"
        f"{base_prompt}\n\n"
        "Return ONLY JSON:\n"
        '{"text": "response", "validation_complete": false, "question_resolved": false, "next_agent": null}\n'
        'If the user asks a question that requires immediate retrieval or analysis, set "next_agent" to the agent name (e.g., "retriever", "analyzer"). Otherwise, leave it null.\n'
    )
    
    return full_prompt


def communicator_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState") -> "GraphValidatorState":
    # Consume self from queue
    state = consume_agent(state, AGENT_COMMUNICATOR)
    
    messages = state.get(STATE_MESSAGES, [])
    questions = state.get(STATE_QUESTIONS, [])
    current_question = state.get(STATE_CURRENT_QUESTION)
    validation_complete = state.get(STATE_VALIDATION_COMPLETE, False)
    current_plan = state.get(STATE_PLAN)
    
    user_msg_obj = get_last_message(messages, MessageRole.USER)
    system_msg_obj = get_last_message(messages, MessageRole.SYSTEM)
    
    user_message = get_message_content(user_msg_obj) if user_msg_obj else None
    system_message = get_message_content(system_msg_obj) if system_msg_obj else None
    
    # Main LLM Logic
    prompt = build_communicator_prompt(validator, state, user_message, system_message, str(current_plan))
    response = validator.api_repo.chat(prompt)
    response_text = extract_text_from_response(response)
    response_data = JsonHelper.parse_json(response_text) or {"text": response_text}
    
    bot_message = response_data.get("text", response_text)
    question_resolved = response_data.get("question_resolved", False)
    next_agent = response_data.get("next_agent")

    updated_questions = questions
    new_current_question = current_question
    
    if question_resolved and current_question:
        updated_questions = [q for q in questions if q.id != current_question.id]
        new_current_question = updated_questions[0] if updated_questions else None
    
    if not updated_questions:
        validation_complete = True
    
    # Handle communicator-driven queue updates
    agent_queue = state.get(STATE_AGENT_QUEUE, [])
    if next_agent and next_agent not in agent_queue:
        agent_queue = agent_queue + [next_agent]
    
    return {
        **state,
        STATE_MESSAGES: messages + [Message(role=MessageRole.BOT, content=bot_message)],
        STATE_QUESTIONS: updated_questions,
        STATE_CURRENT_QUESTION: new_current_question,
        STATE_VALIDATION_COMPLETE: validation_complete,
        STATE_AGENT_QUEUE: agent_queue,
    }
