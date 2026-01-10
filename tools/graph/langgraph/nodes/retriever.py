"""
Retriever node - fetches detailed information about entities and triples.
"""

import json
from typing import TYPE_CHECKING, Union,Dict
from tools.helper.json_helper import JsonHelper
from tools.graph.langgraph.state import GraphValidatorState,get_last_message
from tools.graph.langgraph.message import Message, MessageRole
from tools.graph.langgraph.prompts import get_registry
from tools.graph.constants_graph import (
    AGENT_COMMUNICATOR,
    AGENT_RETRIEVER,
    STATE_MESSAGES,
    STATE_AGENT_QUEUE,
    STATE_NEXT_AGENT,
    MESSAGE_ROLE_USER,
    MESSAGE_ROLE_BOT,
    KEY_ROLE,
    KEY_CONTENT,
    KEY_ACTION,
    KEY_PARAMETERS,
    KEY_RELATED_TRIPLES,
    KEY_SEARCH_RESULTS,
    KEY_ERROR,
    KEY_NAME,
    ACTION_GET_ENTITY_INFO,
    ACTION_GET_TRIPLE_INFO,
    ACTION_GET_RELATED_TRIPLES,
    KEY_INDEX,
    ACTION_SEARCH_ENTITIES,
    KEY_RETRIEVED_INFO_MARKER,
    DEFAULT_REASON,
    DEFAULT_N_A,
    KEY_ENTITY_ID,
    STATE_CURRENT_QUESTION_ID,
    STATE_CHAT_CONTEXT_INFORMATION,
    STATE_CURRENT_QUESTION
)

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph

def retriever_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState", id_to_name: Dict[str, str]) -> "GraphValidatorState":

    """Retrieval agent - fetches detailed information about entities and triples."""
    messages = state.get(STATE_MESSAGES, [])
    question = state.get(STATE_CURRENT_QUESTION)
    user_message = get_last_message(messages, MessageRole.USER)
    last_bot_message = get_last_message(messages, MessageRole.BOT)
    
    registry = get_registry()
    prompt = registry.build_prompt(
        AGENT_RETRIEVER,
        user_message=user_message or "",
        last_bot_message=last_bot_message or "",
        current_question=str(question),
        id_to_name=id_to_name


    )
    
    response = validator.api_repo.chat(prompt)
    action_data = JsonHelper.parse_json(str(response))
    if not action_data:
        action_data = {KEY_ACTION: ACTION_GET_ENTITY_INFO, KEY_PARAMETERS: {}, "reason": DEFAULT_REASON}
    
    action = action_data.get(KEY_ACTION, ACTION_GET_ENTITY_INFO)
    params = action_data.get(KEY_PARAMETERS, {})
    entity_id = params.get(KEY_ENTITY_ID, "")
    
    if action == ACTION_GET_ENTITY_INFO:
        entity_info = validator.tools.get_entity_info(params.get(KEY_NAME, ""), id = entity_id)
        print("Name of entity: ", params.get(KEY_NAME, ""))
        info = entity_info
    elif action == ACTION_GET_TRIPLE_INFO:
        triple_info = validator.tools.get_triple_info(params.get(KEY_INDEX, -1))
        info = triple_info
    elif action == ACTION_GET_RELATED_TRIPLES:
        related_triples = validator.tools.get_related_triples(params.get(KEY_NAME, ""), id = entity_id)
        info = related_triples
    elif action == ACTION_SEARCH_ENTITIES:
        info = {KEY_SEARCH_RESULTS: validator.tools.search_entities(params.get("query", ""))}
    else:
        info = {KEY_ERROR: f"Unknown action: {action}"}
    
    info_text = json.dumps(info, indent=2)
    retrieval_message = f"{KEY_RETRIEVED_INFO_MARKER}\n{info_text}\n\nReason: {action_data.get('reason', DEFAULT_N_A)}"
    
    # Consume current agent from queue if it's at the front
    agent_queue = state.get(STATE_AGENT_QUEUE, [])
    if agent_queue and agent_queue[0] == AGENT_RETRIEVER:
        agent_queue = agent_queue[1:]

    context_information = ChatContextInformation(info = info)
    
    return {
        STATE_CHAT_CONTEXT_INFORMATION: [ChatContextInformation(info = info)],
        **state,
        STATE_MESSAGES: messages + [Message(role=MessageRole.SYSTEM, content=retrieval_message, data=context_information )],
        STATE_AGENT_QUEUE: agent_queue,
    }
