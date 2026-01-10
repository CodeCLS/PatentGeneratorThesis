"""
Modifier node - applies graph modifications based on hidden actions.
"""

from typing import TYPE_CHECKING
from tools.graph.Triple import Triple
from tools.sentence.entity import Entity
from tools.graph.langgraph.state import GraphValidatorState,get_last_message
from tools.graph.langgraph.helpers import get_triple_head_id, get_triple_tail_id
from tools.graph.langgraph.message import MessageRole
from tools.graph.langgraph.prompts import get_registry
from tools.helper.json_helper import JsonHelper
from tools.graph.langgraph.helpers.modifier_actions import ModifierActions
from tools.graph.langgraph.helpers import ModifierActions
from tools.graph.langgraph.nodes.chat_changes_information import ChatChangesInformation
from tools.graph.constants_graph import (
    AGENT_COMMUNICATOR,
    ACTION_ADD_TRIPLES,
    ACTION_DELETE_TRIPLES,
    ACTION_MERGE_ENTITIES,
    ACTION_RENAME_ENTITY,
    ACTION_UPDATE_ENTITY_LABEL,
    ACTION_MODIFY_TRIPLE,
    STATE_HIDDEN_ACTIONS,
    STATE_CHANGES_SUMMARY,
    STATE_STATS,
    STATE_NEXT_AGENT,
    STATE_AGENT_QUEUE,
    AGENT_MODIFIER,
    KEY_TYPE,
    KEY_PARAMETERS,
    KEY_TRIPLES,
    KEY_HEAD,
    KEY_TAIL,
    KEY_RELATION,
    KEY_NAME,
    STATE_INTERNAL_NODE_TYPE,
    DEFAULT_UNKNOWN,
    STATE_MESSAGES,
    STATE_CURRENT_QUESTION,
    STATE_CHAT_CHANGES_INFORMATION
)

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def modifier_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState") -> "GraphValidatorState":
    """Modification agent - applies graph modifications based on hidden actions."""
    messages = state.get(STATE_MESSAGES, [])
    question = state.get(STATE_CURRENT_QUESTION)
    user_message = get_last_message(messages, MessageRole.USER)
    last_bot_message = get_last_message(messages, MessageRole.BOT)
    
    registry = get_registry()
    prompt = registry.build_prompt(
        AGENT_MODIFIER,
        user_message=user_message or "",
        last_bot_message=last_bot_message or "",
        current_question=str(question),
        id_to_name=id_to_name
    )
    
    response = validator.api_repo.chat(prompt)
    action_data = JsonHelper.parse_json(str(response))
    changes = state.get(STATE_CHAT_CHANGES_INFORMATION, [])
    for i in action_data:
        changes.append(ChatChangesInformation.from_dict(i))
    changes_summary = []
    graph = validator.graph
    triples = validator.triples.copy()
    id_to_name = validator.id_to_name.copy()



    
    for change in changes:
        action_type = change.type
        triples = change.triples
        if not isinstance(params, dict):
            params = {}
        ModifierActions.apply_change(action_type, triples)
    
    validator.triples = triples
    validator.id_to_name = id_to_name
    if graph:
        validator.graph = graph
    
    validator.tools.triples = triples
    validator.tools.id_to_name = id_to_name
    if graph:
        validator.tools.graph = graph
    
    stats = validator.tools.calculate_stats()
    
    # Consume current agent from queue if it's at the front
    agent_queue = state.get(STATE_AGENT_QUEUE, [])
    if agent_queue and agent_queue[0] == AGENT_MODIFIER:
        agent_queue = agent_queue[1:]
    
    return {
        **state,
        STATE_HIDDEN_ACTIONS: [],
        STATE_CHANGES_SUMMARY: state.get(STATE_CHANGES_SUMMARY, []) + changes_summary,
        STATE_STATS: stats,
        STATE_AGENT_QUEUE: agent_queue,
    }
