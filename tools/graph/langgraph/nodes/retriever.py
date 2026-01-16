"""
Retriever node - fetches detailed information about entities and triples.
"""

import json
from typing import TYPE_CHECKING, Dict, List, Any
from tools.helper.json_helper import JsonHelper
from tools.graph.langgraph.state import GraphValidatorState, get_last_message, consume_agent
from tools.graph.langgraph.message import Message, MessageRole
from tools.graph.langgraph.prompts import get_registry
from tools.graph.langgraph.nodes.chat_context_information import ChatContextInformation
from tools.graph.constants_graph import (
    AGENT_RETRIEVER,
    AGENT_VISUALIZER,
    AGENT_COMMUNICATOR,
    STATE_MESSAGES,
    STATE_AGENT_QUEUE,
    ACTION_GET_ENTITY_INFO,
    ACTION_GET_TRIPLE_INFO,
    ACTION_GET_RELATED_TRIPLES,
    ACTION_SEARCH_ENTITIES,
    KEY_RETRIEVED_INFO_MARKER,
    DEFAULT_REASON,
    DEFAULT_N_A,
    KEY_ENTITY_ID,
    STATE_CHAT_CONTEXT_INFORMATION,
    STATE_CURRENT_QUESTION,
    STATE_PLAN,
    KEY_NAME,
    KEY_ACTION,
    KEY_PARAMETERS,
    KEY_INDEX,
    KEY_ERROR,
    KEY_SEARCH_RESULTS
)

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph

def _serialize(obj: Any) -> Any:
    """Helper to convert objects to serializable format."""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj

def retriever_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState", id_to_name: Dict[str, str]) -> "GraphValidatorState":
    """Retrieval agent - fetches detailed information about entities and triples."""
    print(f"\n[Retriever] ========== RUNNING ==========")
    messages = state.get(STATE_MESSAGES, [])
    question = state.get(STATE_CURRENT_QUESTION)
    current_plan = state.get(STATE_PLAN)
    user_message = get_last_message(messages, MessageRole.USER)
    
    # 1. Prepare context for prompt
    q_ids = question.entities_referenced if question and question.entities_referenced else []
    q_names = [id_to_name[eid] for eid in q_ids if eid in id_to_name]
    
    mapping_str = "\n".join([f"  '{name}' -> {eid}" for eid, name in list(id_to_name.items())[:30]])
    q_info = f"Entities in question: {', '.join(q_names)} (IDs: {', '.join(q_ids)})" if q_names else ""

    prompt = get_registry().build_prompt(
        AGENT_RETRIEVER,
        plan=current_plan,
        user_message=user_message.content if user_message else "",
        current_question=question.text if question else "N/A",
        id_to_name=id_to_name,
        entity_list_sample=", ".join(list(id_to_name.values())[:20]),
        entity_count=len(id_to_name),
        question_entities_info=q_info,
        question_entity_ids=", ".join(q_ids) if q_ids else "None",
        id_to_name_mapping=mapping_str,
        last_bot_message=""
    )
    
    # 2. Get actions from LLM
    response = validator.api_repo.chat(prompt)
    raw_actions = JsonHelper.parse_json(str(response))
    actions = raw_actions if isinstance(raw_actions, list) else [raw_actions] if raw_actions else []
    
    if not actions and q_ids:
        actions = [{KEY_ACTION: ACTION_GET_RELATED_TRIPLES, KEY_PARAMETERS: {KEY_ENTITY_ID: q_ids[0]}, "reason": "Default retrieval"}]

    # 3. Execute actions
    combined_results = []
    all_entities = []
    all_triples = []
    has_data = False

    for action_data in actions:
        if not isinstance(action_data, dict): continue
        
        action = action_data.get(KEY_ACTION, ACTION_GET_ENTITY_INFO)
        params = action_data.get(KEY_PARAMETERS, {})
        eid = params.get(KEY_ENTITY_ID, "")
        name = params.get(KEY_NAME, "")
        
        # Resolve ID if missing
        if not eid and name:
            eid = next((k for k, v in id_to_name.items() if v.lower() == name.lower()), "")
        if not eid and q_ids: eid = q_ids[0]

        info = None
        if action == ACTION_GET_ENTITY_INFO:
            info = validator.tools.get_entity_info(name, id=eid)
            if name: all_entities.append(name)
        elif action == ACTION_GET_TRIPLE_INFO:
            idx = params.get(KEY_INDEX, -1)
            info = validator.tools.get_triple_info(idx)
            all_triples.append(idx)
        elif action == ACTION_GET_RELATED_TRIPLES:
            info = validator.tools.get_related_triples(name, id=eid)
            if name: all_entities.append(name)
        elif action == ACTION_SEARCH_ENTITIES:
            info = {KEY_SEARCH_RESULTS: validator.tools.search_entities(params.get("query", ""))}
        
        if info:
            combined_results.append(f"Action: {action}\nResult: {json.dumps(_serialize(info), indent=2)}")
            if isinstance(info, list) and len(info) > 0: has_data = True
            elif hasattr(info, 'triples') and info.triples: has_data = True
            elif isinstance(info, dict) and (info.get("triples") or info.get("index") is not None): has_data = True

    # 4. Update state
    res_text = "\n\n---\n\n".join(combined_results)
    retrieval_msg = f"{KEY_RETRIEVED_INFO_MARKER}\n{res_text}"
    
    state = consume_agent(state, AGENT_RETRIEVER)
    context_info = ChatContextInformation(
        intent="multi_retrieval",
        entities_in_focus=list(set(all_entities)),
        relevant_triples=all_triples,
        additional_context={"actions_count": len(actions)}
    )
    
    queue = state.get(STATE_AGENT_QUEUE, [])
    if has_data and AGENT_VISUALIZER not in queue:
        idx = queue.index(AGENT_COMMUNICATOR) if AGENT_COMMUNICATOR in queue else len(queue)
        queue.insert(idx, AGENT_VISUALIZER)
    
    return {
        **state,
        STATE_AGENT_QUEUE: queue,
        STATE_CHAT_CONTEXT_INFORMATION: state.get(STATE_CHAT_CONTEXT_INFORMATION, []) + [context_info],
        STATE_MESSAGES: messages + [Message(role=MessageRole.SYSTEM, content=retrieval_msg)],
    }
