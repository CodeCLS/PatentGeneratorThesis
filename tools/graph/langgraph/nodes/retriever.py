"""
Retriever node - fetches detailed information about entities and triples.
"""

import json
from typing import TYPE_CHECKING, Union, Dict, List, Any
from tools.helper.json_helper import JsonHelper
from tools.graph.langgraph.state import GraphValidatorState, get_last_message, consume_agent
from tools.graph.langgraph.message import Message, MessageRole
from tools.graph.langgraph.prompts import get_registry
from tools.graph.langgraph.nodes.chat_context_information import ChatContextInformation
from tools.graph.constants_graph import (
    AGENT_COMMUNICATOR,
    AGENT_RETRIEVER,
    AGENT_VISUALIZER,
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
    STATE_CURRENT_QUESTION,
    STATE_PLAN
)

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph

def _to_serializable(obj: Any) -> Any:
    """Helper to convert dataclasses to dicts for JSON serialization."""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    if isinstance(obj, list):
        return [_to_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    return obj

def retriever_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState", id_to_name: Dict[str, str]) -> "GraphValidatorState":
    """Retrieval agent - fetches detailed information about entities and triples."""
    messages = state.get(STATE_MESSAGES, [])
    question = state.get(STATE_CURRENT_QUESTION)
    current_plan = state.get(STATE_PLAN)
    user_message = get_last_message(messages, MessageRole.USER)
    
    registry = get_registry()
    
    # Format entity list for the prompt
    entity_list_sample = ""
    entity_count = 0
    id_to_name_mapping = ""
    if id_to_name:
        entity_names = list(id_to_name.values())
        entity_count = len(entity_names)
        # Show first 20 entities as sample
        sample_names = entity_names[:20]
        entity_list_sample = ", ".join(sample_names)
        if entity_count > 20:
            entity_list_sample += f" ... (and {entity_count - 20} more)"
        
        # Format id_to_name mapping for the prompt (show first 30 entries to avoid overwhelming)
        mapping_items = []
        for i, (eid, name) in enumerate(id_to_name.items()):
            if i < 30:
                mapping_items.append(f"  '{name}' -> {eid}")
            else:
                break
        id_to_name_mapping = "\n".join(mapping_items)
        if entity_count > 30:
            id_to_name_mapping += f"\n  ... (and {entity_count - 30} more mappings)"
    
    # Extract entities referenced in the question
    question_entity_ids = []
    question_entity_names = []
    if question:
        # Handle both Question object and dict
        if isinstance(question, dict):
            question_entity_ids = question.get('entities_referenced', [])
        elif hasattr(question, 'entities_referenced') and question.entities_referenced:
            question_entity_ids = question.entities_referenced
        
        print(f"[Retriever] Question entity IDs from question: {question_entity_ids}")
        print(f"[Retriever] id_to_name keys sample: {list(id_to_name.keys())[:5]}")
        
        # Map IDs to names
        for eid in question_entity_ids:
            if eid in id_to_name:
                question_entity_names.append(id_to_name[eid])
            else:
                print(f"[Retriever] WARNING: Question entity ID '{eid}' not found in id_to_name")
    
    question_entities_info = ""
    if question_entity_names:
        question_entities_info = f"Entities mentioned in current question: {', '.join(question_entity_names)} (IDs: {', '.join(question_entity_ids)})\n"
        question_entities_info += "PRIORITY: Use these entity IDs when matching the user's request.\n"
    else:
        print(f"[Retriever] WARNING: No question entity names found. IDs: {question_entity_ids}")
    
    # Get question text (handle both Question object and dict)
    question_text = "N/A"
    if question:
        if isinstance(question, dict):
            question_text = question.get('text', 'N/A')
        elif hasattr(question, 'text'):
            question_text = question.text
    
    prompt = registry.build_prompt(
        AGENT_RETRIEVER,
        plan=current_plan,
        user_message=user_message.content if isinstance(user_message, Message) else user_message.get("content", "") if user_message else "",
        current_question=question_text,
        id_to_name=id_to_name,
        entity_list_sample=entity_list_sample,
        entity_count=entity_count,
        question_entities_info=question_entities_info,
        question_entity_ids=", ".join(question_entity_ids) if question_entity_ids else "None",
        id_to_name_mapping=id_to_name_mapping,
        last_bot_message=""  # Not used anymore, but kept for backward compatibility with cached templates
    )
    
    # Debug: Print the full prompt
    print("\n" + "="*80)
    print("[Retriever] PROMPT:")
    print("="*80)
    print(prompt)
    print("="*80 + "\n")
    
    response = validator.api_repo.chat(prompt)
    action_data = JsonHelper.parse_json(str(response))
    
    # Debug logging
    print("\n" + "="*80)
    print("[Retriever] LLM RESPONSE:")
    print("="*80)
    print(f"Raw response: {response}")
    print(f"Parsed action_data: {action_data}")
    print("="*80 + "\n")
    
    if not action_data:
        action_data = {KEY_ACTION: ACTION_GET_ENTITY_INFO, KEY_PARAMETERS: {}, "reason": DEFAULT_REASON}
    
    action = action_data.get(KEY_ACTION, ACTION_GET_ENTITY_INFO)
    params = action_data.get(KEY_PARAMETERS, {})
    entity_id = params.get(KEY_ENTITY_ID, "")
    
    # Check if entity_id is actually a valid ID (exists in id_to_name keys)
    is_valid_id = entity_id in id_to_name if entity_id else False
    
    # If entity_id is not a valid ID, try to resolve it
    if entity_id and not is_valid_id:
        # Try to find ID by name (entity_id might be a name, not an ID)
        entity_name = entity_id
        matching_ids = []
        
        # First, check if any question entity IDs match this name
        if question_entity_ids:
            for qeid in question_entity_ids:
                if qeid in id_to_name and id_to_name[qeid].lower() == entity_name.lower():
                    matching_ids.append(qeid)
        
        # If no question entity matches, search all entities
        if not matching_ids:
            for eid, name in id_to_name.items():
                if name.lower() == entity_name.lower():
                    matching_ids.append(eid)
        
        # Use the first matching ID
        if matching_ids:
            entity_id = matching_ids[0]
            params[KEY_ENTITY_ID] = entity_id
            print(f"[Retriever] Resolved entity name '{entity_name}' to ID: {entity_id}")
        else:
            print(f"[Retriever] WARNING: Could not resolve entity name '{entity_name}' to an ID")
    
    # If still no entity_id but we have question entities, use the first one
    if not entity_id and question_entity_ids:
        entity_id = question_entity_ids[0]
        params[KEY_ENTITY_ID] = entity_id
        # Also set entity_name if not provided
        if not params.get(KEY_NAME, ""):
            params[KEY_NAME] = id_to_name.get(entity_id, "")
        print(f"[Retriever] Fallback: Using question entity ID: {entity_id} (name: {params.get(KEY_NAME, '')})")
    
    # Debug logging
    print(f"[Retriever] Final Action: {action}")
    print(f"[Retriever] Final Params: {params}")
    print(f"[Retriever] Entity ID: {entity_id}")
    print(f"[Retriever] Entity ID is valid: {entity_id in id_to_name if entity_id else False}")
    if question_entity_ids:
        print(f"[Retriever] Question entity IDs available: {question_entity_ids}")
    
    info = None
    entities_in_focus = []
    relevant_triples = []

    if action == ACTION_GET_ENTITY_INFO:
        entity_name = params.get(KEY_NAME, "")
        info = validator.tools.get_entity_info(entity_name, id=entity_id)
        entities_in_focus = [entity_name]
    elif action == ACTION_GET_TRIPLE_INFO:
        triple_index = params.get(KEY_INDEX, -1)
        info = validator.tools.get_triple_info(triple_index)
        relevant_triples = [triple_index]
    elif action == ACTION_GET_RELATED_TRIPLES:
        entity_name = params.get(KEY_NAME, "")
        info = validator.tools.get_related_triples(entity_name, id=entity_id)
        entities_in_focus = [entity_name]
    elif action == ACTION_SEARCH_ENTITIES:
        info = {KEY_SEARCH_RESULTS: validator.tools.search_entities(params.get("query", ""))}
    else:
        info = {KEY_ERROR: f"Unknown action: {action}"}
    
    serializable_info = _to_serializable(info)
    info_text = json.dumps(serializable_info, indent=2)
    retrieval_message = f"{KEY_RETRIEVED_INFO_MARKER}\n{info_text}\n\nReason: {action_data.get('reason', DEFAULT_N_A)}"
    
    # Consume current agent from queue
    state = consume_agent(state, AGENT_RETRIEVER)

    context_info = ChatContextInformation(
        intent=action,
        entities_in_focus=entities_in_focus,
        relevant_triples=relevant_triples,
        additional_context={"retrieval_reason": action_data.get("reason")}
    )
    
    # If triples were retrieved, add visualizer to queue before communicator
    # This ensures widgets are shown when triples are retrieved
    agent_queue = state.get(STATE_AGENT_QUEUE, [])
    if action in (ACTION_GET_RELATED_TRIPLES, ACTION_GET_TRIPLE_INFO) and info:
        # Check if triples were actually retrieved (not empty)
        has_triples = False
        if isinstance(info, list) and len(info) > 0:
            has_triples = True
        elif isinstance(info, dict):
            # EntityInfo with triples or single TripleInfo
            if info.get("triples") or info.get("index") is not None:
                has_triples = True
        
        if has_triples and AGENT_VISUALIZER not in agent_queue:
            # Insert before communicator if communicator exists, otherwise append
            if AGENT_COMMUNICATOR in agent_queue:
                comm_index = agent_queue.index(AGENT_COMMUNICATOR)
                agent_queue.insert(comm_index, AGENT_VISUALIZER)
                print(f"[Retriever] Added visualizer to queue before communicator (triples retrieved)")
            else:
                agent_queue.append(AGENT_VISUALIZER)
                print(f"[Retriever] Added visualizer to queue (triples retrieved)")
    
    return {
        **state,
        STATE_AGENT_QUEUE: agent_queue,
        STATE_CHAT_CONTEXT_INFORMATION: state.get(STATE_CHAT_CONTEXT_INFORMATION, []) + [context_info],
        STATE_MESSAGES: messages + [Message(role=MessageRole.SYSTEM, content=retrieval_message)],
    }
