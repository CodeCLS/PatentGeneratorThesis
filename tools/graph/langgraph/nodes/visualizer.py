"""
Visualizer node - decides which widgets to show and with what data.
"""

import json
from typing import TYPE_CHECKING, Union
from tools.helper.json_helper import JsonHelper
from tools.graph.langgraph.state import GraphValidatorState, consume_agent
from tools.graph.langgraph.prompts import get_registry
from tools.graph.langgraph.nodes.chat_visual_info import ChatVisualInfo
from tools.graph.langgraph.message import Message, MessageRole
from tools.graph.constants_graph import (
    AGENT_COMMUNICATOR,
    AGENT_VISUALIZER,
    STATE_MESSAGES,
    STATE_CURRENT_QUESTION,
    STATE_DISPLAY_ACTIONS,
    STATE_AGENT_QUEUE,
    STATE_PLAN,
    STATE_CHAT_CONTEXT_INFORMATION,
    KEY_RETRIEVED_INFO_MARKER,
    KEY_RELATED_TRIPLES
)

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def visualizer_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState") -> "GraphValidatorState":
    """Visualization agent - decides which widgets to show."""
    print(f"\n[Visualizer] ========== VISUALIZER NODE RUNNING ==========")
    messages = state.get(STATE_MESSAGES, [])
    current_question = state.get(STATE_CURRENT_QUESTION)
    current_plan = state.get(STATE_PLAN)
    
    # Get retrieved information from context (not conversation)
    context_info_list = state.get(STATE_CHAT_CONTEXT_INFORMATION, [])
    retrieved_info = ""
    retrieved_triples_data = []  # Actual triple data for widget
    
    if context_info_list:
        last_context = context_info_list[-1]
        if isinstance(last_context, dict):
            retrieved_info = (
                f"Retrieved information:\n"
                f"- Intent: {last_context.get('intent', 'N/A')}\n"
                f"- Entities in focus: {', '.join(last_context.get('entities_in_focus', []))}\n"
                f"- Relevant triples: {', '.join(map(str, last_context.get('relevant_triples', [])))}\n"
            )
        else:
            # Handle ChatContextInformation object
            retrieved_info = (
                f"Retrieved information:\n"
                f"- Intent: {getattr(last_context, 'intent', 'N/A')}\n"
                f"- Entities in focus: {', '.join(getattr(last_context, 'entities_in_focus', []))}\n"
                f"- Relevant triples: {', '.join(map(str, getattr(last_context, 'relevant_triples', [])))}\n"
            )
    
    # Get last SYSTEM message (retrieval results) and extract triple data
    system_messages = [msg for msg in messages if (msg.role if isinstance(msg, Message) else msg.get("role")) == MessageRole.SYSTEM]
    if system_messages:
        last_system = system_messages[-1]
        system_content = last_system.content if isinstance(last_system, Message) else last_system.get("content", "")
        
        # Extract triple data from JSON in SYSTEM message
        if KEY_RETRIEVED_INFO_MARKER in system_content:
            try:
                # Parse JSON from SYSTEM message
                json_start = system_content.find(KEY_RETRIEVED_INFO_MARKER) + len(KEY_RETRIEVED_INFO_MARKER)
                json_end = system_content.find("\n\nReason:", json_start)
                if json_end == -1:
                    json_end = len(system_content)
                
                json_str = system_content[json_start:json_end].strip()
                retrieved_data = json.loads(json_str)
                
                # Extract triples from retrieved data
                # Could be a list (get_related_triples) or dict with 'triples' key (get_entity_info)
                if isinstance(retrieved_data, list):
                    retrieved_triples_data = retrieved_data
                elif isinstance(retrieved_data, dict):
                    if KEY_RELATED_TRIPLES in retrieved_data:
                        retrieved_triples_data = retrieved_data[KEY_RELATED_TRIPLES]
                    elif "triples" in retrieved_data:
                        retrieved_triples_data = retrieved_data["triples"]
                    elif retrieved_data.get("index") is not None:
                        # Single triple info
                        retrieved_triples_data = [retrieved_data]
                
                retrieved_info += f"\nRetrieved data:\n{system_content[:1000]}\n"  # Show first 1000 chars
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[Visualizer] Warning: Could not parse triple data from SYSTEM message: {e}")
                retrieved_info += f"\nRetrieved data:\n{system_content[:1000]}\n"
        else:
            retrieved_info += f"\nRetrieved data:\n{system_content[:1000]}\n"
    
    if not retrieved_info:
        retrieved_info = "No data retrieved yet"
    
    registry = get_registry()
    
    # Format triple data for the prompt (so LLM can create widget with correct data)
    triples_data_str = ""
    if retrieved_triples_data:
        triples_data_str = f"\n\nRetrieved triples data (use this for widget_data.triples):\n{json.dumps(retrieved_triples_data, indent=2)}"
        retrieved_info += triples_data_str
    
    prompt = registry.build_prompt(
        AGENT_VISUALIZER,
        plan=current_plan,
        current_question=current_question.text if current_question else "N/A",
        retrieved_info=retrieved_info
    )
    
    response = validator.api_repo.chat(prompt)
    response_str = str(response) if response else ""
    
    print(f"[Visualizer] Raw LLM Response (first 500 chars): {response_str[:500]}")
    print(f"[Visualizer] Retrieved triples count: {len(retrieved_triples_data)}")
    
    widget_data = JsonHelper.parse_json(response_str) if response_str else None
    
    if widget_data is None:
        print(f"[Visualizer] ERROR: Failed to parse JSON from LLM response")
        print(f"[Visualizer] Full response: {response_str}")
        # Create a default widget if we have triples but parsing failed
        if retrieved_triples_data:
            print(f"[Visualizer] Creating default widget with {len(retrieved_triples_data)} triples")
            # Ensure triples are in the correct format (list of dicts with index, head, relation, tail)
            formatted_triples = []
            for i, triple in enumerate(retrieved_triples_data):
                if isinstance(triple, dict):
                    # Already a dict, use it
                    formatted_triples.append(triple)
                else:
                    # Try to convert to dict
                    if hasattr(triple, 'to_dict'):
                        formatted_triples.append(triple.to_dict())
                    else:
                        print(f"[Visualizer] Warning: Could not format triple {i}: {type(triple)}")
            
            if formatted_triples:
                widget_data = {
                    "show_widget": True,
                    "widget_type": "edges_widget",
                    "widget_data": {"triples": formatted_triples}
                }
            else:
                widget_data = {"show_widget": False}
        else:
            widget_data = {"show_widget": False}
    
    print(f"[Visualizer] Parsed widget_data: {widget_data}")
    
    visual_info = ChatVisualInfo.from_dict(widget_data or {})
    
    # Update display actions list
    # ChatVisualInfo contains a widget if show_widget was True in the LLM response
    # The widget is created via Widget.from_dict() which determines the widget type
    new_display_actions = state.get(STATE_DISPLAY_ACTIONS, [])
    if visual_info.widget:  # Check if widget exists (means show_widget was True)
        new_display_actions = new_display_actions + [visual_info]
        print(f"[Visualizer] Widget created: {visual_info.widget.widget_type if visual_info.widget else 'None'}")
    else:
        print(f"[Visualizer] No widget created (show_widget was False or widget_type missing)")
    
    # Use consume_agent helper which returns updated state
    updated_state = consume_agent(state, AGENT_VISUALIZER)
    
    print(f"[Visualizer] ===========================================\n")
    
    return {
        **updated_state,
        STATE_DISPLAY_ACTIONS: new_display_actions,
    }
