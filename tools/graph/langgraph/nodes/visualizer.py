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
        ctx = context_info_list[-1]
        retrieved_info = (
            f"Retrieved information:\n"
            f"- Intent: {getattr(ctx, 'intent', 'N/A')}\n"
            f"- Entities in focus: {', '.join(getattr(ctx, 'entities_in_focus', []))}\n"
            f"- Relevant triples: {', '.join(map(str, getattr(ctx, 'relevant_triples', [])))}\n"
        )
    
    # Get last SYSTEM message (retrieval results) and extract triple data
    system_messages = [msg for msg in messages if msg.role == MessageRole.SYSTEM]
    if system_messages:
        last_system = system_messages[-1]
        system_content = last_system.content
        
        # Extract triple data from JSON in SYSTEM message
        if KEY_RETRIEVED_INFO_MARKER in system_content:
            try:
                # Parse JSON from SYSTEM message
                marker_idx = system_content.find(KEY_RETRIEVED_INFO_MARKER)
                json_start = marker_idx + len(KEY_RETRIEVED_INFO_MARKER)
                json_end = system_content.find("\n\nReason:", json_start)
                if json_end == -1:
                    json_end = len(system_content)
                
                json_str = system_content[json_start:json_end].strip()
                retrieved_data = json.loads(json_str)
                
                # Extract triples from retrieved data
                if isinstance(retrieved_data, list):
                    retrieved_triples_data = retrieved_data
                elif isinstance(retrieved_data, dict):
                    retrieved_triples_data = retrieved_data.get(KEY_RELATED_TRIPLES, retrieved_data.get("triples", []))
                    if not retrieved_triples_data and retrieved_data.get("index") is not None:
                        retrieved_triples_data = [retrieved_data]
                
                retrieved_info += f"\nRetrieved data:\n{system_content[:1000]}\n"
            except Exception as e:
                print(f"[Visualizer] Warning: Could not parse triple data: {e}")
                retrieved_info += f"\nRetrieved data (raw):\n{system_content[:1000]}\n"
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
    
    widget_data = JsonHelper.parse_json(response_str) if response_str else {"show_widget": False}
     
    print(f"[Visualizer] Parsed widget_data: {widget_data}")
     
    visual_info = ChatVisualInfo.from_dict(widget_data or {})
     
    # Update display actions list
    new_display_actions = state.get(STATE_DISPLAY_ACTIONS, [])
    if visual_info.widget:
        new_display_actions = new_display_actions + [visual_info]
        print(f"[Visualizer] Widget created: {visual_info.widget.widget_type}")
    
    updated_state = consume_agent(state, AGENT_VISUALIZER)
    
    return {
        **updated_state,
        STATE_DISPLAY_ACTIONS: new_display_actions,
    }
