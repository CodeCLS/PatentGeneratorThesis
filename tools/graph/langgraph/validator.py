"""
Main LangGraph-based Graph Validator class.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
import networkx as nx

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    raise ImportError("LangGraph is required. Install it with: pip install langgraph")

from tools.api.llm_api_repo import LLmApi_Repo
from tools.graph.data.Triple import Triple
from tools.graph.langgraph.state import GraphValidatorState, create_state
from tools.graph.langgraph.tools import GraphValidatorTools
from tools.graph.constants_graph import *

# Ensure GraphValidatorState is available in module globals for LangGraph evaluation
globals()['GraphValidatorState'] = GraphValidatorState

from tools.graph.langgraph.message import Message, MessageRole
from tools.graph.langgraph.nodes.orchestrator import orchestrator_node
from tools.graph.langgraph.nodes.communicator import communicator_node
from tools.graph.langgraph.nodes.retriever import retriever_node
from tools.graph.langgraph.nodes.visualizer import visualizer_node
from tools.graph.langgraph.nodes.analyzer import analyzer_node
from tools.graph.langgraph.nodes.modifier import modifier_node
from tools.graph.langgraph.routing import (
    route_from_orchestrator,
    route_from_communicator,
    route_from_retriever,
    route_from_visualizer,
    route_from_analyzer,
    route_from_modifier,
)


class GraphValidatorLangGraph:
    def __init__(
        self,
        graph: Optional[nx.MultiDiGraph] = None,
        triples: Optional[List[Triple]] = None,
        id_to_name: Optional[Dict[str, str]] = None,
        api_repo: Optional[LLmApi_Repo] = None,
    ):
        self.api_repo = api_repo or LLmApi_Repo()
        self.graph = graph
        self.triples = triples or []
        self.id_to_name = id_to_name or {}
        
        self.tools = GraphValidatorTools(
            graph=graph or nx.MultiDiGraph(),
            triples=self.triples,
            id_to_name=self.id_to_name,
        )
        self.workflow = self._build_graph()
        self.app = self.workflow.compile()
        self.recursion_limit = 10
        self._initial_questions = []
        self.config = {"recursion_limit": self.recursion_limit}

        self.initial_analysis_complete = False
        
        self._current_state = create_state(
            graph_nodes_count=self.graph.number_of_nodes() if self.graph else 0,
            graph_edges_count=self.graph.number_of_edges() if self.graph else 0,
            triples_count=len(self.triples),
            entities_count=len(self.id_to_name),
        )

    def run_initial_analysis(self):
        """Run the initial analysis stream in the background."""
        initial_state = self._current_state.copy()
        try:
            for state_update in self.app.stream(initial_state, config=self.config):
                if state_update:
                    last_node = list(state_update.keys())[-1]
                    self._current_state = state_update[last_node]
            self.initial_analysis_complete = True
        except Exception as e:
            print(f"Error in background analysis: {e}")
            self.initial_analysis_complete = True
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph[GraphValidatorState, None, GraphValidatorState, GraphValidatorState](GraphValidatorState)

        workflow.add_node(AGENT_COMMUNICATOR, lambda state: communicator_node(self, state))
        workflow.add_node(AGENT_RETRIEVER, lambda state: retriever_node(self, state, self.id_to_name))
        workflow.add_node(AGENT_VISUALIZER, lambda state: visualizer_node(self, state))
        workflow.add_node(AGENT_ANALYZER, lambda state: analyzer_node(self, state))
        workflow.add_node(AGENT_MODIFIER, lambda state: modifier_node(self, state))
        workflow.add_node(AGENT_ORCHESTRATOR, lambda state: orchestrator_node(self, state))

        workflow.set_entry_point(AGENT_ORCHESTRATOR)        
        routing_map = {
            AGENT_ORCHESTRATOR: AGENT_ORCHESTRATOR,
            AGENT_RETRIEVER: AGENT_RETRIEVER,
            AGENT_VISUALIZER: AGENT_VISUALIZER,
            AGENT_ANALYZER: AGENT_ANALYZER,
            AGENT_MODIFIER: AGENT_MODIFIER,
            AGENT_COMMUNICATOR: AGENT_COMMUNICATOR,
            END: END,
            "__end__": END,
        }
        
        workflow.add_conditional_edges(AGENT_ORCHESTRATOR, route_from_orchestrator, routing_map)
        workflow.add_conditional_edges(AGENT_COMMUNICATOR, route_from_communicator, routing_map)
        workflow.add_conditional_edges(AGENT_RETRIEVER, route_from_retriever, routing_map)
        workflow.add_conditional_edges(AGENT_VISUALIZER, route_from_visualizer, routing_map)
        workflow.add_conditional_edges(AGENT_ANALYZER, route_from_analyzer, routing_map)
        workflow.add_conditional_edges(AGENT_MODIFIER, route_from_modifier, routing_map)
        
        return workflow
    
    def _to_serializable(self, state_values: Dict[str, Any]) -> Dict[str, Any]:
        """Convert state values to serializable format for API response."""
        result = {}
        for key, value in state_values.items():
            if isinstance(value, list):
                result[key] = [item.to_dict() if hasattr(item, 'to_dict') else item for item in value]
            elif hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def chat(self, user_message: str, config: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a user message through the agent graph."""
        initial_state = self._current_state.copy()
        
        # Add new user message
        new_message = Message(role=MessageRole.USER, content=user_message)
        initial_state[STATE_MESSAGES] = initial_state.get(STATE_MESSAGES, []) + [new_message]
        initial_state[STATE_CONVERSATION_TURN] = initial_state.get(STATE_CONVERSATION_TURN, 0) + 1
        
        final_state_values = initial_state
        try:
            for state_update in self.app.stream(initial_state, config=self.config):
                for node_name, node_state in state_update.items():
                    print(f"--- Node '{node_name}' finished ---")
                    queue = node_state.get(STATE_AGENT_QUEUE, [])
                    print(f"Agent Queue: {queue}")
                    final_state_values = node_state
            
            self._current_state = final_state_values
            
        except Exception as e:
            import traceback
            error_msg = f"Error in chat: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return {
                STATE_MESSAGES: [m.to_dict() if hasattr(m, 'to_dict') else m for m in initial_state.get(STATE_MESSAGES, [])] + 
                                [Message(role=MessageRole.SYSTEM, content=f"Error: {str(e)}").to_dict()],
                STATE_TEXT: f"Error: {str(e)}",
                STATE_VALIDATION_COMPLETE: False,
            }
        
        # Find the last bot message for the 'text' response
        messages = final_state_values.get(STATE_MESSAGES, [])
        last_bot_msg_content = "Response processed."
        for msg in reversed(messages):
            role = msg.role if isinstance(msg, Message) else msg.get("role")
            if role == MessageRole.BOT or role == "bot":
                last_bot_msg_content = msg.content if isinstance(msg, Message) else msg.get("content", "")
                break
       
        # Convert all complex objects to dicts for JSON serialization
        serializable_state = self._to_serializable(final_state_values)
        
        # Prepare response for API
        response = {
            STATE_TEXT: last_bot_msg_content,
            **serializable_state,
            "stats": self.tools.calculate_stats().to_dict() if hasattr(self.tools, 'calculate_stats') else {}
        }
        
        # Extract widgets from display_actions and add to response
        display_actions = serializable_state.get(STATE_DISPLAY_ACTIONS, [])
        response["widgets"] = []
        
        for widget in display_actions:
            # Convert to dict if it's an object
            if hasattr(widget, "to_dict"):
                widget_dict = widget.to_dict()
            elif isinstance(widget, dict):
                widget_dict = widget
            else:
                continue
            
            if widget_dict.get("show_widget"):
                response["widgets"].append({
                    "widget_type": widget_dict.get("widget_type"),
                    "widget_data": widget_dict.get("widget_data", {})
                })
        
        # Keep old fields for backward compatibility with frontend
        if response["widgets"]:
            last_w = response["widgets"][-1]
            response["show_widget"] = True
            response["widget_type"] = last_w["widget_type"]
            response["widget_data"] = last_w["widget_data"]
        else:
            response["show_widget"] = False
        
        # Ensure 'next_question' is set if 'current_question' exists (for frontend compatibility)
        if STATE_CURRENT_QUESTION in serializable_state and serializable_state[STATE_CURRENT_QUESTION]:
            response[STATE_NEXT_QUESTION] = serializable_state[STATE_CURRENT_QUESTION].get("text")
            
        return response
