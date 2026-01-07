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
from tools.graph.Triple import Triple
from tools.graph.langgraph.state import GraphValidatorState
from tools.graph.langgraph.tools import GraphValidatorTools

# Ensure GraphValidatorState is available in module globals for LangGraph evaluation
globals()['GraphValidatorState'] = GraphValidatorState

from tools.graph.langgraph.nodes.communicator import communicator_node
from tools.graph.langgraph.nodes.retriever import retriever_node
from tools.graph.langgraph.nodes.visualizer import visualizer_node
from tools.graph.langgraph.nodes.analyzer import analyzer_node
from tools.graph.langgraph.nodes.modifier import modifier_node
from tools.graph.langgraph.routing import (
    route_from_communicator,
    route_from_retriever,
    route_from_visualizer,
    route_from_analyzer,
    route_from_modifier,
)


class GraphValidatorLangGraph:
    """Multi-agent graph validator using LangGraph."""
    
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
        self._current_state = None
        self._initial_questions = []
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph with all agent nodes."""
        workflow = StateGraph(GraphValidatorState)
        
        def communicator_wrapper(state: "GraphValidatorState") -> "GraphValidatorState":
            return communicator_node(self, state)
        
        def retriever_wrapper(state: "GraphValidatorState") -> "GraphValidatorState":
            return retriever_node(self, state)
        
        def visualizer_wrapper(state: "GraphValidatorState") -> "GraphValidatorState":
            return visualizer_node(self, state)
        
        def analyzer_wrapper(state: "GraphValidatorState") -> "GraphValidatorState":
            return analyzer_node(self, state)
        
        def modifier_wrapper(state: "GraphValidatorState") -> "GraphValidatorState":
            return modifier_node(self, state)
        
        workflow.add_node("communicator", communicator_wrapper)
        workflow.add_node("retriever", retriever_wrapper)
        workflow.add_node("visualizer", visualizer_wrapper)
        workflow.add_node("analyzer", analyzer_wrapper)
        workflow.add_node("modifier", modifier_wrapper)
        
        workflow.set_entry_point("communicator")
        
        routing_map = {
            "retriever": "retriever",
            "visualizer": "visualizer",
            "analyzer": "analyzer",
            "modifier": "modifier",
            END: END,
            "__end__": END,
        }
        
        workflow.add_conditional_edges("communicator", route_from_communicator, routing_map)
        workflow.add_conditional_edges("retriever", route_from_retriever, {**routing_map, "communicator": "communicator"})
        workflow.add_conditional_edges("visualizer", route_from_visualizer, {**routing_map, "communicator": "communicator"})
        workflow.add_conditional_edges("analyzer", route_from_analyzer, {**routing_map, "communicator": "communicator"})
        workflow.add_conditional_edges("modifier", route_from_modifier, {**routing_map, "communicator": "communicator"})
        
        return workflow
    
    def chat(self, user_message: str, config: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a user message through the agent graph."""
        if not self._current_state:
            initial_state: Dict[str, Any] = {
                "messages": [],
                "current_question_id": None,
                "current_question_text": None,
                "questions": self._initial_questions,
                "graph_nodes_count": self.graph.number_of_nodes() if self.graph else 0,
                "graph_edges_count": self.graph.number_of_edges() if self.graph else 0,
                "triples_count": len(self.triples),
                "entities_count": len(self.id_to_name),
                "next_agent": None,
                "validation_complete": False,
                "hidden_actions": [],
                "display_actions": [],
                "show_widget": False,
                "widget_type": None,
                "widget_data": {},
                "conversation_turn": 0,
                "changes_summary": [],
                "stats": {},
            }
            self._current_state = initial_state
        else:
            initial_state = self._current_state.copy()
        
        initial_state["messages"] = initial_state.get("messages", []) + [
            {"role": "user", "content": user_message}
        ]
        initial_state["conversation_turn"] = initial_state.get("conversation_turn", 0) + 1
        
        final_state = None
        try:
            config = {"recursion_limit": self.recursion_limit}
            for state in self.app.stream(initial_state, config=config):
                final_state = state
                if state:
                    last_node = list(state.keys())[-1] if state else None
                    if last_node:
                        self._current_state = state[last_node]
        except Exception as e:
            return {
                "text": f"Error: {str(e)}",
                "next_question": None,
                "validation_complete": False,
                "hidden_actions": [],
                "show_widget": False,
                "widget_type": None,
                "widget_data": {},
                "changes_summary": [],
                "stats": {},
            }
        
        if not final_state:
            return {
                "text": "No response generated.",
                "next_question": None,
                "validation_complete": False,
                "hidden_actions": [],
                "show_widget": False,
                "widget_type": None,
                "widget_data": {},
                "changes_summary": [],
                "stats": {},
            }
        
        last_node = list(final_state.keys())[-1] if final_state else None
        state_values = final_state[last_node] if last_node else initial_state
        
        messages = state_values.get("messages", [])
        last_bot_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "bot":
                last_bot_msg = msg.get("content", "")
                break
        
        return {
            "text": last_bot_msg or "Response processed.",
            "next_question": state_values.get("current_question_text"),
            "validation_complete": state_values.get("validation_complete", False),
            "hidden_actions": state_values.get("hidden_actions", []),
            "show_widget": state_values.get("show_widget", False),
            "widget_type": state_values.get("widget_type"),
            "widget_data": state_values.get("widget_data", {}),
            "changes_summary": state_values.get("changes_summary", []),
            "stats": state_values.get("stats", {}),
        }
    
    def analyze(
        self,
        graph: Optional[nx.MultiDiGraph] = None,
        triples: Optional[List[Triple]] = None,
        id_to_name: Optional[Dict[str, str]] = None,
    ) -> None:
        """Analyze a graph and/or triples to find issues."""
        if graph is not None:
            self.graph = graph
        if triples is not None:
            self.triples = triples
        if id_to_name is not None:
            self.id_to_name = id_to_name
        
        self.tools = GraphValidatorTools(
            graph=self.graph or nx.MultiDiGraph(),
            triples=self.triples,
            id_to_name=self.id_to_name,
        )
        
        from tools.graph.langgraph.nodes.analyzer import generate_questions
        self._initial_questions = generate_questions(self)
        self._current_state = None
