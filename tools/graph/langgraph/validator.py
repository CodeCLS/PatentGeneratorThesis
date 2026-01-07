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
from tools.graph.langgraph.state import GraphValidatorState, create_state
from tools.graph.langgraph.tools import GraphValidatorTools
from tools.graph.constants_graph import (
    AGENT_COMMUNICATOR,
    AGENT_ANALYZER,
    AGENT_MODIFIER,
    AGENT_RETRIEVER,
    AGENT_VISUALIZER,
    STATE_MESSAGES,
    STATE_CONVERSATION_TURN,
    STATE_CURRENT_QUESTION_TEXT,
    STATE_VALIDATION_COMPLETE,
    STATE_HIDDEN_ACTIONS,
    STATE_SHOW_WIDGET,
    STATE_WIDGET_TYPE,
    STATE_WIDGET_DATA,
    STATE_CHANGES_SUMMARY,
    STATE_STATS,
)

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
        self._initial_questions = []
        
        # Run initial stream to generate questions (starts at analyzer entry point)
        initial_state = create_state(
            graph_nodes_count=self.graph.number_of_nodes() if self.graph else 0,
            graph_edges_count=self.graph.number_of_edges() if self.graph else 0,
            triples_count=len(self.triples),
            entities_count=len(self.id_to_name),
        )
        
        # Run stream once to generate questions
        try:
            config = {"recursion_limit": self.recursion_limit}
            for state in self.app.stream(initial_state, config=config):
                if state:
                    last_node = list(state.keys())[-1] if state else None
                    if last_node:
                        self._current_state = state[last_node]
        except Exception as e:
            # If initialization fails, start with empty state
            self._current_state = initial_state
            print(f"Warning: Failed to initialize questions: {e}")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph with all agent nodes."""
        workflow = StateGraph[GraphValidatorState, None, GraphValidatorState, GraphValidatorState](GraphValidatorState)
        
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
        
        workflow.add_node(AGENT_COMMUNICATOR, communicator_wrapper)
        workflow.add_node(AGENT_RETRIEVER, retriever_wrapper)
        workflow.add_node(AGENT_VISUALIZER, visualizer_wrapper)
        workflow.add_node(AGENT_ANALYZER, analyzer_wrapper)
        workflow.add_node(AGENT_MODIFIER, modifier_wrapper)
        
        workflow.set_entry_point(AGENT_ANALYZER)
        
        routing_map = {
            AGENT_RETRIEVER: AGENT_RETRIEVER,
            AGENT_VISUALIZER: AGENT_VISUALIZER,
            AGENT_ANALYZER: AGENT_ANALYZER,
            AGENT_MODIFIER: AGENT_MODIFIER,
            END: END,
            "__end__": END,
        }
        
        workflow.add_conditional_edges(AGENT_COMMUNICATOR, route_from_communicator, routing_map)
        workflow.add_conditional_edges(AGENT_RETRIEVER, route_from_retriever, {**routing_map, AGENT_COMMUNICATOR: AGENT_COMMUNICATOR})
        workflow.add_conditional_edges(AGENT_VISUALIZER, route_from_visualizer, {**routing_map, AGENT_COMMUNICATOR: AGENT_COMMUNICATOR})
        workflow.add_conditional_edges(AGENT_ANALYZER, route_from_analyzer, {**routing_map, AGENT_COMMUNICATOR: AGENT_COMMUNICATOR})
        workflow.add_conditional_edges(AGENT_MODIFIER, route_from_modifier, {**routing_map, AGENT_COMMUNICATOR: AGENT_COMMUNICATOR})
        
        return workflow
    
    def chat(self, user_message: str, config: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a user message through the agent graph."""
        # _current_state is already initialized in __init__
        initial_state = self._current_state.copy()
        
        initial_state[STATE_MESSAGES] = initial_state.get(STATE_MESSAGES, []) + [
            {"role": "user", "content": user_message}
        ]
        initial_state[STATE_CONVERSATION_TURN] = initial_state.get(STATE_CONVERSATION_TURN, 0) + 1
        
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
                STATE_VALIDATION_COMPLETE: False,
                STATE_HIDDEN_ACTIONS: [],
                STATE_SHOW_WIDGET: False,
                STATE_WIDGET_TYPE: None,
                STATE_WIDGET_DATA: {},
                STATE_CHANGES_SUMMARY: [],
                STATE_STATS: {},
            }
        
        if not final_state:
            return {
                "text": "No response generated.",
                "next_question": None,
                STATE_VALIDATION_COMPLETE: False,
                STATE_HIDDEN_ACTIONS: [],
                STATE_SHOW_WIDGET: False,
                STATE_WIDGET_TYPE: None,
                STATE_WIDGET_DATA: {},
                STATE_CHANGES_SUMMARY: [],
                STATE_STATS: {},
            }
        
        last_node = list(final_state.keys())[-1] if final_state else None
        state_values = final_state[last_node] if last_node else initial_state
        
        messages = state_values.get(STATE_MESSAGES, [])
        last_bot_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "bot":
                last_bot_msg = msg.get("content", "")
                break
        
        return {
            "text": last_bot_msg or "Response processed.",
            "next_question": state_values.get(STATE_CURRENT_QUESTION_TEXT),
            STATE_VALIDATION_COMPLETE: state_values.get(STATE_VALIDATION_COMPLETE, False),
            STATE_HIDDEN_ACTIONS: state_values.get(STATE_HIDDEN_ACTIONS, []),
            STATE_SHOW_WIDGET: state_values.get(STATE_SHOW_WIDGET, False),
            STATE_WIDGET_TYPE: state_values.get(STATE_WIDGET_TYPE),
            STATE_WIDGET_DATA: state_values.get(STATE_WIDGET_DATA, {}),
            STATE_CHANGES_SUMMARY: state_values.get(STATE_CHANGES_SUMMARY, []),
            STATE_STATS: state_values.get(STATE_STATS, {}),
        }
    
