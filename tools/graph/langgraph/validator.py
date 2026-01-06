"""
Main LangGraph-based Graph Validator class.

This module implements the multi-agent system for graph validation using LangGraph.
Each agent handles a specific task (communication, retrieval, visualization, etc.)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
import networkx as nx

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    raise ImportError(
        "LangGraph is required. Install it with: pip install langgraph"
    )

# Import LLM API repo - must be before other imports that might use it
try:
    from tools.api.llm_api_repo import LLmApi_Repo
except ImportError as e:
    raise ImportError(f"Failed to import LLmApi_Repo: {e}")

from tools.graph.Triple import Triple

from tools.graph.langgraph.state import GraphValidatorState
from tools.graph.langgraph.tools import GraphValidatorTools

# Ensure GraphValidatorState is available in module globals for LangGraph evaluation
# This is needed when LangGraph tries to evaluate type annotations at runtime
# Store it in globals() to make it available during forward reference evaluation
globals()['GraphValidatorState'] = GraphValidatorState

# Import node functions
from tools.graph.langgraph.nodes.communicator import communicator_node
from tools.graph.langgraph.nodes.retriever import retriever_node
from tools.graph.langgraph.nodes.visualizer import visualizer_node
from tools.graph.langgraph.nodes.analyzer import analyzer_node
from tools.graph.langgraph.nodes.modifier import modifier_node

# Import routing functions
from tools.graph.langgraph.routing import (
    route_from_communicator,
    route_from_retriever,
    route_from_visualizer,
    route_from_analyzer,
    route_from_modifier,
)


class GraphValidatorLangGraph:
    """
    Multi-agent graph validator using LangGraph.
    
    Agents:
    - communicator: Main communication agent (entry point)
    - retriever: Retrieves information about entities and triples
    - visualizer: Decides which widgets to show
    - analyzer: Analyzes graph and generates questions
    - modifier: Applies graph modifications
    """
    
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
        
        # Initialize tools
        self.tools = GraphValidatorTools(
            graph=graph or nx.MultiDiGraph(),
            triples=self.triples,
            id_to_name=self.id_to_name,
        )
        
        # Build the graph
        self.workflow = self._build_graph()
        # Compile without checkpointing to avoid serialization issues
        # Checkpointing requires all state to be serializable, which is problematic with NetworkX graphs
        # Set recursion limit to 10 to prevent infinite loops
        self.app = self.workflow.compile()
        self.recursion_limit = 10
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph with all agent nodes."""
        # GraphValidatorState is already imported at module level
        # Use it directly - LangGraph should be able to access it
        workflow = StateGraph(GraphValidatorState)
        
        # Create wrapper functions that bind the validator instance
        # LangGraph expects node functions to take only state, so we wrap them
        # Use string annotations to avoid evaluation issues
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
        
        # Add nodes
        workflow.add_node("communicator", communicator_wrapper)
        workflow.add_node("retriever", retriever_wrapper)
        workflow.add_node("visualizer", visualizer_wrapper)
        workflow.add_node("analyzer", analyzer_wrapper)
        workflow.add_node("modifier", modifier_wrapper)
        
        # Set entry point
        workflow.set_entry_point("communicator")
        
        # Add edges based on agent decisions
        # Note: LangGraph may convert END to '__end__' string internally
        # We include both to handle both cases
        def make_routing_map(agents_to_communicator=True):
            """Create routing map with both END constant and '__end__' string keys."""
            routing = {
                "retriever": "retriever",
                "visualizer": "visualizer",
                "analyzer": "analyzer",
                "modifier": "modifier",
                END: END,  # END constant as key
                "__end__": END,  # Also include string key for compatibility
            }
            if agents_to_communicator:
                routing["communicator"] = "communicator"
            return routing
        
        workflow.add_conditional_edges(
            "communicator",
            route_from_communicator,
            make_routing_map(agents_to_communicator=False)
        )
        
        workflow.add_conditional_edges(
            "retriever",
            route_from_retriever,
            make_routing_map(agents_to_communicator=True)
        )
        
        workflow.add_conditional_edges(
            "visualizer",
            route_from_visualizer,
            make_routing_map(agents_to_communicator=True)
        )
        
        workflow.add_conditional_edges(
            "analyzer",
            route_from_analyzer,
            make_routing_map(agents_to_communicator=True)
        )
        
        workflow.add_conditional_edges(
            "modifier",
            route_from_modifier,
            make_routing_map(agents_to_communicator=True)
        )
        
        return workflow
    
    # ========================================================================
    # Public API
    # ========================================================================
    
    def chat(self, user_message: str, config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process a user message through the agent graph.
        
        Args:
            user_message: User's message
            config: Optional LangGraph config (for checkpointing)
            
        Returns:
            Dict with response text, next_question, validation_complete, etc.
        """
        # Manage state manually (without checkpointing to avoid serialization issues)
        # Initialize state if needed
        if not hasattr(self, '_current_state') or not self._current_state:
            # Initial state (don't include graph/triples directly - they're stored in self.graph/self.triples)
            # Use dict type hint to avoid evaluation issues
            # Use initial questions from analyze() if available
            initial_questions = getattr(self, '_initial_questions', [])
            initial_state: Dict[str, Any] = {
                "messages": [],
                "current_question_id": None,
                "current_question_text": None,
                "questions": initial_questions,  # Use questions generated in analyze()
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
        
        # Add user message
        initial_state["messages"] = initial_state.get("messages", []) + [
            {"role": "user", "content": user_message}
        ]
        initial_state["conversation_turn"] = initial_state.get("conversation_turn", 0) + 1
        
        # Run the graph (without checkpointing config)
        final_state = None
        try:
            # Stream with recursion limit config (no checkpointing to avoid serialization issues)
            config = {"recursion_limit": self.recursion_limit}
            for state in self.app.stream(initial_state, config=config):
                final_state = state
                # Update our stored state
                if state:
                    last_node = list(state.keys())[-1] if state else None
                    if last_node:
                        self._current_state = state[last_node]
        except Exception as e:
            print(f"Error in LangGraph stream: {e}")
            import traceback
            traceback.print_exc()
            # Return error response
            return {
                "text": f"I encountered an error processing your message: {str(e)}",
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
                "text": "No response generated. Please try again.",
                "next_question": None,
                "validation_complete": False,
                "hidden_actions": [],
                "show_widget": False,
                "widget_type": None,
                "widget_data": {},
                "changes_summary": [],
                "stats": {},
            }
        
        # Extract final state
        last_node = list(final_state.keys())[-1] if final_state else None
        if last_node:
            state_values = final_state[last_node]
            # Handle case where state_values might be None
            if state_values is None:
                state_values = initial_state
        else:
            state_values = initial_state
        
        # Get last bot message
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
        """
        Analyze a graph and/or triples to find issues.
        
        This method:
        1. Updates the graph data and tools
        2. Generates initial validation questions (via analyzer_node logic)
        3. Resets conversation state
        
        The questions are generated using the same logic as analyzer_node,
        but are stored in the initial state for the first chat() call.
        
        Args:
            graph: Optional NetworkX graph to analyze
            triples: Optional list of Triple objects to analyze
            id_to_name: Optional mapping from entity ID to display name
        """
        if graph is not None:
            self.graph = graph
        if triples is not None:
            self.triples = triples
        if id_to_name is not None:
            self.id_to_name = id_to_name
        
        # Update tools with new graph data
        # The tools need fresh references to the updated graph/triples
        self.tools = GraphValidatorTools(
            graph=self.graph or nx.MultiDiGraph(),
            triples=self.triples,
            id_to_name=self.id_to_name,
        )
        
        # Generate initial questions by analyzing the graph
        # This matches the behavior of the original GraphValidator.analyze()
        from tools.graph.langgraph.nodes.analyzer import build_context, generate_questions
        
        context = build_context(self)
        initial_questions = generate_questions(self, context)
        
        # Reset conversation state when graph is updated
        # Store initial questions in state for first chat() call
        if hasattr(self, '_current_state'):
            self._current_state = None
        
        # Store initial questions for use in first chat() call
        # The communicator_node will use these if no questions exist in state
        self._initial_questions = initial_questions
        
        # Note: We don't rebuild the workflow here because:
        # 1. The workflow structure (nodes, edges, routing) doesn't change based on graph data
        # 2. The workflow was already built in __init__ and compiled into self.app
        # 3. The nodes access graph data through self.graph/self.triples, not through the workflow
        # 4. Rebuilding is expensive and unnecessary
