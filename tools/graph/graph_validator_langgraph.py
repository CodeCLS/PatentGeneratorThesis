"""
LangGraph-based Graph Validator with multiple agent nodes.

This module implements a multi-agent system for graph validation using LangGraph.
Each agent handles a specific task (communication, retrieval, visualization, etc.)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, TypedDict, Annotated, Literal
from dataclasses import dataclass, field
import json
import uuid
import networkx as nx

try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    raise ImportError(
        "LangGraph is required. Install it with: pip install langgraph"
    )

from tools.api.llm_api_repo import LLmApi_Repo
from tools.graph.Triple import Triple
from tools.sentence.entity import Entity


# ============================================================================
# State Definition
# ============================================================================

class GraphValidatorState(TypedDict):
    """State passed between agent nodes in the graph."""
    # Current conversation
    messages: Annotated[List[Dict[str, str]], "append"]  # Chat messages: [{"role": "user/bot", "content": "..."}]
    
    # Current question being handled
    current_question_id: Optional[str]  # ID of the current question
    current_question_text: Optional[str]  # Text of the current question
    questions: List[Dict[str, Any]]  # All available questions
    
    # Graph data (metadata only - actual graph stored separately to avoid serialization issues)
    graph_nodes_count: int  # Number of nodes in graph
    graph_edges_count: int  # Number of edges in graph
    triples_count: int  # Number of triples
    entities_count: int  # Number of entities
    
    # Agent decisions
    next_agent: Optional[str]  # Which agent to route to next
    validation_complete: bool  # Whether validation is done
    
    # Actions to perform
    hidden_actions: List[Dict[str, Any]]  # Graph modification actions
    display_actions: List[Dict[str, Any]]  # UI display actions
    
    # Widget information
    show_widget: bool
    widget_type: Optional[str]
    widget_data: Dict[str, Any]
    
    # Context and metadata
    conversation_turn: int  # Current turn number
    changes_summary: List[str]  # Summary of changes made
    stats: Dict[str, Any]  # Graph statistics


# ============================================================================
# Tools for Agents
# ============================================================================

class GraphValidatorTools:
    """Tools that agents can use to interact with the graph."""
    
    def __init__(
        self,
        graph: nx.MultiDiGraph,
        triples: List[Triple],
        id_to_name: Dict[str, str],
    ):
        self.graph = graph
        self.triples = triples
        self.id_to_name = id_to_name
        self._original_graph = graph.copy() if graph else None
        self._original_triples = triples.copy()
    
    def get_entity_info(self, entity_name: str) -> Dict[str, Any]:
        """Retrieve detailed information about an entity by name."""
        # Find entity ID from name
        entity_id = None
        for eid, name in self.id_to_name.items():
            if name.lower() == entity_name.lower():
                entity_id = eid
                break
        
        if not entity_id:
            return {"error": f"Entity '{entity_name}' not found"}
        
        info = {
            "name": entity_name,
            "id": entity_id,
            "connections": 0,
            "properties": {},
            "connected_entities": [],
            "triples": [],
        }
        
        if self.graph and self.graph.has_node(entity_id):
            node_data = self.graph.nodes[entity_id]
            info["properties"] = {k: v for k, v in node_data.items() 
                                 if k not in ("node_type", "name") and not k.startswith("_")}
            info["label"] = node_data.get("node_type", "UNKNOWN")
            info["connections"] = self.graph.degree(entity_id)
            
            # Get connected entities
            for neighbor in self.graph.neighbors(entity_id):
                neighbor_name = self.id_to_name.get(neighbor, neighbor)
                edge_data = self.graph.get_edge_data(entity_id, neighbor)
                if edge_data:
                    for key, data in edge_data.items():
                        relation = data.get("label", "")
                        info["connected_entities"].append({
                            "name": neighbor_name,
                            "relation": relation,
                            "direction": "outgoing"
                        })
            
            # Get incoming connections
            for predecessor in self.graph.predecessors(entity_id):
                pred_name = self.id_to_name.get(predecessor, predecessor)
                edge_data = self.graph.get_edge_data(predecessor, entity_id)
                if edge_data:
                    for key, data in edge_data.items():
                        relation = data.get("label", "")
                        info["connected_entities"].append({
                            "name": pred_name,
                            "relation": relation,
                            "direction": "incoming"
                        })
        
        # Find triples involving this entity
        for i, triple in enumerate(self.triples):
            head_id = getattr(triple.head, "id", str(triple.head))
            tail_id = getattr(triple.tail, "id", str(triple.tail))
            if head_id == entity_id or tail_id == entity_id:
                head_name = self.id_to_name.get(head_id, str(triple.head))
                tail_name = self.id_to_name.get(tail_id, str(triple.tail))
                info["triples"].append({
                    "index": i,
                    "head": head_name,
                    "relation": triple.relation,
                    "tail": tail_name,
                })
        
        return info
    
    def get_triple_info(self, triple_index: int) -> Dict[str, Any]:
        """Retrieve detailed information about a triple by index."""
        if triple_index < 0 or triple_index >= len(self.triples):
            return {"error": f"Triple index {triple_index} out of range"}
        
        triple = self.triples[triple_index]
        head_id = getattr(triple.head, "id", str(triple.head))
        tail_id = getattr(triple.tail, "id", str(triple.tail))
        head_name = self.id_to_name.get(head_id, str(triple.head))
        tail_name = self.id_to_name.get(tail_id, str(triple.tail))
        
        info = {
            "index": triple_index,
            "head": head_name,
            "head_id": head_id,
            "relation": triple.relation,
            "tail": tail_name,
            "tail_id": tail_id,
        }
        
        # Get additional context from graph
        if self.graph:
            if self.graph.has_node(head_id):
                info["head_properties"] = dict(self.graph.nodes[head_id])
            if self.graph.has_node(tail_id):
                info["tail_properties"] = dict(self.graph.nodes[tail_id])
            
            # Check if edge exists in graph
            if self.graph.has_edge(head_id, tail_id):
                edge_data = self.graph.get_edge_data(head_id, tail_id)
                if edge_data:
                    info["edge_data"] = dict(list(edge_data.values())[0])
        
        return info
    
    def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """Search for entities by name (fuzzy match)."""
        query_lower = query.lower()
        results = []
        
        for eid, name in self.id_to_name.items():
            if query_lower in name.lower():
                results.append({"id": eid, "name": name})
                if len(results) >= limit:
                    break
        
        return results
    
    def get_related_triples(self, entity_name: str, max_depth: int = 1) -> List[Dict[str, Any]]:
        """Get triples related to an entity (including neighbors)."""
        entity_id = None
        for eid, name in self.id_to_name.items():
            if name.lower() == entity_name.lower():
                entity_id = eid
                break
        
        if not entity_id or not self.graph:
            return []
        
        # Get direct and neighbor entities
        related_entities = {entity_id}
        if self.graph.has_node(entity_id):
            for neighbor in list(self.graph.neighbors(entity_id))[:5]:  # Limit neighbors
                related_entities.add(neighbor)
            for predecessor in list(self.graph.predecessors(entity_id))[:5]:
                related_entities.add(predecessor)
        
        # Find triples involving these entities
        related_triples = []
        for i, triple in enumerate(self.triples):
            head_id = getattr(triple.head, "id", str(triple.head))
            tail_id = getattr(triple.tail, "id", str(triple.tail))
            if head_id in related_entities or tail_id in related_entities:
                head_name = self.id_to_name.get(head_id, str(triple.head))
                tail_name = self.id_to_name.get(tail_id, str(triple.tail))
                related_triples.append({
                    "index": i,
                    "head": head_name,
                    "relation": triple.relation,
                    "tail": tail_name,
                })
        
        return related_triples
    
    def calculate_stats(self) -> Dict[str, Any]:
        """Calculate current graph statistics."""
        return {
            "total_triples": len(self.triples),
            "total_entities": len(self.id_to_name),
            "graph_nodes": self.graph.number_of_nodes() if self.graph else 0,
            "graph_edges": self.graph.number_of_edges() if self.graph else 0,
            "triples_changed": len(self.triples) - len(self._original_triples),
            "entities_changed": len(self.id_to_name) - len(self._original_triples),  # Simplified
        }


# ============================================================================
# Agent Nodes
# ============================================================================

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
        self.app = self.workflow.compile()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph with all agent nodes."""
        workflow = StateGraph(GraphValidatorState)
        
        # Add nodes
        workflow.add_node("communicator", self._communicator_node)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("visualizer", self._visualizer_node)
        workflow.add_node("analyzer", self._analyzer_node)
        workflow.add_node("modifier", self._modifier_node)
        
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
            self._route_from_communicator,
            make_routing_map(agents_to_communicator=False)
        )
        
        workflow.add_conditional_edges(
            "retriever",
            self._route_from_retriever,
            make_routing_map(agents_to_communicator=True)
        )
        
        workflow.add_conditional_edges(
            "visualizer",
            self._route_from_visualizer,
            make_routing_map(agents_to_communicator=True)
        )
        
        workflow.add_conditional_edges(
            "analyzer",
            self._route_from_analyzer,
            make_routing_map(agents_to_communicator=True)
        )
        
        workflow.add_conditional_edges(
            "modifier",
            self._route_from_modifier,
            make_routing_map(agents_to_communicator=True)
        )
        
        return workflow
    
    # ========================================================================
    # Node Implementations
    # ========================================================================
    
    def _communicator_node(self, state: GraphValidatorState) -> GraphValidatorState:
        """
        Main communication agent - handles user messages and coordinates other agents.
        This is the entry point for all conversations.
        """
        messages = state.get("messages", [])
        current_question_id = state.get("current_question_id")
        current_question_text = state.get("current_question_text")
        questions = state.get("questions", [])
        validation_complete = state.get("validation_complete", False)
        
        # Get the last user message
        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        if not user_message and not current_question_text:
            # Initial state - generate first question
            if not questions:
                # Need to analyze first - but check if we've already tried
                # Look at messages to see if we already tried to generate questions
                recent_bot_msgs = [msg for msg in messages[-2:] if msg.get("role") == "bot"]
                if recent_bot_msgs and any("analyzing" in msg.get("content", "").lower() or "analyzed" in msg.get("content", "").lower() for msg in recent_bot_msgs):
                    # We already tried - end to prevent recursion
                    return {
                        **state,
                        "messages": messages + [{"role": "bot", "content": "I'm ready to help you validate your graph. Please ask me a question or provide feedback."}],
                        "next_agent": None,  # End conversation
                    }
                # Need to analyze first
                return {
                    **state,
                    "next_agent": "analyzer",
                }
            else:
                # Use first question
                first_q = questions[0]
                question_text = f"I'm ready to help you validate and improve your knowledge graph.\n\nLet me start by asking: {first_q.get('text', '')}"
                return {
                    **state,
                    "messages": messages + [{"role": "bot", "content": question_text}],
                    "current_question_id": first_q.get("id"),
                    "current_question_text": first_q.get("text"),
                    "next_agent": None,  # Wait for user response - None will route to END
                }
        
        # Build prompt for LLM
        prompt = self._build_communicator_prompt(state, user_message)
        
        # Call LLM
        response = self.api_repo.chat(prompt)
        response_text = self._extract_text_from_response(response)
        
        # Parse LLM response (JSON)
        try:
            response_data = json.loads(response_text.replace("```json", "").replace("```", "").strip())
        except json.JSONDecodeError:
            # Fallback: treat as plain text
            response_data = {
                "text": response_text,
                "next_agent": None,
                "hidden_actions": [],
                "next_question": None,
            }
        
        # Update state
        bot_message = response_data.get("text", response_text)
        next_agent = response_data.get("next_agent")
        hidden_actions = response_data.get("hidden_actions", [])
        next_question = response_data.get("next_question")
        
        # Determine next agent
        # Check if user explicitly asked for questions
        user_asked_for_questions = user_message and (
            "ask" in user_message.lower() and "question" in user_message.lower()
        ) or "ask questions" in user_message.lower() if user_message else False
        
        # Normalize next_agent (handle 'null' string from JSON)
        if next_agent and isinstance(next_agent, str) and next_agent.lower() == "null":
            next_agent = None
        
        if next_agent:
            agent_to_call = next_agent
        elif user_asked_for_questions:
            agent_to_call = "analyzer"  # Generate questions when requested
        elif hidden_actions:
            agent_to_call = "modifier"
        elif "retrieve" in bot_message.lower() or "get information" in bot_message.lower():
            agent_to_call = "retriever"
        elif "show" in bot_message.lower() or "display" in bot_message.lower() or "widget" in bot_message.lower():
            agent_to_call = "visualizer"
        elif next_question or not validation_complete:
            agent_to_call = "analyzer"  # Generate next question
        else:
            agent_to_call = END  # Use END constant instead of None
        
        return {
            **state,
            "messages": messages + [{"role": "bot", "content": bot_message}],
            "next_agent": agent_to_call,
            "hidden_actions": hidden_actions,
            "current_question_text": next_question,
            "validation_complete": response_data.get("validation_complete", validation_complete),
        }
    
    def _retriever_node(self, state: GraphValidatorState) -> GraphValidatorState:
        """
        Retrieval agent - fetches detailed information about entities and triples.
        """
        messages = state.get("messages", [])
        last_bot_message = None
        for msg in reversed(messages):
            if msg.get("role") == "bot":
                last_bot_message = msg.get("content", "")
                break
        
        # Parse what to retrieve from the last message
        prompt = (
            "You are a retrieval agent. Your job is to identify what information needs to be retrieved.\n\n"
            f"Last bot message: {last_bot_message}\n\n"
            "Based on the conversation, determine what needs to be retrieved:\n"
            "- Entity information (use get_entity_info)\n"
            "- Triple information (use get_triple_info)\n"
            "- Related triples (use get_related_triples)\n"
            "- Entity search (use search_entities)\n\n"
            "Return JSON with:\n"
            '{"action": "get_entity_info|get_triple_info|get_related_triples|search_entities", '
            '"parameters": {"entity_name": "...", "triple_index": 0, etc.}, '
            '"reason": "Why this information is needed"}\n'
        )
        
        response = self.api_repo.chat(prompt)
        try:
            action_data = json.loads(response.replace("```json", "").replace("```", "").strip())
        except:
            action_data = {"action": "get_entity_info", "parameters": {}, "reason": "Default"}
        
        # Execute retrieval
        action = action_data.get("action", "get_entity_info")
        params = action_data.get("parameters", {})
        
        if action == "get_entity_info":
            entity_name = params.get("entity_name", "")
            if entity_name:
                info = self.tools.get_entity_info(entity_name)
            else:
                info = {"error": "No entity name provided"}
        elif action == "get_triple_info":
            triple_index = params.get("triple_index", -1)
            info = self.tools.get_triple_info(triple_index)
        elif action == "get_related_triples":
            entity_name = params.get("entity_name", "")
            info = {"related_triples": self.tools.get_related_triples(entity_name)}
        elif action == "search_entities":
            query = params.get("query", "")
            info = {"search_results": self.tools.search_entities(query)}
        else:
            info = {"error": f"Unknown action: {action}"}
        
        # Format retrieved information for the communicator
        info_text = json.dumps(info, indent=2)
        retrieval_message = f"[Retrieved Information]\n{info_text}\n\nReason: {action_data.get('reason', 'N/A')}"
        
        return {
            **state,
            "messages": messages + [{"role": "system", "content": retrieval_message}],
            "next_agent": "communicator",  # Return to communicator with retrieved info
        }
    
    def _visualizer_node(self, state: GraphValidatorState) -> GraphValidatorState:
        """
        Visualization agent - decides which widgets to show and with what data.
        """
        messages = state.get("messages", [])
        current_question_id = state.get("current_question_id")
        # Get triples from instance, not state (to avoid serialization)
        triples = self.triples
        
        # Analyze what should be visualized
        prompt = (
            "You are a visualization agent. Your job is to decide what widgets to show.\n\n"
            f"Current question: {state.get('current_question_text', 'N/A')}\n"
            f"Recent conversation: {messages[-3:] if len(messages) >= 3 else messages}\n\n"
            "Widget types available:\n"
            "- triple_editor: Edit a specific triple\n"
            "- entity_selector: Select entities from a list\n"
            "- importance_selector: Rate importance of a triple\n"
            "- graph_viewer: Show a subgraph\n"
            "- confirmation_dialog: Confirm an action\n\n"
            "Return JSON:\n"
            '{"show_widget": true/false, '
            '"widget_type": "triple_editor|entity_selector|...", '
            '"widget_data": {"triple_index": 0, "entities": [...], etc.}, '
            '"reason": "Why this widget is needed"}\n'
        )
        
        response = self.api_repo.chat(prompt)
        try:
            widget_data = json.loads(response.replace("```json", "").replace("```", "").strip())
        except:
            widget_data = {"show_widget": False, "widget_type": None, "widget_data": {}}
        
        return {
            **state,
            "show_widget": widget_data.get("show_widget", False),
            "widget_type": widget_data.get("widget_type"),
            "widget_data": widget_data.get("widget_data", {}),
            "next_agent": "communicator",
        }
    
    def _analyzer_node(self, state: GraphValidatorState) -> GraphValidatorState:
        """
        Analysis agent - generates new questions and analyzes the graph.
        """
        # Build context
        context = self._build_context()
        
        # Generate questions using LLM
        questions = self._generate_questions(context)
        
        # Update state with new questions
        next_question = None
        question_id = None
        messages = state.get("messages", [])
        
        if questions:
            first_q = questions[0]
            if isinstance(first_q, dict):
                next_question = first_q.get("text")
                question_id = first_q.get("id")
            else:
                next_question = getattr(first_q, "text", None)
                question_id = getattr(first_q, "id", None)
            
            # Update current question
            if next_question:
                # Add question to messages if not already there
                question_msg = f"Let me ask you: {next_question}"
                # Check if this question is already in messages
                if not any(next_question in msg.get("content", "") for msg in messages):
                    messages.append({"role": "bot", "content": question_msg})
        else:
            # No questions generated - add a message and end to prevent recursion
            messages.append({"role": "bot", "content": "I've analyzed your graph. Please provide feedback or ask me questions."})
        
        return {
            **state,
            "messages": messages,
            "questions": questions if questions else [],  # Ensure questions is always a list
            "current_question_text": next_question,
            "current_question_id": question_id,
            "next_agent": None if not questions else "communicator",  # End if no questions to prevent loop
        }
    
    def _modifier_node(self, state: GraphValidatorState) -> GraphValidatorState:
        """
        Modification agent - applies graph modifications based on hidden actions.
        """
        from tools.graph.graph_validator import ActionType
        
        hidden_actions = state.get("hidden_actions", [])
        changes_summary = []
        # Get graph/triples from instance, not state (to avoid serialization issues)
        graph = self.graph
        triples = self.triples.copy()  # Work with a copy
        id_to_name = self.id_to_name.copy()  # Work with a copy
        
        # Apply each action
        for action in hidden_actions:
            action_type = action.get("type")
            params = action.get("parameters", {})
            
            try:
                if action_type == "add_triples":
                    # Add new triples
                    new_triples_data = params.get("triples", [])
                    added_count = 0
                    for triple_data in new_triples_data:
                        # Create Triple objects from data
                        head_name = triple_data.get("head", "")
                        tail_name = triple_data.get("tail", "")
                        relation = triple_data.get("relation", "")
                        
                        # Find or create entities
                        head_id = None
                        tail_id = None
                        for eid, name in id_to_name.items():
                            if name == head_name:
                                head_id = eid
                            if name == tail_name:
                                tail_id = eid
                        
                        if head_id and tail_id:
                            # Create entities and triple
                            head_entity = Entity(
                                id=head_id,
                                name=head_name,
                                label=graph.nodes[head_id].get("node_type", "UNKNOWN") if graph and graph.has_node(head_id) else "UNKNOWN"
                            )
                            tail_entity = Entity(
                                id=tail_id,
                                name=tail_name,
                                label=graph.nodes[tail_id].get("node_type", "UNKNOWN") if graph and graph.has_node(tail_id) else "UNKNOWN"
                            )
                            new_triple = Triple(head=head_entity, relation=relation, tail=tail_entity)
                            triples.append(new_triple)
                            added_count += 1
                    
                    if added_count > 0:
                        changes_summary.append(f"Added {added_count} triples")
                
                elif action_type == "delete_triples":
                    # Delete triples by index (in reverse order to maintain indices)
                    indices = sorted(params.get("triple_indices", []), reverse=True)
                    deleted_count = 0
                    for idx in indices:
                        if 0 <= idx < len(triples):
                            triples.pop(idx)
                            deleted_count += 1
                    if deleted_count > 0:
                        changes_summary.append(f"Deleted {deleted_count} triples")
                
                elif action_type == "merge_entities":
                    # Merge entities
                    entity_names = params.get("entity_names", [])
                    if len(entity_names) >= 2:
                        # Keep first, merge others into it
                        target_name = entity_names[0]
                        source_names = entity_names[1:]
                        
                        # Find IDs
                        target_id = None
                        source_ids = []
                        for eid, name in id_to_name.items():
                            if name == target_name:
                                target_id = eid
                            elif name in source_names:
                                source_ids.append(eid)
                        
                        if target_id and source_ids:
                            # Update triples to point to target
                            for triple in triples:
                                head_id = getattr(triple.head, "id", str(triple.head))
                                tail_id = getattr(triple.tail, "id", str(triple.tail))
                                
                                if head_id in source_ids:
                                    triple.head.id = target_id
                                if tail_id in source_ids:
                                    triple.tail.id = target_id
                            
                            # Remove source entities from id_to_name
                            for sid in source_ids:
                                if sid in id_to_name:
                                    del id_to_name[sid]
                            
                            changes_summary.append(f"Merged {len(source_names)} entities into '{target_name}'")
                
                elif action_type == "rename_entity":
                    old_name = params.get("old_name")
                    new_name = params.get("new_name")
                    if old_name and new_name:
                        # Find entity ID
                        for eid, name in list(id_to_name.items()):
                            if name == old_name:
                                id_to_name[eid] = new_name
                                changes_summary.append(f"Renamed '{old_name}' to '{new_name}'")
                                break
                
                elif action_type == "modify_triple":
                    triple_index = params.get("triple_index")
                    new_relation = params.get("new_relation")
                    if triple_index is not None and 0 <= triple_index < len(triples):
                        triple = triples[triple_index]
                        if new_relation:
                            triple.relation = new_relation
                            changes_summary.append(f"Modified triple {triple_index}")
                
            except Exception as e:
                changes_summary.append(f"Error applying action {action_type}: {str(e)}")
        
        # Update instance data (not state, to avoid serialization issues)
        self.triples = triples
        self.id_to_name = id_to_name
        if graph:
            self.graph = graph
        
        # Update tools with new data
        self.tools.triples = triples
        self.tools.id_to_name = id_to_name
        if graph:
            self.tools.graph = graph
        
        # Update state metadata only (for tracking)
        state["triples_count"] = len(triples)
        state["entities_count"] = len(id_to_name)
        if graph:
            state["graph_nodes_count"] = graph.number_of_nodes()
            state["graph_edges_count"] = graph.number_of_edges()
        
        # Calculate stats
        stats = self.tools.calculate_stats()
        
        return {
            **state,
            "hidden_actions": [],  # Clear after processing
            "changes_summary": state.get("changes_summary", []) + changes_summary,
            "stats": stats,
            "next_agent": "communicator",
        }
    
    # ========================================================================
    # Routing Functions
    # ========================================================================
    
    def _route_from_communicator(self, state: GraphValidatorState):
        """Route from communicator to next agent."""
        next_agent = state.get("next_agent")
        # Handle 'null' string (from JSON), None, or END constant
        if next_agent:
            # Check if it's the END constant (compare by identity, not value)
            if next_agent is END:
                return END
            # Handle string 'null'
            if isinstance(next_agent, str) and next_agent.lower() == "null":
                next_agent = None
            # Normalize agent name
            elif next_agent in ("retriever", "visualizer", "analyzer", "modifier"):
                return next_agent
        
        if state.get("validation_complete", False):
            return END
        
        # Check if we just came from analyzer (to prevent infinite loop)
        # Look at the last few messages to see if analyzer was recently called
        messages = state.get("messages", [])
        recent_bot_messages = [msg for msg in messages[-3:] if msg.get("role") == "bot"]
        
        # If no questions exist AND we haven't just tried to generate them, route to analyzer
        questions = state.get("questions", [])
        if not questions:
            # Only route to analyzer if we haven't recently tried (prevent recursion)
            # Check if the last bot message suggests we just tried to generate questions
            if not recent_bot_messages or "analyzing" not in recent_bot_messages[-1].get("content", "").lower():
                return "analyzer"
            else:
                # We just tried, but questions weren't generated - end to prevent infinite loop
                return END
        
        # Otherwise, end (wait for user response) - return END constant
        # LangGraph will handle the conversion to '__end__' internally
        return END
    
    def _route_from_retriever(self, state: GraphValidatorState) -> str:
        """Route from retriever."""
        return "communicator"
    
    def _route_from_visualizer(self, state: GraphValidatorState) -> str:
        """Route from visualizer."""
        return "communicator"
    
    def _route_from_analyzer(self, state: GraphValidatorState) -> str:
        """Route from analyzer."""
        return "communicator"
    
    def _route_from_modifier(self, state: GraphValidatorState) -> str:
        """Route from modifier."""
        return "communicator"
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _build_communicator_prompt(self, state: GraphValidatorState, user_message: Optional[str]) -> str:
        """Build prompt for the communicator agent."""
        messages = state.get("messages", [])
        current_question = state.get("current_question_text")
        # Get triples and id_to_name from instance, not state (to avoid serialization)
        triples = self.triples
        id_to_name = self.id_to_name
        
        # Build triples summary
        triples_text = ""
        for i, triple in enumerate(triples[:50]):  # Limit to 50
            head_name = getattr(triple.head, "name", str(triple.head))
            tail_name = getattr(triple.tail, "name", str(triple.tail))
            triples_text += f"  {i}. {head_name} --[{triple.relation}]--> {tail_name}\n"
        
        # Build conversation history
        conv_text = ""
        for msg in messages[-10:]:  # Last 10 messages
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")[:200]
            conv_text += f"{role}: {content}\n"
        
        prompt = (
            "You are an intelligent knowledge graph validator having a natural conversation.\n"
            "Your goal is to help validate and improve the knowledge graph.\n\n"
            "IMPORTANT: Use ONLY human-readable entity names. NEVER mention IDs, UUIDs, or hashes.\n\n"
            f"CURRENT GRAPH STATE:\n"
            f"- {len(triples)} triples\n"
            f"- {len(id_to_name)} entities\n\n"
            f"TRIPLES:\n{triples_text}\n\n"
            f"CONVERSATION HISTORY:\n{conv_text}\n\n"
        )
        
        if current_question:
            prompt += f"CURRENT QUESTION: {current_question}\n\n"
        
        if user_message:
            prompt += f"USER MESSAGE: {user_message}\n\n"
        
        prompt += (
            "YOUR CAPABILITIES:\n"
            "1. Ask questions about the graph\n"
            "2. Request information retrieval (set next_agent='retriever')\n"
            "3. Request visualization (set next_agent='visualizer')\n"
            "4. Suggest graph modifications (add to hidden_actions)\n"
            "5. Generate next question (set next_question)\n\n"
            "Return JSON:\n"
            '{\n'
            '  "text": "Your response to the user",\n'
            '  "next_agent": "retriever|visualizer|analyzer|modifier|null",\n'
            '  "hidden_actions": [...],\n'
            '  "next_question": "Optional next question",\n'
            '  "validation_complete": false\n'
            '}\n'
        )
        
        return prompt
    
    def _build_context(self) -> Dict[str, Any]:
        """Build context for analysis."""
        return {
            "num_nodes": self.graph.number_of_nodes() if self.graph else 0,
            "num_edges": self.graph.number_of_edges() if self.graph else 0,
            "num_triples": len(self.triples),
            "entities": list(self.id_to_name.keys()),
        }
    
    def _generate_questions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate validation questions using LLM."""
        prompt = (
            "You are a knowledge graph validation expert.\n\n"
            f"Graph has {context['num_nodes']} nodes, {context['num_edges']} edges, {context['num_triples']} triples.\n\n"
            "Generate 3-5 validation questions about potential issues.\n"
            "Return JSON array:\n"
            '[{"id": "q1", "text": "Question text", "category": "mistake|discrepancy|unclear", "priority": 8}]\n'
        )
        
        response = self.api_repo.chat(prompt)
        try:
            questions = json.loads(response.replace("```json", "").replace("```", "").strip())
            if not isinstance(questions, list):
                questions = [questions]
        except:
            questions = []
        
        return questions
    
    def _extract_text_from_response(self, response: Any) -> str:
        """Extract text from LLM response."""
        if isinstance(response, dict):
            return response.get("content", response.get("text", response.get("message", "")))
        return str(response) if response else ""
    
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
            initial_state: GraphValidatorState = {
                "messages": [],
                "current_question_id": None,
                "current_question_text": None,
                "questions": [],
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
            # Stream without config since we're not using checkpointing
            for state in self.app.stream(initial_state):
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
        """Initialize or update the graph/triples for validation."""
        if graph is not None:
            self.graph = graph
        if triples is not None:
            self.triples = triples
        if id_to_name is not None:
            self.id_to_name = id_to_name
        
        # Update tools
        self.tools = GraphValidatorTools(
            graph=self.graph or nx.MultiDiGraph(),
            triples=self.triples,
            id_to_name=self.id_to_name,
        )
        
        # Rebuild graph (in case structure changed)
        self.workflow = self._build_graph()
        self.app = self.workflow.compile(checkpointer=MemorySaver())

