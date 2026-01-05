"""
Adapter to make GraphValidatorLangGraph compatible with the existing server interface.
"""

from typing import Dict, Any, Optional, List
import networkx as nx
from tools.graph.Triple import Triple
from tools.graph.graph_validator_langgraph import GraphValidatorLangGraph


class GraphValidatorLangGraphAdapter:
    """
    Adapter that wraps GraphValidatorLangGraph to match the interface
    expected by the existing server and frontend.
    """
    
    def __init__(
        self,
        graph: Optional[nx.MultiDiGraph] = None,
        triples: Optional[List[Triple]] = None,
        id_to_name: Optional[Dict[str, str]] = None,
        api_repo=None,
    ):
        self.validator = GraphValidatorLangGraph(
            graph=graph,
            triples=triples or [],
            id_to_name=id_to_name or {},
            api_repo=api_repo,
        )
        self.global_conversation_history = []
        self.questions = []
        self.responses = []  # For backward compatibility with original validator interface
    
    def analyze(
        self,
        graph: Optional[nx.MultiDiGraph] = None,
        triples: Optional[List[Triple]] = None,
        id_to_name: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize or update the graph/triples for validation."""
        self.validator.analyze(graph=graph, triples=triples, id_to_name=id_to_name)
        
        # Generate initial questions
        context = self.validator._build_context()
        self.questions = self.validator._generate_questions(context)
        
        # Initialize conversation history
        if not self.global_conversation_history:
            initial_message = "I'm ready to help you validate and improve your knowledge graph."
            if self.questions and len(self.questions) > 0:
                first_question = self.questions[0]
                initial_message += f"\n\nLet me start by asking: {first_question.get('text', '')}"
            self.global_conversation_history.append({
                "role": "bot",
                "content": initial_message
            })
    
    def chat(
        self,
        user_message: str,
        generate_next_question: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a user message through the LangGraph agent system.
        
        Returns a dict compatible with the existing server interface.
        """
        # Add user message to history (only if not empty)
        if user_message:
            self.global_conversation_history.append({"role": "user", "content": user_message})
        
        # Call the LangGraph validator
        try:
            # For empty message (initial state), pass empty string
            chat_message = user_message if user_message else ""
            result = self.validator.chat(chat_message)
            
            # Handle error responses
            if "error" in result:
                error_text = result.get("error", "Unknown error")
                return {
                    "text": f"I encountered an error: {error_text}",
                    "hidden_actions": [],
                    "next_question": None,
                    "validation_complete": False,
                    "actions": [],
                    "show_widget": False,
                    "widget_type": None,
                    "widget_data": {},
                    "changes_summary": [],
                    "stats": {},
                }
            
            # Add bot response to history
            bot_text = result.get("text", "Response processed.")
            self.global_conversation_history.append({"role": "bot", "content": bot_text})
            
            # Format response to match expected interface
            return {
                "text": bot_text,
                "hidden_actions": result.get("hidden_actions", []),
                "next_question": result.get("next_question"),
                "validation_complete": result.get("validation_complete", False),
                "actions": result.get("display_actions", []),
                "show_widget": result.get("show_widget", False),
                "widget_type": result.get("widget_type"),
                "widget_data": result.get("widget_data", {}),
                "changes_summary": result.get("changes_summary", []),
                "stats": result.get("stats", {}),
            }
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"Error in adapter chat: {error_msg}")
            traceback.print_exc()
            return {
                "text": f"I encountered an error processing your message: {error_msg}",
                "hidden_actions": [],
                "next_question": None,
                "validation_complete": False,
                "actions": [],
                "show_widget": False,
                "widget_type": None,
                "widget_data": {},
                "changes_summary": [],
                "stats": {},
            }
    
    def getFirstQuestion(self):
        """Get the first unanswered question (for backward compatibility)."""
        if self.questions:
            # Convert to Question-like object
            first_q = self.questions[0]
            from tools.graph.graph_validator import Question
            # Handle both dict and Question object
            if isinstance(first_q, dict):
                return Question(
                    id=first_q.get("id", "q1"),
                    text=first_q.get("text", ""),
                    category=first_q.get("category", "unclear"),
                    context=first_q.get("context", {}),
                    priority=first_q.get("priority", 5),
                )
            else:
                # Already a Question object
                return first_q
        return None
    
    def getUnansweredQuestions(self):
        """Get all unanswered questions (for backward compatibility)."""
        from tools.graph.graph_validator import Question
        unanswered = []
        for q in self.questions:
            if isinstance(q, dict):
                # Check if answered (if dict has 'answered' key)
                if not q.get("answered", False):
                    unanswered.append(Question(
                        id=q.get("id", "q1"),
                        text=q.get("text", ""),
                        category=q.get("category", "unclear"),
                        context=q.get("context", {}),
                        priority=q.get("priority", 5),
                    ))
            else:
                # Already a Question object
                if not getattr(q, "answered", False):
                    unanswered.append(q)
        return unanswered
    
    def getAllQuestions(self):
        """Get all questions (for backward compatibility)."""
        from tools.graph.graph_validator import Question
        return [
            Question(
                id=q.get("id", f"q{i}"),
                text=q.get("text", ""),
                category=q.get("category", "unclear"),
                context=q.get("context", {}),
                priority=q.get("priority", 5),
            )
            for i, q in enumerate(self.questions)
        ]
    
    def getUpdatedGraph(self) -> Optional[nx.MultiDiGraph]:
        """Get the current (possibly modified) graph."""
        return self.validator.graph
    
    def getUpdatedTriples(self) -> List[Triple]:
        """Get the current (possibly modified) triples."""
        return self.validator.triples
    
    @property
    def graph(self):
        """Access to graph for compatibility."""
        return self.validator.graph
    
    @property
    def triples(self):
        """Access to triples for compatibility."""
        return self.validator.triples
    
    @property
    def id_to_name(self):
        """Access to id_to_name for compatibility."""
        return self.validator.id_to_name
    
    def getChanges(self) -> Dict[str, Any]:
        """Get summary of changes made (for backward compatibility)."""
        changes_summary = self.global_conversation_history  # Simplified
        stats = self.validator.tools.calculate_stats() if hasattr(self.validator, 'tools') else {}
        
        return {
            "triples_added": stats.get("triples_changed", 0) if stats.get("triples_changed", 0) > 0 else 0,
            "triples_deleted": abs(stats.get("triples_changed", 0)) if stats.get("triples_changed", 0) < 0 else 0,
            "entities_merged": 0,  # Would need to track this
            "entities_renamed": 0,  # Would need to track this
            "changes_summary": getattr(self, '_changes_summary', []),
        }

