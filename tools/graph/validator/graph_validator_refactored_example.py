"""
Example of how GraphValidator would look after refactoring to use component classes.

This is a demonstration - the actual refactoring would be done incrementally.
"""
from typing import List, Dict, Any, Optional
import networkx as nx

from tools.graph.Triple import Triple
from tools.graph.graph_validator import Question, Response, Action
from tools.api.llm_api_repo import LLmApi_Repo

# Import the component classes
from tools.graph.validator.graph_analyzer import GraphAnalyzer
from tools.graph.validator.entity_mapper import EntityMapper
from tools.graph.validator.question_manager import QuestionManager
from tools.graph.validator.question_generator import QuestionGenerator
from tools.graph.validator.conversation_manager import ConversationManager


class GraphValidatorRefactored:
    """
    Refactored GraphValidator using component classes.
    
    This class orchestrates the validation process by delegating to specialized components.
    """
    
    def __init__(self, api_repo: Optional[LLmApi_Repo] = None):
        """Initialize the graph validator with component classes."""
        self.api_repo = api_repo or LLmApi_Repo()
        
        # Graph data
        self.graph: Optional[nx.MultiDiGraph] = None
        self.triples: List[Triple] = []
        self.id_to_name: Dict[str, str] = {}
        self._original_graph: Optional[nx.MultiDiGraph] = None
        self._original_triples: List[Triple] = []
        
        # Initialize component classes
        self.analyzer: Optional[GraphAnalyzer] = None
        self.entity_mapper: Optional[EntityMapper] = None
        self.question_manager = QuestionManager()
        self.question_generator = QuestionGenerator(self.api_repo)
        self.conversation_manager = ConversationManager()
        
        # For backward compatibility
        self.responses: List[Response] = []
        self.conversation_mode: bool = False
    
    def analyze(
        self,
        graph: Optional[nx.MultiDiGraph] = None,
        triples: Optional[List[Triple]] = None,
        id_to_name: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Analyze a graph and/or triples to find issues.
        
        This method now delegates to component classes:
        - GraphAnalyzer: builds context
        - QuestionGenerator: generates questions
        - QuestionManager: stores questions
        """
        # Store original for reference
        if graph is not None:
            self._original_graph = graph.copy()
        self._original_triples = (triples or []).copy()
        
        # Update graph data
        self.graph = graph
        self.triples = triples or []
        self.id_to_name = id_to_name or {}
        
        # Initialize component classes with current data
        self.analyzer = GraphAnalyzer(
            graph=self.graph,
            triples=self.triples,
            id_to_name=self.id_to_name,
        )
        self.entity_mapper = EntityMapper(
            id_to_name=self.id_to_name,
            triples=self.triples,
        )
        
        # Clear previous questions
        self.question_manager.clear()
        self.conversation_manager.clear_all()
        
        # Build context using GraphAnalyzer
        context = self.analyzer.build_context()
        
        # Generate questions using QuestionGenerator
        questions = self.question_generator.generate_questions(context)
        
        # Store questions using QuestionManager
        self.question_manager.add_questions(questions)
        
        print(f"✅ Generated {len(questions)} questions for validation")
        
        # Initialize global conversation with a greeting and first question
        if not self.conversation_manager.get_global_history():
            initial_message = "I'm ready to help you validate and improve your knowledge graph."
            first_question = self.question_manager.get_first_question()
            if first_question:
                initial_message += f"\n\nLet me start by asking: {first_question.text}"
                print(f"✅ Initial message includes question: {first_question.text[:50]}...")
            else:
                initial_message += " I've analyzed your graph and will ask you questions about potential issues."
                print("⚠️  No questions generated - graph might be empty or have no issues")
            
            self.conversation_manager.add_global_message("bot", initial_message)
    
    def getFirstQuestion(self) -> Optional[Question]:
        """Get the first (highest priority) unanswered question."""
        return self.question_manager.get_first_question()
    
    def getAllQuestions(self) -> List[Question]:
        """Get all questions."""
        return self.question_manager.get_all_questions()
    
    def getUnansweredQuestions(self) -> List[Question]:
        """Get all unanswered questions."""
        return self.question_manager.get_unanswered_questions()
    
    def getQuestionById(self, question_id: str) -> Optional[Question]:
        """Get a question by its ID."""
        return self.question_manager.get_question_by_id(question_id)
    
    def _name_to_id(self, entity_name: str) -> Optional[str]:
        """Convert entity name to ID - delegates to EntityMapper."""
        if self.entity_mapper:
            return self.entity_mapper.name_to_id(entity_name)
        return None
    
    # Note: answerQuestion, _apply_hidden_actions, chat, etc. would also be refactored
    # to use the component classes, but those are more complex and would require
    # creating ResponseHandler and GraphModifier classes first.

