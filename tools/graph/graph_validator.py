"""
Graph Validator: Interactive QA system for finding mistakes, discrepancies, and unclear connections.
Uses LLM to analyze graphs/triples and generate questions, with support for interactive responses.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
import json
import logging
from tools.graph.langgraph.state import GraphValidatorState

# Set up logger
logger = logging.getLogger(__name__)

from tools.graph.Triple import Triple
from tools.api.llm_api_repo import LLmApi_Repo
from tools.sentence.entity import Entity

# Import shared types (to avoid circular imports)
from tools.graph.validator_types import (
    ActionType,
    Action,
    Question,
    ConversationTurn,
    Response,
)

# Import component classes (after types to avoid circular import)
from tools.graph.validator import (
    GraphAnalyzer,
    EntityMapper,
    QuestionManager,
    QuestionGenerator,
    ConversationManager,
    ResponseHandler,
    GraphModifier,
)
from tools.graph.validator.debug_utils import open_debug_browser, format_agent_output


class GraphValidator:
    """
    Validates graphs and triples by finding mistakes, discrepancies, and unclear connections.
    Uses LLM to generate questions and handle interactive responses. Can modify the graph/triples.
    """
    
    def __init__(self, api_repo: Optional[LLmApi_Repo] = None, debug: bool = False):
        """
        Initialize the graph validator.
        
        Args:
            api_repo: Optional LLM API repository (defaults to LLmApi_Repo())
            debug: If True, opens browser windows with agent outputs for debugging
        """
        logger.info("Initializing GraphValidator")
        self.api_repo = api_repo or LLmApi_Repo()
        self.debug = debug
        
        # Graph data
        self.graph: Optional[nx.MultiDiGraph] = None
        self.triples: List[Triple] = []
        self.id_to_name: Dict[str, str] = {}
        self._original_graph: Optional[nx.MultiDiGraph] = None
        self._original_triples: List[Triple] = []
        
        # Initialize component classes
        logger.debug("Initializing component classes")
        self.analyzer: Optional[GraphAnalyzer] = None
        self.entity_mapper: Optional[EntityMapper] = None
        self.question_manager = QuestionManager()
        self.question_generator = QuestionGenerator(self.api_repo, debug=debug)
        self.conversation_manager = ConversationManager()
        self.response_handler: Optional[ResponseHandler] = None
        self.graph_modifier: Optional[GraphModifier] = None
        
        # For backward compatibility - delegate to components
        self.questions: List[Question] = []  # Will sync with question_manager
        self.responses: List[Response] = []
        self.conversation_history: Dict[str, List[ConversationTurn]] = {}  # Will sync with conversation_manager
        self.global_conversation_history: List[Dict[str, str]] = []  # Will sync with conversation_manager
        self.conversation_mode: bool = False  # Whether to use flexible conversation mode
        logger.info("GraphValidator initialized successfully")
        
    def analyze(
        self,
        graph: Optional[nx.MultiDiGraph] = None,
        triples: Optional[List[Triple]] = None,
        id_to_name: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Analyze a graph and/or triples to find issues.
        
        Args:
            graph: Optional NetworkX graph to analyze
            triples: Optional list of Triple objects to analyze
            id_to_name: Optional mapping from entity ID to display name
        """
        logger.info("=" * 80)
        logger.info("Starting graph analysis")
        logger.info(f"Graph: {graph.number_of_nodes() if graph else 0} nodes, {graph.number_of_edges() if graph else 0} edges")
        logger.info(f"Triples: {len(triples) if triples else 0}")
        logger.info(f"Entity mappings: {len(id_to_name) if id_to_name else 0}")
        
        # Store original for reference
        if graph is not None:
            self._original_graph = graph.copy()
            logger.debug(f"Stored original graph: {self._original_graph.number_of_nodes()} nodes")
        self._original_triples = (triples or []).copy()
        logger.debug(f"Stored original triples: {len(self._original_triples)}")
        
        self.graph = graph
        self.triples = triples or []
        self.id_to_name = id_to_name or {}
        
        # Initialize component classes with current data
        logger.debug("Initializing component classes with graph data")
        self.analyzer = GraphAnalyzer(
            graph=self.graph,
            triples=self.triples,
            id_to_name=self.id_to_name,
        )
        self.entity_mapper = EntityMapper(
            id_to_name=self.id_to_name,
            triples=self.triples,
        )
        self.graph_modifier = GraphModifier(
            graph=self.graph,
            triples=self.triples,
            entity_mapper=self.entity_mapper,
        )
        self.response_handler = ResponseHandler(
            api_repo=self.api_repo,
            question_manager=self.question_manager,
            conversation_manager=self.conversation_manager,
            id_to_name=self.id_to_name,
            debug=self.debug,
        )
        logger.debug("Component classes initialized")
        
        # Clear previous state
        logger.debug("Clearing previous state")
        self.question_manager.clear()
        self.conversation_manager.clear_all()
        self.questions = []
        self.responses = []
        self.conversation_history = {}
        self.global_conversation_history = []
        
        # Build context using GraphAnalyzer
        logger.info("Building context for LLM")
        context = self.analyzer.build_context()
        logger.debug(f"Context built: {context['num_nodes']} nodes, {context['num_edges']} edges, {context['num_triples']} triples, {len(context['entities'])} entities")
        
        # Generate questions using QuestionGenerator
        logger.info("Generating validation questions using LLM")
        questions = self.question_generator.generate_questions(context)
        logger.info(f"Generated {len(questions)} questions")
        
        # Store questions using QuestionManager
        self.question_manager.add_questions(questions)
        self.questions = questions  # Sync for backward compatibility
        
        print(f"✅ Generated {len(questions)} questions for validation")
        logger.info(f"Generated {len(questions)} questions for validation")
        
        # Initialize global conversation with a greeting and first question
        logger.info("Initializing conversation")
        initial_message = ""
        first_question = self.question_manager.get_first_question()
        if first_question:
            initial_message += f"\n\nLet me start by asking: {first_question.text}"
            logger.info(f"First question: {first_question.text[:100]}...")
            print(f"✅ Initial message includes question: {first_question.text[:50]}...")
        else:
            initial_message += "Zero issues found thus no questions"
            logger.warning("No questions generated - graph might be empty or have no issues")
            print("⚠️  No questions generated - graph might be empty or have no issues")
        
        self.conversation_manager.add_global_message("bot", initial_message)
        self.global_conversation_history = self.conversation_manager.get_global_history()  # Sync
        logger.info("Analysis complete")
        logger.info("=" * 80)
    
    def _build_context(self) -> Dict[str, Any]:
        """Build context information for the LLM - delegates to GraphAnalyzer."""
        if self.analyzer:
            return self.analyzer.build_context()
        # Fallback if analyzer not initialized
        return {
            "num_nodes": 0,
            "num_edges": 0,
            "num_triples": len(self.triples),
            "entities": [],
            "triples_summary": [],
            "potential_issues": [],
        }
    
    def _generate_questions(self, context: Dict[str, Any]) -> List[Question]:
        """Use LLM to generate validation questions - delegates to QuestionGenerator."""
        return self.question_generator.generate_questions(context)
    
   
    def getFirstQuestion(self) -> Optional[Question]:
        """Get the first (highest priority) unanswered question - delegates to QuestionManager."""
        return self.question_manager.get_first_question()
    
    def getAllQuestions(self) -> List[Question]:
        """Get all questions - delegates to QuestionManager."""
        return self.question_manager.get_all_questions()
    
    def getUnansweredQuestions(self) -> List[Question]:
        """Get all unanswered questions - delegates to QuestionManager."""
        return self.question_manager.get_unanswered_questions()
    
    def answerQuestion(
        self,
        question_id: str,
        answer_text: str,
    ) -> Response:
        """
        Answer a question and get a response with actions - delegates to ResponseHandler.
        
        Args:
            question_id: ID of the question to answer
            answer_text: The user's answer text
            
        Returns:
            Response object with text and actions
        """
        logger.info("=" * 80)
        logger.info(f"Processing answer for question: {question_id}")
        logger.info(f"Answer text: {answer_text[:100]}{'...' if len(answer_text) > 100 else ''}")
        
        if not self.response_handler:
            logger.error("Response handler not initialized. Call analyze() first.")
            return Response(
                question_id=question_id,
                text="Response handler not initialized. Call analyze() first.",
            )
        
        # Update response handler's entity mapper if needed
        self.response_handler.entity_mapper = self.entity_mapper
        self.response_handler.id_to_name = self.id_to_name
        
        # Process answer using ResponseHandler
        logger.debug("Delegating to ResponseHandler.process_answer()")
        response_obj, updated_graph, updated_triples = self.response_handler.process_answer(
            question_id=question_id,
            answer_text=answer_text,
            apply_hidden_actions_callback=self._apply_hidden_actions,
        )
        
        # Update graph and triples if modified
        if updated_graph is not None:
            logger.info(f"Graph updated: {updated_graph.number_of_nodes()} nodes, {updated_graph.number_of_edges()} edges")
            self.graph = updated_graph
            # Update analyzer and modifier with new graph
            if self.analyzer:
                self.analyzer.graph = updated_graph
            if self.graph_modifier:
                self.graph_modifier.graph = updated_graph
        if updated_triples is not None:
            logger.info(f"Triples updated: {len(updated_triples)} triples")
            self.triples = updated_triples
            # Update analyzer and modifier with new triples
            if self.analyzer:
                self.analyzer.triples = updated_triples
            if self.graph_modifier:
                self.graph_modifier.triples = updated_triples
        
        # Sync conversation history for backward compatibility
        self.conversation_history = self.conversation_manager.conversation_history
        self.global_conversation_history = self.conversation_manager.get_global_history()
        
        # Store response
        self.responses.append(response_obj)
        logger.info(f"Response generated: {response_obj.text[:100]}{'...' if len(response_obj.text) > 100 else ''}")
        logger.info(f"Question completed: {response_obj.question_completed}")
        logger.info(f"Actions: {len(response_obj.actions)} display, {len(response_obj.hidden_actions)} hidden")
        logger.info("=" * 80)
        
        return response_obj
    
    def answerQuestion_original(
        self,
        question_id: str,
        answer_text: str,
    ) -> Response:
        """Original implementation - kept for reference."""
        # Find the question
        question = None
        for q in self.questions:
            if q.id == question_id:
                question = q
                break
        
        if not question:
            return Response(
                question_id=question_id,
                text="Question not found.",
            )
        
        # Build prompt for LLM to process the answer
        # Get entity names for user-friendly context
        context_for_llm = question.context.copy()
        # Replace entity_ids with entity_names if present
        if "entity_ids" in context_for_llm:
            entity_names = []
            for entity_id in context_for_llm.get("entity_ids", []):
                entity_name = self.id_to_name.get(entity_id, entity_id)
                entity_names.append(entity_name)
            context_for_llm["entity_names"] = entity_names
            # Don't show IDs to LLM
            context_for_llm.pop("entity_ids", None)
        
        # Get conversation history for this question
        conversation_history_text = ""
        if question_id in self.conversation_history and self.conversation_history[question_id]:
            turns = self.conversation_history[question_id]
            conversation_history_text = "\n\nCONVERSATION HISTORY FOR THIS QUESTION:\n"
            conversation_history_text += "=" * 60 + "\n"
            for turn in turns[-5:]:  # Show last 5 turns
                conversation_history_text += f"\nTurn {turn.turn_number}:\n"
                conversation_history_text += f"  User: {turn.user_answer}\n"
                conversation_history_text += f"  Bot: {turn.bot_response[:200]}{'...' if len(turn.bot_response) > 200 else ''}\n"
            conversation_history_text += "=" * 60 + "\n"
        
        prompt = (
            "You are processing a user's answer to a knowledge graph validation question.\n\n"
            "IMPORTANT: Use ONLY human-readable entity names in your responses. NEVER mention IDs, UUIDs, or hashes.\n"
            "Write in plain, user-friendly language.\n\n"
            f"QUESTION: {question.text}\n"
            f"CATEGORY: {question.category}\n"
            f"NUMBER OF PREVIOUS RESPONSES: {question.num_responses}\n"
            f"CONTEXT: {json.dumps(context_for_llm, indent=2)}\n"
            f"{conversation_history_text}\n"
            f"USER'S CURRENT ANSWER: {answer_text}\n\n"
            "Based on the user's answer, determine:\n"
            "1. What actions should be taken (show triples, highlight entities, ask follow-up, etc.)\n"
            "2. What response text to show the user (use entity names, NOT IDs)\n"
            "3. Whether this question is FULLY ANSWERED and can be marked as complete\n"
            "4. Any metadata needed for the actions\n\n"
            "QUESTION COMPLETION RULES (CRITICAL - READ CAREFULLY):\n"
            "- Set 'question_completed' to TRUE when ANY of these conditions are met:\n"
            "  1. The user explicitly confirms or acknowledges (e.g., 'yes', 'correct', 'that's right', 'confirmed', 'ok', 'understood', 'I agree')\n"
            "  2. The user has provided clear, specific information that resolves the issue\n"
            "  3. Graph modifications have been made via hidden_actions and the user's answer indicates acceptance\n"
            "  4. The user says the information is correct or accurate\n"
            "  5. After 2+ turns, if the user provides any form of confirmation or acknowledgment\n"
            "- Set 'question_completed' to FALSE ONLY when:\n"
            "  * The answer is genuinely unclear or ambiguous (not just asking for clarification)\n"
            "  * The user explicitly asks a NEW question that requires a different answer\n"
            "  * The user explicitly says they don't understand AND you haven't explained yet\n"
            "\n"
            "IMPORTANT: If the user confirms something is correct, acknowledges understanding, or provides clear information, "
            "you MUST set question_completed to TRUE. Do NOT keep asking for clarification if the user has already confirmed.\n\n"
            "Return a JSON object with this structure:\n"
            "{\n"
            '  "text": "Response text to show the user (visible)",\n'
            '  "question_completed": false,  // true if question is fully answered, false if more info needed\n'
            '  "show_widget": false,  // Whether to show a widget\n'
            '  "widget_type": "importance_selector",  // Optional widget type\n'
            '  "actions": [\n'
            '    {\n'
            '      "type": "show_triples|hide_triples|highlight_entities|ask_importance|open_widget|close_widget|show_entity_details|show_contradictions",\n'
            '      "parameters": {\n'
            '        "triple_indices": [0, 5],  // For show_triples, hide_triples\n'
            '        "entity_names": ["water tank", "air bubble generator"],  // Use entity NAMES for highlight_entities, show_entity_details\n'
            '        "triple_index": 12,  // For ask_importance\n'
            '        "widget_type": "importance_selector",  // For open_widget\n'
            '        "message": "Additional message"\n'
            '      },\n'
            '      "description": "What this action does (visible to user)"\n'
            '    }\n'
            '  ],\n'
            '  "hidden_actions": [\n'
            '    {\n'
            '      "type": "add_triples|delete_triples|modify_triple|merge_entities|delete_entity|rename_entity|change_entity_label|update_entity_properties|add_relation|remove_relation|change_relation|split_entity|create_entity|update_triple_relation|update_triple_head|update_triple_tail",\n'
            '      "parameters": {\n'
            '        "triples": [{"head": "water tank", "relation": "connects", "tail": "air bubble generator"}],  // Use entity NAMES, not IDs\n'
            '        "triple_indices": [0, 5],  // For delete_triples\n'
            '        "triple_index": 12,  // For modify_triple, update_triple_*\n'
            '        "new_relation": "connects to",  // For modify_triple, update_triple_relation, change_relation\n'
            '        "new_head": "water storage",  // Use entity NAME, not ID\n'
            '        "new_tail": "opening portion",  // Use entity NAME, not ID\n'
            '        "entity_names": ["water tank", "air bubble generator"],  // Use entity NAMES for merge_entities, split_entity\n'
            '        "target_entity_name": "water tank",  // Use entity NAME for merge_entities\n'
            '        "entity_name": "water pipe",  // Use entity NAME for delete_entity, rename_entity, etc.\n'
            '        "new_name": "New Entity Name",  // For rename_entity\n'
            '        "new_label": "COMPONENT",  // For change_entity_label\n'
            '        "properties": {"key": "value"},  // For update_entity_properties\n'
            '        "head_name": "water tank",  // Use entity NAME for add_relation, remove_relation\n'
            '        "tail_name": "air bubble generator",  // Use entity NAME for add_relation, remove_relation\n'
            '        "relation": "connects",  // For add_relation, remove_relation, change_relation\n'
            '        "split_into": ["water storage", "opening portion"],  // Use entity NAMES for split_entity\n'
            '        "entity_data": {"name": "water pipe", "label": "COMPONENT"}  // For create_entity\n'
            '      },\n'
            '      "description": "Hidden action that modifies the graph"\n'
            '    }\n'
            '  ],\n'
            '  "metadata": {\n'
            '    "triples": [{"head": "...", "relation": "...", "tail": "..."}],  // New triples to add\n'
            '    "follow_up_question": "Optional follow-up question",\n'
            '    "confidence": 0.8,\n'
            '    "next_actions": ["action1", "action2"]\n'
            '  }\n'
            '}\n\n'
            "Return ONLY the JSON object. No markdown fences, no commentary.\n"
        )
        
        try:
            response = self.api_repo.chat(prompt)
            
            # Parse response
            if isinstance(response, dict):
                response_text = response.get("content", response.get("text", response.get("message", "")))
                if not response_text and "choices" in response:
                    response_text = response["choices"][0].get("message", {}).get("content", "")
            else:
                response_text = str(response) if response else ""
            
            # Clean and parse JSON
            response_text = response_text.strip()
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            response_data = json.loads(response_text)
            
            # Parse visible actions
            actions = []
            for action_data in response_data.get("actions", []):
                try:
                    action_type = ActionType(action_data.get("type", "show_triples"))
                    actions.append(Action(
                        type=action_type,
                        parameters=action_data.get("parameters", {}),
                        description=action_data.get("description"),
                    ))
                except ValueError:
                    continue
            
            # Parse hidden actions (that modify graph)
            hidden_actions = []
            for action_data in response_data.get("hidden_actions", []):
                try:
                    action_type = ActionType(action_data.get("type", "add_triples"))
                    hidden_actions.append(Action(
                        type=action_type,
                        parameters=action_data.get("parameters", {}),
                        description=action_data.get("description"),
                    ))
                except ValueError:
                    continue
            
            # Check if question is completed
            question_completed = response_data.get("question_completed", False)
            
            # Heuristic: Check if user's answer contains confirmation phrases
            # If so, force completion even if LLM didn't set it
            answer_lower = answer_text.lower().strip()
            confirmation_phrases = [
                "yes", "correct", "right", "confirmed", "ok", "okay", "understood", 
                "i agree", "that's right", "that is correct", "exactly", "precisely",
                "sounds good", "looks good", "fine", "good", "perfect", "agreed",
                "i understand", "got it", "makes sense", "that makes sense", "clear",
                "no problem", "no issues", "no concerns", "all good", "all set"
            ]
            
            # Check if answer is a confirmation
            is_confirmation = any(phrase in answer_lower for phrase in confirmation_phrases)
            
            # Also check if answer is very short and seems like an acknowledgment
            is_short_acknowledgment = len(answer_text.strip()) < 20 and any(
                word in answer_lower for word in ["yes", "no", "ok", "okay", "sure", "fine", "good"]
            )
            
            # If user confirms and we have graph modifications, or if it's a clear confirmation
            if (is_confirmation or is_short_acknowledgment) and (hidden_actions or question.num_responses >= 1):
                if not question_completed:
                    print(f"⚠ User confirmed/acknowledged - forcing question completion")
                    question_completed = True
            
            # Create Response object
            # Check if Response class supports question_completed field
            # (handles case where module wasn't reloaded in notebook)
            response_kwargs = {
                "question_id": question_id,
                "text": response_data.get("text", "Response processed."),
                "actions": actions,
                "hidden_actions": hidden_actions,
                "metadata": response_data.get("metadata", {}),
                "show_widget": response_data.get("show_widget", False),
                "widget_type": response_data.get("widget_type"),
            }
            
            # Apply hidden actions to modify graph/triples BEFORE creating response
            # (so we can check if modifications happened and update question_completed accordingly)
            updated_graph, updated_triples, graph_was_modified = self._apply_hidden_actions(hidden_actions, response_kwargs.get("metadata", {}))
            
            # If graph was modified via hidden actions, automatically mark question as complete
            # (user has taken action, so the question is resolved)
            # Check if any action is a modification action
            if hidden_actions:
                modification_actions = {
                    ActionType.ADD_TRIPLES, ActionType.DELETE_TRIPLES, ActionType.MODIFY_TRIPLE,
                    ActionType.MERGE_ENTITIES, ActionType.DELETE_ENTITY, ActionType.RENAME_ENTITY,
                    ActionType.CHANGE_ENTITY_LABEL, ActionType.ADD_RELATION, ActionType.REMOVE_RELATION,
                    ActionType.CHANGE_RELATION, ActionType.SPLIT_ENTITY, ActionType.CREATE_ENTITY,
                    ActionType.UPDATE_TRIPLE_RELATION, ActionType.UPDATE_TRIPLE_HEAD, ActionType.UPDATE_TRIPLE_TAIL
                }
                has_modification_action = any(action.type in modification_actions for action in hidden_actions)
                
                # If we have modification actions OR graph was modified, complete the question
                if graph_was_modified or has_modification_action:
                    if not question_completed:
                        question_completed = True
                        print(f"✓ Graph modified via hidden actions ({len(hidden_actions)} actions, types={[a.type.value for a in hidden_actions]}, was_modified={graph_was_modified}) - auto-completing question")
            
            # Only add question_completed if the field exists in the Response class
            # (question_completed may have been updated by confirmation detection or graph modification above)
            if hasattr(Response, '__dataclass_fields__') and 'question_completed' in Response.__dataclass_fields__:
                response_kwargs["question_completed"] = question_completed
            
            response_obj = Response(**response_kwargs)
            
            # Update graph and triples after response object is created
            if updated_graph is not None:
                self.graph = updated_graph
            if updated_triples is not None:
                self.triples = updated_triples
            
            # Store conversation turn BEFORE storing response (so we can use response_obj.text)
            if question_id not in self.conversation_history:
                self.conversation_history[question_id] = []
            
            turn_number = len(self.conversation_history[question_id]) + 1
            self.conversation_history[question_id].append(
                ConversationTurn(
                    user_answer=answer_text,
                    bot_response=response_obj.text,
                    turn_number=turn_number,
                )
            )
            
            # Mark question as answered if LLM says it's complete or user confirmed
            # IMPORTANT: Update response object's question_completed if we set it here
            if question_completed:
                question.answered = True
                # Force update response object's question_completed field (in case it wasn't set correctly)
                if hasattr(response_obj, 'question_completed'):
                    response_obj.question_completed = True
                print(f"✓ Question '{question.text[:50]}...' marked as completed (question_completed={question_completed}, response_obj.question_completed={getattr(response_obj, 'question_completed', 'N/A')}, question.answered={question.answered})")
            else:
                question.num_responses += 1
                # Auto-complete after 2 responses if user provided clear information
                # (reduced from 3 to prevent loops)
                if question.num_responses >= 2:
                    # Check if user provided substantial information (not just "I don't understand")
                    has_substantial_info = (
                        len(answer_text.strip()) > 30 and 
                        "don't understand" not in answer_lower and
                        "don't know" not in answer_lower and
                        "unclear" not in answer_lower
                    )
                    if has_substantial_info or hidden_actions:
                        question.answered = True
                        question_completed = True
                        # Update response object if possible
                        if hasattr(response_obj, 'question_completed'):
                            response_obj.question_completed = True
                        print(f"⚠ Question '{question.text[:50]}...' auto-completed after {question.num_responses} responses (user provided info)")
                    elif question.num_responses >= 3:
                        # Force complete after 3 responses regardless
                        question.answered = True
                        question_completed = True
                        if hasattr(response_obj, 'question_completed'):
                            response_obj.question_completed = True
                        print(f"⚠ Question '{question.text[:50]}...' force-completed after 3 responses (preventing loop)")
            
            # Store response
            self.responses.append(response_obj)
            
            return response_obj
            
        except Exception as e:
            print(f"Error processing answer: {e}")
            return Response(
                question_id=question_id,
                text=f"Error processing answer: {e}",
            )
    
    def getQuestionById(self, question_id: str) -> Optional[Question]:
        """Get a question by its ID - delegates to QuestionManager."""
        return self.question_manager.get_question_by_id(question_id)
    
    def getResponsesForQuestion(self, question_id: str) -> List[Response]:
        """Get all responses for a specific question."""
        return [r for r in self.responses if r.question_id == question_id]
    
    def _name_to_id(self, entity_name: str) -> Optional[str]:
        """Convert entity name to ID - delegates to EntityMapper."""
        if self.entity_mapper:
            return self.entity_mapper.name_to_id(entity_name)
        return None
    
    def _apply_hidden_actions(
        self,
        hidden_actions: List[Action],
        metadata: Dict[str, Any],
    ) -> tuple[Optional[nx.MultiDiGraph], Optional[List[Triple]], bool]:
        """
        Apply hidden actions to modify the graph and triples - delegates to GraphModifier.
        
        Returns:
            Tuple of (updated_graph, updated_triples, was_modified) where was_modified indicates if any changes were made
        """
        logger.debug(f"Applying {len(hidden_actions)} hidden actions")
        if not self.graph_modifier:
            logger.warning("GraphModifier not initialized, skipping action application")
            return self.graph, self.triples, False
        
        # Update modifier with current graph/triples
        self.graph_modifier.graph = self.graph
        self.graph_modifier.triples = self.triples
        self.graph_modifier.entity_mapper = self.entity_mapper
        
        # Log action types
        action_types = [action.type.value for action in hidden_actions]
        logger.debug(f"Action types: {action_types}")
        
        # Apply actions
        updated_graph, updated_triples, was_modified = self.graph_modifier.apply_actions(
            hidden_actions, metadata
        )
        
        if was_modified:
            logger.info(f"Graph modified: {updated_graph.number_of_nodes() if updated_graph else 0} nodes, {len(updated_triples) if updated_triples else 0} triples")
        else:
            logger.debug("No modifications made to graph")
        
        # Update id_to_name mapping if entities were added/modified
        self.id_to_name = self.entity_mapper.id_to_name
        
        return updated_graph, updated_triples, was_modified
    
    def _apply_hidden_actions_original(
        self,
        hidden_actions: List[Action],
        metadata: Dict[str, Any],
    ) -> tuple[Optional[nx.MultiDiGraph], Optional[List[Triple]], bool]:
        """Original implementation - kept for reference."""
        graph_modified = False
        triples_modified = False
        updated_graph = self.graph.copy() if self.graph is not None else None
        updated_triples = self.triples.copy()
        
        for action in hidden_actions:
            # Convert entity names to IDs for internal processing (LLM uses names, we need IDs)
            if "entity_name" in action.parameters:
                entity_name = action.parameters["entity_name"]
                entity_id = self._name_to_id(entity_name)
                if entity_id:
                    action.parameters["entity_id"] = entity_id
            
            if "entity_names" in action.parameters:
                entity_names = action.parameters["entity_names"]
                entity_ids = []
                for name in entity_names:
                    eid = self._name_to_id(name)
                    if eid:
                        entity_ids.append(eid)
                if entity_ids:
                    action.parameters["entity_ids"] = entity_ids
            
            if "target_entity_name" in action.parameters:
                target_name = action.parameters["target_entity_name"]
                target_id = self._name_to_id(target_name)
                if target_id:
                    action.parameters["target_entity_id"] = target_id
            
            if "head_name" in action.parameters:
                head_name = action.parameters["head_name"]
                head_id = self._name_to_id(head_name)
                if head_id:
                    action.parameters["head_id"] = head_id
            
            if "tail_name" in action.parameters:
                tail_name = action.parameters["tail_name"]
                tail_id = self._name_to_id(tail_name)
                if tail_id:
                    action.parameters["tail_id"] = tail_id
            
            if "new_head" in action.parameters and isinstance(action.parameters["new_head"], str):
                new_head_name = action.parameters["new_head"]
                new_head_id = self._name_to_id(new_head_name)
                if new_head_id:
                    action.parameters["new_head"] = new_head_id
            
            if "new_tail" in action.parameters and isinstance(action.parameters["new_tail"], str):
                new_tail_name = action.parameters["new_tail"]
                new_tail_id = self._name_to_id(new_tail_name)
                if new_tail_id:
                    action.parameters["new_tail"] = new_tail_id
            
            if "split_into" in action.parameters:
                split_names = action.parameters["split_into"]
                if isinstance(split_names, list) and split_names and isinstance(split_names[0], str):
                    split_ids = []
                    for name in split_names:
                        eid = self._name_to_id(name)
                        if eid:
                            split_ids.append(eid)
                    if split_ids:
                        action.parameters["split_into"] = split_ids
            if action.type == ActionType.ADD_TRIPLES:
                # Add new triples from metadata or parameters
                new_triples_data = action.parameters.get("triples", [])
                if not new_triples_data:
                    new_triples_data = metadata.get("triples", [])
                
                for triple_data in new_triples_data:
                    if not isinstance(triple_data, dict):
                        continue
                    
                    # LLM provides entity names, convert to IDs
                    head_name = triple_data.get("head")
                    tail_name = triple_data.get("tail")
                    relation = triple_data.get("relation")
                    
                    if not all([head_name, tail_name, relation]):
                        continue
                    
                    # Convert names to IDs (or use name as ID if not found)
                    head_id = self._name_to_id(head_name) or head_name
                    tail_id = self._name_to_id(tail_name) or tail_name
                    
                    # Try to find entities in the graph or create new ones
                    from tools.sentence.entity import Entity
                    import uuid
                    
                    # Get or create head entity
                    if updated_graph and updated_graph.has_node(head_id):
                        node_data = updated_graph.nodes[head_id]
                        head_ent = Entity(
                            id=head_id,
                            name=self.id_to_name.get(head_id, head_name),
                            label=node_data.get("node_type", "UNKNOWN"),
                            ref_short=head_id[-4:] if len(head_id) >= 4 else head_id,
                        )
                    else:
                        # Create new entity with the name provided by LLM
                        head_ent = Entity(
                            id=head_id,
                            name=head_name,  # Use the name from LLM, not the ID
                            label="UNKNOWN",
                            ref_short=str(uuid.uuid4())[-4:],
                        )
                        if updated_graph:
                            updated_graph.add_node(head_id, node_type="UNKNOWN", name=head_name)
                        # Update id_to_name mapping
                        self.id_to_name[head_id] = head_name
                    
                    # Get or create tail entity
                    if updated_graph and updated_graph.has_node(tail_id):
                        node_data = updated_graph.nodes[tail_id]
                        tail_ent = Entity(
                            id=tail_id,
                            name=self.id_to_name.get(tail_id, tail_name),
                            label=node_data.get("node_type", "UNKNOWN"),
                            ref_short=tail_id[-4:] if len(tail_id) >= 4 else tail_id,
                        )
                    else:
                        # Create new entity with the name provided by LLM
                        tail_ent = Entity(
                            id=tail_id,
                            name=tail_name,  # Use the name from LLM, not the ID
                            label="UNKNOWN",
                            ref_short=str(uuid.uuid4())[-4:],
                        )
                        if updated_graph:
                            updated_graph.add_node(tail_id, node_type="UNKNOWN", name=tail_name)
                        # Update id_to_name mapping
                        self.id_to_name[tail_id] = tail_name
                        # Update id_to_name mapping
                        self.id_to_name[tail_id] = tail_name
                    
                    # Create triple
                    new_triple = Triple(head=head_ent, relation=relation, tail=tail_ent)
                    updated_triples.append(new_triple)
                    triples_modified = True
                    
                    # Add edge to graph if graph exists
                    if updated_graph:
                        updated_graph.add_edge(head_id, tail_id, label=relation)
                        graph_modified = True
            
            elif action.type == ActionType.DELETE_TRIPLES:
                # Delete triples by indices
                triple_indices = action.parameters.get("triple_indices", [])
                if triple_indices:
                    # Sort indices in reverse to delete from end
                    for idx in sorted(triple_indices, reverse=True):
                        if 0 <= idx < len(updated_triples):
                            triple = updated_triples[idx]
                            # Remove from graph if it exists
                            if updated_graph:
                                head_id = getattr(triple.head, "ref", None) or getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None)
                                tail_id = getattr(triple.tail, "ref", None) or getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None)
                                if head_id and tail_id and updated_graph.has_edge(head_id, tail_id):
                                    updated_graph.remove_edge(head_id, tail_id)
                                    graph_modified = True
                            
                            del updated_triples[idx]
                            triples_modified = True
            
            elif action.type == ActionType.MODIFY_TRIPLE:
                # Modify an existing triple (change relation, head, or tail)
                triple_index = action.parameters.get("triple_index")
                new_relation = action.parameters.get("new_relation")
                new_head = action.parameters.get("new_head")
                new_tail = action.parameters.get("new_tail")
                
                if triple_index is not None and 0 <= triple_index < len(updated_triples):
                    triple = updated_triples[triple_index]
                    old_head_id = getattr(triple.head, "ref", None) or getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None)
                    old_tail_id = getattr(triple.tail, "ref", None) or getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None)
                    
                    # Update relation
                    if new_relation:
                        triple.relation = new_relation
                        triples_modified = True
                        if updated_graph and old_head_id and old_tail_id:
                            # Update edge label
                            if updated_graph.has_edge(old_head_id, old_tail_id):
                                edge_data = updated_graph.get_edge_data(old_head_id, old_tail_id)
                                if edge_data:
                                    for key in list(edge_data.keys()):
                                        updated_graph[old_head_id][old_tail_id][key]["label"] = new_relation
                                    graph_modified = True
                    
                    # Update head
                    if new_head:
                        from tools.sentence.entity import Entity
                        if isinstance(new_head, str):
                            # Find or create entity
                            new_head_ent = self._get_or_create_entity(new_head, updated_graph)
                            triple.head = new_head_ent
                            triples_modified = True
                            
                            if updated_graph and old_head_id and old_tail_id:
                                # Update graph edge
                                if updated_graph.has_edge(old_head_id, old_tail_id):
                                    edge_data = updated_graph.get_edge_data(old_head_id, old_tail_id)
                                    if edge_data:
                                        for key, data in edge_data.items():
                                            updated_graph.add_edge(new_head_ent.id, old_tail_id, key=key, **data)
                                        updated_graph.remove_edge(old_head_id, old_tail_id)
                                        graph_modified = True
                    
                    # Update tail
                    if new_tail:
                        from tools.sentence.entity import Entity
                        if isinstance(new_tail, str):
                            # Find or create entity
                            new_tail_ent = self._get_or_create_entity(new_tail, updated_graph)
                            triple.tail = new_tail_ent
                            triples_modified = True
                            
                            if updated_graph:
                                head_id = getattr(triple.head, "ref", None) or getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None)
                                if head_id and old_tail_id:
                                    # Update graph edge
                                    if updated_graph.has_edge(head_id, old_tail_id):
                                        edge_data = updated_graph.get_edge_data(head_id, old_tail_id)
                                        if edge_data:
                                            for key, data in edge_data.items():
                                                updated_graph.add_edge(head_id, new_tail_ent.id, key=key, **data)
                                            updated_graph.remove_edge(head_id, old_tail_id)
                                            graph_modified = True
            
            elif action.type == ActionType.MERGE_ENTITIES:
                # Merge entities (merge source into target)
                entity_ids = action.parameters.get("entity_ids", [])
                target_entity_id = action.parameters.get("target_entity_id")
                
                if len(entity_ids) >= 2 and target_entity_id:
                    source_ids = [eid for eid in entity_ids if eid != target_entity_id]
                    
                    if updated_graph:
                        for source_id in source_ids:
                            if updated_graph.has_node(source_id):
                                # Redirect all edges from source to target
                                for successor in list(updated_graph.successors(source_id)):
                                    edge_data = updated_graph.get_edge_data(source_id, successor)
                                    if edge_data:
                                        for key, data in edge_data.items():
                                            updated_graph.add_edge(target_entity_id, successor, key=key, **data)
                                
                                for predecessor in list(updated_graph.predecessors(source_id)):
                                    edge_data = updated_graph.get_edge_data(predecessor, source_id)
                                    if edge_data:
                                        for key, data in edge_data.items():
                                            updated_graph.add_edge(predecessor, target_entity_id, key=key, **data)
                                
                                # Remove source node
                                updated_graph.remove_node(source_id)
                                graph_modified = True
                    
                    # Update triples to use target entity
                    for triple in updated_triples:
                        head_id = getattr(triple.head, "ref", None) or getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None)
                        tail_id = getattr(triple.tail, "ref", None) or getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None)
                        
                        if head_id in source_ids:
                            triple.head.ref = target_entity_id
                            triple.head.ref_short = target_entity_id[-4:] if len(target_entity_id) >= 4 else target_entity_id
                            triples_modified = True
                        
                        if tail_id in source_ids:
                            triple.tail.ref = target_entity_id
                            triple.tail.ref_short = target_entity_id[-4:] if len(target_entity_id) >= 4 else target_entity_id
                            triples_modified = True
            
            elif action.type == ActionType.DELETE_ENTITY:
                # Delete entity and all connected triples
                entity_id = action.parameters.get("entity_id")
                
                if entity_id:
                    # Remove from graph
                    if updated_graph and updated_graph.has_node(entity_id):
                        updated_graph.remove_node(entity_id)
                        graph_modified = True
                    
                    # Remove all triples involving this entity
                    triples_to_remove = []
                    for i, triple in enumerate(updated_triples):
                        head_id = getattr(triple.head, "ref", None) or getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None)
                        tail_id = getattr(triple.tail, "ref", None) or getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None)
                        
                        if head_id == entity_id or tail_id == entity_id:
                            triples_to_remove.append(i)
                    
                    # Remove in reverse order
                    for idx in sorted(triples_to_remove, reverse=True):
                        del updated_triples[idx]
                        triples_modified = True
            
            elif action.type == ActionType.RENAME_ENTITY:
                # Rename entity (change name/label)
                entity_id = action.parameters.get("entity_id")
                new_name = action.parameters.get("new_name")
                
                if entity_id and new_name:
                    # Update in graph node data
                    if updated_graph and updated_graph.has_node(entity_id):
                        updated_graph.nodes[entity_id]["name"] = new_name
                        graph_modified = True
                    
                    # Update id_to_name mapping
                    self.id_to_name[entity_id] = new_name
                    
                    # Update in triples
                    for triple in updated_triples:
                        head_id = getattr(triple.head, "ref", None) or getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None)
                        tail_id = getattr(triple.tail, "ref", None) or getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None)
                        
                        if head_id == entity_id:
                            triple.head.name = new_name
                            triples_modified = True
                        
                        if tail_id == entity_id:
                            triple.tail.name = new_name
                            triples_modified = True
            
            elif action.type == ActionType.CHANGE_ENTITY_LABEL:
                # Change entity type/label
                entity_id = action.parameters.get("entity_id")
                new_label = action.parameters.get("new_label")
                
                if entity_id and new_label:
                    # Update in graph node data
                    if updated_graph and updated_graph.has_node(entity_id):
                        updated_graph.nodes[entity_id]["node_type"] = new_label
                        graph_modified = True
                    
                    # Update in triples
                    for triple in updated_triples:
                        head_id = getattr(triple.head, "ref", None) or getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None)
                        tail_id = getattr(triple.tail, "ref", None) or getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None)
                        
                        if head_id == entity_id:
                            triple.head.label = new_label
                            triples_modified = True
                        
                        if tail_id == entity_id:
                            triple.tail.label = new_label
                            triples_modified = True
            
            elif action.type == ActionType.UPDATE_ENTITY_PROPERTIES:
                # Update entity metadata/properties
                entity_id = action.parameters.get("entity_id")
                properties = action.parameters.get("properties", {})
                
                if entity_id and properties:
                    # Update in graph node data
                    if updated_graph and updated_graph.has_node(entity_id):
                        for key, value in properties.items():
                            updated_graph.nodes[entity_id][key] = value
                        graph_modified = True
            
            elif action.type == ActionType.ADD_RELATION:
                # Add relation between existing entities
                head_id = action.parameters.get("head_id")
                tail_id = action.parameters.get("tail_id")
                relation = action.parameters.get("relation")
                
                if head_id and tail_id and relation:
                    # Add to graph
                    if updated_graph:
                        updated_graph.add_edge(head_id, tail_id, label=relation)
                        graph_modified = True
                    
                    # Add to triples
                    from tools.sentence.entity import Entity
                    head_ent = self._get_or_create_entity(head_id, updated_graph)
                    tail_ent = self._get_or_create_entity(tail_id, updated_graph)
                    
                    new_triple = Triple(head=head_ent, relation=relation, tail=tail_ent)
                    updated_triples.append(new_triple)
                    triples_modified = True
            
            elif action.type == ActionType.REMOVE_RELATION:
                # Remove specific relation edge
                head_id = action.parameters.get("head_id")
                tail_id = action.parameters.get("tail_id")
                relation = action.parameters.get("relation")
                
                if head_id and tail_id:
                    # Remove from graph
                    if updated_graph and updated_graph.has_edge(head_id, tail_id):
                        if relation:
                            # Remove specific relation
                            edge_data = updated_graph.get_edge_data(head_id, tail_id)
                            if edge_data:
                                for key, data in edge_data.items():
                                    if data.get("label") == relation:
                                        updated_graph.remove_edge(head_id, tail_id, key)
                                        graph_modified = True
                        else:
                            # Remove all edges between these nodes
                            updated_graph.remove_edge(head_id, tail_id)
                            graph_modified = True
                    
                    # Remove from triples
                    triples_to_remove = []
                    for i, triple in enumerate(updated_triples):
                        t_head_id = getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None)
                        t_tail_id = getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None)
                        
                        if t_head_id == head_id and t_tail_id == tail_id:
                            if not relation or triple.relation == relation:
                                triples_to_remove.append(i)
                    
                    for idx in sorted(triples_to_remove, reverse=True):
                        del updated_triples[idx]
                        triples_modified = True
            
            elif action.type == ActionType.CHANGE_RELATION:
                # Change relation name for specific edge
                head_id = action.parameters.get("head_id")
                tail_id = action.parameters.get("tail_id")
                new_relation = action.parameters.get("new_relation")
                
                if head_id and tail_id and new_relation:
                    # Update in graph
                    if updated_graph and updated_graph.has_edge(head_id, tail_id):
                        edge_data = updated_graph.get_edge_data(head_id, tail_id)
                        if edge_data:
                            for key in list(edge_data.keys()):
                                updated_graph[head_id][tail_id][key]["label"] = new_relation
                            graph_modified = True
                    
                    # Update in triples
                    for triple in updated_triples:
                        t_head_id = getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None)
                        t_tail_id = getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None)
                        
                        if t_head_id == head_id and t_tail_id == tail_id:
                            triple.relation = new_relation
                            triples_modified = True
            
            elif action.type == ActionType.SPLIT_ENTITY:
                # Split one entity into multiple entities
                entity_id = action.parameters.get("entity_id")
                split_into = action.parameters.get("split_into", [])
                
                if entity_id and len(split_into) >= 2:
                    # Distribute edges to new entities
                    if updated_graph and updated_graph.has_node(entity_id):
                        # Get all edges
                        successors = list(updated_graph.successors(entity_id))
                        predecessors = list(updated_graph.predecessors(entity_id))
                        
                        # Distribute to new entities (round-robin)
                        for i, successor in enumerate(successors):
                            target_id = split_into[i % len(split_into)]
                            edge_data = updated_graph.get_edge_data(entity_id, successor)
                            if edge_data:
                                for key, data in edge_data.items():
                                    updated_graph.add_edge(target_id, successor, key=key, **data)
                        
                        for i, predecessor in enumerate(predecessors):
                            target_id = split_into[i % len(split_into)]
                            edge_data = updated_graph.get_edge_data(predecessor, entity_id)
                            if edge_data:
                                for key, data in edge_data.items():
                                    updated_graph.add_edge(predecessor, target_id, key=key, **data)
                        
                        # Remove original entity
                        updated_graph.remove_node(entity_id)
                        graph_modified = True
                    
                    # Update triples (distribute to new entities)
                    for triple in updated_triples:
                        head_id = getattr(triple.head, "ref", None) or getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None)
                        tail_id = getattr(triple.tail, "ref", None) or getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None)
                        
                        if head_id == entity_id:
                            # Distribute to new entities
                            target_id = split_into[0]  # Use first split entity
                            triple.head.ref = target_id
                            triple.head.ref_short = target_id[-4:] if len(target_id) >= 4 else target_id
                            triples_modified = True
                        
                        if tail_id == entity_id:
                            # Distribute to new entities
                            target_id = split_into[0]  # Use first split entity
                            triple.tail.ref = target_id
                            triple.tail.ref_short = target_id[-4:] if len(target_id) >= 4 else target_id
                            triples_modified = True
            
            elif action.type == ActionType.CREATE_ENTITY:
                # Create new entity
                entity_data = action.parameters.get("entity_data", {})
                entity_id = action.parameters.get("entity_id") or entity_data.get("id")
                
                if entity_data or entity_id:
                    from tools.sentence.entity import Entity
                    import uuid
                    
                    if not entity_id:
                        entity_id = str(uuid.uuid4())
                    
                    name = entity_data.get("name", entity_id)
                    label = entity_data.get("label", "UNKNOWN")
                    
                    new_entity = Entity(
                        id=entity_id,
                        name=name,
                        label=label,
                        ref_short=entity_id[-4:] if len(entity_id) >= 4 else entity_id,
                    )
                    
                    # Add to graph
                    if updated_graph:
                        updated_graph.add_node(entity_id, node_type=label, name=name)
                        graph_modified = True
                    
                    # Update id_to_name
                    self.id_to_name[entity_id] = name
            
            elif action.type == ActionType.UPDATE_TRIPLE_RELATION:
                # Update only the relation of a triple
                triple_index = action.parameters.get("triple_index")
                new_relation = action.parameters.get("new_relation")
                
                if triple_index is not None and 0 <= triple_index < len(updated_triples) and new_relation:
                    triple = updated_triples[triple_index]
                    triple.relation = new_relation
                    triples_modified = True
                    
                    # Update graph edge
                    if updated_graph:
                        head_id = getattr(triple.head, "ref", None) or getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None)
                        tail_id = getattr(triple.tail, "ref", None) or getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None)
                        if head_id and tail_id and updated_graph.has_edge(head_id, tail_id):
                            edge_data = updated_graph.get_edge_data(head_id, tail_id)
                            if edge_data:
                                for key in list(edge_data.keys()):
                                    updated_graph[head_id][tail_id][key]["label"] = new_relation
                                graph_modified = True
            
            elif action.type == ActionType.UPDATE_TRIPLE_HEAD:
                # Update only the head entity of a triple
                triple_index = action.parameters.get("triple_index")
                new_head = action.parameters.get("new_head")
                
                if triple_index is not None and 0 <= triple_index < len(updated_triples) and new_head:
                    triple = updated_triples[triple_index]
                    old_head_id = getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None)
                    tail_id = getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None)
                    
                    new_head_ent = self._get_or_create_entity(new_head, updated_graph)
                    triple.head = new_head_ent
                    triples_modified = True
                    
                    # Update graph edge
                    if updated_graph and old_head_id and tail_id:
                        if updated_graph.has_edge(old_head_id, tail_id):
                            edge_data = updated_graph.get_edge_data(old_head_id, tail_id)
                            if edge_data:
                                for key, data in edge_data.items():
                                    updated_graph.add_edge(new_head_ent.id, tail_id, key=key, **data)
                                updated_graph.remove_edge(old_head_id, tail_id)
                                graph_modified = True
            
            elif action.type == ActionType.UPDATE_TRIPLE_TAIL:
                # Update only the tail entity of a triple
                triple_index = action.parameters.get("triple_index")
                new_tail = action.parameters.get("new_tail")
                
                if triple_index is not None and 0 <= triple_index < len(updated_triples) and new_tail:
                    triple = updated_triples[triple_index]
                    head_id = getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None)
                    old_tail_id = getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None)
                    
                    new_tail_ent = self._get_or_create_entity(new_tail, updated_graph)
                    triple.tail = new_tail_ent
                    triples_modified = True
                    
                    # Update graph edge
                    if updated_graph and head_id and old_tail_id:
                        if updated_graph.has_edge(head_id, old_tail_id):
                            edge_data = updated_graph.get_edge_data(head_id, old_tail_id)
                            if edge_data:
                                for key, data in edge_data.items():
                                    updated_graph.add_edge(head_id, new_tail_ent.id, key=key, **data)
                                updated_graph.remove_edge(head_id, old_tail_id)
                                graph_modified = True
        
        if graph_modified or triples_modified:
            was_modified = graph_modified or triples_modified
            return updated_graph, updated_triples, was_modified
        
        return None, None, False
    
    def _get_or_create_entity(self, entity_id: str, graph: Optional[nx.MultiDiGraph]) -> Entity:
        """
        Get existing entity or create a new one.
        
        Args:
            entity_id: Entity ID
            graph: Optional graph to check for entity
            
        Returns:
            Entity object
        """
        from tools.sentence.entity import Entity
        import uuid
        
        # Check if entity exists in graph
        if graph and graph.has_node(entity_id):
            node_data = graph.nodes[entity_id]
            return Entity(
                id=entity_id,
                name=self.id_to_name.get(entity_id, node_data.get("name", entity_id)),
                label=node_data.get("node_type", "UNKNOWN"),
                ref_short=entity_id[-4:] if len(entity_id) >= 4 else entity_id,
            )
        
        # Check if we have name mapping
        name = self.id_to_name.get(entity_id, entity_id)
        
        # Create new entity
        return Entity(
            id=entity_id,
            name=name,
            label="UNKNOWN",
            ref_short=entity_id[-4:] if len(entity_id) >= 4 else entity_id,
        )
    
    def getUpdatedGraph(self) -> Optional[nx.MultiDiGraph]:
        """Get the current (possibly modified) graph."""
        return self.graph
    
    def getUpdatedTriples(self) -> List[Triple]:
        """Get the current (possibly modified) triples."""
        return self.triples
    
    def chat(
        self,
        user_message: str,
        generate_next_question: bool = True,
    ) -> Dict[str, Any]:
        """
        Flexible conversation mode: LLM maintains full context and decides what to do next.
        Can ask new questions, modify graph, or continue conversation.
        
        Args:
            user_message: User's message/answer
            generate_next_question: Whether LLM should generate next question if current topic is resolved
            
        Returns:
            Dict with:
                - "text": Response text
                - "hidden_actions": Actions to modify graph
                - "next_question": Optional new question to ask
                - "validation_complete": Whether validation is done
                - "actions": Display actions
        """
        logger.info("=" * 80)
        logger.info("CHAT: Processing user message")
        logger.info(f"User message: {user_message[:200]}{'...' if len(user_message) > 200 else ''}")
        logger.info(f"Generate next question: {generate_next_question}")
        
        # Initialize global_conversation_history if it doesn't exist (for backward compatibility)
        if not hasattr(self, 'global_conversation_history'):
            logger.debug("Initializing global_conversation_history")
            self.global_conversation_history = []
        
        # Add user message to global history
        logger.debug("Adding user message to conversation history")
        self.global_conversation_history.append({"role": "user", "content": user_message})
        
        # Build full context for LLM
        logger.info("Building context for LLM")
        context = self._build_context()
        logger.debug(f"Context: {context['num_nodes']} nodes, {context['num_edges']} edges, {context['num_triples']} triples, {len(context['entities'])} entities")
        
        # Build conversation history text
        logger.debug("Building conversation history text")
        conversation_text = ""
        if self.global_conversation_history:
            conversation_text = "\n\nFULL CONVERSATION HISTORY:\n"
            conversation_text += "=" * 60 + "\n"
            for i, msg in enumerate(self.global_conversation_history[-10:], 1):  # Last 10 messages
                role = msg["role"].upper()
                content = msg["content"][:300] + "..." if len(msg["content"]) > 300 else msg["content"]
                conversation_text += f"\n{i}. {role}: {content}\n"
            conversation_text += "=" * 60 + "\n"
        logger.debug(f"Conversation history: {len(self.global_conversation_history)} messages")
        
        # Build triples summary (user-friendly, no IDs)
        logger.debug("Building triples summary")
        triples_summary = []
        for i, triple in enumerate(self.triples[:100]):  # Limit to 100 for prompt size
            head_name = getattr(triple.head, "name", str(triple.head))
            tail_name = getattr(triple.tail, "name", str(triple.tail))
            triples_summary.append({
                "index": i,
                "head": head_name,
                "relation": triple.relation,
                "tail": tail_name,
            })
        logger.debug(f"Triples summary: {len(triples_summary)} triples")
        
        # Build entities summary with properties
        logger.debug("Building entities summary")
        entities_summary = []
        for entity_id, entity_name in list(self.id_to_name.items())[:100]:  # Limit to 100
            entity_info = {"name": entity_name, "id": entity_id}
            
            # Get entity properties from graph if available
            if self.graph and self.graph.has_node(entity_id):
                node_data = self.graph.nodes[entity_id]
                entity_info["label"] = node_data.get("node_type", "UNKNOWN")
                entity_info["properties"] = {k: v for k, v in node_data.items() 
                                            if k not in ("node_type", "name") and not k.startswith("_")}
            
            entities_summary.append(entity_info)
        logger.debug(f"Entities summary: {len(entities_summary)} entities")
        
        logger.info("Building LLM prompt for chat")
        prompt = (
            "You are an intelligent knowledge graph validator having a natural conversation with a user.\n"
            "Your goal is to help validate and improve the knowledge graph through conversation.\n\n"
            "IMPORTANT: Use ONLY human-readable entity names. NEVER mention IDs, UUIDs, or hashes.\n"
            "Write in plain, user-friendly, conversational language.\n\n"
            f"CURRENT GRAPH STATE:\n"
            f"- {context['num_nodes']} nodes, {context['num_edges']} edges\n"
            f"- {context['num_triples']} triples\n"
            f"- {len(context['entities'])} entities\n\n"
            f"ALL TRIPLES (showing only names, no IDs):\n"
        )
        
        # Show ALL triples (or at least more of them)
        for triple_info in triples_summary[:100]:  # Increased from 50 to 100
            prompt += f"  {triple_info['index']}. {triple_info['head']} --[{triple_info['relation']}]--> {triple_info['tail']}\n"
        
        if len(triples_summary) > 100:
            prompt += f"  ... and {len(triples_summary) - 100} more triples\n"
        
        prompt += f"\nENTITIES AND THEIR PROPERTIES:\n"
        for entity_info in entities_summary[:50]:  # Show first 50 entities with properties
            props_text = ""
            if entity_info.get("properties"):
                props_text = f" (Properties: {', '.join([f'{k}={v}' for k, v in list(entity_info['properties'].items())[:3]])})"
            prompt += f"  - {entity_info['name']} (Type: {entity_info.get('label', 'UNKNOWN')}){props_text}\n"
        
        if len(entities_summary) > 50:
            prompt += f"  ... and {len(entities_summary) - 50} more entities\n"
        
        prompt += f"{conversation_text}\n"
        prompt += (
            "YOUR CAPABILITIES (YOU HAVE FULL FREEDOM):\n"
            "1. Ask questions about the graph to find issues\n"
            "2. Modify the graph proactively via hidden_actions to improve it:\n"
            "   - Add missing triples that should exist\n"
            "   - Delete incorrect or redundant triples\n"
            "   - Modify triples (change relations, entities)\n"
            "   - Merge duplicate or similar entities\n"
            "   - Split entities that represent multiple concepts\n"
            "   - Rename entities for clarity\n"
            "   - Change entity labels/types\n"
            "   - Create new entities if needed\n"
            "   - Remove entities that don't make sense\n"
            "3. Continue the conversation naturally\n"
            "4. Decide when to move to a new topic or when validation is complete\n\n"
            "IMPORTANT: BE PROACTIVE AND IMPROVE CLARITY!\n"
            "- Don't just ask questions - if you see an obvious issue, fix it via hidden_actions\n"
            "- Even when the user says something is 'correct', you can still improve clarity:\n"
            "  * Rename entities to be more descriptive or clear\n"
            "  * Clarify relation names (e.g., 'connects' -> 'connects to', 'has' -> 'contains')\n"
            "  * Add missing triples that logically should exist\n"
            "  * Merge entities that are duplicates or very similar\n"
            "  * Split entities that represent multiple concepts\n"
            "  * Change entity labels/types for better categorization\n"
            "- If you notice missing connections that should exist, add them\n"
            "- If entity names are unclear or ambiguous, rename them for clarity\n"
            "- If relations are vague or could be more specific, improve them\n"
            "- Quality over quantity: Make meaningful improvements, not many small changes\n"
            "- You have full freedom to improve the graph - use it proactively!\n"
            "- Think: 'Even if this is correct, can I make it clearer or more precise?'\n\n"
            "DECISION LOGIC:\n"
            "- ALWAYS ask questions about the graph - be proactive in finding issues\n"
            "- If the user asks you to ask questions, IMMEDIATELY generate and ask a specific question about the graph\n"
            "- After each user response, analyze the current graph state (which may have changed)\n"
            "- If the user says something is 'correct' or confirms something, you can still:\n"
            "  * Improve clarity (rename entities, clarify relations)\n"
            "  * Merge duplicate entities they mentioned\n"
            "  * Add missing connections\n"
            "  * Then ask a NEW question about a different issue\n"
            "- If you made changes, check if they introduced new issues\n"
            "- If the user's message resolves a topic, you MUST:\n"
            "  * Modify the graph proactively if you see improvements needed (via hidden_actions)\n"
            "  * Ask a NEW question about a different issue (if generate_next_question=true)\n"
            "  * Don't just wait - be proactive!\n"
            "- After making changes, always check if new issues were introduced\n"
            "- If validation seems complete, set validation_complete=true\n"
            "- Be natural and conversational - don't be rigid\n"
            "- Think like a human expert reviewing the graph - what would you fix?\n"
            "- IMPORTANT: If the user asks you to ask questions, immediately generate and ask a question about the graph\n\n"
            "Return a JSON object:\n"
            "{\n"
            '  "text": "Your response to the user (conversational, friendly)",\n'
            '  "hidden_actions": [\n'
            '    {\n'
            '      "type": "add_triples|delete_triples|modify_triple|merge_entities|delete_entity|rename_entity|change_entity_label|...",\n'
            '      "parameters": {\n'
            '        "triple_indices": [0, 5],  // For delete_triples\n'
            '        "triples": [{"head": "water tank", "relation": "connects", "tail": "air bubble generator"}],  // Use entity NAMES\n'
            '        "entity_names": ["water tank", "air bubble generator"],  // Use entity NAMES\n'
            '        // ... other parameters\n'
            '      },\n'
            '      "description": "What this action does"\n'
            '    }\n'
            '  ],\n'
            '  "next_question": "A specific question to ask about the graph (REQUIRED if user asked for questions, otherwise optional)",\n'
            '  "validation_complete": false,  // true if validation is done\n'
            '  "actions": [],  // Display actions (show_triples, highlight_entities, etc.)\n'
            '  "show_widget": false,\n'
            '  "widget_type": null\n'
            '}\n\n'
            "CRITICAL: If the user asks you to ask questions, you MUST:\n"
            "1. Immediately analyze the graph and find an issue\n"
            "2. Ask a specific, concrete question about that issue\n"
            "3. Include the question in your 'text' response AND in 'next_question'\n"
            "4. Don't just say 'I'll ask questions' - actually ASK a question!\n\n"
            "Be intelligent: If the user confirms something or you've made graph modifications, "
            "you can move to a new question or topic. Don't keep asking about the same thing.\n"
            "Always be proactive - ask questions about the graph, don't wait for the user to guide you.\n"
        )
        
        try:
            response = self.api_repo.chat(prompt)
            
            # Parse response
            if isinstance(response, dict):
                response_text = response.get("content", response.get("text", response.get("message", "")))
                if not response_text and "choices" in response:
                    response_text = response["choices"][0].get("message", {}).get("content", "")
            else:
                response_text = str(response) if response else ""
            
            # Clean and parse JSON
            response_text = response_text.strip()
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            try:
                response_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"JSON decode error in chat method: {e}")
                print(f"Response text (first 500 chars): {response_text[:500]}")
                # Return a safe default response
                return {
                    "text": "I apologize, but I encountered an error processing my response. The user said: '" + user_message + "'. Could you please rephrase or provide more details?",
                    "hidden_actions": [],
                    "next_question": None,
                    "validation_complete": False,
                    "actions": [],
                    "show_widget": False,
                    "widget_type": None,
                    "changes_summary": [],
                    "stats": {
                        "triples_before": len(self.triples),
                        "triples_after": len(self.triples),
                        "triples_changed": 0,
                        "entities_before": len(self.id_to_name),
                        "entities_after": len(self.id_to_name),
                        "entities_changed": 0,
                        "total_entities": len(self.id_to_name),
                        "total_triples": len(self.triples),
                    },
                }
            
            # Parse hidden actions
            hidden_actions = []
            for action_data in response_data.get("hidden_actions", []):
                try:
                    action_type = ActionType(action_data.get("type", "add_triples"))
                    hidden_actions.append(Action(
                        type=action_type,
                        parameters=action_data.get("parameters", {}),
                        description=action_data.get("description"),
                    ))
                except ValueError:
                    continue
            
            # Apply hidden actions
            updated_graph, updated_triples, graph_was_modified = self._apply_hidden_actions(hidden_actions, {})
            if updated_graph is not None:
                self.graph = updated_graph
            if updated_triples is not None:
                self.triples = updated_triples
            
            # Add bot response to global history
            bot_response_text = response_data.get("text", "Response processed.")
            self.global_conversation_history.append({"role": "bot", "content": bot_response_text})
            
            # Regenerate questions after each answer, especially if graph was modified
            # This ensures we catch new issues that may have been introduced
            next_question = response_data.get("next_question")
            validation_complete = response_data.get("validation_complete", False)
            
            # If graph was modified or user resolved a topic, regenerate questions
            # Also regenerate if user explicitly asked for questions
            user_asked_for_questions = "ask" in user_message.lower() and "question" in user_message.lower()
            
            if graph_was_modified or (generate_next_question and not validation_complete) or user_asked_for_questions:
                # Rebuild context with updated graph/triples
                updated_context = self._build_context()
                
                # Generate new questions based on current state
                new_questions = self._generate_questions(updated_context)
                
                # If we got new questions and no next_question was provided, use the first new one
                if new_questions and not next_question:
                    next_question = new_questions[0].text
                    # Store the new questions for potential future use
                    self.questions = new_questions
                    # If user asked for questions or we regenerated, append the question to the response
                    if next_question:
                        # Check if question is already in the response
                        if next_question.lower() not in bot_response_text.lower():
                            bot_response_text += f"\n\n{next_question}"
                            response_data["text"] = bot_response_text
                            # Also update the next_question field
                            response_data["next_question"] = next_question
            
            return {
                "text": bot_response_text,
                "hidden_actions": [
                    {
                        "type": action.type.value,
                        "parameters": action.parameters,
                        "description": action.description,
                    }
                    for action in hidden_actions
                ],
                "next_question": next_question,
                "validation_complete": validation_complete,
                "actions": response_data.get("actions", []),
                "show_widget": response_data.get("show_widget", False),
                "widget_type": response_data.get("widget_type"),
                "graph_modified": graph_was_modified,  # Indicate if graph was changed
            }
            
        except Exception as e:
            print(f"Error in chat: {e}")
            return {
                "text": f"Error processing message: {e}",
                "hidden_actions": [],
                "next_question": None,
                "validation_complete": False,
                "actions": [],
            }
    
    def getChanges(self) -> Dict[str, Any]:
        """Get summary of changes made to graph/triples."""
        return {
            "original_triples_count": len(self._original_triples),
            "current_triples_count": len(self.triples),
            "triples_added": len(self.triples) - len(self._original_triples),
            "original_graph_nodes": self._original_graph.number_of_nodes() if self._original_graph else 0,
            "current_graph_nodes": self.graph.number_of_nodes() if self.graph else 0,
            "original_graph_edges": self._original_graph.number_of_edges() if self._original_graph else 0,
            "current_graph_edges": self.graph.number_of_edges() if self.graph else 0,
        }

