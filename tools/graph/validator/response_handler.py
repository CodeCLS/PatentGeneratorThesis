"""
Response Handler: Handles user responses and generates bot responses using LLM.
"""
from typing import Optional, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

from tools.graph.validator_types import Question, Response, Action, ActionType, ConversationTurn
from tools.api.llm_api_repo import LLmApi_Repo
from tools.graph.validator.conversation_manager import ConversationManager
from tools.graph.validator.question_manager import QuestionManager
from tools.graph.validator.entity_mapper import EntityMapper
from tools.graph.validator.debug_utils import open_debug_browser, format_agent_output


class ResponseHandler:
    """Handles user responses and generates bot responses using LLM."""
    
    def __init__(
        self,
        api_repo: LLmApi_Repo,
        question_manager: QuestionManager,
        conversation_manager: ConversationManager,
        id_to_name: Dict[str, str],
        debug: bool = False,
    ):
        self.api_repo = api_repo
        self.question_manager = question_manager
        self.conversation_manager = conversation_manager
        self.id_to_name = id_to_name
        self.entity_mapper: Optional[EntityMapper] = None
        self.debug = debug
    
    def process_answer(
        self,
        question_id: str,
        answer_text: str,
        apply_hidden_actions_callback,  # Callback to apply hidden actions
    ) -> Response:
        """
        Process a user's answer to a question and generate a response.
        
        Args:
            question_id: ID of the question being answered
            answer_text: The user's answer text
            apply_hidden_actions_callback: Function to apply hidden actions (takes hidden_actions, metadata)
            
        Returns:
            Response object with text and actions
        """
        logger.info(f"ResponseHandler: Processing answer for question {question_id}")
        logger.debug(f"Answer text: {answer_text[:100]}{'...' if len(answer_text) > 100 else ''}")
        
        # Find the question
        question = self.question_manager.get_question_by_id(question_id)
        
        if not question:
            logger.warning(f"Question {question_id} not found")
            return Response(
                question_id=question_id,
                text="Question not found.",
            ), None, None
        
        logger.debug(f"Found question: {question.text[:100]}...")
        
        # Build prompt for LLM to process the answer
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
        logger.debug("Formatting conversation history")
        conversation_history_text = self.conversation_manager.format_conversation_history(question_id, max_turns=5)
        
        logger.info("Building LLM prompt for response generation")
        prompt = self._build_response_prompt(question, context_for_llm, conversation_history_text, answer_text)
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        try:
            logger.info("Calling LLM API for response generation")
            prompt = self._build_response_prompt(question, context_for_llm, conversation_history_text, answer_text)
            logger.debug(f"Prompt length: {len(prompt)} characters")
            response = self.api_repo.chat(prompt)
            logger.info("Received response from LLM")
            
            # Parse response
            logger.debug("Parsing LLM response")
            response_data = self._parse_llm_response(response)
            
            # Parse actions
            logger.debug("Parsing actions from response")
            actions = self._parse_actions(response_data.get("actions", []))
            hidden_actions = self._parse_actions(response_data.get("hidden_actions", []))
            logger.info(f"Parsed {len(actions)} display actions and {len(hidden_actions)} hidden actions")
            
            # Debug: Open browser with output
            if self.debug:
                debug_content = format_agent_output(
                    agent_name="ResponseHandler",
                    input_data={
                        "question": question.text,
                        "answer": answer_text,
                        "prompt_preview": prompt[:1000] + "..." if len(prompt) > 1000 else prompt,
                    },
                    output_data=response_data,
                    metadata={
                        "question_id": question_id,
                        "prompt_length": len(prompt),
                        "actions_count": len(actions),
                        "hidden_actions_count": len(hidden_actions),
                    }
                )
                open_debug_browser(debug_content, title="ResponseHandler Output")
            
            # Check if question is completed
            question_completed = response_data.get("question_completed", False)
            logger.debug(f"Question completed flag: {question_completed}")
            
            # Apply hidden actions to modify graph/triples
            if hidden_actions:
                logger.info(f"Applying {len(hidden_actions)} hidden actions to graph")
                updated_graph, updated_triples, graph_was_modified = apply_hidden_actions_callback(
                    hidden_actions, response_data.get("metadata", {})
                )
                logger.info(f"Graph modification result: modified={graph_was_modified}")
            else:
                logger.debug("No hidden actions to apply")
                updated_graph, updated_triples, graph_was_modified = None, None, False
            
            # Check for confirmation and update question_completed
            logger.debug("Checking question completion status")
            question_completed = self._check_completion(
                answer_text, question, question_completed, hidden_actions, graph_was_modified
            )
            logger.info(f"Question completion status: {question_completed}")
            
            # Create Response object
            response_text = response_data.get("text", "Response processed.")
            logger.info(f"Response text: {response_text[:200]}{'...' if len(response_text) > 200 else ''}")
            response_obj = Response(
                question_id=question_id,
                text=response_text,
                actions=actions,
                hidden_actions=hidden_actions,
                metadata=response_data.get("metadata", {}),
                show_widget=response_data.get("show_widget", False),
                widget_type=response_data.get("widget_type"),
                question_completed=question_completed,
            )
            
            # Store conversation turn
            logger.debug("Storing conversation turn")
            turns = self.conversation_manager.get_turns(question_id)
            turn_number = len(turns) + 1
            self.conversation_manager.add_turn(question_id, answer_text, response_obj.text, turn_number)
            logger.debug(f"Stored turn {turn_number} for question {question_id}")
            
            # Update question state
            if question_completed:
                logger.info(f"Marking question {question_id} as answered")
                question.answered = True
                self.question_manager.mark_question_answered(question_id)
            else:
                question.num_responses += 1
                logger.debug(f"Question {question_id} has {question.num_responses} responses")
                # Auto-complete after 2-3 responses
                if question.num_responses >= 2:
                    has_substantial_info = (
                        len(answer_text.strip()) > 30 and 
                        "don't understand" not in answer_text.lower() and
                        "don't know" not in answer_text.lower() and
                        "unclear" not in answer_text.lower()
                    )
                    if has_substantial_info or hidden_actions:
                        logger.info(f"Auto-completing question {question_id} after {question.num_responses} responses (substantial info provided)")
                        question.answered = True
                        question_completed = True
                        response_obj.question_completed = True
                    elif question.num_responses >= 3:
                        logger.info(f"Force-completing question {question_id} after 3 responses")
                        question.answered = True
                        question_completed = True
                        response_obj.question_completed = True
            
            logger.info("ResponseHandler: Answer processing complete")
            return response_obj, updated_graph, updated_triples
            
        except Exception as e:
            logger.error(f"Error processing answer: {e}", exc_info=True)
            print(f"Error processing answer: {e}")
            error_response = Response(
                question_id=question_id,
                text=f"Error processing answer: {e}",
            )
            return error_response, None, None
    
    def _build_response_prompt(
        self,
        question: Question,
        context_for_llm: Dict[str, Any],
        conversation_history_text: str,
        answer_text: str,
    ) -> str:
        """Build the LLM prompt for processing a user's answer."""
        return (
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
            '  "actions": [...],  // Display actions\n'
            '  "hidden_actions": [...],  // Graph modification actions\n'
            '  "metadata": {...}  // Additional data\n'
            '}\n\n'
            "Return ONLY the JSON object. No markdown fences, no commentary.\n"
        )
    
    def _parse_llm_response(self, response: Any) -> Dict[str, Any]:
        """Parse LLM response into a dictionary."""
        if isinstance(response, dict):
            response_text = response.get("content", response.get("text", response.get("message", "")))
            if not response_text and "choices" in response:
                response_text = response["choices"][0].get("message", {}).get("content", "")
        else:
            response_text = str(response) if response else ""
        
        # Clean and parse JSON
        response_text = response_text.strip()
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        return json.loads(response_text)
    
    def _parse_actions(self, actions_data: list) -> list[Action]:
        """Parse action data into Action objects."""
        actions = []
        for action_data in actions_data:
            try:
                action_type = ActionType(action_data.get("type", "show_triples"))
                actions.append(Action(
                    type=action_type,
                    parameters=action_data.get("parameters", {}),
                    description=action_data.get("description"),
                ))
            except ValueError:
                continue
        return actions
    
    def _check_completion(
        self,
        answer_text: str,
        question: Question,
        question_completed: bool,
        hidden_actions: list[Action],
        graph_was_modified: bool,
    ) -> bool:
        """Check if question should be marked as completed based on answer and actions."""
        answer_lower = answer_text.lower().strip()
        confirmation_phrases = [
            "yes", "correct", "right", "confirmed", "ok", "okay", "understood", 
            "i agree", "that's right", "that is correct", "exactly", "precisely",
            "sounds good", "looks good", "fine", "good", "perfect", "agreed",
            "i understand", "got it", "makes sense", "that makes sense", "clear",
            "no problem", "no issues", "no concerns", "all good", "all set"
        ]
        
        is_confirmation = any(phrase in answer_lower for phrase in confirmation_phrases)
        is_short_acknowledgment = len(answer_text.strip()) < 20 and any(
            word in answer_lower for word in ["yes", "no", "ok", "okay", "sure", "fine", "good"]
        )
        
        # If user confirms and we have graph modifications, or if it's a clear confirmation
        if (is_confirmation or is_short_acknowledgment) and (hidden_actions or question.num_responses >= 1):
            if not question_completed:
                print(f"⚠ User confirmed/acknowledged - forcing question completion")
                question_completed = True
        
        # If graph was modified via hidden actions, automatically mark question as complete
        if hidden_actions:
            modification_actions = {
                ActionType.ADD_TRIPLES, ActionType.DELETE_TRIPLES, ActionType.MODIFY_TRIPLE,
                ActionType.MERGE_ENTITIES, ActionType.DELETE_ENTITY, ActionType.RENAME_ENTITY,
                ActionType.CHANGE_ENTITY_LABEL, ActionType.ADD_RELATION, ActionType.REMOVE_RELATION,
                ActionType.CHANGE_RELATION, ActionType.SPLIT_ENTITY, ActionType.CREATE_ENTITY,
                ActionType.UPDATE_TRIPLE_RELATION, ActionType.UPDATE_TRIPLE_HEAD, ActionType.UPDATE_TRIPLE_TAIL
            }
            has_modification_action = any(action.type in modification_actions for action in hidden_actions)
            
            if graph_was_modified or has_modification_action:
                if not question_completed:
                    question_completed = True
                    print(f"✓ Graph modified via hidden actions - auto-completing question")
        
        return question_completed

