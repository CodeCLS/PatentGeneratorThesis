"""
Communicator node - main communication agent that handles user messages and coordinates other agents.
"""

from typing import TYPE_CHECKING, Optional, Union

try:
    from langgraph.graph import END
except ImportError:
    END = "__end__"

import json
from tools.helper.json_helper import JsonHelper
from tools.graph.langgraph.helpers import (
    extract_text_from_response,
    get_triple_head_name,
    get_triple_tail_name,
    process_retrieved_info_for_widget,
    extract_retrieved_info,
)
from tools.graph.langgraph.state import GraphValidatorState
from tools.graph.langgraph.question import Question
from tools.graph.langgraph.message import Message, MessageRole
from tools.graph.langgraph.message.widgets import Widget, EdgesWidget
from tools.graph.langgraph.prompts import get_registry
from tools.graph.constants_graph import (
    AGENT_COMMUNICATOR,
    AGENT_ANALYZER,
    AGENT_MODIFIER,
    AGENT_VISUALIZER,
    AGENT_RETRIEVER,
    ACTION_DELETE_TRIPLES,
    ACTION_UPDATE_ENTITY_LABEL,
    ACTION_MERGE_ENTITIES,
    STATE_MESSAGES,
    STATE_CURRENT_QUESTION_TEXT,
    STATE_QUESTIONS,
    STATE_CHANGES_SUMMARY,
    STATE_CURRENT_QUESTION_ID,
    STATE_VALIDATION_COMPLETE,
    STATE_NEXT_AGENT,
    STATE_AGENT_QUEUE,
    STATE_HIDDEN_ACTIONS,
    STATE_SHOW_WIDGET,
    STATE_WIDGET_TYPE,
    STATE_WIDGET_DATA,
    STATE_TEXT,
    STATE_NEXT_QUESTION,
    MESSAGE_ROLE_USER,
    MESSAGE_ROLE_SYSTEM,
    KEY_CONTENT,
    KEY_ROLE,
    KEY_RELATED_TRIPLES,
    KEY_SEARCH_RESULTS,
    KEY_TRIPLES,
    KEY_NAME,
    KEY_ERROR,
    KEY_INDEX,
    KEY_HEAD,
    KEY_TAIL,
    KEY_RELATION,
    KEY_RETRIEVED_INFO_MARKER,
    KEY_REASON,
    WIDGET_TYPE_EDGES,
    WIDGET_TYPE_VISUALIZATION,
    STATE_INTERNAL_RETRIEVED_TRIPLES,
    STATE_INTERNAL_RETRIEVED_INFO_PROCESSED,
    STATE_INTERNAL_NEEDS_WIDGET,
    DEFAULT_ENTITY,
)

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def _format_retrieved_info(info_text: str, user_message: Optional[str] = None) -> str:
    """Format retrieved information into a readable response."""
    info_data = extract_retrieved_info(info_text)
    
    if info_data:
        # Format based on data type
        if KEY_RELATED_TRIPLES in info_data:
            triples = info_data[KEY_RELATED_TRIPLES]
            if not triples:
                return "No triples found."
            response = "Here are the related triples:\n\n"
            for t in triples[:30]:
                response += f"  {t.get(KEY_INDEX, '?')}. {t.get(KEY_HEAD, '')} --[{t.get(KEY_RELATION, '')}]--> {t.get(KEY_TAIL, '')}\n"
            if len(triples) > 30:
                response += f"\n  ... and {len(triples) - 30} more"
            return response
        
        if KEY_SEARCH_RESULTS in info_data:
            results = info_data[KEY_SEARCH_RESULTS]
            if not results:
                return "No entities found."
            response = "Found entities:\n\n"
            for r in results[:20]:
                response += f"  - {r.get(KEY_NAME, '')}\n"
            return response
        
        if KEY_TRIPLES in info_data:
            entity_name = info_data.get(KEY_NAME, DEFAULT_ENTITY)
            triples = info_data.get(KEY_TRIPLES, [])
            response = f"Information about '{entity_name}':\n"
            if triples:
                response += "\nRelated triples:\n"
                for t in triples[:20]:
                    response += f"  {t.get(KEY_INDEX, '?')}. {t.get(KEY_HEAD, '')} --[{t.get(KEY_RELATION, '')}]--> {t.get(KEY_TAIL, '')}\n"
            return response
        
        if KEY_ERROR in info_data:
            return f"Error: {info_data[KEY_ERROR]}"
        
        # Default: show the data
        return f"Retrieved information:\n{json.dumps(info_data, indent=2)}"
    
    # Fallback if parsing fails
    return "I've retrieved the information. Here it is:\n\n" + info_text


def _get_current_question(questions: list, current_question_id: Optional[str]) -> Optional[Question]:
    """Get the current question object from the questions list by ID."""
    if not current_question_id or not questions:
        return None
    
    for q in questions:
        if isinstance(q, Question):
            question = q
        elif isinstance(q, dict):
            question = Question.from_dict(q)
        else:
            question = Question.from_dict(q.to_dict() if hasattr(q, 'to_dict') else {"id": "", "text": str(q)})
        
        if question.id == current_question_id:
            return question
    
    return None


def _get_message_content(msg: Union[Message, dict]) -> str:
    """Extract content from a Message object or dict."""
    if isinstance(msg, Message):
        return msg.content
    elif isinstance(msg, dict):
        return msg.get(KEY_CONTENT, "")
    return ""


def _get_message_role(msg: Union[Message, dict]) -> str:
    """Extract role from a Message object or dict."""
    if isinstance(msg, Message):
        return msg.role.value if isinstance(msg.role, MessageRole) else str(msg.role)
    elif isinstance(msg, dict):
        return msg.get(KEY_ROLE, "")
    return ""


def build_communicator_prompt(validator: "GraphValidatorLangGraph", state: "GraphValidatorState", user_message: Optional[str]) -> str:
    """Build prompt for the communicator agent."""
    messages = state.get(STATE_MESSAGES, [])
    current_question_id = state.get(STATE_CURRENT_QUESTION_ID)
    questions = state.get(STATE_QUESTIONS, [])
    current_question_obj = _get_current_question(questions, current_question_id)
    current_question = current_question_obj.text if current_question_obj else None
    remaining_count = len(questions)
    triples = validator.triples
    id_to_name = validator.id_to_name
    
    # Build conversation history
    conv_text = ""
    for msg in messages[-15:]:
        role = _get_message_role(msg)
        content = _get_message_content(msg)
        if role:
            conv_text += f"{role.upper()}: {content[:150]}\n"
    
    # Add changes that were actually made
    changes_summary = state.get(STATE_CHANGES_SUMMARY, [])
    changes_text = ""
    if changes_summary:
        changes_text = "\n".join([f"- {c}" for c in changes_summary[-5:]])
    
    # Build context-specific additions
    context_additions = ""
    if changes_text:
        context_additions += (
            f"CHANGES MADE TO GRAPH:\n{changes_text}\n\n"
            "IMPORTANT: These are the ACTUAL changes that were applied to the graph. "
            "If an entity was deleted or merged, it will NOT appear in the ENTITIES list above.\n"
            "If a change is listed here, it has already been applied. Do NOT assume changes that aren't listed.\n\n"
        )
    
    if remaining_count > 0:
        context_additions += f"REMAINING QUESTIONS: {remaining_count} questions left to review.\n\n"
    
    if current_question:
        context_additions += (
            f"CURRENT QUESTION: {current_question}\n\n"
            "The user is responding to this question. Engage naturally.\n"
            "If the conversation history shows that the question has already been asked, do not ask it again.\n"
            "IMPORTANT: If you rephrase or ask about the question, you MUST ask about the SAME entity mentioned in the CURRENT QUESTION above.\n"
            "Do NOT change the focus to a different entity just because you see it in the graph.\n\n"
        )
    
    if user_message:
        context_additions += (
            f"USER JUST SAID: {user_message}\n\n"
            "Determine if the user is:\n"
            "- ASKING FOR INFORMATION (triples, connections, details about entities) - route to 'retriever'\n"
            "- ASKING A FOLLOW-UP (clarification, more info, etc.) - keep current question active, route to 'retriever' if asking for specific information\n"
            "- RESOLVING/ANSWERING the question (yes/no, explicit answer, 'move on', 'next', etc.) - mark question as resolved\n"
            "- REQUESTING CHANGES (delete, merge, update, etc.) - route to 'modifier'\n\n"
            "If they're asking about connections, triples, or details about an entity, you MUST route to 'retriever'.\n"
            "If they RESOLVED the question AND there are more questions, you can mention moving to the next one.\n"
            "If they're asking a FOLLOW-UP, stay focused on the current question.\n\n"
        )
    else:
        context_additions += "This is the start of the conversation so perhaps ask the question in detail if you deem it necessary.\n\n"
    
    # Get base prompt from registry
    registry = get_registry()
    base_prompt = registry.build_prompt(
        AGENT_COMMUNICATOR,
        current_question=current_question or "None",
        remaining_count=remaining_count,
        triples_count=len(triples),
        entities_count=len(id_to_name)
    )
    
    # Add dynamic context
    full_prompt = (
        f"GRAPH: {len(triples)} triples, {len(id_to_name)} entities\n"
        f"HISTORY:\n{conv_text}\n\n"
        f"{context_additions}"
        f"{base_prompt}\n\n"
        "Return ONLY JSON:\n"
        '{"text": "response", "next_agent": null, "hidden_actions": [], "next_question": null, "validation_complete": false, "question_resolved": false}\n'
        "Available agents for next_agent: 'retriever' get triple/entity information, 'visualizer' (show widgets), 'analyzer' (generate questions), 'modifier' (make changes), null/END (finish conversation)\n"
        "question_resolved: true only if user clearly answered/resolved the current question, false for follow-ups.\n"
    )
    
    return full_prompt


def communicator_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState") -> "GraphValidatorState":
    """Main communication agent - handles user messages and coordinates other agents."""
    messages = state.get(STATE_MESSAGES, [])
    current_question_id = state.get(STATE_CURRENT_QUESTION_ID)
    questions = state.get(STATE_QUESTIONS, [])
    current_question_obj = _get_current_question(questions, current_question_id)
    current_question_text = current_question_obj.text if current_question_obj else None
    validation_complete = state.get(STATE_VALIDATION_COMPLETE, False)
    
    # Get the last user message
    user_message = None
    for msg in reversed(messages):
        role = _get_message_role(msg)
        if role == MESSAGE_ROLE_USER:
            user_message = _get_message_content(msg)
            break
    
    # Initial state - present first question (check this first, before widget checks)
    if not user_message and not current_question_text and questions:
        first_q = questions[0]
        if isinstance(first_q, Question):
            question = first_q
       
        return {
            **state,
            STATE_MESSAGES: messages + [Message(role=MessageRole.BOT, content=question.text)],
            STATE_CURRENT_QUESTION_ID: question.id,
            STATE_CURRENT_QUESTION_TEXT: question.text,
            STATE_NEXT_AGENT: END,
        }
    
    # Handle retrieved information from retriever
    retrieved_info = None
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get(KEY_ROLE) == MESSAGE_ROLE_SYSTEM:
            content = msg.get(KEY_CONTENT, "")
            if KEY_RETRIEVED_INFO_MARKER in content:
                retrieved_info = content
                break
    
    if retrieved_info:
        try:
            lines = retrieved_info.split('\n')
            json_lines = []
            in_json = False
            
            for line in lines:
                if line.strip().startswith('{'):
                    in_json = True
                if in_json:
                    if line.strip().startswith('Reason:'):
                        break
                    json_lines.append(line)
            
            if json_lines:
                json_text = '\n'.join(json_lines)
                info_data = json.loads(json_text)
                
                if KEY_RELATED_TRIPLES in info_data:
                    triples = info_data[KEY_RELATED_TRIPLES]
                    # Route to visualizer to show edges_widget
                    return {
                        **state,
                        STATE_MESSAGES: messages,  # Don't add bot message yet
                        STATE_NEXT_AGENT: AGENT_VISUALIZER,
                        STATE_INTERNAL_RETRIEVED_TRIPLES: triples,  # Pass triples to visualizer
                        STATE_INTERNAL_RETRIEVED_INFO_PROCESSED: True,  # Mark as processed to prevent loops
                    }
        except:
            pass
        
        # Fallback: format and show normally
        response_text = _format_retrieved_info(retrieved_info, user_message)
        return {
            **state,
            STATE_MESSAGES: messages + [Message(role=MessageRole.BOT, content=response_text)],
            STATE_NEXT_AGENT: END,
        }
    
    # Check if visualizer just created a widget from retrieved triples
    show_widget = state.get(STATE_SHOW_WIDGET, False)
    widget_type = state.get(STATE_WIDGET_TYPE)
    # Check if we already processed retrieved_info (to prevent loops)
    already_processed = state.get(STATE_INTERNAL_RETRIEVED_INFO_PROCESSED, False)
    
    if show_widget and widget_type == WIDGET_TYPE_EDGES and not user_message and not already_processed:
        # Visualizer just showed a widget, now generate a natural response
        # Get the original user message from messages
        original_user_msg = None
        for msg in reversed(messages):
            role = _get_message_role(msg)
            if role == MESSAGE_ROLE_USER:
                original_user_msg = _get_message_content(msg)
                break
        
        prompt = build_communicator_prompt(validator, state, original_user_msg)
        # Widget context is already included in the prompt registry templates
        response = validator.api_repo.chat(prompt)
        response_text = extract_text_from_response(response)
        response_data = JsonHelper.parse_json(response_text)
        bot_message = response_data.get("text", response_text) if response_data else response_text
        
        return {
            **state,
            STATE_MESSAGES: messages + [Message(role=MessageRole.BOT, content=bot_message)],
            STATE_NEXT_AGENT: END,
            STATE_INTERNAL_RETRIEVED_INFO_PROCESSED: True,  # Mark as processed to prevent loops
        }
    
    retrieved_info = None
    if not show_widget and not already_processed:
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get(KEY_ROLE) == MESSAGE_ROLE_SYSTEM:
                content = msg.get(KEY_CONTENT, "")
                if KEY_RETRIEVED_INFO_MARKER in content:
                    retrieved_info = content
                    break
    
    if retrieved_info:
        # Use helper to process retrieved info
        triples, info_data = process_retrieved_info_for_widget(retrieved_info)
        
        # If we have related_triples, show them as a widget
        if triples:
            return {
                **state,
                STATE_MESSAGES: messages,  # Don't add bot message yet
                STATE_NEXT_AGENT: AGENT_VISUALIZER,
                STATE_INTERNAL_RETRIEVED_TRIPLES: triples,  # Pass triples to visualizer
                STATE_INTERNAL_RETRIEVED_INFO_PROCESSED: True,  # Mark as processed to prevent loops
            }
        
        # Fallback: format and show normally
        response_text = _format_retrieved_info(retrieved_info, user_message)
        return {
            **state,
            STATE_MESSAGES: messages + [Message(role=MessageRole.BOT, content=response_text)],
            STATE_NEXT_AGENT: END,
        }
    
    # Check for merged results from parallel execution (both changes and widget)
    changes_summary = state.get(STATE_CHANGES_SUMMARY, [])
    
    if changes_summary and show_widget:
        # Both modifier and visualizer have completed in parallel
        changes_text = "\n".join([f"- {c}" for c in changes_summary[-3:]])
        widget_type = state.get(STATE_WIDGET_TYPE, WIDGET_TYPE_VISUALIZATION)
        bot_message = f"I've made these changes:\n{changes_text}\n\n"
        bot_message += f"I've also prepared a {widget_type} for you to interact with."
        return {
            **state,
            STATE_MESSAGES: messages + [Message(role=MessageRole.BOT, content=bot_message)],
            STATE_NEXT_AGENT: END,
        }
    elif show_widget:
        # Only visualizer completed
        return {
            **state,
            STATE_MESSAGES: messages + [Message(role=MessageRole.BOT, content="I've prepared a visualization. Please interact with it.")],
            STATE_NEXT_AGENT: END,
        }
    elif changes_summary:
        # Only modifier completed
        changes_text = "\n".join([f"- {c}" for c in changes_summary[-3:]])
        return {
            **state,
            STATE_MESSAGES: messages + [Message(role=MessageRole.BOT, content=f"I've made these changes:\n{changes_text}\n\nWhat would you like to do next?")],
            STATE_NEXT_AGENT: END,
        }
    
    # Build prompt and call LLM
    prompt = build_communicator_prompt(validator, state, user_message)
    response = validator.api_repo.chat(prompt)
    response_text = extract_text_from_response(response)
    
    response_data = JsonHelper.parse_json(response_text)
    if not response_data:
        response_data = {
            STATE_TEXT: response_text[:500],
            STATE_NEXT_AGENT: None,
            STATE_HIDDEN_ACTIONS: [],
            STATE_NEXT_QUESTION: None,
            "question_resolved": False,
        }
    
    bot_message = response_data.get(STATE_TEXT, response_text)
    next_agent = response_data.get(STATE_NEXT_AGENT)
    hidden_actions = response_data.get(STATE_HIDDEN_ACTIONS, [])
    question_resolved = response_data.get("question_resolved", False)
    # Don't use next_question from LLM - it might just be an index like "1"
    # Always get the actual question text from the questions list
    
    # Only remove question if it's actually resolved, not just any message
    updated_questions = []
    for q in questions:
        if isinstance(q, Question):
            question = q
        elif isinstance(q, dict):
            question = Question.from_dict(q)
        else:
            question = Question.from_dict(q.to_dict() if hasattr(q, 'to_dict') else {"id": "", "text": str(q)})
        
        # Only remove if question is resolved AND it matches current question
        should_remove = False
        if question_resolved and current_question_id and question.id == current_question_id:
            should_remove = True
        
        if not should_remove:
            updated_questions.append(question)
    
    # Only clear current question if resolved, otherwise keep it active for follow-ups
    if question_resolved and current_question_id:
        current_question_id = None
        current_question_text = None
    
    # Determine next agent - check for parallel execution needs
    needs_widget = False
    if next_agent == AGENT_VISUALIZER and hidden_actions:
        # User wants both changes and visualization - use parallel execution
        needs_widget = True
        agent_to_call = None  # Will be handled by routing via fork
    elif next_agent:
        agent_to_call = next_agent
    elif hidden_actions:
        agent_to_call = AGENT_MODIFIER
    else:
        # End gracefully - orchestrator will decide what to do next when user chats again
        agent_to_call = END
    
    # Only prepare next question if current one was resolved, otherwise keep current question
    # Derive final question from the updated questions list
    final_question_obj = _get_current_question(updated_questions, current_question_id) if not question_resolved else None
    if not final_question_obj and updated_questions:
        # Get first question if current was resolved or doesn't exist
        next_q = updated_questions[0]
        if isinstance(next_q, Question):
            final_question_obj = next_q
        elif isinstance(next_q, dict):
            final_question_obj = Question.from_dict(next_q)
        else:
            final_question_obj = Question.from_dict(next_q.to_dict() if hasattr(next_q, 'to_dict') else {"id": "", "text": str(next_q)})
        
        # Add next question to bot message so it's presented to user
        if final_question_obj:
            bot_message = bot_message + f"\n\n{final_question_obj.text}"
    
    final_question_text = final_question_obj.text if final_question_obj else None
    final_question_id = final_question_obj.id if final_question_obj else None
    
    # Mark complete if no more questions
    if not updated_questions:
        validation_complete = True
    
    # Don't add bot message if routing to retriever - retriever will handle the response
    updated_messages = messages
    if agent_to_call != AGENT_RETRIEVER:
        updated_messages = messages + [Message(role=MessageRole.BOT, content=bot_message)]
    
    # Clear agent queue if we're ending the conversation
    agent_queue = state.get(STATE_AGENT_QUEUE, [])
    if agent_to_call == END or agent_to_call is None:
        agent_queue = []
    elif AGENT_COMMUNICATOR in agent_queue:
        # Remove communicator from queue since we just ran it
        agent_queue = [a for a in agent_queue if a != AGENT_COMMUNICATOR]
    
    return {
        **state,
        STATE_MESSAGES: updated_messages,
        STATE_QUESTIONS: updated_questions,
        STATE_NEXT_AGENT: agent_to_call,
        STATE_AGENT_QUEUE: agent_queue,
        STATE_HIDDEN_ACTIONS: hidden_actions,
        STATE_CURRENT_QUESTION_TEXT: final_question_text,
        STATE_CURRENT_QUESTION_ID: final_question_id,
        STATE_VALIDATION_COMPLETE: validation_complete,
        STATE_INTERNAL_NEEDS_WIDGET: needs_widget,  # Flag for parallel execution
    }
