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
    STATE_PLAN,
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


def build_communicator_prompt(validator: "GraphValidatorLangGraph", state: "GraphValidatorState", user_message: Optional[str], system_message: Optional[str], current_plan: Optional[str]) -> str:
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
    if system_message:
        context_additions += f"SYSTEM MESSAGE/System Retrieved Information: {system_message}\n\n"
    
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
        f"CURRENT PLAN, FOLLOW this Plan strictly: {current_plan}\n"
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
    # 1. ALWAYS consume queue and define variables at the VERY START
    agent_queue = state.get(STATE_AGENT_QUEUE, [])
    if agent_queue and agent_queue[0] == AGENT_COMMUNICATOR:
        agent_queue = agent_queue[1:]
    
    messages = state.get(STATE_MESSAGES, [])
    questions = state.get(STATE_QUESTIONS, [])
    current_question_id = state.get(STATE_CURRENT_QUESTION_ID)
    validation_complete = state.get(STATE_VALIDATION_COMPLETE, False)
    current_plan = state.get(STATE_PLAN, False)

    # Debug print at the top so it always shows
    print(f"\n[DEBUG] Communicator starting. Queue: {agent_queue}")

    # ... (Keep your message extraction logic) ...
    user_message = None
    system_message = None
    for msg in reversed(messages):
        if _get_message_role(msg) == MESSAGE_ROLE_USER:
            user_message = _get_message_content(msg)
            break
        if _get_message_role(msg) == MESSAGE_ROLE_SYSTEM:
            system_message = _get_message_content(msg)
            break
    # 3. Main LLM Logic
    prompt = build_communicator_prompt(validator, state, user_message,system_message,current_plan)
    response = validator.api_repo.chat(prompt)
    response_text = extract_text_from_response(response)
    response_data = JsonHelper.parse_json(response_text) or {"text": response_text}
    
    bot_message = response_data.get("text", response_text)
    question_resolved = response_data.get("question_resolved", False)

    # 4. Question Management (Safe from IndexErrors)
    updated_questions = [q for q in questions if not (question_resolved and q.id == current_question_id)]
    
    next_q_text = updated_questions[0].text if updated_questions else None
    next_q_id = updated_questions[0].id if updated_questions else None
    
    if not updated_questions:
        validation_complete = True

    # Final Debug and Return
    print(f"[DEBUG] Communicator finished. Next Agent: {response_data.get('next_agent')}")
    
    return {
        **state,
        STATE_MESSAGES: messages + [Message(role=MessageRole.BOT, content=bot_message + " system: " + str(system_message))],
        STATE_QUESTIONS: updated_questions,
        STATE_AGENT_QUEUE: agent_queue,
        STATE_CURRENT_QUESTION_TEXT: next_q_text,
        STATE_CURRENT_QUESTION_ID: next_q_id,
        STATE_VALIDATION_COMPLETE: validation_complete,
    }