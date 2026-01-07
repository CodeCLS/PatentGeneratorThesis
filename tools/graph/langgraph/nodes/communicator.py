"""
Communicator node - main communication agent that handles user messages and coordinates other agents.
"""

from typing import TYPE_CHECKING, Optional

try:
    from langgraph.graph import END
except ImportError:
    END = "__end__"

from tools.helper.json_helper import JsonHelper
from tools.graph.langgraph.helpers import extract_text_from_response, get_triple_head_name, get_triple_tail_name
from tools.graph.langgraph.state import GraphValidatorState
from tools.graph.langgraph.question import Question
from tools.graph.constants_graph import (
    AGENT_ANALYZER,
    AGENT_MODIFIER,
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
    STATE_HIDDEN_ACTIONS,
    STATE_SHOW_WIDGET,
)

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def build_communicator_prompt(validator: "GraphValidatorLangGraph", state: "GraphValidatorState", user_message: Optional[str]) -> str:
    """Build prompt for the communicator agent."""
    messages = state.get(STATE_MESSAGES, [])
    current_question = state.get(STATE_CURRENT_QUESTION_TEXT)
    questions = state.get(STATE_QUESTIONS, [])
    remaining_count = len(questions)
    triples = validator.triples
    id_to_name = validator.id_to_name
    
    # Build triples summary
    triples_text = ""
    for i, triple in enumerate(triples[:30]):
        head_name = get_triple_head_name(triple)
        tail_name = get_triple_tail_name(triple)
        triples_text += f"  {i}. {head_name} --[{triple.relation}]--> {tail_name}\n"
    
    if len(triples) > 30:
        triples_text += f"\n  ... and {len(triples) - 30} more triples (total: {len(triples)})\n"
    
    # Build conversation history
    conv_text = ""
    for msg in messages[-5:]:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")[:150]
        conv_text += f"{role}: {content}\n"
    
    # Build entity list
    all_entities = list(id_to_name.values())
    if len(all_entities) > 50:
        entity_list = ", ".join(all_entities[:50]) + f"\n  ... and {len(all_entities) - 50} more"
    else:
        entity_list = ", ".join(all_entities)
    
    prompt = (
        "You are a knowledge graph validator having a conversational dialogue.\n\n"
        f"GRAPH: {len(triples)} triples, {len(id_to_name)} entities\n"
        f"ENTITIES: {entity_list}\n\n"
        f"TRIPLES:\n{triples_text}\n\n"
        f"HISTORY:\n{conv_text}\n\n"
    )
    
    # Add changes that were actually made
    changes_summary = state.get(STATE_CHANGES_SUMMARY, [])
    if changes_summary:
        changes_text = "\n".join([f"- {c}" for c in changes_summary[-5:]])  # Last 5 changes
        prompt += (
            f"CHANGES MADE TO GRAPH:\n{changes_text}\n\n"
            "IMPORTANT: These are the ACTUAL changes that were applied to the graph. "
            "If an entity was deleted or merged, it will NOT appear in the ENTITIES list above.\n"
            "If a change is listed here, it has already been applied. Do NOT assume changes that aren't listed.\n\n"
        )
    
    prompt += "RULES: Use entity names (not IDs), reference triples by index, be conversational.\n\n"
    
    if remaining_count > 0:
        prompt += f"REMAINING QUESTIONS: {remaining_count} questions left to review.\n\n"
    
    if current_question:
        prompt += (
            f"CURRENT QUESTION: {current_question}\n\n"
            "The user is responding to this question. Engage naturally.\n"
            "DO NOT repeat the question - it's already been asked.\n"
            "IMPORTANT: If you rephrase or ask about the question, you MUST ask about the SAME entity mentioned in the CURRENT QUESTION above.\n"
            "Do NOT change the focus to a different entity just because you see it in the graph.\n\n"
        )
    
    if user_message:
        prompt += (
            f"USER JUST SAID: {user_message}\n\n"
            "Determine if the user is:\n"
            "- ASKING A FOLLOW-UP (clarification, more info, etc.) - keep current question active\n"
            "- RESOLVING/ANSWERING the question (yes/no, explicit answer, 'move on', 'next', etc.) - mark question as resolved\n\n"
            "If they want to DELETE/REMOVE a triple, include hidden_actions: "
            f'[{{"type": "{ACTION_DELETE_TRIPLES}", "parameters": {{"triple_indices": [0]}}}}]\n'
            "If they want to UPDATE an entity label, include: "
            f'[{{"type": "{ACTION_UPDATE_ENTITY_LABEL}", "parameters": {{"entity_name": "Entity Name", "new_label": "NEW_LABEL"}}}}]\n'
            "If they want to MERGE entities, include: "
            f'[{{"type": "{ACTION_MERGE_ENTITIES}", "parameters": {{"entity_names": ["Entity1", "Entity2"]}}}}]\n'
            "If they RESOLVED the question AND there are more questions, you can mention moving to the next one.\n"
            "If they're asking a FOLLOW-UP, stay focused on the current question.\n\n"
        )
    else:
        prompt += "This is the start of the conversation or you're presenting a question.\n\n"
    
    prompt += (
        "Return ONLY JSON:\n"
        '{"text": "response", "next_agent": null, "hidden_actions": [], "next_question": null, "validation_complete": false, "question_resolved": false}\n'
        "question_resolved: true only if user clearly answered/resolved the current question, false for follow-ups.\n"
    )
    
    return prompt


def communicator_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState") -> "GraphValidatorState":
    """Main communication agent - handles user messages and coordinates other agents."""
    messages = state.get(STATE_MESSAGES, [])
    current_question_id = state.get(STATE_CURRENT_QUESTION_ID)
    current_question_text = state.get(STATE_CURRENT_QUESTION_TEXT)
    questions = state.get(STATE_QUESTIONS, [])
    validation_complete = state.get(STATE_VALIDATION_COMPLETE, False)
    
    # Get the last user message
    user_message = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break
    
    # Handle results from other agents
    last_message = messages[-1] if messages else None
    if not user_message and last_message and last_message.get("role") == "system":
        if "[Retrieved Information]" in last_message.get("content", ""):
            return {
                **state,
                STATE_MESSAGES: messages + [{"role": "bot", "content": "I've retrieved the information. How would you like to proceed?"}],
                STATE_NEXT_AGENT: END,
            }
        elif state.get(STATE_SHOW_WIDGET):
            return {
                **state,
                STATE_MESSAGES: messages + [{"role": "bot", "content": "I've prepared a visualization. Please interact with it."}],
                STATE_NEXT_AGENT: END,
            }
        elif state.get(STATE_CHANGES_SUMMARY):
            changes = state.get(STATE_CHANGES_SUMMARY, [])
            changes_text = "\n".join([f"- {c}" for c in changes[-3:]])
            return {
                **state,
                STATE_MESSAGES: messages + [{"role": "bot", "content": f"I've made these changes:\n{changes_text}\n\nWhat would you like to do next?"}],
                STATE_NEXT_AGENT: END,
            }
    
    # Initial state - present first question
    if not user_message and not current_question_text and questions:
        first_q = questions[0]
        if isinstance(first_q, dict):
            question = Question.from_dict(first_q)
        else:
            question = first_q if isinstance(first_q, Question) else Question.from_dict(first_q.to_dict() if hasattr(first_q, 'to_dict') else {"id": "", "text": str(first_q)})
        
        return {
            **state,
            STATE_MESSAGES: messages + [{"role": "bot", "content": question.text}],
            STATE_CURRENT_QUESTION_ID: question.id,
            STATE_CURRENT_QUESTION_TEXT: question.text,
            STATE_NEXT_AGENT: END,
        }
    
    # No questions - route to analyzer
    if not questions:
        return {
            **state,
            STATE_NEXT_AGENT: AGENT_ANALYZER,
        }
    
    # Build prompt and call LLM
    prompt = build_communicator_prompt(validator, state, user_message)
    response = validator.api_repo.chat(prompt)
    response_text = extract_text_from_response(response)
    
    response_data = JsonHelper.parse_json(response_text)
    if not response_data:
        response_data = {
            "text": response_text[:500],
            "next_agent": None,
            "hidden_actions": [],
            "next_question": None,
            "question_resolved": False,
        }
    
    bot_message = response_data.get("text", response_text)
    next_agent = response_data.get("next_agent")
    hidden_actions = response_data.get("hidden_actions", [])
    question_resolved = response_data.get("question_resolved", False)
    # Don't use next_question from LLM - it might just be an index like "1"
    # Always get the actual question text from the questions list
    
    # Only remove question if it's actually resolved, not just any message
    updated_questions = []
    for q in questions:
        if isinstance(q, dict):
            question = Question.from_dict(q)
        elif isinstance(q, Question):
            question = q
        else:
            question = Question.from_dict(q.to_dict() if hasattr(q, 'to_dict') else {"id": "", "text": str(q)})
        
        # Only remove if question is resolved AND it matches current question
        should_remove = False
        if question_resolved and current_question_id and question.id == current_question_id:
            should_remove = True
        elif question_resolved and current_question_text and question.text == current_question_text:
            should_remove = True
        
        if not should_remove:
            updated_questions.append(question)
    
    # Only clear current question if resolved, otherwise keep it active for follow-ups
    if question_resolved and current_question_id:
        current_question_id = None
        current_question_text = None
    
    # Determine next agent
    if next_agent:
        agent_to_call = next_agent
    elif hidden_actions:
        agent_to_call = AGENT_MODIFIER
    elif not validation_complete and not updated_questions:
        # Only route to analyzer if there are NO questions (to generate new ones)
        agent_to_call = AGENT_ANALYZER
    else:
        agent_to_call = END
    
    # Only prepare next question if current one was resolved, otherwise keep current question
    final_question_text = current_question_text  # Keep current if not resolved
    final_question_id = current_question_id
    
    if question_resolved and updated_questions:
        # Current question resolved, move to next
        next_q = updated_questions[0]
        if isinstance(next_q, Question):
            final_question_text = next_q.text
            final_question_id = next_q.id
        else:
            question = Question.from_dict(next_q if isinstance(next_q, dict) else next_q.to_dict() if hasattr(next_q, 'to_dict') else {"id": "", "text": str(next_q)})
            final_question_text = question.text
            final_question_id = question.id
        
        # Add next question to bot message so it's presented to user
        bot_message = bot_message + f"\n\n{final_question_text}"
    
    # Mark complete if no more questions
    if not updated_questions:
        validation_complete = True
    
    return {
        **state,
        STATE_MESSAGES: messages + [{"role": "bot", "content": bot_message}],
        STATE_QUESTIONS: updated_questions,
        STATE_NEXT_AGENT: agent_to_call,
        STATE_HIDDEN_ACTIONS: hidden_actions,
        STATE_CURRENT_QUESTION_TEXT: final_question_text,
        STATE_CURRENT_QUESTION_ID: final_question_id,
        STATE_VALIDATION_COMPLETE: validation_complete,
    }
