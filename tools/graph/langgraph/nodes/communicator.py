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

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def build_communicator_prompt(validator: "GraphValidatorLangGraph", state: "GraphValidatorState", user_message: Optional[str]) -> str:
    """Build prompt for the communicator agent."""
    messages = state.get("messages", [])
    current_question = state.get("current_question_text")
    questions = state.get("questions", [])
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
        "RULES: Use entity names (not IDs), reference triples by index, be conversational.\n\n"
    )
    
    if remaining_count > 0:
        prompt += f"REMAINING QUESTIONS: {remaining_count} questions left to review.\n\n"
    
    if current_question:
        prompt += (
            f"CURRENT QUESTION: {current_question}\n\n"
            "The user is responding to this question. Engage naturally.\n"
            "DO NOT repeat the question - it's already been asked.\n\n"
        )
    
    if user_message:
        prompt += (
            f"USER JUST SAID: {user_message}\n\n"
            "Respond naturally. If they answered a question, acknowledge it briefly.\n"
            "If they want to DELETE/REMOVE a triple, include hidden_actions: "
            '[{"type": "delete_triples", "parameters": {"triple_indices": [0]}}]\n'
            f"If there are more questions ({remaining_count} remaining), mention moving to the next one.\n\n"
        )
    else:
        prompt += "This is the start of the conversation or you're presenting a question.\n\n"
    
    prompt += (
        "Return ONLY JSON:\n"
        '{"text": "response", "next_agent": null, "hidden_actions": [], "next_question": null, "validation_complete": false}\n'
    )
    
    return prompt


def communicator_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState") -> "GraphValidatorState":
    """Main communication agent - handles user messages and coordinates other agents."""
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
    
    # Handle results from other agents
    last_message = messages[-1] if messages else None
    if not user_message and last_message and last_message.get("role") == "system":
        if "[Retrieved Information]" in last_message.get("content", ""):
            return {
                **state,
                "messages": messages + [{"role": "bot", "content": "I've retrieved the information. How would you like to proceed?"}],
                "next_agent": END,
            }
        elif state.get("show_widget"):
            return {
                **state,
                "messages": messages + [{"role": "bot", "content": "I've prepared a visualization. Please interact with it."}],
                "next_agent": END,
            }
        elif state.get("changes_summary"):
            changes = state.get("changes_summary", [])
            changes_text = "\n".join([f"- {c}" for c in changes[-3:]])
            return {
                **state,
                "messages": messages + [{"role": "bot", "content": f"I've made these changes:\n{changes_text}\n\nWhat would you like to do next?"}],
                "next_agent": END,
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
            "messages": messages + [{"role": "bot", "content": question.text}],
            "current_question_id": question.id,
            "current_question_text": question.text,
            "next_agent": END,
        }
    
    # No questions - route to analyzer
    if not questions:
        return {
            **state,
            "next_agent": "analyzer",
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
        }
    
    bot_message = response_data.get("text", response_text)
    next_agent = response_data.get("next_agent")
    hidden_actions = response_data.get("hidden_actions", [])
    # Don't use next_question from LLM - it might just be an index like "1"
    # Always get the actual question text from the questions list
    
    # Remove answered question
    updated_questions = []
    for q in questions:
        if isinstance(q, dict):
            question = Question.from_dict(q)
        elif isinstance(q, Question):
            question = q
        else:
            question = Question.from_dict(q.to_dict() if hasattr(q, 'to_dict') else {"id": "", "text": str(q)})
        
        # Remove if this is the current question being answered
        should_remove = False
        if user_message and current_question_id and question.id == current_question_id:
            should_remove = True
        elif user_message and current_question_text and question.text == current_question_text:
            should_remove = True
        
        if not should_remove:
            updated_questions.append(question)
    
    if user_message and current_question_id:
        current_question_id = None
        current_question_text = None
    
    # Determine next agent
        # Determine next agent
    if next_agent:
        agent_to_call = next_agent
    elif hidden_actions:
        agent_to_call = "modifier"
    elif not validation_complete and not updated_questions:
        # Only route to analyzer if there are NO questions (to generate new ones)
        agent_to_call = "analyzer"
    else:
        agent_to_call = END
    
    # Prepare next question if available - always get from questions list, not LLM response
    final_question_text = None
    final_question_id = None
    if updated_questions:
        next_q = updated_questions[0]
        if isinstance(next_q, Question):
            final_question_text = next_q.text
            final_question_id = next_q.id
        else:
            question = Question.from_dict(next_q if isinstance(next_q, dict) else next_q.to_dict() if hasattr(next_q, 'to_dict') else {"id": "", "text": str(next_q)})
            final_question_text = question.text
            final_question_id = question.id
    
    # Mark complete if no more questions
    if not updated_questions:
        validation_complete = True
    
    return {
        **state,
        "messages": messages + [{"role": "bot", "content": bot_message}],
        "questions": updated_questions,
        "next_agent": agent_to_call,
        "hidden_actions": hidden_actions,
        "current_question_text": final_question_text,
        "current_question_id": final_question_id,
        "validation_complete": validation_complete,
    }
