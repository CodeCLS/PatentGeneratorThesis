"""
Communicator node - main communication agent that handles user messages and coordinates other agents.
"""

from typing import TYPE_CHECKING, Optional

try:
    from langgraph.graph import END
except ImportError:
    END = "__end__"  # Fallback if END is not available

from tools.helper.json_helper import JsonHelper
from tools.graph.langgraph.helpers import extract_text_from_response

# Import GraphValidatorState at runtime (not just TYPE_CHECKING)
# This is needed because LangGraph might inspect type hints at runtime
from tools.graph.langgraph.state import GraphValidatorState

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def build_communicator_prompt(validator: "GraphValidatorLangGraph", state: "GraphValidatorState", user_message: Optional[str]) -> str:
    """Build prompt for the communicator agent."""
    messages = state.get("messages", [])
    current_question = state.get("current_question_text")
    questions = state.get("questions", [])
    remaining_count = len(questions)
    # Get triples and id_to_name from instance, not state (to avoid serialization)
    triples = validator.triples
    id_to_name = validator.id_to_name
    
    # Build triples summary - limit to reduce token usage
    triples_text = ""
    num_triples_to_show = min(30, len(triples))  # Reduced from 100 to 30
    for i, triple in enumerate(triples[:num_triples_to_show]):
        head_name = getattr(triple.head, "name", str(triple.head))
        tail_name = getattr(triple.tail, "name", str(triple.tail))
        triples_text += f"  {i}. {head_name} --[{triple.relation}]--> {tail_name}\n"
    
    if len(triples) > num_triples_to_show:
        triples_text += f"\n  ... and {len(triples) - num_triples_to_show} more triples (total: {len(triples)})\n"
    
    # Build conversation history - limit to reduce tokens
    conv_text = ""
    for msg in messages[-5:]:  # Reduced from 10 to 5 messages
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")[:150]  # Reduced from 200 to 150 chars
        conv_text += f"{role}: {content}\n"
    
    # Build entity list - limit to reduce tokens
    all_entities = list(id_to_name.values())
    if len(all_entities) > 50:  # Reduced from 100 to 50
        entity_list = ", ".join(all_entities[:50]) + f"\n  ... and {len(all_entities) - 50} more (total: {len(all_entities)})"
    else:
        entity_list = ", ".join(all_entities)
    
    prompt = (
        "You are a knowledge graph validator having a conversational dialogue.\n\n"
        f"GRAPH: {len(triples)} triples, {len(id_to_name)} entities\n"
        f"ENTITIES: {entity_list}\n\n"
        f"TRIPLES:\n{triples_text}\n\n"
        f"HISTORY:\n{conv_text}\n\n"
        "RULES: Use entity names (not IDs), reference triples by index (e.g., 'triple 5'), be specific and conversational.\n\n"
    )
    
    if remaining_count > 0:
        prompt += f"REMAINING QUESTIONS: {remaining_count} questions left to review.\n\n"
    
    if current_question:
        prompt += (
            f"CURRENT QUESTION BEING DISCUSSED: {current_question}\n\n"
            "The user is responding to this question. Engage with their response naturally.\n"
            "DO NOT repeat the question - it's already been asked. Just respond to what the user said.\n"
            f"After acknowledging their response, if there are more questions ({remaining_count} remaining), briefly mention moving to the next one.\n\n"
        )
    
    if user_message:
        prompt += f"USER JUST SAID: {user_message}\n\n"
        prompt += (
            "Respond naturally to what the user said. This is a conversation - be conversational!\n"
            "If they answered a question, acknowledge it briefly.\n"
            f"If there are more questions ({remaining_count} remaining), mention moving to the next question.\n"
            "If they asked something, answer it using the graph data you have access to.\n\n"
        )
    else:
        prompt += (
            "This is the start of the conversation or you're presenting a question.\n"
            "Be natural and conversational. Reference specific triples and entities from the graph.\n\n"
        )
    
    prompt += (
        "Return ONLY JSON (no text before/after):\n"
        '{"text": "response", "next_agent": null, "hidden_actions": [], "next_question": null, "validation_complete": false}\n'
        "CRITICAL: Keep response under 1000 tokens. Be concise.\n"
        "IMPORTANT: After the user answers a question, acknowledge it briefly and move to the next question if there are more."
    )
    
    return prompt


def communicator_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState") -> "GraphValidatorState":
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
    
    # Check if we just received results from another agent (and no new user message)
    # (indicated by last message being a system message from retriever/visualizer/etc)
    last_message = messages[-1] if messages else None
    just_received_results = (
        not user_message and  # No new user message
        last_message and 
        last_message.get("role") == "system" and 
        ("[Retrieved Information]" in last_message.get("content", "") or 
         state.get("show_widget") or
         state.get("changes_summary"))
    )
    
    # If we just received results from another agent (and no user message), present them and end
    # This ensures communicator is the final step - it presents results and waits for user
    if just_received_results:
        # Format the results nicely for the user
        if "[Retrieved Information]" in last_message.get("content", ""):
            # Results from retriever - extract and format
            result_text = last_message.get("content", "")
            # Present results to user
            return {
                **state,
                "messages": messages + [{"role": "bot", "content": "I've retrieved the information you requested. How would you like to proceed?"}],
                "next_agent": END,  # End - wait for user response (communicator comes last)
            }
        elif state.get("show_widget"):
            # Results from visualizer - widget is already set in state
            return {
                **state,
                "messages": messages + [{"role": "bot", "content": "I've prepared a visualization for you. Please interact with it or let me know what you'd like to do next."}],
                "next_agent": END,  # End - wait for user response (communicator comes last)
            }
        elif state.get("changes_summary"):
            # Results from modifier - changes were made
            changes = state.get("changes_summary", [])
            changes_text = "\n".join([f"- {c}" for c in changes[-3:]])  # Show last 3 changes
            return {
                **state,
                "messages": messages + [{"role": "bot", "content": f"I've made the following changes to the graph:\n{changes_text}\n\nWhat would you like to do next?"}],
                "next_agent": END,  # End - wait for user response (communicator comes last)
            }
        else:
            # Generic case - just end
            return {
                **state,
                "messages": messages + [{"role": "bot", "content": "I've completed the requested action. How can I help you next?"}],
                "next_agent": END,  # End - wait for user response (communicator comes last)
            }
    
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
                    "messages": messages + [{"role": "bot", "content": "I've analyzed your graph. Please ask me a question or provide feedback."}],
                    "next_agent": None,  # End conversation
                }
            # Need to analyze first - don't add message, analyzer will handle it
            return {
                **state,
                "next_agent": "analyzer",
            }
        else:
            # Use first question - check if it was already asked
            first_q = questions[0]
            question_text = first_q.get('text', '') if isinstance(first_q, dict) else getattr(first_q, 'text', '')
            
            # Clean the question text first (strip prefixes)
            clean_question = question_text
         
            # Check if this question (or cleaned version) was already asked
            # Only check exact matches in recent messages to avoid false positives
            already_asked = any(
                clean_question == msg.get("content", "") or 
                question_text == msg.get("content", "")
                for msg in messages[-10:]  # Check last 10 messages
                if msg.get("role") == "bot"
            )
            
            question_id = first_q.get("id") if isinstance(first_q, dict) else getattr(first_q, "id", None)
            
            if not already_asked:
                # First time asking - present the cleaned question directly
                return {
                    **state,
                    "messages": messages + [{"role": "bot", "content": clean_question}],
                    "current_question_id": question_id,
                    "current_question_text": clean_question,
                    "next_agent": None,  # Wait for user response - None will route to END
                }
            else:
                # Question already asked, just update state and don't add message
                return {
                    **state,
                    "current_question_id": question_id,
                    "current_question_text": clean_question,
                    "next_agent": None,
                }
    
    # Build prompt for LLM
    prompt = build_communicator_prompt(validator, state, user_message)
    
    # Call LLM
    response = validator.api_repo.chat(prompt)
    response_text = extract_text_from_response(response)
    
    # Use JsonHelper for robust JSON parsing (handles fences and extraction automatically)
    response_data = JsonHelper.parse_json(response_text)
    if response_data is None:
        # Fallback: treat as plain text response
        print(f"⚠️  Failed to parse JSON from LLM response")
        print(f"   Response text: {response_text[:200]}...")
        response_data = {
            "text": response_text[:500],  # Limit length
            "next_agent": None,
            "hidden_actions": [],
            "next_question": None,
        }
    
    # Update state
    bot_message = response_data.get("text", response_text)
    next_agent = response_data.get("next_agent")
    hidden_actions = response_data.get("hidden_actions", [])
    next_question = response_data.get("next_question")
    
    # If user just responded to a question, remove it from the questions list
    updated_questions = list(questions)  # Make a copy
    if user_message and current_question_id:
        # User responded - remove the current question from the list
        updated_questions = [q for q in updated_questions if q.get("id") != current_question_id]
        # Clear current question since it's been answered
        current_question_id = None
        current_question_text = None
    
   
    
    # Check for duplicates - only check exact matches, not substring matches
    # This prevents false positives when legitimate messages contain similar text
    is_duplicate = any(
        msg.get("content") == bot_message
        for msg in messages[-5:]  # Only check last 5 messages to avoid false positives
        if msg.get("role") == "bot"
    )
    
    if is_duplicate:
        print(f"⚠️  Duplicate message detected, skipping: {bot_message[:100]}...")
        # Determine next agent before returning
        if next_agent:
            agent_to_call = next_agent
        elif hidden_actions:
            agent_to_call = "modifier"
        else:
            agent_to_call = END
        return {
            **state,
            "questions": updated_questions,  # Update questions list
            "next_agent": agent_to_call,
            "hidden_actions": hidden_actions,
            "current_question_text": next_question or current_question_text,
            "current_question_id": current_question_id,
            "validation_complete": response_data.get("validation_complete", validation_complete),
        }
    
    # Check if bot_message already contains the current question (to avoid duplicates)
    if current_question_text:
        # Check if the question text appears in the bot message (exact match or significant overlap)
        if current_question_text in bot_message or bot_message in current_question_text:
            # The LLM already included the question in its response, don't add it again
            pass
    
    # ========================================================================
    # ROUTING DECISION LOGIC - This determines which node to visit next
    # The value set here (agent_to_call) is stored in state['next_agent']
    # and later read by _route_from_communicator() to route to the next node
    # ========================================================================
    # Check if user explicitly asked for questions
    user_asked_for_questions = user_message and (
        "ask" in user_message.lower() and "question" in user_message.lower()
    ) or "ask questions" in user_message.lower() if user_message else False
    
    # Normalize next_agent (handle 'null' string from JSON)
    if next_agent and isinstance(next_agent, str) and next_agent.lower() == "null":
        next_agent = None
    
    # Priority order for routing decisions:
    if next_agent:
        # 1. LLM explicitly specified next_agent in JSON response
        agent_to_call = next_agent
    elif user_asked_for_questions:
        # 2. User asked for questions -> route to analyzer
        agent_to_call = "analyzer"
    elif hidden_actions:
        # 3. LLM suggested graph modifications -> route to modifier
        agent_to_call = "modifier"
    elif "retrieve" in bot_message.lower() or "get information" in bot_message.lower():
        # 4. Bot message mentions retrieval -> route to retriever
        agent_to_call = "retriever"
    elif "show" in bot_message.lower() or "display" in bot_message.lower() or "widget" in bot_message.lower():
        # 5. Bot message mentions showing/displaying -> route to visualizer
        agent_to_call = "visualizer"
    elif next_question or not validation_complete:
        # 6. Need to generate next question -> route to analyzer
        agent_to_call = "analyzer"
    else:
        # 7. No action needed -> end conversation (wait for user)
        agent_to_call = END
    
    # bot_message is already cleaned above, so just add it
    # If we have remaining questions and no current question, get the next one
    final_current_question_text = next_question or current_question_text
    final_current_question_id = current_question_id
    
    # If user just answered and we have more questions, prepare the next one
    if user_message and updated_questions and not final_current_question_text:
        # Get the next question
        next_q = updated_questions[0]
        if isinstance(next_q, dict):
            final_current_question_text = next_q.get("text", "")
            final_current_question_id = next_q.get("id")
        else:
            final_current_question_text = getattr(next_q, "text", None)
            final_current_question_id = getattr(next_q, "id", None)
    
    # Mark validation as complete if no more questions
    if not updated_questions and not validation_complete:
        validation_complete = True
    
    return {
        **state,
        "messages": messages + [{"role": "bot", "content": bot_message}],
        "questions": updated_questions,  # Update questions list (remove answered ones)
        "next_agent": agent_to_call,
        "hidden_actions": hidden_actions,
        "current_question_text": final_current_question_text,
        "current_question_id": final_current_question_id,
        "validation_complete": validation_complete,
    }

