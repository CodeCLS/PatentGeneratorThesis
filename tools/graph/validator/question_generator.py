"""
Question Generator: Generates validation questions using LLM.
"""
from typing import List, Dict, Any
import json
import logging
import re

logger = logging.getLogger(__name__)

from tools.graph.validator_types import Question, Action, ActionType
from tools.api.llm_api_repo import LLmApi_Repo
from tools.graph.validator.debug_utils import open_debug_browser, format_agent_output
from tools.helper.json_helper import JsonHelper


class QuestionGenerator:
    """Generates validation questions using LLM."""
    
    def __init__(self, api_repo: LLmApi_Repo, debug: bool = False):
        self.api_repo = api_repo
        self.debug = debug
        self._repair_recursion_limit = 10
    
    def generate_questions(self, context: Dict[str, Any]) -> List[Question]:
        """Use LLM to generate validation questions."""
        logger.info("QuestionGenerator: Generating validation questions")
        logger.debug(f"Context: {context['num_nodes']} nodes, {context['num_edges']} edges, {context['num_triples']} triples")
        prompt = (
            "You are a knowledge graph validation expert analyzing a patent knowledge graph.\n\n"
            "TASK: Identify mistakes, discrepancies, unclear connections, and unclear entities.\n\n"
            "GRAPH STATISTICS:\n"
            f"- Graph has {context['num_nodes']} nodes and {context['num_edges']} edges\n"
            f"- {context['num_triples']} triples provided\n"
            f"- {len(context['entities'])} entities\n\n"
            "ALL ENTITIES IN THE GRAPH:\n"
        )
        
        # Add all entities with their properties
        for entity_info in context.get("entities", []):
            entity_line = f"  - {entity_info.get('name', 'UNKNOWN')}"
            if entity_info.get('type'):
                entity_line += f" (Type: {entity_info['type']})"
            if entity_info.get('connections') is not None:
                entity_line += f" [Connections: {entity_info['connections']}]"
            if entity_info.get('properties'):
                props_str = ", ".join([f"{k}={v}" for k, v in list(entity_info['properties'].items())[:3]])
                entity_line += f" (Properties: {props_str})"
            prompt += entity_line + "\n"
        
        prompt += (
            "\n"
            "TYPES OF ISSUES TO FIND:\n"
            "1. MISTAKES: Incorrect relations, wrong entity connections, factual errors\n"
            "2. DISCREPANCIES: Contradictory triples, conflicting information\n"
            "3. UNCLEAR_CONNECTIONS: Vague relations, ambiguous relationships\n"
            "4. UNCLEAR_ENTITIES: Entities with unclear names, duplicates, or missing context\n"
            "5. IMPORTANCE: Triples that might be unimportant or redundant\n\n"
            "ALL TRIPLES IN THE GRAPH:\n"
        )
        
        # Add ALL triple summaries - use human-readable names only, NO IDs
        triples_summary = context.get("triples_summary", [])
        for triple_info in triples_summary:  # Include ALL triples, not just first 50
            prompt += (
                f"  {triple_info['index']}. {triple_info['head']} --[{triple_info['relation']}]--> "
                f"{triple_info['tail']}\n"
            )
        
        if len(triples_summary) == 0:
            prompt += "  (No triples found in graph)\n"
        
        prompt += (
            "\n\n"
            "IMPORTANT GUIDELINES:\n"
            "- Use ONLY human-readable entity names in your questions (e.g., 'water tank', 'air bubble generator')\n"
            "- NEVER mention internal IDs, UUIDs, hashes, or technical identifiers in questions\n"
            "- Write questions in plain, user-friendly language that a non-technical person can understand\n"
            "- Focus on the meaning and relationships, not technical implementation details\n"
            "- When referring to entities, use their descriptive names, not IDs\n\n"
            "For each issue you find, generate a question in this JSON format:\n"
            "{\n"
            '  "id": "q1",\n'
            '  "text": "The question text to ask the user (use entity names, NOT IDs)",\n'
            '  "category": "mistake|discrepancy|unclear_connection|unclear_entity|importance",\n'
            '  "context": {\n'
            '    "triple_indices": [0, 5, 12],  // Indices of related triples\n'
            '    "entity_names": ["water tank", "air bubble generator"],  // Use entity NAMES, not IDs\n'
            '    "issue_description": "Brief description of the issue (user-friendly language)"\n'
            '  },\n'
            '  "priority": 8,  // 1-10, higher = more important\n'
            '  "show_widget": false,  // Whether to show a widget when asking\n'
            '  "widget_type": "importance_selector",  // Optional widget type\n'
            '  "widget_parameters": {},  // Optional widget parameters\n'
            '  "suggested_actions": [\n'
            '    {\n'
            '      "type": "show_triples",\n'
            '      "parameters": {"triple_indices": [0, 5]}\n'
            '    },\n'
            '    {\n'
            '      "type": "ask_importance",\n'
            '      "parameters": {"triple_index": 12}\n'
            '    }\n'
            '  ]\n'
            '}\n\n'
            "CRITICAL: Return ONLY valid JSON. No markdown fences, no commentary, no explanation.\n"
            "CRITICAL: Ensure all strings are properly closed with double quotes.\n"
            "CRITICAL: Do not include newlines or special characters in string values that break JSON.\n"
            "CRITICAL: If a question text is long, keep it on a single line or properly escape it.\n"
            "Focus on the most important issues first (high priority).\n"
            "Generate 5-10 questions maximum.\n"
        )
        
        try:
            logger.info("Calling LLM API for question generation")
            logger.debug(f"Prompt length: {len(prompt)} characters")
            response = self.api_repo.chat(prompt)
            logger.info("Received response from LLM")
            
            # Parse response
            logger.debug("Parsing LLM response")
            if isinstance(response, dict):
                response_text = response.get("content", response.get("text", response.get("message", "")))
                if not response_text and "choices" in response:
                    response_text = response["choices"][0].get("message", {}).get("content", "")
            else:
                response_text = str(response) if response else ""
            
            # Clean and parse JSON
            response_text = response_text.strip()
            logger.debug(f"Raw response text length: {len(response_text)} characters")
            
            # Try to parse using JsonHelper (handles markdown fences and Python literals)
            questions_data = JsonHelper.parse_json(response_text)
            
            # If JsonHelper fails, try to repair common JSON issues
            if questions_data is None:
                logger.warning("JsonHelper.parse_json failed, attempting JSON repair...")
                print(f"⚠️  JsonHelper.parse_json failed, attempting JSON repair...")
                questions_data = self._repair_json(response_text, recursion_depth=0)
            
            # If still None, log error with full details
            if questions_data is None:
                logger.error(f"Failed to parse questions JSON after repair attempts")
                
                # Try to get the actual JSON error
                json_error_msg = "Unknown error"
                json_error_pos = None
                try:
                    json.loads(response_text)  # This will raise the actual error
                except json.JSONDecodeError as e:
                    json_error_msg = e.msg
                    json_error_pos = e.pos
                
                print(f"⚠️  Failed to parse questions JSON: {json_error_msg}")
                if json_error_pos is not None:
                    print(f"   Error at position: {json_error_pos}")
                print(f"   Response length: {len(response_text)} characters")
                print(f"\n   === FULL RESPONSE ===")
                print(response_text)
                print(f"   === END OF RESPONSE ===\n")
                
                # Show context around error if we have position
                if json_error_pos is not None and json_error_pos < len(response_text):
                    start = max(0, json_error_pos - 100)
                    end = min(len(response_text), json_error_pos + 100)
                    print(f"   Context around error position {json_error_pos}:")
                    print(f"   ...{response_text[start:end]}...")
                
                # Save full response to file for debugging
                import os
                debug_file = "question_generator_debug_response.json"
                try:
                    with open(debug_file, "w", encoding="utf-8") as f:
                        f.write(response_text)
                    print(f"\n   Full response saved to: {debug_file}")
                except Exception as save_error:
                    logger.debug(f"Could not save debug file: {save_error}")
                
                return []
            
            logger.debug(f"Parsed {len(questions_data) if isinstance(questions_data, list) else 1} question(s)")
            
            # Debug: Open browser with output
            if self.debug:
                debug_content = format_agent_output(
                    agent_name="QuestionGenerator",
                    input_data=prompt[:2000] + "..." if len(prompt) > 2000 else prompt,  # Limit prompt size
                    output_data=questions_data,
                    metadata={
                        "prompt_length": len(prompt),
                        "response_length": len(response_text),
                        "questions_generated": len(questions_data) if isinstance(questions_data, list) else 1,
                    }
                )
                open_debug_browser(debug_content, title="QuestionGenerator Output")
            
            if not isinstance(questions_data, list):
                questions_data = [questions_data]
            
            # Convert to Question objects
            questions = []
            for i, q_data in enumerate(questions_data):
                if not isinstance(q_data, dict):
                    continue
                
                question_id = q_data.get("id", f"q{i+1}")
                text = q_data.get("text", "")
                category = q_data.get("category", "unclear_connection")
                context_data = q_data.get("context", {})
                priority = q_data.get("priority", 5)
                show_widget = q_data.get("show_widget", False)
                widget_type = q_data.get("widget_type")
                widget_parameters = q_data.get("widget_parameters", {})
                
                # Parse suggested actions
                suggested_actions = []
                for action_data in q_data.get("suggested_actions", []):
                    try:
                        action_type = ActionType(action_data.get("type", "show_triples"))
                        suggested_actions.append(Action(
                            type=action_type,
                            parameters=action_data.get("parameters", {}),
                            description=action_data.get("description"),
                        ))
                    except ValueError:
                        continue
                
                questions.append(Question(
                    id=question_id,
                    text=text,
                    category=category,
                    context=context_data,
                    priority=priority,
                    suggested_actions=suggested_actions,
                    show_widget=show_widget,
                    widget_type=widget_type,
                    widget_parameters=widget_parameters,
                ))
            
            # Sort by priority (highest first)
            questions.sort(key=lambda q: q.priority, reverse=True)
            logger.info(f"QuestionGenerator: Generated {len(questions)} questions (sorted by priority)")
            if questions:
                logger.debug(f"Highest priority question: {questions[0].text[:100]}...")
            
            return questions
            
        except Exception as e:
            logger.error(f"Error generating questions: {e}", exc_info=True)
            print(f"⚠️  Error generating questions: {e}")
            if hasattr(e, 'pos') and hasattr(e, 'msg'):
                # JSON decode error with position info
                print(f"   JSON error at position {e.pos}: {e.msg}")
                if response_text:
                    start = max(0, e.pos - 50)
                    end = min(len(response_text), e.pos + 50)
                    print(f"   Context: ...{response_text[start:end]}...")
            return []
    
    def _repair_json(self, text: str, recursion_depth: int = 0) -> Any:
        """
        Attempt to repair common JSON issues like unterminated strings.
        Uses a simpler approach: extract complete JSON objects from the array.
        """
        if recursion_depth >= self._repair_recursion_limit:
            logger.warning(f"JSON repair recursion limit ({self._repair_recursion_limit}) reached")
            print(f"⚠️  JSON repair recursion limit ({self._repair_recursion_limit}) reached")
            return None
        
        logger.debug(f"Attempting JSON repair (depth: {recursion_depth})")
        try:
            # Remove markdown fences if present
            text = text.strip()
            if text.startswith("```"):
                first_newline = text.find("\n")
                if first_newline != -1:
                    text = text[first_newline + 1:].strip()
                if text.endswith("```"):
                    text = text[:-3].strip()
            
            # Find the start of the JSON array
            array_start = text.find("[")
            if array_start == -1:
                return None
            
            # Strategy: Use regex to find complete JSON objects
            # Look for patterns like {"id": "...", "text": "...", ...}
            # This is simpler than trying to parse character by character
            
            # Try to find all complete objects using regex
            # Pattern: { ... } where braces are balanced
            objects = []
            i = array_start + 1
            brace_count = 0
            obj_start = None
            in_string = False
            escape = False
            
            while i < len(text):
                char = text[i]
                
                if escape:
                    escape = False
                    i += 1
                    continue
                
                if char == "\\":
                    escape = True
                    i += 1
                    continue
                
                if char == '"' and not escape:
                    in_string = not in_string
                    i += 1
                    continue
                
                if not in_string:
                    if char == "{":
                        if brace_count == 0:
                            obj_start = i
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0 and obj_start is not None:
                            # Found a complete object
                            obj_text = text[obj_start:i+1]
                            try:
                                obj = json.loads(obj_text)
                                objects.append(obj)
                            except json.JSONDecodeError:
                                # Try to repair this specific object
                                repaired = self._repair_json_object(obj_text, recursion_depth + 1)
                                if repaired:
                                    objects.append(repaired)
                            obj_start = None
                    elif char == "]" and brace_count == 0:
                        # End of array
                        break
                
                i += 1
            
            # Check if we ended while in a string (text was cut off mid-string)
            # This can happen even if obj_start is None (if cut off very early)
            if in_string:
                # We have an incomplete object with an unterminated string
                # Find where the incomplete object starts
                if obj_start is None:
                    # Text was cut off before we found a complete object start
                    # Look backwards from current position to find the last "{"
                    for j in range(i - 1, array_start, -1):
                        if text[j] == "{":
                            obj_start = j
                            break
                
                if obj_start is not None:
                    incomplete_obj_text = text[obj_start:]
                    logger.debug(f"Detected incomplete object with unterminated string (length: {len(incomplete_obj_text)})")
                    print(f"⚠️  Detected incomplete object with unterminated string, attempting repair...")
                    
                    # Try to repair this incomplete object
                    repaired_obj = self._repair_json_object(incomplete_obj_text, recursion_depth + 1)
                    if repaired_obj:
                        objects.append(repaired_obj)
                        logger.info("Successfully repaired incomplete object")
                        print(f"✓ Successfully repaired incomplete object")
                    else:
                        logger.warning("Could not repair incomplete object")
                        print(f"⚠️  Could not repair incomplete object")
                else:
                    # Couldn't find object start, try to construct minimal object from what we have
                    logger.debug("Could not find object start, attempting minimal object construction")
                    print(f"⚠️  Could not find object start, attempting minimal object construction...")
                    # Look for the last complete field before the unterminated string
                    # Find where we are in the text
                    last_complete_field = text.rfind(',', array_start, i)
                    if last_complete_field != -1:
                        # Try to extract up to the last complete field and add minimal structure
                        partial_text = text[:last_complete_field + 1]
                        # Try to find what field we're in
                        # Look for the pattern: "text": "...
                        text_field_match = re.search(r'"text":\s*"([^"]*)$', partial_text)
                        if text_field_match:
                            # We're in the text field, close it and the object
                            repaired_text = partial_text.rstrip(',') + '}"}'
                            try:
                                # Try to parse as a complete object
                                repaired_obj = json.loads(repaired_text)
                                objects.append(repaired_obj)
                                logger.info("Successfully constructed minimal object")
                                print(f"✓ Successfully constructed minimal object")
                            except json.JSONDecodeError:
                                pass
            
            if objects:
                logger.info(f"JSON repair successful: extracted {len(objects)} complete object(s)")
                print(f"✓ JSON repair successful: extracted {len(objects)} complete object(s)")
                return objects
            
            # Fallback: Try to extract objects even if the array is incomplete
            # Handle case where text is cut off mid-string in the last object
            # Look for the last complete object by finding the last "}" that's not inside a string
            last_brace = -1
            brace_count = 0
            in_string = False
            escape = False
            
            for i in range(len(text) - 1, array_start, -1):
                char = text[i]
                
                if escape:
                    escape = False
                    continue
                
                if char == "\\":
                    escape = True
                    continue
                
                if char == '"' and not escape:
                    in_string = not in_string
                    continue
                
                if not in_string:
                    if char == "}":
                        last_brace = i
                        break
            
            if last_brace > array_start:
                # Extract up to the last complete brace
                truncated = text[:last_brace + 1]
                
                # Try to repair the last object if it has issues
                last_obj_start = truncated.rfind("{")
                if last_obj_start != -1:
                    last_obj_text = truncated[last_obj_start:]
                    repaired_obj = self._repair_json_object(last_obj_text, recursion_depth + 1)
                    if repaired_obj:
                        # Replace with repaired object
                        truncated = truncated[:last_obj_start] + json.dumps(repaired_obj)
                
                if not truncated.endswith("]"):
                    truncated += "]"
                try:
                    result = json.loads(truncated)
                    if isinstance(result, list):
                        return result
                except json.JSONDecodeError:
                    # If still failing, return what we have
                    if objects:
                        return objects
                    pass
            
            return None
            
        except Exception as e:
            logger.debug(f"JSON repair failed: {e}", exc_info=True)
            print(f"⚠️  JSON repair failed: {e}")
            return None
    
    def _repair_json_object(self, obj_text: str, recursion_depth: int = 0) -> Dict[str, Any]:
        """
        Attempt to repair a single JSON object with unterminated strings.
        """
        if recursion_depth >= self._repair_recursion_limit:
            return None
        
        try:
            # First, check if we're in an unterminated string at the end
            # Track string state from the beginning
            in_string = False
            escape = False
            last_string_start = -1
            last_string_field = None
            
            for i, char in enumerate(obj_text):
                if escape:
                    escape = False
                    continue
                
                if char == "\\":
                    escape = True
                    continue
                
                if char == '"' and not escape:
                    if not in_string:
                        in_string = True
                        last_string_start = i
                        # Look backwards to find the field name
                        lookback = obj_text[:i]
                        field_match = re.search(r'"(\w+)":\s*"$', lookback)
                        if field_match:
                            last_string_field = field_match.group(1)
                    else:
                        in_string = False
                        last_string_start = -1
                        last_string_field = None
            
            # If we're still in a string at the end, we have an unterminated string
            if in_string and last_string_start != -1:
                logger.debug(f"Detected unterminated string in field '{last_string_field}' starting at position {last_string_start}")
                print(f"⚠️  Detected unterminated string in field '{last_string_field}', attempting to close...")
                
                # Strategy 1: Close the string and add minimal required fields
                # Find where the string content starts
                string_content_start = last_string_start + 1
                string_content = obj_text[string_content_start:]
                
                # Close the string and complete the object
                # If we have at least "id" and "text" fields, we can complete it
                has_id = '"id"' in obj_text[:last_string_start]
                has_text = last_string_field == "text"
                
                if has_id and has_text:
                    # We have id and text (even if incomplete), close the text field and object
                    repaired = obj_text + '"'  # Close the string
                    # Add closing brace if missing
                    if not repaired.rstrip().endswith("}"):
                        repaired += "}"
                    try:
                        result = json.loads(repaired)
                        logger.info(f"Successfully closed unterminated string in '{last_string_field}' field")
                        print(f"✓ Successfully closed unterminated string")
                        return result
                    except json.JSONDecodeError as e:
                        logger.debug(f"Strategy 1 failed: {e}")
                
                # Strategy 2: If the string is very long, truncate it
                max_string_length = 500
                if len(string_content) > max_string_length:
                    truncated_content = string_content[:max_string_length]
                    repaired = obj_text[:string_content_start] + truncated_content + '..."' + "}"
                    try:
                        result = json.loads(repaired)
                        logger.info(f"Successfully truncated and closed long unterminated string")
                        print(f"✓ Successfully truncated and closed long unterminated string")
                        return result
                    except json.JSONDecodeError as e:
                        logger.debug(f"Strategy 2 failed: {e}")
                
                # Strategy 3: Close the string with ellipsis to indicate truncation
                repaired = obj_text + '..."'  # Close with ellipsis
                if not repaired.rstrip().endswith("}"):
                    repaired += "}"
                try:
                    result = json.loads(repaired)
                    logger.info(f"Successfully closed unterminated string with ellipsis")
                    print(f"✓ Successfully closed unterminated string with ellipsis")
                    return result
                except json.JSONDecodeError as e:
                    logger.debug(f"Strategy 3 failed: {e}")
            
            # Strategy: Find the last field that might be unterminated and close it
            # First, try to find where the object should end
            last_brace = obj_text.rfind("}")
            if last_brace == -1:
                # No closing brace - try to add one
                # Find the last complete field
                last_comma = obj_text.rfind(",")
                if last_comma != -1:
                    # Truncate at last comma and add closing brace
                    repaired = obj_text[:last_comma] + "}"
                    try:
                        return json.loads(repaired)
                    except json.JSONDecodeError:
                        pass
                return None
            
            # Check if there's an unterminated string before the last brace
            before_brace = obj_text[:last_brace].rstrip()
            
            # Track string state to find unterminated strings
            in_string = False
            escape = False
            last_string_start = -1
            string_field_name = None
            i = 0
            
            while i < len(before_brace):
                char = before_brace[i]
                
                if escape:
                    escape = False
                    i += 1
                    continue
                
                if char == "\\":
                    escape = True
                    i += 1
                    continue
                
                if char == '"' and not escape:
                    if not in_string:
                        in_string = True
                        last_string_start = i
                        # Look backwards to find the field name
                        lookback = before_brace[:i]
                        field_match = re.search(r'"(\w+)":\s*"$', lookback)
                        if field_match:
                            string_field_name = field_match.group(1)
                    else:
                        in_string = False
                        last_string_start = -1
                        string_field_name = None
                    i += 1
                    continue
                
                i += 1
            
            # If we're still in a string at the end, we have an unterminated string
            if in_string and last_string_start != -1:
                # The string was cut off - we need to close it
                # Strategy 1: Close the string right before the brace
                repaired = before_brace + '"' + obj_text[last_brace:]
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass
                
                # Strategy 2: If the string is very long, truncate it and close
                # Find where the string content starts (after the opening quote)
                string_content_start = last_string_start + 1
                # Truncate the string content to a reasonable length (e.g., 200 chars)
                max_string_length = 200
                if len(before_brace) - string_content_start > max_string_length:
                    # Truncate the string
                    truncated_string = before_brace[:string_content_start] + before_brace[string_content_start:string_content_start + max_string_length] + '..."' + obj_text[last_brace:]
                    try:
                        return json.loads(truncated_string)
                    except json.JSONDecodeError:
                        pass
                
                # Strategy 3: Remove the unterminated field entirely if it's the last one
                # Find the last comma before the unterminated string
                before_string = before_brace[:last_string_start]
                last_comma = before_string.rfind(",")
                if last_comma != -1:
                    # Truncate at the last complete field
                    truncated = before_brace[:last_comma] + "}"
                    try:
                        return json.loads(truncated)
                    except json.JSONDecodeError:
                        pass
                
                # Strategy 4: Last resort - close with ellipsis
                repaired = before_brace + '..."' + obj_text[last_brace:]
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass
            
            # Fallback: Try to truncate the last field if it's unterminated
            last_comma = before_brace.rfind(",")
            if last_comma != -1:
                truncated = before_brace[:last_comma] + "}"
                try:
                    return json.loads(truncated)
                except json.JSONDecodeError:
                    pass
            
            logger.debug("Object repair: all strategies failed")
            return None
            
        except Exception as e:
            logger.debug(f"Object repair exception: {e}", exc_info=True)
            return None

