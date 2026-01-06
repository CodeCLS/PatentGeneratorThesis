"""
Analyzer node - generates new questions and analyzes the graph.
"""

import json
from typing import TYPE_CHECKING, Dict, List, Optional, Any

from tools.helper.json_helper import JsonHelper

# Import GraphValidatorState at runtime (not just TYPE_CHECKING)
# This is needed because LangGraph might inspect type hints at runtime
from tools.graph.langgraph.state import GraphValidatorState

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def build_context(validator: "GraphValidatorLangGraph") -> Dict[str, Any]:
    """Build context for analysis by actually analyzing the graph structure."""
    context = {
        "num_nodes": validator.graph.number_of_nodes() if validator.graph else 0,
        "num_edges": validator.graph.number_of_edges() if validator.graph else 0,
        "num_triples": len(validator.triples),
        "entities": list(validator.id_to_name.keys()),
    }
    
    # Actually analyze the graph to find specific issues
    issues = []
    
    if validator.graph:
        # Find duplicate relations (same head->tail with same relation)
        duplicate_relations = []
        relation_counts = {}
        for triple in validator.triples:
            head_id = getattr(triple.head, "ref", None) or getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None) or str(triple.head)
            tail_id = getattr(triple.tail, "ref", None) or getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None) or str(triple.tail)
            key = (head_id, tail_id, triple.relation)
            if key in relation_counts:
                duplicate_relations.append({
                    "head": validator.id_to_name.get(head_id, head_id),
                    "tail": validator.id_to_name.get(tail_id, tail_id),
                    "relation": triple.relation
                })
            relation_counts[key] = relation_counts.get(key, 0) + 1
        
        if duplicate_relations:
            issues.append({
                "type": "duplicate_relations",
                "count": len(duplicate_relations),
                "examples": duplicate_relations[:3]
            })
        
        # Find nodes with many incoming edges of same type
        incoming_by_relation = {}
        for triple in validator.triples:
            tail_id = getattr(triple.tail, "ref", None) or getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None) or str(triple.tail)
            head_id = getattr(triple.head, "ref", None) or getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None) or str(triple.head)
            key = (tail_id, triple.relation, head_id)
            if key not in incoming_by_relation:
                incoming_by_relation[key] = []
            incoming_by_relation[key].append(head_id)
        
        multiple_same_relation = []
        for (tail_id, relation, head_id), sources in incoming_by_relation.items():
            if len(sources) > 1:
                multiple_same_relation.append({
                    "target": validator.id_to_name.get(tail_id, tail_id),
                    "relation": relation,
                    "sources": [validator.id_to_name.get(s, s) for s in sources[:3]]
                })
        
        if multiple_same_relation:
            issues.append({
                "type": "multiple_same_relation",
                "count": len(multiple_same_relation),
                "examples": multiple_same_relation[:3]
            })
        
        # Find isolated nodes (no connections)
        isolated = []
        if validator.graph and hasattr(validator.graph, 'nodes') and hasattr(validator.graph, 'degree'):
            try:
                for node_id in list(validator.graph.nodes()):
                    try:
                        if validator.graph.degree(node_id) == 0:
                            isolated.append(validator.id_to_name.get(node_id, node_id))
                    except (KeyError, AttributeError):
                        # Node might not exist or graph structure issue
                        continue
            except (AttributeError, TypeError):
                # Graph might not be properly initialized
                pass
        
        if isolated:
            issues.append({
                "type": "isolated_nodes",
                "count": len(isolated),
                "examples": isolated[:5]
            })
        
        # Find triples with complex relations (those that might need simplification)
        complex_relations = []
        for triple in validator.triples:
            if hasattr(triple, "properties") and triple.properties:
                # Already simplified, check if original was complex
                original = triple.properties.get("original_relation", "")
                if original and len(original.split()) > 5:
                    complex_relations.append({
                        "head": validator.id_to_name.get(getattr(triple.head, "ref", None) or getattr(triple.head, "id", None) or "", ""),
                        "tail": validator.id_to_name.get(getattr(triple.tail, "ref", None) or getattr(triple.tail, "id", None) or "", ""),
                        "simplified": triple.relation,
                        "original": original
                    })
            elif len(triple.relation.split()) > 5:
                complex_relations.append({
                    "head": validator.id_to_name.get(getattr(triple.head, "ref", None) or getattr(triple.head, "id", None) or "", ""),
                    "tail": validator.id_to_name.get(getattr(triple.tail, "ref", None) or getattr(triple.tail, "id", None) or "", ""),
                    "relation": triple.relation
                })
        
        if complex_relations:
            issues.append({
                "type": "complex_relations",
                "count": len(complex_relations),
                "examples": complex_relations[:3]
            })
    
    context["issues"] = issues
    return context


def repair_questions_json(json_text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Attempt to repair JSON with unterminated strings.
    Similar to QuestionGenerator._repair_json but simpler.
    """
    try:
        # Check if we're in an unterminated string at the end
        in_string = False
        escape = False
        last_string_start = -1
        
        for i, char in enumerate(json_text):
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
                else:
                    in_string = False
                    last_string_start = -1
        
        # If we're still in a string at the end, try to close it
        if in_string and last_string_start != -1:
            # Strategy 1: Close the string and complete the object/array
            repaired = json_text + '"'  # Close the string
            
            # Count braces to see if we need to close the object
            open_braces = json_text.count('{') - json_text.count('}')
            open_brackets = json_text.count('[') - json_text.count(']')
            
            # Close braces if needed
            if open_braces > 0:
                repaired += '}' * open_braces
            if open_brackets > 0:
                repaired += ']' * open_brackets
            
            # Try JsonHelper first, then fall back to json.loads
            result = JsonHelper.parse_json(repaired)
            if result is not None and isinstance(result, list):
                print(f"✓ Successfully repaired unterminated string in JSON")
                return result
            # Fallback to json.loads for validation
            try:
                result = json.loads(repaired)
                if isinstance(result, list):
                    print(f"✓ Successfully repaired unterminated string in JSON")
                    return result
            except json.JSONDecodeError:
                pass
            
            # Strategy 2: Truncate the string if it's very long
            string_content = json_text[last_string_start + 1:]
            if len(string_content) > 500:
                truncated = json_text[:last_string_start + 1] + string_content[:500] + '..."'
                if open_braces > 0:
                    truncated += '}' * open_braces
                if open_brackets > 0:
                    truncated += ']' * open_brackets
                # Try JsonHelper first, then fall back to json.loads
                result = JsonHelper.parse_json(truncated)
                if result is not None and isinstance(result, list):
                    print(f"✓ Successfully repaired by truncating long string")
                    return result
                # Fallback to json.loads for validation
                try:
                    result = json.loads(truncated)
                    if isinstance(result, list):
                        print(f"✓ Successfully repaired by truncating long string")
                        return result
                except json.JSONDecodeError:
                    pass
        
        return None
        
    except Exception as e:
        return None


def generate_questions(validator: "GraphValidatorLangGraph", context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate specific validation questions based on actual graph analysis."""
    issues = context.get("issues", [])
    
    # Build examples - reduced for token efficiency
    detailed_examples = []
    for i, triple in enumerate(validator.triples):
        head_id = getattr(triple.head, "ref", None) or getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None) or str(triple.head)
        tail_id = getattr(triple.tail, "ref", None) or getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None) or str(triple.tail)
        head_name = validator.id_to_name.get(head_id, getattr(triple.head, "name", str(triple.head)))
        tail_name = validator.id_to_name.get(tail_id, getattr(triple.tail, "name", str(triple.tail)))
        
        detailed_examples.append({
            "index": i,
            "head": head_name,
            "relation": triple.relation,
            "tail": tail_name
        })
    
    if not issues:
        # No issues found - validation is complete
        # Return empty questions list - the analyzer_node will mark validation_complete = True
        return []
    else:
        # Generate specific questions based on found issues with triple details
        issues_text = "\n".join([
            f"- {issue['type']}: {issue['count']} instances.\n"
            + "\n".join([
                f"  Example: {ex}" for ex in issue['examples'][:3]
            ])
            for issue in issues
        ])
        
        # Find triples that match the issues for specific questions
        issue_triples = []
        for issue in issues:
            if issue['type'] == 'duplicate_relations' and issue['examples']:
                # Find triples matching the duplicate examples
                for ex in issue['examples'][:2]:
                    head_name = ex.get('head', '')
                    tail_name = ex.get('tail', '')
                    relation = ex.get('relation', '')
                    for t in detailed_examples:
                        if (t['head'] == head_name and t['tail'] == tail_name and t['relation'] == relation):
                            issue_triples.append(t)
                            break
        
        if not issue_triples:
            issue_triples = detailed_examples[:3]  # Fallback to first 3
        
        specific_issue_triples = "\n".join([
            f"  Triple {t['index']}: {t['head']} --[{t['relation']}]--> {t['tail']}"
            for t in issue_triples[:5]
        ])
        
        prompt = (
            f"Graph: {context['num_nodes']} nodes, {context['num_edges']} edges, {context['num_triples']} triples\n\n"
            f"Issues found:\n{issues_text}\n\n"
            f"Triples with issues:\n{specific_issue_triples}\n\n"
            "Generate 3-4 SPECIFIC questions. Each must:\n"
            "- Include triple index (e.g., 'triple 5')\n"
            "- Use actual entity names\n"
            "- Focus on ONE issue/triple\n"
            "- Reference the issue found\n"
            "- Be conversational, not generic\n\n"
            "Return ONLY JSON array (no text before/after):\n"
            '[{"id": "q1", "text": "I found duplicate relations. Triple 5: Entity A --[connects]--> Entity B. Should this be removed?", "category": "mistake", "priority": 8}]\n'
            "CRITICAL: Keep JSON output under 1000 tokens. Use short question text (max 200 chars per question)."
        )
    
    # Make multiple smaller API calls to get 8-12 questions total while keeping each response under 1K tokens
    all_questions = []
    max_batches = 3  # Generate up to 3 batches of 3-4 questions each = 9-12 questions total
    
    for batch_num in range(max_batches):
        # Update prompt to request fewer questions per batch
        
        response = validator.api_repo.chat(prompt)
        response_text = str(response)
        
        # Use JsonHelper for robust JSON parsing
        questions = JsonHelper.parse_json(response_text)
        
        # If JsonHelper fails, try to repair unterminated strings
        if questions is None:
            questions = repair_questions_json(response_text)
        
        # If still None, log error and continue to next batch
        if questions is None:
            if batch_num == 0:  # Only log error on first batch
                print(f"⚠️  Failed to parse questions JSON in batch {batch_num + 1}")
                print(f"   Response text (first 500 chars): {response_text[:500]}")
                print(f"   Response text (last 500 chars): {response_text[-500:]}")
                # Save full response to file for debugging
                try:
                    import os
                    debug_file = "question_generation_debug_response.json"
                    with open(debug_file, "w", encoding="utf-8") as f:
                        f.write(response_text)
                    print(f"   Full response saved to: {debug_file}")
                except Exception as e:
                    print(f"   Could not save debug file: {e}")
            break
        
        if not isinstance(questions, list):
            questions = [questions]
        
        # Add batch number to question IDs to avoid duplicates
        for q in questions:
            if isinstance(q, dict):
                q["id"] = f"q{len(all_questions) + 1}"
        
        all_questions.extend(questions)
        
    
    questions = all_questions[:30]  # Cap at 12 questions total
    
    # Filter out abstract/generic questions
   
    
    
    return questions


def analyzer_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState") -> "GraphValidatorState":
    """
    Analysis agent - generates new questions and analyzes the graph.
    """
    # Build context
    context = build_context(validator)
    
    # Generate questions using LLM
    questions = generate_questions(validator, context)
    
    # Update state with new questions
    next_question = None
    question_id = None
    messages = state.get("messages", [])
    
    if questions:
        first_q = questions[0]
        if isinstance(first_q, dict):
            next_question = first_q.get("text")
            question_id = first_q.get("id")
        else:
            next_question = getattr(first_q, "text", None)
            question_id = getattr(first_q, "id", None)
        
        # Don't add question to messages here - let communicator handle it
        # This prevents duplicate questions
    else:
        # No questions generated - no issues found, validation is complete
        messages.append({"role": "bot", "content": "I've analyzed your graph and found no issues. Validation is complete!"})
    
    return {
        **state,
        "messages": messages,
        "questions": questions if questions else [],  # Ensure questions is always a list
        "current_question_text": next_question,
        "current_question_id": question_id,
        "validation_complete": not questions,  # Mark complete if no questions (no issues found)
        "next_agent": None if not questions else "communicator",  # End if no questions to prevent loop
    }

