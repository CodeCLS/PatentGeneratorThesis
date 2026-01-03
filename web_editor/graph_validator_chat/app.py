"""
Flask server for Graph Validator Chat Interface.
"""
# Fix Jinja2 compatibility issue - must be imported BEFORE Flask
import web_editor.jinja2_compat  # noqa: F401

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
from typing import List, Dict, Any, Optional
import threading
import webbrowser
import time
import networkx as nx

from tools.graph.Triple import Triple
from tools.graph.graph_validator import GraphValidator

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Global validator instance
validator: Optional[GraphValidator] = None


def initialize_validator(
    graph: Optional[nx.MultiDiGraph] = None,
    triples: Optional[List[Triple]] = None,
    id_to_name: Optional[Dict[str, str]] = None,
) -> Optional[GraphValidator]:
    """
    Initialize the validator with graph and/or triples.
    
    Args:
        graph: Optional NetworkX graph
        triples: Optional list of Triple objects
        id_to_name: Optional mapping from entity ID to display name
    
    Returns:
        GraphValidator instance
    """
    global validator
    
    validator = GraphValidator()
    validator.analyze(graph=graph, triples=triples, id_to_name=id_to_name)
    
    return validator


def get_validator() -> Optional[GraphValidator]:
    """Get the current validator instance."""
    return validator


@app.route('/')
def index():
    """Serve the main chat interface."""
    return render_template('index.html')


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current validator status."""
    if not validator:
        return jsonify({
            "initialized": False,
            "message": "Validator not initialized"
        })
    
    return jsonify({
        "initialized": True,
        "num_questions": len(validator.questions),
        "num_responses": len(validator.responses),
        "has_graph": validator.graph is not None,
        "num_triples": len(validator.triples),
    })


@app.route('/api/questions', methods=['GET'])
def get_questions():
    """Get all questions."""
    if not validator:
        return jsonify({"error": "Validator not initialized"}), 400
    
    questions_data = []
    for q in validator.getAllQuestions():
        questions_data.append({
            "id": q.id,
            "text": q.text,
            "category": q.category,
            "priority": q.priority,
            "show_widget": q.show_widget,
            "widget_type": q.widget_type,
            "widget_parameters": q.widget_parameters,
        })
    
    return jsonify({"questions": questions_data})


@app.route('/api/questions/first', methods=['GET'])
def get_first_question():
    """Get the first (highest priority) question."""
    if not validator:
        return jsonify({"error": "Validator not initialized"}), 400
    
    question = validator.getFirstQuestion()
    if not question:
        return jsonify({"question": None})
    
    return jsonify({
        "question": {
            "id": question.id,
            "text": question.text,
            "category": question.category,
            "priority": question.priority,
            "show_widget": question.show_widget,
            "widget_type": question.widget_type,
            "widget_parameters": question.widget_parameters,
        }
    })


@app.route('/api/questions/<question_id>/answer', methods=['POST'])
def answer_question(question_id: str):
    """Answer a question and get response."""
    if not validator:
        return jsonify({"error": "Validator not initialized"}), 400
    
    data = request.get_json()
    answer_text = data.get("answer", "")
    
    if not answer_text:
        return jsonify({"error": "Answer text is required"}), 400
    
    # Get response from validator
    response = validator.answerQuestion(question_id, answer_text)
    
    # Format response for frontend
    response_data = {
        "question_id": response.question_id,
        "text": response.text,
        "show_widget": response.show_widget,
        "widget_type": response.widget_type,
        "actions": [
            {
                "type": action.type.value,
                "parameters": action.parameters,
                "description": action.description,
            }
            for action in response.actions
        ],
        "hidden_actions": [
            {
                "type": action.type.value,
                "parameters": action.parameters,
                "description": action.description,
            }
            for action in response.hidden_actions
        ],
    }
    
    return jsonify(response_data)


@app.route('/api/state', methods=['GET'])
def get_state():
    """Get current validator state (graph, triples, entities)."""
    if not validator:
        return jsonify({"error": "Validator not initialized"}), 400
    
    # Get updated graph and triples
    updated_graph = validator.getUpdatedGraph()
    updated_triples = validator.getUpdatedTriples()
    changes = validator.getChanges()
    
    # Serialize graph info (just counts for display)
    graph_info = None
    if updated_graph:
        graph_info = {
            "num_nodes": updated_graph.number_of_nodes(),
            "num_edges": updated_graph.number_of_edges(),
        }
    
    # Extract entities from triples
    entities = []
    entity_ids = set()
    for triple in updated_triples:
        head_id = getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None)
        tail_id = getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None)
        
        if head_id and head_id not in entity_ids:
            entities.append({
                "id": head_id,
                "name": getattr(triple.head, "name", ""),
                "label": getattr(triple.head, "label", ""),
            })
            entity_ids.add(head_id)
        
        if tail_id and tail_id not in entity_ids:
            entities.append({
                "id": tail_id,
                "name": getattr(triple.tail, "name", ""),
                "label": getattr(triple.tail, "label", ""),
            })
            entity_ids.add(tail_id)
    
    return jsonify({
        "graph": graph_info,
        "num_triples": len(updated_triples),
        "num_entities": len(entities),
        "entities": entities[:100],  # Limit to first 100
        "id_to_name": validator.id_to_name,
        "changes": changes,
    })


@app.route('/api/export', methods=['GET'])
def export_data():
    """Export full graph, triples, and entities for notebook use."""
    if not validator:
        return jsonify({"error": "Validator not initialized"}), 400
    
    import pickle
    import base64
    
    # Get updated data
    updated_graph = validator.getUpdatedGraph()
    updated_triples = validator.getUpdatedTriples()
    
    # Serialize graph as base64 pickle
    graph_data = None
    if updated_graph:
        graph_bytes = pickle.dumps(updated_graph)
        graph_data = base64.b64encode(graph_bytes).decode('utf-8')
    
    # Serialize triples (convert to dict format)
    triples_data = []
    for triple in updated_triples:
        head_id = getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None)
        tail_id = getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None)
        
        triples_data.append({
            "head": {
                "id": head_id,
                "name": getattr(triple.head, "name", ""),
                "label": getattr(triple.head, "label", ""),
                "ref_short": getattr(triple.head, "ref_short", ""),
            },
            "relation": triple.relation,
            "tail": {
                "id": tail_id,
                "name": getattr(triple.tail, "name", ""),
                "label": getattr(triple.tail, "label", ""),
                "ref_short": getattr(triple.tail, "ref_short", ""),
            },
        })
    
    # Extract all entities
    entities = []
    entity_ids = set()
    for triple in updated_triples:
        head_id = getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None)
        tail_id = getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None)
        
        if head_id and head_id not in entity_ids:
            entities.append({
                "id": head_id,
                "name": getattr(triple.head, "name", ""),
                "label": getattr(triple.head, "label", ""),
                "ref_short": getattr(triple.head, "ref_short", ""),
            })
            entity_ids.add(head_id)
        
        if tail_id and tail_id not in entity_ids:
            entities.append({
                "id": tail_id,
                "name": getattr(triple.tail, "name", ""),
                "label": getattr(triple.tail, "label", ""),
                "ref_short": getattr(triple.tail, "ref_short", ""),
            })
            entity_ids.add(tail_id)
    
    return jsonify({
        "graph": graph_data,
        "triples": triples_data,
        "entities": entities,
        "id_to_name": validator.id_to_name,
        "changes": validator.getChanges(),
    })


def run_server(port: int = 5001, debug: bool = False, open_browser: bool = True):
    """Run the Flask server."""
    app.run(host='127.0.0.1', port=port, debug=debug, use_reloader=False)

