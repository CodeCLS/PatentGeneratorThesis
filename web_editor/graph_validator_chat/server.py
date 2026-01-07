"""
Simple HTTP server for Graph Validator Chat Interface.
Single file with everything needed - no Flask, no Jinja2 dependencies!
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import json
import threading
import webbrowser
import time
import socket
import pickle
import base64
import os
from typing import Optional, Dict, Any, List
import networkx as nx

from tools.graph.Triple import Triple
from tools.graph.langgraph.validator import GraphValidatorLangGraph
from tools.graph.langgraph.helpers import get_triple_head_id, get_triple_tail_id
from tools.graph.langgraph.question import Question


# Global validator instance
validator: Optional[GraphValidatorLangGraph] = None


def initialize_validator(
    graph: Optional[nx.MultiDiGraph] = None,
    triples: Optional[List[Triple]] = None,
    id_to_name: Optional[Dict[str, str]] = None,
) -> GraphValidatorLangGraph:
    """Initialize the validator with graph and/or triples."""
    global validator
    validator = GraphValidatorLangGraph(
        graph=graph,
        triples=triples,
        id_to_name=id_to_name,
    )
    return validator


def get_validator() -> Optional[GraphValidatorLangGraph]:
    """Get the current validator instance."""
    return validator


class GraphValidatorHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the graph validator chat interface."""
    
    def log_message(self, format, *args):
        """Suppress server logs."""
        pass
    
    def _send_json(self, data: Dict[str, Any], status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def _send_error(self, message: str, status: int = 400):
        """Send error response."""
        self._send_json({"error": message}, status)
    
    def _read_file(self, filename: str) -> str:
        """Read a file from the static directory."""
        file_path = os.path.join(os.path.dirname(__file__), 'static', filename)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        return ""
    
    def do_GET(self):
        """Handle GET requests."""
        path = urlparse(self.path).path
        
        if path == '/' or path == '/index.html':
            self._serve_html()
        elif path == '/api/status':
            self._send_json(self._get_status())
        elif path == '/api/questions/first':
            self._send_json(self._get_first_question())
        elif path == '/api/state':
            self._send_json(self._get_state())
        elif path == '/api/export':
            self._send_json(self._export_data())
        elif path == '/static/app.js':
            self._serve_file('app.js', 'application/javascript')
        elif path == '/static/style.css':
            self._serve_file('style.css', 'text/css')
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests."""
        path = urlparse(self.path).path
        
        if path == '/api/chat':
            self._handle_chat()
        else:
            self.send_error(404)
    
    def _serve_file(self, filename: str, content_type: str):
        """Serve a static file."""
        content = self._read_file(filename)
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))
    
    def _serve_html(self):
        """Serve the main HTML page."""
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graph Validator Chat</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>Graph Validator Chat</h1>
            <div class="status" id="status">Initializing...</div>
        </header>
        <div class="chat-container">
            <div class="messages" id="messages">
                <div class="message bot">
                    <div class="message-content">
                        <strong>Bot:</strong> Analyzing your graph and triples...
                    </div>
                </div>
            </div>
            <div class="input-area">
                <input type="text" id="answerInput" placeholder="Type your message here..." disabled>
                <button id="sendButton" disabled>Send</button>
            </div>
        </div>
        <div class="sidebar">
            <h3>Current Question</h3>
            <div id="currentQuestion" class="question-display"><p>Loading question...</p></div>
            <h3>Recent Changes</h3>
            <div id="changesDisplay" class="state-display"><p>No changes yet</p></div>
            <h3>Graph Statistics</h3>
            <div id="graphStats" class="state-display"><p>Loading...</p></div>
        </div>
    </div>
    <script src="/static/app.js"></script>
</body>
</html>"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def _get_status(self) -> Dict[str, Any]:
        """Get current validator status."""
        if not validator:
            return {"initialized": False, "message": "Validator not initialized"}
        
        questions = validator._current_state.get("questions", []) if hasattr(validator, '_current_state') and validator._current_state else []
        unanswered = [q for q in questions if not (q.answered if isinstance(q, Question) else q.get("answered", False))]
        
        return {
            "initialized": True,
            "num_questions": len(questions),
            "num_unanswered": len(unanswered),
            "has_graph": validator.graph is not None,
            "num_triples": len(validator.triples),
        }
    
    def _get_first_question(self) -> Dict[str, Any]:
        """Get the first unanswered question."""
        if not validator:
            return {"error": "Validator not initialized"}
        
        questions = validator._current_state.get("questions", []) if hasattr(validator, '_current_state') and validator._current_state else []
        if not questions:
            return {"question": None}
        
        first_q = questions[0]
        if isinstance(first_q, Question):
            return {"question": first_q.to_dict()}
        elif isinstance(first_q, dict):
            return {"question": first_q}
        else:
            return {"question": first_q.to_dict() if hasattr(first_q, 'to_dict') else {"id": "", "text": str(first_q)}}
    
    def _get_state(self) -> Dict[str, Any]:
        """Get current validator state."""
        if not validator:
            return {"error": "Validator not initialized"}
        
        graph = validator.graph
        triples = validator.triples
        changes_summary = validator._current_state.get("changes_summary", []) if hasattr(validator, '_current_state') and validator._current_state else []
        stats = validator.tools.calculate_stats() if hasattr(validator, 'tools') else {}
        
        changes = {
            "triples_added": max(0, stats.get("triples_changed", 0)),
            "triples_deleted": max(0, -stats.get("triples_changed", 0)),
            "entities_merged": 0,
            "entities_renamed": 0,
            "changes_summary": changes_summary,
        }
        
        entities = self._extract_entities(triples)
        
        return {
            "graph": {
                "num_nodes": graph.number_of_nodes() if graph else 0,
                "num_edges": graph.number_of_edges() if graph else 0,
            },
            "num_triples": len(triples),
            "num_entities": len(entities),
            "entities": entities[:100],
            "id_to_name": validator.id_to_name,
            "changes": changes,
        }
    
    def _handle_chat(self):
        """Handle chat messages."""
        if not validator:
            self._send_error("Validator not initialized")
            return
        
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(content_length).decode('utf-8'))
            user_message = data.get("message", "")
            
            if not user_message:
                self._send_error("Message text is required")
                return
            
            response = validator.chat(user_message)
            self._send_json(response)
        except json.JSONDecodeError:
            self._send_error("Invalid JSON")
        except Exception as e:
            self._send_error(f"Error: {str(e)}")
    
    def _export_data(self) -> Dict[str, Any]:
        """Export full graph, triples, and entities."""
        if not validator:
            return {"error": "Validator not initialized"}
        
        graph = validator.graph
        triples = validator.triples
        entities = self._extract_entities(triples)
        
        graph_data = None
        if graph:
            graph_data = base64.b64encode(pickle.dumps(graph)).decode('utf-8')
        
        triples_data = [
            {
                "head": {"id": get_triple_head_id(t), "name": t.head.name, "label": t.head.label},
                "relation": t.relation,
                "tail": {"id": get_triple_tail_id(t), "name": t.tail.name, "label": t.tail.label},
            }
            for t in triples
        ]
        
        changes_summary = validator._current_state.get("changes_summary", []) if hasattr(validator, '_current_state') and validator._current_state else []
        stats = validator.tools.calculate_stats() if hasattr(validator, 'tools') else {}
        
        changes = {
            "triples_added": max(0, stats.get("triples_changed", 0)),
            "triples_deleted": max(0, -stats.get("triples_changed", 0)),
            "entities_merged": 0,
            "entities_renamed": 0,
            "changes_summary": changes_summary,
        }
        
        return {
            "graph": graph_data,
            "triples": triples_data,
            "entities": entities,
            "id_to_name": validator.id_to_name,
            "changes": changes,
        }
    
    def _extract_entities(self, triples: List[Triple]) -> List[Dict[str, Any]]:
        """Extract unique entities from triples."""
        entities = []
        entity_ids = set()
        
        for triple in triples:
            head_id = get_triple_head_id(triple)
            tail_id = get_triple_tail_id(triple)
            
            if head_id and head_id not in entity_ids:
                entities.append({
                    "id": head_id,
                    "name": triple.head.name,
                    "label": triple.head.label,
                })
                entity_ids.add(head_id)
            
            if tail_id and tail_id not in entity_ids:
                entities.append({
                    "id": tail_id,
                    "name": triple.tail.name,
                    "label": triple.tail.label,
                })
                entity_ids.add(tail_id)
        
        return entities


def _is_port_available(port: int) -> bool:
    """Check if a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False


def _find_available_port(start_port: int) -> Optional[int]:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + 20):
        if _is_port_available(port):
            return port
    for port in [8000, 8001, 8080, 8888]:
        if _is_port_available(port):
            return port
    return None


def start_validator_chat(
    graph: Optional[nx.MultiDiGraph] = None,
    triples: Optional[List[Triple]] = None,
    id_to_name: Optional[Dict[str, str]] = None,
    port: int = 5001,
    open_browser: bool = True,
    debug: bool = False,
) -> None:
    """Start the graph validator chat interface."""
    initialize_validator(graph, triples, id_to_name)
    
    actual_port = _find_available_port(port)
    if not actual_port:
        print(f"❌ ERROR: Could not find an available port starting from {port}")
        return
    
    if actual_port != port:
        print(f"⚠️  Port {port} not available, using port {actual_port}")
    
    def run_server():
        server = HTTPServer(('127.0.0.1', actual_port), GraphValidatorHandler)
        print(f"✓ Graph Validator Chat: http://localhost:{actual_port}")
        
        if open_browser:
            def open_browser_delayed():
                time.sleep(1.5)
                webbrowser.open(f"http://localhost:{actual_port}")
            threading.Thread(target=open_browser_delayed, daemon=True).start()
        
        server.serve_forever()
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(0.5)


# Helper functions for getting data from validator
def get_validator_state() -> Dict[str, Any]:
    """Get validator state from running server or directly from validator."""
    try:
        import requests
        response = requests.get('http://localhost:5001/api/export', timeout=2)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    v = get_validator()
    if not v:
        return {
            "graph": None,
            "triples": [],
            "entities": [],
            "id_to_name": {},
            "changes": {},
        }
    
    graph = v.graph
    triples = v.triples
    changes_summary = v._current_state.get("changes_summary", []) if hasattr(v, '_current_state') and v._current_state else []
    stats = v.tools.calculate_stats() if hasattr(v, 'tools') else {}
    
    changes = {
        "triples_added": max(0, stats.get("triples_changed", 0)),
        "triples_deleted": max(0, -stats.get("triples_changed", 0)),
        "entities_merged": 0,
        "entities_renamed": 0,
        "changes_summary": changes_summary,
    }
    
    graph_data = None
    if graph:
        graph_data = base64.b64encode(pickle.dumps(graph)).decode('utf-8')
    
    entities = []
    entity_ids = set()
    for triple in triples:
        head_id = get_triple_head_id(triple)
        tail_id = get_triple_tail_id(triple)
        
        if head_id and head_id not in entity_ids:
            entities.append({
                "id": head_id,
                "name": triple.head.name,
                "label": triple.head.label,
            })
            entity_ids.add(head_id)
        
        if tail_id and tail_id not in entity_ids:
            entities.append({
                "id": tail_id,
                "name": triple.tail.name,
                "label": triple.tail.label,
            })
            entity_ids.add(tail_id)
    
    return {
        "graph": graph_data,
        "triples": triples,
        "entities": entities,
        "id_to_name": v.id_to_name,
        "changes": changes,
    }


def get_updated_graph() -> Optional[nx.MultiDiGraph]:
    """Get the updated graph from the validator."""
    v = get_validator()
    return v.graph if v else None


def get_updated_triples() -> List[Triple]:
    """Get the updated triples from the validator."""
    v = get_validator()
    return v.triples if v else []


def get_changes_summary() -> Dict[str, Any]:
    """Get a summary of changes made to the graph."""
    v = get_validator()
    if not v:
        return {}
    
    changes_summary = v._current_state.get("changes_summary", []) if hasattr(v, '_current_state') and v._current_state else []
    stats = v.tools.calculate_stats() if hasattr(v, 'tools') else {}
    
    return {
        "triples_added": max(0, stats.get("triples_changed", 0)),
        "triples_deleted": max(0, -stats.get("triples_changed", 0)),
        "entities_merged": 0,
        "entities_renamed": 0,
        "changes_summary": changes_summary,
    }
