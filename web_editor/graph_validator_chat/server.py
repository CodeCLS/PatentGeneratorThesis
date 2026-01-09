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
import subprocess
from typing import Optional, Dict, Any, List
import networkx as nx

from tools.graph.Triple import Triple
from tools.graph.langgraph.validator import GraphValidatorLangGraph
from tools.graph.langgraph.helpers import get_triple_head_id, get_triple_tail_id
from tools.graph.langgraph.question import Question


# Global validator instance
validator: Optional[GraphValidatorLangGraph] = None
_server_running = False
_api_server: Optional[HTTPServer] = None
_api_thread: Optional[threading.Thread] = None
_nextjs_process: Optional[subprocess.Popen] = None
_nextjs_thread: Optional[threading.Thread] = None


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
        """Log server requests."""
        print(f"[{self.address_string()}] {format % args}")
    
    def _send_json(self, data: Dict[str, Any], status: int = 200):
        """Send JSON response."""
        try:
            self.send_response(status)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode('utf-8'))
        except (ConnectionAbortedError, BrokenPipeError, OSError) as e:
            # Client disconnected - this is normal, just ignore it
            pass
    
    def _send_error(self, message: str, status: int = 400):
        """Send error response."""
        self._send_json({"error": message}, status)
    
    
    def do_GET(self):
        """Handle GET requests - API only."""
        path = urlparse(self.path).path
        print(f"[API] GET {path}")
        
        if path == '/api/status':
            self._send_json(self._get_status())
        elif path == '/api/questions/first':
            result = self._get_first_question()
            print(f"[API] /api/questions/first: question={result.get('question') is not None}, generating={result.get('generating', False)}")
            self._send_json(result)
        elif path == '/api/state':
            self._send_json(self._get_state())
        elif path == '/api/export':
            self._send_json(self._export_data())
        elif path == '/api/triples':
            self._send_json(self._get_triples())
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests."""
        try:
            path = urlparse(self.path).path
            print(f"[API] POST {path}")
            
            if path == '/api/chat':
                self._handle_chat()
            else:
                self._send_error("Not found", 404)
        except Exception as e:
            import traceback
            print(f"[API] Error in do_POST: {type(e).__name__}: {str(e)}")
            print(f"[API] Traceback: {traceback.format_exc()}")
            try:
                self._send_error(f"Internal error: {str(e)}", 500)
            except:
                # If even sending error fails, send basic response
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"error":"Internal server error"}')
    
    
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
        
        # Get current questions from state
        if hasattr(validator, '_current_state') and validator._current_state:
            questions = validator._current_state.get("questions", [])
        else:
            questions = []
        
        # If no questions, generate them directly
        if not questions:
            print(f"[API] No questions found, generating... (triples: {len(validator.triples)})")
            from tools.graph.langgraph.nodes.analyzer import generate_questions
            questions = generate_questions(validator)
            print(f"[API] Generated {len(questions)} questions")
            if questions:
                if not hasattr(validator, '_current_state') or not validator._current_state:
                    from tools.graph.langgraph.state import create_state
                    validator._current_state = create_state(
                        graph_nodes_count=validator.graph.number_of_nodes() if validator.graph else 0,
                        graph_edges_count=validator.graph.number_of_edges() if validator.graph else 0,
                        triples_count=len(validator.triples),
                        entities_count=len(validator.id_to_name),
                    )
                validator._current_state["questions"] = questions
            else:
                print(f"[API] Warning: generate_questions returned empty list. Triples: {len(validator.triples)}")
        
        if not questions:
            return {"question": None, "all_completed": False}
        
        # Get first unanswered question
        first_unanswered = None
        for q in questions:
            if isinstance(q, Question):
                if not q.answered:
                    first_unanswered = q
                    break
            elif isinstance(q, dict):
                if not q.get("answered", False):
                    first_unanswered = q
                    break
        
        if not first_unanswered:
            return {"question": None, "all_completed": True}
        
        if isinstance(first_unanswered, Question):
            return {"question": first_unanswered.to_dict()}
        elif isinstance(first_unanswered, dict):
            return {"question": first_unanswered}
        else:
            return {"question": first_unanswered.to_dict() if hasattr(first_unanswered, 'to_dict') else {"id": "", "text": str(first_unanswered)}}
    
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
            
            print(f"[API] Processing chat message: {user_message[:100]}...")
            response = validator.chat(user_message)
            if not response:
                raise ValueError("validator.chat() returned None")
            if not isinstance(response, dict):
                raise ValueError(f"validator.chat() returned {type(response)}, expected dict")
            from tools.graph.constants_graph import STATE_TEXT, STATE_NEXT_QUESTION
            print(f"[API] Chat response keys: {list(response.keys())}")
            if response.get(STATE_TEXT, "").startswith("Error:"):
                print(f"[API] Warning: Response contains error: {response.get(STATE_TEXT)}")
            self._send_json(response)
        except json.JSONDecodeError as e:
            print(f"[API] JSON decode error: {e}")
            self._send_error("Invalid JSON")
        except Exception as e:
            import traceback
            print(f"[API] Error in _handle_chat: {type(e).__name__}: {str(e)}")
            print(f"[API] Traceback: {traceback.format_exc()}")
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
    
    def _get_triples(self) -> Dict[str, Any]:
        """Get all triples for display."""
        if not validator:
            return {"error": "Validator not initialized", "triples": []}
        
        triples = validator.triples
        triples_data = []
        
        for i, triple in enumerate(triples):
            head_id = get_triple_head_id(triple)
            tail_id = get_triple_tail_id(triple)
            
            triples_data.append({
                "index": i,
                "head": {
                    "id": head_id,
                    "name": triple.head.name,
                    "label": triple.head.label,
                },
                "relation": triple.relation,
                "tail": {
                    "id": tail_id,
                    "name": triple.tail.name,
                    "label": triple.tail.label,
                },
            })
        
        return {"triples": triples_data}
    
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


def stop_validator_chat():
    """Stop the running validator chat servers."""
    global _server_running, _api_server, _api_thread, _nextjs_process, _nextjs_thread
    
    if not _server_running:
        return
    
    print("[Server] Stopping validator chat servers...")
    
    # Stop API server
    if _api_server:
        try:
            _api_server.shutdown()
            print("[Server] API server stopped")
        except Exception as e:
            print(f"[Server] Error stopping API server: {e}")
        _api_server = None
    
    # Stop Next.js process
    if _nextjs_process:
        try:
            _nextjs_process.terminate()
            _nextjs_process.wait(timeout=2)
            print("[Server] Next.js server stopped")
        except subprocess.TimeoutExpired:
            _nextjs_process.kill()
            print("[Server] Next.js server force-killed")
        except Exception as e:
            print(f"[Server] Error stopping Next.js server: {e}")
        _nextjs_process = None
    
    _api_thread = None
    _nextjs_thread = None
    _server_running = False
    print("[Server] All servers stopped")


def start_validator_chat(
    graph: Optional[nx.MultiDiGraph] = None,
    triples: Optional[List[Triple]] = None,
    id_to_name: Optional[Dict[str, str]] = None,
    port: int = 5000,
    open_browser: bool = True,
    debug: bool = False,
) -> None:
    from pathlib import Path
    global validator, _server_running, _api_server, _api_thread, _nextjs_process, _nextjs_thread
    api_port_in_use = not _is_port_available(port) if port else False
    nextjs_port_in_use = not _is_port_available(3000)
    if _server_running or api_port_in_use or nextjs_port_in_use:
        print(f"[Server] Restarting server (detected running servers or ports in use)...")
        stop_validator_chat()
        time.sleep(2)
        _server_running = False
        _api_server = None
        _api_thread = None
        _nextjs_process = None
        _nextjs_thread = None
    
    if validator is None or graph is not None or triples is not None:
        print(f"[Server] Initializing validator...")
        initialize_validator(graph, triples, id_to_name)
    else:
        print(f"[Server] Validator already exists, skipping initialization")
    _server_running = True
    print(f"[Server] Requested API port: {port}")
    api_port = _find_available_port(port)
    if not api_port:
        print(f"❌ ERROR: Could not find an available port starting from {port}")
        return
    
    if api_port != port:
        print(f"⚠️  Port {port} not available, using port {api_port} for API server")
    else:
        print(f"✓ Using requested port {api_port} for API server")
    
    def run_api_server():
        global _api_server
        _api_server = HTTPServer(('127.0.0.1', api_port), GraphValidatorHandler)
        print(f"✓ API Server running on http://localhost:{api_port}")
        _api_server.serve_forever()
    
    _api_thread = threading.Thread(target=run_api_server, daemon=True)
    _api_thread.start()
    time.sleep(0.5)
    
    # Start Next.js dev server
    nextjs_dir = Path(__file__).parent / "nextjs"
    if not nextjs_dir.exists():
        print(f"❌ ERROR: Next.js directory not found at {nextjs_dir}")
        return
    
    # Determine Next.js port (check if 3000 is available, use alternative if not)
    nextjs_port = _find_available_port(3000)
    if not nextjs_port:
        print(f"❌ ERROR: Could not find an available port for Next.js")
        return
    
    if nextjs_port != 3000:
        print(f"⚠️  Port 3000 in use, Next.js will use port {nextjs_port}")
    else:
        print(f"✓ Using port {nextjs_port} for Next.js frontend")
    
    # Update next.config.js to use the correct API port
    next_config_path = nextjs_dir / "next.config.js"
    if next_config_path.exists():
        with open(next_config_path, 'r') as f:
            config_content = f.read()
        # Replace the port in the config
        import re
        config_content = re.sub(
            r"destination: 'http://localhost:\d+/api/:path\*'",
            f"destination: 'http://localhost:{api_port}/api/:path*'",
            config_content
        )
        with open(next_config_path, 'w') as f:
            f.write(config_content)
    
    def run_nextjs():
        import platform
        is_windows = platform.system() == "Windows"
        shell_flag = is_windows
        
        # Check if node_modules exists, if not, install dependencies
        node_modules = nextjs_dir / "node_modules"
        if not node_modules.exists():
            print("📦 Installing Next.js dependencies...")
            try:
                subprocess.run(
                    ["npm", "install"],
                    cwd=str(nextjs_dir),
                    check=True,
                    shell=shell_flag,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            except subprocess.CalledProcessError as e:
                print(f"❌ ERROR: Failed to install dependencies: {e}")
                return
        
        # Start Next.js dev server on the determined port
        print(f"🚀 Starting Next.js dev server on port {nextjs_port}...")
        try:
            global _nextjs_process
            _nextjs_process = subprocess.Popen(
                ["npm", "run", "dev", "--", "-p", str(nextjs_port)],
                cwd=str(nextjs_dir),
                shell=shell_flag,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            # Wait for process to complete (or be terminated)
            _nextjs_process.wait()
        except KeyboardInterrupt:
            pass
    
    _nextjs_thread = threading.Thread(target=run_nextjs, daemon=True)
    _nextjs_thread.start()
    
    # Wait a bit for Next.js to start, then open browser
    if open_browser:
        def open_browser_delayed():
            time.sleep(4)  # Give Next.js time to start
            url = f"http://localhost:{nextjs_port}"
            print(f"[Server] Opening browser to {url}")
            webbrowser.open(url)
        threading.Thread(target=open_browser_delayed, daemon=True).start()
    
    print(f"✓ Graph Validator Chat: http://localhost:{nextjs_port}")
    print(f"✓ API Server: http://localhost:{api_port}")


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
