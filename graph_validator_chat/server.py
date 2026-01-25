"""
Simple HTTP server for Graph Validator Chat Interface.
Single file with everything needed - no Flask, no Jinja2 dependencies!
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
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
import re
from typing import Optional, Dict, Any, List
import networkx as nx

from tools.graph.data.Triple import Triple
from tools.graph.langgraph.validator import GraphValidatorLangGraph
from tools.graph.langgraph.helpers import get_triple_head_id, get_triple_tail_id
from tools.graph.langgraph.question import Question
from tools.graph.claim_generation.claim_generator_langchain import ClaimGeneratorLangChain
from tools.graph.rag.graph_rag import GraphRAG
from tools.graph.visualizer import GraphVisualizer


# Global validator instance
validator: Optional[GraphValidatorLangGraph] = None
# Global sentence_split for patent description
_sentence_split: Optional[List[Any]] = None
_server_running = False
_api_server: Optional[HTTPServer] = None
_api_thread: Optional[threading.Thread] = None
_nextjs_process: Optional[subprocess.Popen] = None
_nextjs_thread: Optional[threading.Thread] = None

# Global progress tracking
_claim_generation_progress: Dict[str, Any] = {}
_cached_claims: List[Dict[str, Any]] = []


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    daemon_threads = True


def initialize_validator(
    graph: Optional[nx.MultiDiGraph] = None,
    triples: Optional[List[Triple]] = None,
    id_to_name: Optional[Dict[str, str]] = None,
    sentence_split: Optional[List[Any]] = None,
) -> GraphValidatorLangGraph:
    """Initialize the validator with graph and/or triples.
    
    Args:
        graph: Optional NetworkX graph
        triples: Optional list of Triple objects
        id_to_name: Optional mapping from entity ID to name
        sentence_split: Optional list of Sentence objects with .text attribute for patent description
    """
    global validator, _sentence_split
    # Store sentence_split globally
    _sentence_split = sentence_split
    
    # 1. Create the object instantly
    validator = GraphValidatorLangGraph(
        graph=graph,
        triples=triples,
        id_to_name=id_to_name,
    )
    
    # 2. Run heavy analysis in a background thread
    def background_analysis():
        print("[Server] Starting background analysis...")
        validator.run_initial_analysis()
        print("[Server] Background analysis complete.")
        
    threading.Thread(target=background_analysis, daemon=True).start()
    
    return validator


def get_validator() -> Optional[GraphValidatorLangGraph]:
    """Get the current validator instance."""
    return validator


class GraphValidatorHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the graph validator chat interface."""
    def log_request(self, code='-', size='-'):
        print(f"[REQ] {self.command} {self.path} -> {code} ({size} bytes)")
    
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
    
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        path = urlparse(self.path).path
        print(f"[API] GET {path}")
        
        # Serve static files
        if path.startswith('/static/'):
            self._serve_static_file(path)
            return
        
        # Serve widget showcase page
        if path == '/widget-showcase':
            self._serve_widget_showcase()
            return
        
        # Serve main index page
        if path == '/' or path == '/index.html':
            self._serve_index()
            return
        
        # API endpoints
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
        elif path == '/api/test':
            print("TEST RUN")
            self._send_json({"ok": True, "message": "TEST RUN"})
            return
        elif path == '/api/triples':
            self._send_json(self._get_triples())
        elif path == '/api/graph/html':
            self._send_json(self._get_graph_html())
        elif path == '/api/claims':
            # GET endpoint to retrieve generated claims
            self._send_json(self._get_generated_claims())
        elif path == '/api/claims/progress':
            # GET endpoint to retrieve claim generation progress
            self._send_json(self._get_claim_progress())
        else:
            print(f"[API] 404: Path not found: {path}")
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests."""
        try:
            path = urlparse(self.path).path
            print(f"[API] POST {path}")
            
            if path == '/api/chat':
                print("[API] Routing to _handle_chat")
                self._handle_chat()
            elif path == '/api/cypher/query':
                print("[API] Routing to _handle_cypher_query")
                self._handle_cypher_query()
            elif path == '/api/claims/generate':
                print("[API ] Routing to _handle_generate_claims")
                self._handle_generate_claims()
            elif path == '/api/entities/update':
                print("[API] Routing to _handle_entity_update")
                self._handle_entity_update()
            elif path == '/api/entities/merge':
                print("[API] Routing to _handle_entity_merge")
                self._handle_entity_merge()
            elif path == '/api/entities/delete':
                print("[API] Routing to _handle_entity_delete")
                self._handle_entity_delete()
            elif path == '/api/triples/delete':
                print("[API] Routing to _handle_triple_delete")
                self._handle_triple_delete()
            else:
                print(f"[API] POST 404: Path not found: {path}")
                self._send_error("Not found", 404)
        except Exception as e:
            import traceback
            error_details = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            print(f"[API] Error in do_POST: {error_details}")
            
            # Log to a file specifically for background thread errors
            try:
                with open("server_error.log", "a", encoding="utf-8") as f:
                    import datetime
                    f.write(f"\n[{datetime.datetime.now()}] ERROR in do_POST {path}:\n{error_details}\n")
            except:
                pass
                
            try:
                self._send_error(f"Internal error: {str(e)}", 500)
            except:
                # If even sending error fails, send basic response
                try:
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"error":"Internal server error"}')
                except:
                    pass
    
    
    def _get_status(self) -> Dict[str, Any]:
        """Get current validator status."""
        if not validator or not getattr(validator, 'initial_analysis_complete', False):
            return {"initialized": False, "message": "Validator initializing..."}
        
        questions = validator._current_state.get("questions", []) if hasattr(validator, '_current_state') and validator._current_state else []
        print(q.type for q in questions )
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
            
            # Ensure response is JSON-serializable
            try:
                json.dumps(response)  # Test serialization
            except (TypeError, ValueError) as e:
                print(f"[API] ERROR: Response is not JSON-serializable: {e}")
                print(f"[API] Problematic keys/values:")
                for key, value in response.items():
                    try:
                        json.dumps(value)
                    except (TypeError, ValueError):
                        print(f"[API]   - {key}: {type(value)} = {str(value)[:200]}")
                raise ValueError(f"Response contains non-serializable data: {e}")
            
            self._send_json(response)
        except json.JSONDecodeError as e:
            print(f"[API] JSON decode error: {e}")
            self._send_error("Invalid JSON")
        except Exception as e:
            import traceback
            print(f"[API] Error in _handle_chat: {type(e).__name__}: {str(e)}")
            print(f"[API] Traceback: {traceback.format_exc()}")
            self._send_error(f"Error: {str(e)}")

    @staticmethod
    def _is_cypher_query(text: str) -> bool:
        if not text:
            return False
        text_upper = text.strip().upper()
        cypher_keywords = ("MATCH", "RETURN", "WITH", "WHERE", "CREATE", "MERGE", "CALL", "UNWIND")
        return any(keyword in text_upper for keyword in cypher_keywords)

    @staticmethod
    def _extract_cypher_query(text: str) -> str:
        if not text:
            return ""
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[a-zA-Z]*\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        match = re.search(r"(MATCH|CREATE|MERGE|CALL|WITH|UNWIND)\b[\s\S]*", stripped, re.IGNORECASE)
        if match:
            return match.group(0).strip()
        return stripped

    @staticmethod
    def _collect_schema_context() -> Dict[str, List[str]]:
        relations = set()
        entity_types = set()
        if not validator:
            return {"relations": [], "entity_types": []}

        for triple in validator.triples:
            relation = getattr(triple, "relation", "") or ""
            if relation.strip():
                relations.add(relation.strip())

            head_entity = getattr(triple, "head", None)
            tail_entity = getattr(triple, "tail", None)
            head_label = getattr(head_entity, "label", "") or getattr(head_entity, "entity_type", "") or ""
            tail_label = getattr(tail_entity, "label", "") or getattr(tail_entity, "entity_type", "") or ""
            if head_label.strip():
                entity_types.add(head_label.strip().upper())
            if tail_label.strip():
                entity_types.add(tail_label.strip().upper())

        return {
            "relations": sorted(relations),
            "entity_types": sorted(entity_types),
        }

    def _build_cypher_prompt(self, user_request: str, relations: List[str], entity_types: List[str]) -> str:
        relations_text = ", ".join(relations) if relations else "None"
        entity_types_text = ", ".join(entity_types) if entity_types else "None"
        return (
            "You are a Cypher assistant for a Neo4j graph.\n"
            "Schema:\n"
            "- Nodes are labeled :Entity and have properties: id, name, node_type (entity type).\n"
            "- Relationships are :LINKS_TO with property relation.\n"
            f"Available relation values: {relations_text}\n"
            f"Available entity types: {entity_types_text}\n"
            "Return only a Cypher query with no explanation.\n"
            f"User request: {user_request}\n"
        )

    def _handle_cypher_query(self):
        """Handle Cypher generation or execution request."""
        if not validator:
            self._send_error("Validator not initialized")
            return

        try:
            content_length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(content_length).decode('utf-8'))
            user_query = (data.get("query") or "").strip()

            if not user_query:
                self._send_error("Query text is required")
                return

            if self._is_cypher_query(user_query):
                self._send_json({
                    "query": user_query,
                    "ran": True,
                    "message": "Cypher query run",
                })
                return

            schema_context = self._collect_schema_context()
            prompt = self._build_cypher_prompt(
                user_query,
                schema_context["relations"],
                schema_context["entity_types"],
            )
            from tools.api.llm_api_repo import LLmApi_Repo
            llm = LLmApi_Repo()
            llm_response = llm.chat(prompt)
            suggested_query = self._extract_cypher_query(str(llm_response))
            if not suggested_query:
                suggested_query = user_query

            self._send_json({
                "query": suggested_query,
                "ran": False,
                "message": "Cypher query ready",
            })
        except json.JSONDecodeError:
            self._send_error("Invalid JSON", 400)
        except Exception as e:
            import traceback
            print(f"[API] Error in _handle_cypher_query: {e}")
            print(traceback.format_exc())
            self._send_error(f"Error: {str(e)}", 500)
    
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
    
    def _get_graph_html(self) -> Dict[str, Any]:
        print("[API] Starting _get_graph_html")
        if not validator:
            return {"error": "Validator not initialized", "html": ""}
        
        try:
            from tools.graph.visualizer import GraphVisualizer
            import tempfile
            import os
            
            triples = validator.triples
            id_to_name = validator.id_to_name
            
            if not triples:
                print("[API] Warning: No triples available in _get_graph_html")
                return {"error": "No graph or triples available", "html": ""}
            
            visualizer = GraphVisualizer()
            graph = visualizer.build_graph(triples)
            validator.graph = graph
            
            print(f"[API] Generating graph HTML (nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()})...")
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            # Temporarily disable webbrowser to prevent auto-opening
            import webbrowser
            import contextlib
            
            @contextlib.contextmanager
            def disable_webbrowser():
                original_open = webbrowser.open
                webbrowser.open = lambda *args, **kwargs: None
                try:
                    yield
                finally:
                    webbrowser.open = original_open
            
            with disable_webbrowser():
                visualizer.visualize_pyvis(graph, out_file=temp_path, id_to_name=id_to_name)
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            os.unlink(temp_path)
            print(f"[API] Graph HTML generated successfully ({len(html_content)} chars)")
            
            # Sanitize HTML: remove any file:// links that might cause unwanted navigation
            import re
            html_content = re.sub(r'file:///[^\s"\'<>]*', '', html_content)
            html_content = re.sub(r'href=["\']file:///[^"\']*["\']', 'href="#"', html_content)
            html_content = re.sub(r'src=["\']file:///[^"\']*["\']', 'src="#"', html_content)
            
            return {"html": html_content}
        except Exception as e:
            import traceback
            print(f"[API] Error generating graph HTML: {e}")
            print(traceback.format_exc())
            return {"error": str(e), "html": ""}
    
    def _serve_static_file(self, path: str):
        """Serve static files (CSS, JS, etc.)."""
        try:
            # Remove leading /static/
            file_path = path[8:]  # Remove '/static/'
            static_dir = os.path.join(os.path.dirname(__file__), 'static')
            full_path = os.path.join(static_dir, file_path)
            
            # Security: ensure file is within static directory
            if not os.path.abspath(full_path).startswith(os.path.abspath(static_dir)):
                self.send_error(403)
                return
            
            if not os.path.exists(full_path):
                self.send_error(404)
                return
            
            # Determine content type
            content_type = 'text/plain'
            if file_path.endswith('.css'):
                content_type = 'text/css'
            elif file_path.endswith('.js'):
                content_type = 'application/javascript'
            elif file_path.endswith('.html'):
                content_type = 'text/html'
            elif file_path.endswith('.png'):
                content_type = 'image/png'
            elif file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
                content_type = 'image/jpeg'
            elif file_path.endswith('.svg'):
                content_type = 'image/svg+xml'
            
            with open(full_path, 'rb') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            import traceback
            print(f"[API] Error serving static file: {e}")
            print(traceback.format_exc())
            self.send_error(500)
    
    def _serve_index(self):
        """Serve the main index page."""
        try:
            index_path = os.path.join(os.path.dirname(__file__), 'templates', 'index.html')
            
            if not os.path.exists(index_path):
                self.send_error(404)
                return
            
            with open(index_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html_content.encode('utf-8'))
        except Exception as e:
            import traceback
            print(f"[API] Error serving index: {e}")
            print(traceback.format_exc())
            self.send_error(500)
    
    def _serve_widget_showcase(self):
        """Serve the widget showcase page."""
        try:
            # Read the widget showcase HTML file
            showcase_path = os.path.join(os.path.dirname(__file__), 'templates', 'widget-showcase.html')
            
            if not os.path.exists(showcase_path):
                self.send_error(404)
                return
            
            with open(showcase_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html_content.encode('utf-8'))
        except Exception as e:
            import traceback
            print(f"[API] Error serving widget showcase: {e}")
            print(traceback.format_exc())
            self.send_error(500)
    
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
    
    def _handle_generate_claims(self):
        """Handle claim generation request."""
        if not validator:
            self._send_error("Validator not initialized")
            return
        
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                data = json.loads(self.rfile.read(content_length).decode('utf-8'))
                patent_description = data.get("patent_description", "")
                num_independent = data.get("num_independent", 3)
                num_dependent_per_independent = data.get("num_dependent_per_independent", 2)
                similarity_threshold = data.get("similarity_threshold", 0.3)
            else:
                # Use default values if no body provided
                patent_description = ""
                num_independent = 3
                num_dependent_per_independent = 2
                similarity_threshold = 0.3
            
            # Get patent description from validator state if not provided
            print(f"[DEBUG] _handle_generate_claims: Initial patent_description: '{patent_description[:100] if patent_description else 'EMPTY'}...'")
            print(f"[DEBUG] _handle_generate_claims: patent_description is None: {patent_description is None}")
            print(f"[DEBUG] _handle_generate_claims: patent_description.strip() == '': {patent_description.strip() == '' if patent_description else 'N/A'}")
            
            if not patent_description or patent_description.strip() == "":
                print(f"[DEBUG] _handle_generate_claims: Patent description empty, trying to extract from sentence_split...")
                # First try to get from sentence_split (global variable)
                global _sentence_split
                if _sentence_split and len(_sentence_split) > 0:
                    print(f"[DEBUG] _handle_generate_claims: Found sentence_split with {len(_sentence_split)} sentences")
                    try:
                        # Extract text from Sentence objects
                        description_parts = []
                        for idx, sentence in enumerate(_sentence_split):
                            if hasattr(sentence, 'text'):
                                text = sentence.text
                                if text and text.strip():
                                    description_parts.append(text.strip())
                            elif isinstance(sentence, str):
                                if sentence.strip():
                                    description_parts.append(sentence.strip())
                        
                        if description_parts:
                            patent_description = "\n".join(description_parts)
                            print(f"[DEBUG] _handle_generate_claims: Extracted patent description from sentence_split ({len(patent_description)} chars)")
                        else:
                            print(f"[DEBUG] _handle_generate_claims: sentence_split has no valid text content")
                    except Exception as e:
                        print(f"[DEBUG] _handle_generate_claims: Error extracting from sentence_split: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Fallback: Try to get from validator state (chat messages)
                if not patent_description or patent_description.strip() == "":
                    print(f"[DEBUG] _handle_generate_claims: Trying to extract from validator state...")
                    if hasattr(validator, '_current_state') and validator._current_state:
                        print(f"[DEBUG] _handle_generate_claims: Validator has _current_state")
                        messages = validator._current_state.get("messages", [])
                        print(f"[DEBUG] _handle_generate_claims: Found {len(messages)} messages in state")
                        # Extract text from messages to build description
                        description_parts = []
                        for idx, msg in enumerate(messages):
                            print(f"[DEBUG] _handle_generate_claims: Message {idx}: type={type(msg)}")
                            if isinstance(msg, dict):
                                content = msg.get("content", msg.get("text", ""))
                                role = msg.get("role", "")
                                print(f"[DEBUG] _handle_generate_claims:   Dict message - role={role}, content_length={len(content) if content else 0}")
                                if role == "user" and content:
                                    description_parts.append(content)
                            elif hasattr(msg, "content"):
                                if hasattr(msg, "role") and msg.role == "user":
                                    description_parts.append(str(msg.content))
                        
                        if description_parts:
                            patent_description = "\n\n".join(description_parts)
                            print(f"[DEBUG] _handle_generate_claims: Extracted patent description from chat messages ({len(patent_description)} chars)")
                        else:
                            patent_description = "Patent invention description not provided. Please provide a description of the invention."
                            print(f"[DEBUG] _handle_generate_claims: No user messages found, using placeholder")
                    else:
                        patent_description = "Patent invention description not provided. Please provide a description of the invention."
                        print(f"[DEBUG] _handle_generate_claims: No validator state, using placeholder")
            
            print(f"[DEBUG] _handle_generate_claims: Final patent_description length: {len(patent_description)} chars")
            print(f"[DEBUG] _handle_generate_claims: Final patent_description preview: {patent_description[:200]}...")
            print(f"[DEBUG] _handle_generate_claims: num_independent={num_independent}, num_dependent_per_independent={num_dependent_per_independent}")
            
            # Declare globals at the start of the function
            global _claim_generation_progress, _cached_claims
            
            # Set initial progress BEFORE starting generation
            _claim_generation_progress = {
                "stage": "planning",
                "message": "Planning claim structure...",
                "progress": 0,
            }
            print(f"[API] Initial progress set: {_claim_generation_progress}")
            print(f"[API] Progress stage: {_claim_generation_progress.get('stage')}")
            
            # Initialize GraphRAG
            graph_rag = GraphRAG(
                G=validator.graph,
                triples=validator.triples,
                id_to_name=validator.id_to_name,
            )
            
            # Initialize claim generator
            claim_generator = ClaimGeneratorLangChain(
                graph_rag=graph_rag,
            )
            
            # Progress tracking
            progress_updates = []
            
            def progress_callback(update):
                """Store progress updates."""
                # Need global here too since it's a nested function
                global _claim_generation_progress
                progress_updates.append(update)
                # Store in global variable for GET requests
                _claim_generation_progress = update
                print(f"[API] Progress: {update.get('stage')} - {update.get('message')} ({update.get('progress', 0)}%)")
            
            # Generate claims with progress tracking
            print(f"[API] Starting claim generation...")
            print(f"[API] Triples count: {len(validator.triples)}")
            print(f"[API] Graph nodes: {validator.graph.number_of_nodes() if validator.graph else 0}")
            print(f"[API] Patent description: {patent_description[:200] if patent_description else 'EMPTY'}...")
            
            generated_claims = []
            try:
                generated_claims = claim_generator.generate_all_claims(
                    patent_description=patent_description,
                    triples=validator.triples,
                    graph=validator.graph,
                    id_to_name=validator.id_to_name,
                    num_independent=num_independent,
                    num_dependent_per_independent=num_dependent_per_independent,
                    progress_callback=progress_callback,
                    similarity_threshold=similarity_threshold,
                )
                print(f"[API] generate_all_claims returned {len(generated_claims) if generated_claims else 0} claims")
            except Exception as e:
                import traceback
                print(f"[API] ERROR in generate_all_claims: {e}")
                print(f"[API] Traceback: {traceback.format_exc()}")
                # Create emergency fallback claim
                from tools.graph.claim_generation.claim_generator_langchain import GeneratedClaim
                generated_claims = [GeneratedClaim(
                    claim_number=1,
                    claim_text="1. A system comprising components as described in the patent description.",
                    claim_type="independent",
                    focus="Main invention",
                )]
                print(f"[API] Created emergency fallback claim")
            
            # CRITICAL: Ensure we always have at least one claim
            if not generated_claims or len(generated_claims) == 0:
                print(f"[API] CRITICAL: No claims generated! Creating emergency fallback.")
                from tools.graph.claim_generation.claim_generator_langchain import GeneratedClaim
                generated_claims = [GeneratedClaim(
                    claim_number=1,
                    claim_text="1. A system comprising components as described in the patent description.",
                    claim_type="independent",
                    focus="Main invention",
                )]
            
            # Convert to JSON-serializable format
            claims_data = []
            for claim in generated_claims:
                claims_data.append({
                    "claim_number": claim.claim_number,
                    "claim_text": claim.claim_text,
                    "claim_type": claim.claim_type,
                    "parent_claim_number": claim.parent_claim_number,
                    "focus": claim.focus,
                    "used_triples": claim.used_triples if hasattr(claim, 'used_triples') else [],
                    "prompt": claim.prompt if hasattr(claim, 'prompt') else "",
                    "refinement_iterations": claim.refinement_iterations if hasattr(claim, 'refinement_iterations') else 0,
                    "final_score": claim.final_score if hasattr(claim, 'final_score') else 0.0,
                })
            
            print(f"[API] Final claims_data: {len(claims_data)} claims")
            
            # Cache the generated claims globally
            _cached_claims = claims_data
            
            # Set final progress
            _claim_generation_progress = {
                "stage": "complete",
                "message": f"Successfully generated {len(claims_data)} claims!",
                "progress": 100,
                "num_claims": len(claims_data),
            }
            
            self._send_json({
                "success": True,
                "claims": claims_data,
                "num_claims": len(claims_data),
                "progress": _claim_generation_progress,
            })
            
        except Exception as e:
            import traceback
            print(f"[API] Error generating claims: {type(e).__name__}: {str(e)}")
            print(f"[API] Traceback: {traceback.format_exc()}")
            self._send_error(f"Error generating claims: {str(e)}", 500)
    
    def _get_generated_claims(self) -> Dict[str, Any]:
        """Get previously generated claims from storage."""
        global _cached_claims
        try:
            if _cached_claims:
                return {
                    "success": True,
                    "claims": _cached_claims,
                    "num_claims": len(_cached_claims),
                }
            else:
                return {
                    "success": True,
                    "claims": [],
                    "num_claims": 0,
                }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_claim_progress(self) -> Dict[str, Any]:
        """Get current claim generation progress."""
        global _claim_generation_progress
        try:
            # Check if progress exists and has a stage (not empty dict)
            if (_claim_generation_progress and 
                isinstance(_claim_generation_progress, dict) and 
                _claim_generation_progress.get("stage") and
                _claim_generation_progress.get("stage") != "idle"):
                print(f"[API] Returning progress: {_claim_generation_progress}")
                return {
                    "success": True,
                    "progress": _claim_generation_progress,
                }
            else:
                # Return idle only if truly no progress or empty
                print(f"[API] No active progress found. Current value: {_claim_generation_progress}")
                return {
                    "success": True,
                    "progress": {
                        "stage": "idle",
                        "message": "No generation in progress",
                        "progress": 0,
                    },
                }
        except Exception as e:
            print(f"[API] Error getting progress: {e}")
            import traceback
            print(traceback.format_exc())
            return {"error": str(e)}
    
    def _handle_entity_update(self):
        """Handle entity update request."""
        if not validator:
            self._send_error("Validator not initialized")
            return
        
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(content_length).decode('utf-8'))
            
            update_type = data.get("type")
            
            if update_type == "entity":
                entity_id = data.get("id")
                new_name = data.get("name", "").strip()
                new_label = data.get("label", "").strip()
                
                if not entity_id or not new_name:
                    self._send_error("Entity ID and name are required", 400)
                    return
                
                # 1. Update triples data
                triples_updated = 0
                for triple in validator.triples:
                    if get_triple_head_id(triple) == entity_id:
                        triple.head.name = new_name
                        if new_label:
                            triple.head.label = new_label
                        triples_updated += 1
                    if get_triple_tail_id(triple) == entity_id:
                        triple.tail.name = new_name
                        if new_label:
                            triple.tail.label = new_label
                        triples_updated += 1
                
                # 2. Update mapping
                if entity_id in validator.id_to_name:
                    validator.id_to_name[entity_id] = new_name
                
                # 3. SYNC GRAPH: Rebuild graph from updated triples
                if validator.graph is not None:
                    visualizer = GraphVisualizer()
                    validator.graph = visualizer.build_graph(validator.triples)
                
                print(f"[API] Updated entity {entity_id}: name={new_name}, label={new_label}, triples_updated={triples_updated}")
                self._send_json({
                    "success": True,
                    "message": f"Entity updated successfully",
                    "triples_updated": triples_updated,
                })
                
            elif update_type == "triple":
                triple_index = data.get("index")
                new_relation = data.get("relation", "").strip()
                
                if triple_index is None or not isinstance(triple_index, int):
                    self._send_error("Triple index is required", 400)
                    return
                
                if triple_index < 0 or triple_index >= len(validator.triples):
                    self._send_error("Invalid triple index", 400)
                    return
                
                if not new_relation:
                    self._send_error("Relation is required", 400)
                    return
                
                # 1. Update triple
                triple = validator.triples[triple_index]
                old_relation = triple.relation
                triple.relation = new_relation
                
                # 2. SYNC GRAPH: Rebuild graph from updated triples
                if validator.graph is not None:
                    visualizer = GraphVisualizer()
                    validator.graph = visualizer.build_graph(validator.triples)
                
                print(f"[API] Updated triple {triple_index}: relation={new_relation}")
                self._send_json({
                    "success": True,
                    "message": "Triple updated successfully",
                })
            else:
                self._send_error("Unknown update type", 400)
                
        except json.JSONDecodeError:
            self._send_error("Invalid JSON", 400)
        except Exception as e:
            import traceback
            print(f"[API] Error in _handle_entity_update: {e}")
            print(traceback.format_exc())
            self._send_error(f"Error: {str(e)}", 500)
    
    def _handle_entity_merge(self):
        """Handle entity merge request.
        
        Merges source_id into target_id: all relations pointing to source_id
        will be updated to point to target_id, and source_id triples are removed.
        """
        if not validator:
            self._send_error("Validator not initialized")
            return
        
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(content_length).decode('utf-8'))
            
            source_id = data.get("source_id")
            target_id = data.get("target_id")
            
            if not source_id or not target_id:
                self._send_error("source_id and target_id are required", 400)
                return
            
            if source_id == target_id:
                self._send_error("Cannot merge an entity with itself", 400)
                return
            
            # Find source and target entities
            source_entity = None
            target_entity = None
            
            for triple in validator.triples:
                if get_triple_head_id(triple) == source_id and source_entity is None:
                    source_entity = triple.head
                if get_triple_head_id(triple) == target_id and target_entity is None:
                    target_entity = triple.head
                if get_triple_tail_id(triple) == source_id and source_entity is None:
                    source_entity = triple.tail
                if get_triple_tail_id(triple) == target_id and target_entity is None:
                    target_entity = triple.tail
            
            if source_entity is None:
                self._send_error(f"Source entity {source_id} not found", 404)
                return
            
            if target_entity is None:
                self._send_error(f"Target entity {target_id} not found", 404)
                return
            
            # 1. Update triples list
            relations_transferred = 0
            triples_to_remove = []
            
            for i, triple in enumerate(validator.triples):
                head_is_source = get_triple_head_id(triple) == source_id
                tail_is_source = get_triple_tail_id(triple) == source_id
                
                if head_is_source and tail_is_source:
                    triples_to_remove.append(i)
                elif head_is_source:
                    triple.head = target_entity
                    relations_transferred += 1
                elif tail_is_source:
                    triple.tail = target_entity
                    relations_transferred += 1
            
            for i in sorted(triples_to_remove, reverse=True):
                del validator.triples[i]
            
            # 2. Update mapping
            if source_id in validator.id_to_name:
                del validator.id_to_name[source_id]
            
            # 3. SYNC GRAPH: Rebuild graph from updated triples
            if validator.graph is not None:
                visualizer = GraphVisualizer()
                validator.graph = visualizer.build_graph(validator.triples)
            
            print(f"[API] Merged entity {source_id} into {target_id}: "
                  f"relations_transferred={relations_transferred}, "
                  f"self_references_removed={len(triples_to_remove)}")
            
            self._send_json({
                "success": True,
                "message": f"Entities merged successfully",
                "relations_transferred": relations_transferred,
                "self_references_removed": len(triples_to_remove),
            })
                
        except json.JSONDecodeError:
            self._send_error("Invalid JSON", 400)
        except Exception as e:
            import traceback
            print(f"[API] Error in _handle_entity_merge: {e}")
            print(traceback.format_exc())
            self._send_error(f"Error: {str(e)}", 500)

    def _handle_entity_delete(self):
        """Handle entity delete request.
        
        Deletes the entity and all connected triples.
        """
        print("[API] Starting _handle_entity_delete")
        if not validator:
            self._send_error("Validator not initialized")
            return
        
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(content_length).decode('utf-8'))
            
            entity_id = data.get("id")
            print(f"[API] Deleting entity: {entity_id}")
            
            if not entity_id:
                self._send_error("Entity ID is required", 400)
                return
            
            # 1. Update triples list
            triples_removed = 0
            for i in range(len(validator.triples) - 1, -1, -1):
                triple = validator.triples[i]
                if get_triple_head_id(triple) == entity_id or get_triple_tail_id(triple) == entity_id:
                    del validator.triples[i]
                    triples_removed += 1
            
            # 2. Update mapping
            if entity_id in validator.id_to_name:
                del validator.id_to_name[entity_id]
            
            # 3. SYNC GRAPH: Rebuild graph from updated triples
            if validator.graph is not None:
                print(f"[API] Rebuilding graph after entity delete (triples: {len(validator.triples)})...")
                visualizer = GraphVisualizer()
                validator.graph = visualizer.build_graph(validator.triples)
                print("[API] Graph rebuild complete")
            
            print(f"[API] Deleted entity {entity_id} and {triples_removed} connected triples")
            
            self._send_json({
                "success": True,
                "message": f"Entity and {triples_removed} relations deleted successfully",
                "triples_removed": triples_removed,
            })
                
        except json.JSONDecodeError:
            self._send_error("Invalid JSON", 400)
        except Exception as e:
            import traceback
            print(f"[API] Error in _handle_entity_delete: {e}")
            print(traceback.format_exc())
            self._send_error(f"Error: {str(e)}", 500)

    def _handle_triple_delete(self):
        """Handle triple delete request."""
        print("[API] Starting _handle_triple_delete")
        if not validator:
            self._send_error("Validator not initialized")
            return
        
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(content_length).decode('utf-8'))
            
            triple_index = data.get("index")
            print(f"[API] Deleting triple at index: {triple_index}")
            
            if triple_index is None or not isinstance(triple_index, int):
                self._send_error("Triple index is required", 400)
                return
            
            if triple_index < 0 or triple_index >= len(validator.triples):
                self._send_error("Invalid triple index", 400)
                return
            
            # 1. Get triple info for graph sync
            triple = validator.triples[triple_index]
            head_id = get_triple_head_id(triple)
            tail_id = get_triple_tail_id(triple)
            relation = triple.relation
            
            # 2. Delete from triples list
            del validator.triples[triple_index]
            
            # 3. SYNC GRAPH: Rebuild graph from updated triples
            if validator.graph is not None:
                print(f"[API] Rebuilding graph after triple delete (triples: {len(validator.triples)})...")
                visualizer = GraphVisualizer()
                validator.graph = visualizer.build_graph(validator.triples)
                print("[API] Graph rebuild complete")
            
            print(f"[API] Deleted triple {triple_index}: {head_id} --{relation}--> {tail_id}")
            
            self._send_json({
                "success": True,
                "message": "Triple deleted successfully",
            })
                
        except json.JSONDecodeError:
            self._send_error("Invalid JSON", 400)
        except Exception as e:
            import traceback
            print(f"[API] Error in _handle_triple_delete: {e}")
            print(traceback.format_exc())
            self._send_error(f"Error: {str(e)}", 500)


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
    sentence_split: Optional[List[Any]] = None,
    port: int = 5000,
    open_browser: bool = True,
    debug: bool = False,
) -> None:
    """
    Start the validator chat interface.
    
    Args:
        graph: Optional NetworkX graph
        triples: Optional list of Triple objects
        id_to_name: Optional mapping from entity ID to name
        sentence_split: Optional list of Sentence objects with .text attribute for patent description
        port: Port number for the API server
        open_browser: Whether to automatically open browser
        debug: Enable debug mode
    """
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
        _api_server = ThreadedHTTPServer(('127.0.0.1', api_port), GraphValidatorHandler)
        print(f"✓ API Server running on http://localhost:{api_port}")
        _api_server.serve_forever()
    
    _api_thread = threading.Thread(target=run_api_server, daemon=False)
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
        # Replace the port in the config for API routes
        import re
        config_content = re.sub(
            r"destination: 'http://localhost:\d+/api/:path\*'",
            f"destination: 'http://localhost:{api_port}/api/:path*'",
            config_content
        )
        # Replace the port for widget-showcase route
        config_content = re.sub(
            r"destination: 'http://localhost:\d+/widget-showcase'",
            f"destination: 'http://localhost:{api_port}/widget-showcase'",
            config_content
        )
        # Replace the port for static files route
        config_content = re.sub(
            r"destination: 'http://localhost:\d+/static/:path\*'",
            f"destination: 'http://localhost:{api_port}/static/:path*'",
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
                stdout=None,
                stderr=None
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
    def init_background():
        if validator is None or graph is not None or triples is not None:
            print(f"[Server] Initializing validator in background...")
            initialize_validator(graph, triples, id_to_name, sentence_split)
        else:
            print(f"[Server] Validator already exists")
            # Still update sentence_split if provided
            if sentence_split is not None:
                global _sentence_split
                _sentence_split = sentence_split

    # Start initialization in a separate thread
    threading.Thread(target=init_background, daemon=True).start()
    
    # Keep the main thread alive so the cell doesn't finish
    # This allows debugging and keeps the servers running
    print("\n" + "="*70)
    print("[Server] ✓ Servers are running. This cell will stay ACTIVE.")
    print("[Server] ✓ Debugger will remain attached while this loop runs.")
    print("[Server] ✓ Press Ctrl+C in this cell to stop servers.")
    print("="*70)
    import sys
    import datetime
    start_time = datetime.datetime.now()
    
    # Ensure we enter the blocking loop
    if not _server_running:
        print("[Server] WARNING: _server_running was False, setting to True")
        _server_running = True
    
    try:
        iteration = 0
        print(f"[Server] Entering blocking loop (checking every 0.5s)...")
        print(f"[Server] Current time: {datetime.datetime.now().strftime('%H:%M:%S')}")
        print("[Server] About to enter loop.")
        print("[Server] _server_running =", _server_running)
        print("[Server] api thread alive =", _api_thread.is_alive() if _api_thread else None)
        print("[Server] nextjs thread alive =", _nextjs_thread.is_alive() if _nextjs_thread else None)

        while _server_running:
            time.sleep(0.5)  # Shorter sleep for more responsive checking
            iteration += 1
            # Print status every 5 seconds to show the cell is still running
            if iteration % 10 == 0:  # 0.5s * 10 = 5 seconds
                elapsed = datetime.datetime.now() - start_time
                elapsed_seconds = elapsed.total_seconds()
                minutes = int(elapsed_seconds // 60)
                seconds = int(elapsed_seconds % 60)
                current_time = datetime.datetime.now().strftime('%H:%M:%S')
                print(f"[Server] ⏱️  Still running... ({minutes}m {seconds}s elapsed) [Time: {current_time}]")
            sys.stdout.flush()
            
            # Double-check that _server_running hasn't been changed unexpectedly
            if not _server_running:
                print("[Server] _server_running became False, exiting loop")
                break
    except KeyboardInterrupt:
        print("\n[Server] ⚠️  KeyboardInterrupt received. Stopping servers...")
        stop_validator_chat()
    except Exception as e:
        import traceback
        print(f"\n[Server] ❌ Error in main loop: {e}")
        print(f"[Server] Traceback:\n{traceback.format_exc()}")
        stop_validator_chat()
        raise
    finally:
        print("[Server] Main loop exited. Servers may still be running in background threads.")


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
