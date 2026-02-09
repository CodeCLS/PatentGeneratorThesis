"""
Simple HTTP server for Graph Validator Chat Interface.
Single file with everything needed - no Flask, no Jinja2 dependencies!
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
import json
import threading
import webbrowser
import time
import socket
import pickle
import base64
import io
import os
import subprocess
import re
from typing import Optional, Dict, Any, List
import networkx as nx
import logging
from pathlib import Path
import uuid

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

# Setup logging for server errors
logging.basicConfig(
    filename='server_error.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from tools.graph.data.Triple import Triple
from tools.graph.neo4j_manager import Neo4jManager
from tools.graph.langgraph.validator import GraphValidatorLangGraph
from tools.graph.langgraph.helpers import get_triple_head_id, get_triple_tail_id
from tools.graph.langgraph.question import Question
from tools.graph.claim_generation.claim_generator_langchain import ClaimGeneratorLangChain
from tools.graph.rag.graph_rag import GraphRAG
from tools.graph.visualizer import GraphVisualizer
from tools.api.llm_api_repo import LLmApi_Repo
from tools.helper.json_helper import JsonHelper
from PatentProvider import PatentProvider

try:
    from .pipeline_manager import PipelineManager
except ImportError:  # pragma: no cover
    from pipeline_manager import PipelineManager


# Global validator instance
validator: Optional[GraphValidatorLangGraph] = None
# Global sentence_split for patent description
_sentence_split: Optional[List[Any]] = None
_server_running = False
_api_server: Optional[HTTPServer] = None
_api_thread: Optional[threading.Thread] = None
_nextjs_process: Optional[subprocess.Popen] = None
_nextjs_thread: Optional[threading.Thread] = None
_neo4j_manager: Optional[Neo4jManager] = None
_pipeline_manager: Optional[PipelineManager] = None
_pipeline_lock = threading.Lock()
_pipeline_thread: Optional[threading.Thread] = None
_persist_path = Path(__file__).parent / "persisted_validator.pkl"
_server_session_id = str(uuid.uuid4())

# Global progress tracking
_claim_generation_progress: Dict[str, Any] = {}
_pipeline_progress: Dict[str, Any] = {"stage": "idle", "message": "No pipeline running", "progress": 0}
_pipeline_progress_lock = threading.Lock()
_cached_claims: List[Dict[str, Any]] = []

# Graph HTML cache: only built on explicit Refresh; invalidated when triples/entities change
_cached_graph_html: Optional[str] = None
_cached_graph_triples_count: Optional[int] = None
_cached_graph_layout: Optional[str] = None

# Source metadata for PDF header (patent ID or file used, short abstract)
_source_patent_id: Optional[str] = None
_source_filename: Optional[str] = None
_source_type: Optional[str] = None  # "patent_id" | "pdf" | "text"


def _invalidate_graph_cache() -> None:
    """Call when triples or entities change so the graph page shows 'outdated' until Refresh."""
    global _cached_graph_html, _cached_graph_triples_count, _cached_graph_layout
    _cached_graph_html = None
    _cached_graph_triples_count = None
    _cached_graph_layout = None


if load_dotenv:
    load_dotenv()


def _extract_pdf_text_from_bytes(pdf_bytes: bytes) -> str:
    try:
        from PyPDF2 import PdfReader
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyPDF2 is required for PDF uploads") from exc

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text)
    return "\n\n".join(pages).strip()


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


def _save_persisted_validator(
    *,
    graph: Optional[nx.MultiDiGraph],
    triples: Optional[List[Triple]],
    id_to_name: Optional[Dict[str, str]],
    sentence_split: Optional[List[Any]],
) -> None:
    """Persist validator inputs for reuse across sessions."""
    global _source_patent_id, _source_filename, _source_type
    try:
        payload = {
            "graph": graph,
            "triples": triples or [],
            "id_to_name": id_to_name or {},
            "sentence_split": sentence_split or [],
            "source_metadata": {
                "patent_id": _source_patent_id,
                "filename": _source_filename,
                "source": _source_type,
            },
        }
        with open(_persist_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"[Server] Persisted validator state to {_persist_path}")
    except Exception as e:
        print(f"[Server] Warning: failed to persist validator state: {e}")


def _load_persisted_validator() -> Optional[Dict[str, Any]]:
    """Load persisted validator inputs, if available."""
    if not _persist_path.exists():
        return None
    try:
        with open(_persist_path, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            return None
        return payload
    except Exception as e:
        print(f"[Server] Warning: failed to load persisted validator state: {e}")
        return None


def _get_neo4j_manager() -> Optional[Neo4jManager]:
    """Lazy-load a Neo4j manager from environment variables."""
    global _neo4j_manager
    uri = (os.getenv("NEO4J_URI") or "").strip()
    user = (os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or "").strip()
    password = (os.getenv("NEO4J_PASSWORD") or "").strip()
    database = (os.getenv("NEO4J_DATABASE") or "").strip() or None

    if not uri:
        aura_instance_id = (os.getenv("AURA_INSTANCEID") or "").strip()
        aura_instance_name = (os.getenv("AURA_INSTANCENAME") or "").strip()
        aura_host = aura_instance_id or aura_instance_name
        if aura_host:
            uri = f"neo4j+s://{aura_host}.databases.neo4j.io"

    if not uri or not user or not password:
        return None

    if (
        _neo4j_manager is None
        or _neo4j_manager.uri != uri
        or _neo4j_manager.user != user
        or _neo4j_manager.password != password
        or _neo4j_manager.database != database
    ):
        _neo4j_manager = Neo4jManager(uri=uri, user=user, password=password, database=database)

    return _neo4j_manager


def get_validator() -> Optional[GraphValidatorLangGraph]:
    """Get the current validator instance."""
    return validator


def _get_pipeline_manager() -> PipelineManager:
    """Lazy-load PipelineManager."""
    global _pipeline_manager
    if _pipeline_manager is None:
        _pipeline_manager = PipelineManager()
    return _pipeline_manager


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
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        print(f"[API] GET {path}")
        
        # Serve static files
        if path.startswith('/static/'):
            self._serve_static_file(path)
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
        elif path == '/api/neo4j/stats':
            self._send_json(self._get_neo4j_stats())
        elif path == '/api/graph/status':
            self._send_json(self._get_graph_status())
        elif path == '/api/graph/html':
            layout = (query.get("layout") or [""])[0].strip().lower()
            refresh = (query.get("refresh") or ["0"])[0].strip().lower() in ("1", "true", "yes")
            self._send_json(self._get_graph_html(layout=layout, refresh=refresh))
        elif path == '/api/graph/neo4j':
            self._send_json(self._get_graph_neo4j_html())
        elif path == '/api/claims':
            # GET endpoint to retrieve generated claims
            self._send_json(self._get_generated_claims())
        elif path == '/api/source':
            # GET source metadata for PDF header (patent ID or file, short abstract)
            self._send_json(self._get_source())
        elif path == '/api/claims/progress':
            # GET endpoint to retrieve claim generation progress
            self._send_json(self._get_claim_progress())
        elif path == '/api/pipeline/progress':
            # GET endpoint to retrieve pipeline progress
            self._send_json(self._get_pipeline_progress())
        elif path == '/api/analyze':
            self._send_json(self._get_analyze_data())
        elif path == '/api/session':
            self._send_json({"session_id": _server_session_id})
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
            elif path == '/api/neo4j/upload':
                print("[API] Routing to _handle_neo4j_upload")
                self._handle_neo4j_upload()
            elif path == '/api/pipeline/start':
                print("[API] Routing to _handle_pipeline_start")
                self._handle_pipeline_start()
            elif path == '/api/pipeline/run':
                print("[API] Routing to _handle_pipeline_run")
                self._handle_pipeline_run()
            elif path == '/api/pipeline/restore':
                print("[API] Routing to _handle_pipeline_restore")
                self._handle_pipeline_restore()
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
            elif path == '/api/triples/update':
                print("[API] Routing to _handle_triple_update")
                self._handle_triple_update()
            elif path == '/api/triples/add':
                print("[API] Routing to _handle_triple_add")
                self._handle_triple_add()
            elif path == '/api/triples/suggest':
                print("[API] Routing to _handle_triple_suggest")
                self._handle_triple_suggest()
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
                manager = _get_neo4j_manager()
                if not manager:
                    self._send_json({
                        "query": user_query,
                        "ran": False,
                        "message": "Neo4j not configured",
                        "results": [],
                    })
                    return

                limit = data.get("limit")
                if not isinstance(limit, int) or limit <= 0:
                    limit = None

                result = manager.run_cypher(user_query, limit=limit)
                self._send_json({
                    "query": user_query,
                    "ran": True,
                    "message": "Cypher query run",
                    "results": result.get("records", []),
                    "summary": result.get("summary", {}),
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

    def _handle_neo4j_upload(self):
        """Upload the current validator graph to Neo4j."""
        if not validator:
            self._send_error("Validator not initialized")
            return

        manager = _get_neo4j_manager()
        if not manager:
            self._send_error("Neo4j not configured")
            return

        try:
            content_length = int(self.headers.get('Content-Length', 0))
            payload = {}
            if content_length > 0:
                payload = json.loads(self.rfile.read(content_length).decode('utf-8'))

            node_label = payload.get("node_label", "Entity")
            rel_type = payload.get("rel_type", "LINKS_TO")
            dedupe_edges = payload.get("dedupe_edges", True)
            database = payload.get("database")

            graph = validator.graph
            if graph is None:
                visualizer = GraphVisualizer()
                graph = visualizer.build_graph(validator.triples)
                validator.graph = graph

            id_to_name = validator.id_to_name or GraphVisualizer.build_id_to_name_map_from_triples(validator.triples)

            manager.push_multidigraph_to_aura(
                graph,
                node_label=node_label,
                rel_type=rel_type,
                id_to_name=id_to_name,
                dedupe_edges=dedupe_edges,
                database=database,
            )

            self._send_json({
                "success": True,
                "message": "Graph uploaded to Neo4j",
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
            })
        except json.JSONDecodeError:
            self._send_error("Invalid JSON", 400)
        except Exception as e:
            import traceback
            print(f"[API] Error in _handle_neo4j_upload: {e}")
            print(traceback.format_exc())
            self._send_error(f"Error: {str(e)}", 500)

    @staticmethod
    def _entity_matches_ref(entity: Any, ref_value: str) -> bool:
        if not entity or not ref_value:
            return False
        return (
            getattr(entity, "ref", None) == ref_value
            or getattr(entity, "id", None) == ref_value
            or getattr(entity, "ref_short", None) == ref_value
        )

    @staticmethod
    def _highlight_mentions(text: str, mentions: List[Any]) -> str:
        if not text or not mentions:
            return text
        spans = []
        for ent in mentions:
            try:
                start = int(getattr(ent, "start", 0) or 0)
                end = int(getattr(ent, "end", 0) or 0)
            except (TypeError, ValueError):
                continue
            if 0 <= start < end <= len(text):
                spans.append((start, end))
        if not spans:
            return text
        spans.sort()
        out = []
        cursor = 0
        for start, end in spans:
            if start < cursor:
                continue
            if start > cursor:
                out.append(text[cursor:start])
            out.append(f"[[{text[start:end]}]]")
            cursor = end
        if cursor < len(text):
            out.append(text[cursor:])
        return "".join(out)

    def _collect_entity_context(self, ref_value: str, max_sentences: int = 8) -> Dict[str, Any]:
        if not _sentence_split or not ref_value:
            return {"mentions": [], "total_mentions": 0}

        mentions = []
        total = 0
        for idx, sentence in enumerate(_sentence_split):
            text = getattr(sentence, "text", "") or ""
            entities = getattr(sentence, "entities", []) or []
            matched = [e for e in entities if self._entity_matches_ref(e, ref_value)]
            if not matched:
                continue
            total += len(matched)
            if len(mentions) < max_sentences:
                mentions.append({
                    "sentence_index": idx + 1,
                    "text": self._highlight_mentions(text, matched),
                })

        return {"mentions": mentions, "total_mentions": total}

    def _handle_triple_suggest(self):
        """Generate AI suggestions for triples with textual context."""
        if not validator:
            self._send_error("Validator not initialized")
            return

        try:
            content_length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(content_length).decode('utf-8')) if content_length > 0 else {}
            indices = data.get("indices") or []
            if not isinstance(indices, list) or not indices:
                self._send_error("indices must be a non-empty list", 400)
                return

            # Build prompt payload
            batch = []
            for idx in indices:
                try:
                    idx_int = int(idx)
                except (TypeError, ValueError):
                    continue
                if idx_int < 0 or idx_int >= len(validator.triples):
                    continue
                triple = validator.triples[idx_int]
                head_id = get_triple_head_id(triple)
                tail_id = get_triple_tail_id(triple)
                batch.append({
                    "index": idx_int,
                    "head": triple.head.name,
                    "relation": triple.relation,
                    "tail": triple.tail.name,
                    "head_id": head_id,
                    "tail_id": tail_id,
                })

            if not batch:
                self._send_error("No valid triples found for indices", 400)
                return

            context_payload = []
            for item in batch:
                head_ctx = self._collect_entity_context(item["head_id"])
                tail_ctx = self._collect_entity_context(item["tail_id"])
                context_payload.append({
                    "index": item["index"],
                    "display_index": item["index"] + 1,
                    "triple": f"{item['head']} --[{item['relation']}]--> {item['tail']}",
                    "head_mentions_total": head_ctx["total_mentions"],
                    "head_mentions": head_ctx["mentions"],
                    "tail_mentions_total": tail_ctx["total_mentions"],
                    "tail_mentions": tail_ctx["mentions"],
                })

            prompt = (
                "You are reviewing knowledge-graph triples extracted from a patent.\n"
                "For each triple, suggest one action:\n"
                "- KEEP (no change)\n"
                "- DELETE (bad or irrelevant)\n"
                "- CHANGE (specify exact change, e.g., change head to X, tail to Y, or relation to Z)\n"
                "Use the provided mention context to decide. The mention text highlights the entity as [[...]].\n"
                "Return ONLY a JSON array of objects with fields:\n"
                "index (number, 0-based), action (KEEP|DELETE|CHANGE), suggestion (short text), reason (short text).\n"
                "You will also see display_index (1-based); do NOT use it for the response.\n"
                "If action is KEEP, reason can be empty.\n\n"
                f"Triples with context:\n{json.dumps(context_payload, ensure_ascii=False, indent=2)}\n"
            )

            from tools.api.llm_models.deepseek_model import DeepSeekModel
            llm = LLmApi_Repo(llm_client=DeepSeekModel(max_tokens=2000))
            raw = llm.chat(prompt)
            response_text = raw.get("content") if isinstance(raw, dict) else str(raw or "")
            parsed = JsonHelper.parse_json(response_text)
            if not isinstance(parsed, list):
                parsed = []

            suggestions = []
            batch_indices = {b["index"] for b in batch}
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                try:
                    idx_val = int(item.get("index"))
                except (TypeError, ValueError):
                    continue
                if idx_val not in batch_indices and (idx_val - 1) in batch_indices:
                    idx_val = idx_val - 1
                action = str(item.get("action", "")).strip().upper()
                suggestion = str(item.get("suggestion", "")).strip()
                reason = str(item.get("reason", "")).strip()
                if idx_val in batch_indices:
                    if action == "KEEP":
                        reason = ""
                    suggestions.append({
                        "index": idx_val,
                        "action": action or "KEEP",
                        "suggestion": suggestion or "Keep as-is.",
                        "reason": reason,
                    })

            self._send_json({"suggestions": suggestions})
        except json.JSONDecodeError:
            self._send_error("Invalid JSON", 400)
        except Exception as e:
            import traceback
            print(f"[API] Error in _handle_triple_suggest: {e}")
            print(traceback.format_exc())
            self._send_error(f"Error: {str(e)}", 500)

    def _handle_pipeline_run(self):
        """Run the Main.ipynb pipeline using a patent ID or raw text."""
        print("[API] Starting _handle_pipeline_run")
        try:
            global _pipeline_progress
            def set_pipeline_progress(update: Dict[str, Any]) -> None:
                global _pipeline_progress
                with _pipeline_progress_lock:
                    _pipeline_progress = update

            set_pipeline_progress({
                "stage": "starting",
                "message": "Starting pipeline",
                "progress": 1,
            })

            content_length = int(self.headers.get('Content-Length', 0))
            if content_length <= 0:
                self._send_error("Request body required", 400)
                return

            data = json.loads(self.rfile.read(content_length).decode('utf-8'))
            patent_id = data.get("patent_id") or data.get("patentId") or data.get("patentID")
            text = data.get("text") or data.get("patent_text") or ""
            pdf_base64 = data.get("pdf_base64") or data.get("pdfBase64") or ""
            filename = data.get("filename") or ""

            source = None
            if isinstance(text, str) and text.strip():
                source = "text"
            elif isinstance(pdf_base64, str) and pdf_base64.strip():
                source = "pdf"
            elif patent_id:
                patent_id = str(patent_id).strip()
                if not patent_id:
                    self._send_error("Patent ID is empty", 400)
                    return
                source = "patent_id"
                print(f"[API] Fetching patent description for {patent_id}")
                text = PatentProvider().getDescription(patent_id)

            if source == "pdf":
                try:
                    pdf_bytes = base64.b64decode(pdf_base64)
                except Exception:
                    self._send_error("Invalid PDF data", 400)
                    return
                text = _extract_pdf_text_from_bytes(pdf_bytes)

            if not isinstance(text, str) or not text.strip():
                if source == "pdf":
                    self._send_error("No extractable text found in PDF. Try a text-based PDF or OCR.", 400)
                else:
                    self._send_error("No text provided or fetched", 400)
                return

            manager = _get_pipeline_manager()
            with _pipeline_lock:
                result = manager.run(text, progress_callback=set_pipeline_progress)

            set_pipeline_progress({
                "stage": "validator_init",
                "message": "Initializing validator",
                "progress": 98,
            })

            global _source_patent_id, _source_filename, _source_type
            _source_patent_id = patent_id if source == "patent_id" else None
            _source_filename = filename if filename else None
            _source_type = source

            initialize_validator(
                graph=result.graph,
                triples=result.triples,
                id_to_name=result.id_to_name,
                sentence_split=result.sentence_split,
            )
            _save_persisted_validator(
                graph=result.graph,
                triples=result.triples,
                id_to_name=result.id_to_name,
                sentence_split=result.sentence_split,
            )

            graph_data = None
            if result.graph is not None:
                graph_data = base64.b64encode(pickle.dumps(result.graph)).decode('utf-8')

            self._send_json({
                "success": True,
                "source": source,
                "patent_id": patent_id if source == "patent_id" else None,
                "filename": filename if filename else None,
                "num_nodes": result.graph.number_of_nodes() if result.graph else 0,
                "num_edges": result.graph.number_of_edges() if result.graph else 0,
                "num_triples": len(result.triples),
                "id_to_name": result.id_to_name,
                "graph": graph_data,
                "merge_stats": result.merge_stats,
            })
            set_pipeline_progress({
                "stage": "complete",
                "message": "Pipeline complete",
                "progress": 100,
            })
        except json.JSONDecodeError:
            self._send_error("Invalid JSON", 400)
        except Exception as e:
            import traceback
            print(f"[API] Error in _handle_pipeline_run: {e}")
            print(traceback.format_exc())
            with _pipeline_progress_lock:
                _pipeline_progress = {
                    "stage": "error",
                    "message": f"Pipeline failed: {str(e)}",
                    "progress": 0,
                }
            self._send_error(f"Error: {str(e)}", 500)

    def _handle_pipeline_start(self):
        """Start the pipeline in a background thread and return immediately."""
        print("[API] Starting _handle_pipeline_start")
        try:
            global _pipeline_progress, _pipeline_thread

            with _pipeline_lock:
                if _pipeline_thread and _pipeline_thread.is_alive():
                    self._send_error("Pipeline already running", 409)
                    return

            content_length = int(self.headers.get('Content-Length', 0))
            if content_length <= 0:
                self._send_error("Request body required", 400)
                return

            data = json.loads(self.rfile.read(content_length).decode('utf-8'))
            patent_id = data.get("patent_id") or data.get("patentId") or data.get("patentID")
            text = data.get("text") or data.get("patent_text") or ""
            pdf_base64 = data.get("pdf_base64") or data.get("pdfBase64") or ""
            filename = data.get("filename") or ""

            source = None
            if isinstance(text, str) and text.strip():
                source = "text"
            elif isinstance(pdf_base64, str) and pdf_base64.strip():
                source = "pdf"
            elif patent_id:
                patent_id = str(patent_id).strip()
                if not patent_id:
                    self._send_error("Patent ID is empty", 400)
                    return
                source = "patent_id"
            else:
                self._send_error("No text provided or fetched", 400)
                return

            def set_pipeline_progress(update: Dict[str, Any]) -> None:
                global _pipeline_progress
                with _pipeline_progress_lock:
                    _pipeline_progress = update

            set_pipeline_progress({
                "stage": "starting",
                "message": "Starting pipeline",
                "progress": 1,
            })

            def pipeline_task() -> None:
                global _pipeline_thread
                try:
                    task_text = text
                    task_patent_id = patent_id
                    task_pdf_base64 = pdf_base64
                    task_filename = filename
                    task_source = source

                    if task_source == "patent_id":
                        print(f"[API] Fetching patent description for {task_patent_id}")
                        task_text = PatentProvider().getDescription(task_patent_id)
                    elif task_source == "pdf":
                        try:
                            pdf_bytes = base64.b64decode(task_pdf_base64)
                            task_text = _extract_pdf_text_from_bytes(pdf_bytes)
                        except Exception as exc:
                            set_pipeline_progress({
                                "stage": "error",
                                "message": f"Failed to read PDF: {exc}",
                                "progress": 0,
                            })
                            return

                    if not isinstance(task_text, str) or not task_text.strip():
                        set_pipeline_progress({
                            "stage": "error",
                            "message": (
                                "No extractable text found in PDF. Try a text-based PDF or OCR."
                                if task_source == "pdf"
                                else "No text provided or fetched"
                            ),
                            "progress": 0,
                        })
                        return

                    manager = _get_pipeline_manager()
                    with _pipeline_lock:
                        result = manager.run(task_text, progress_callback=set_pipeline_progress)

                    set_pipeline_progress({
                        "stage": "validator_init",
                        "message": "Initializing validator",
                        "progress": 98,
                    })

                    global _source_patent_id, _source_filename, _source_type
                    _source_patent_id = task_patent_id if task_source == "patent_id" else None
                    _source_filename = task_filename if task_filename else None
                    _source_type = task_source

                    initialize_validator(
                        graph=result.graph,
                        triples=result.triples,
                        id_to_name=result.id_to_name,
                        sentence_split=result.sentence_split,
                    )
                    _save_persisted_validator(
                        graph=result.graph,
                        triples=result.triples,
                        id_to_name=result.id_to_name,
                        sentence_split=result.sentence_split,
                    )

                    set_pipeline_progress({
                        "stage": "complete",
                        "message": "Pipeline complete",
                        "progress": 100,
                    })
                except Exception as e:
                    print(f"[API] Error in pipeline background task: {e}")
                    set_pipeline_progress({
                        "stage": "error",
                        "message": f"Pipeline failed: {str(e)}",
                        "progress": 0,
                    })
                finally:
                    _pipeline_thread = None

            _pipeline_thread = threading.Thread(target=pipeline_task, daemon=True)
            _pipeline_thread.start()

            self._send_json({
                "success": True,
                "started": True,
                "source": source,
                "patent_id": patent_id if source == "patent_id" else None,
                "filename": filename if filename else None,
                "progress": _pipeline_progress,
            })
        except json.JSONDecodeError:
            self._send_error("Invalid JSON", 400)
        except Exception as e:
            import traceback
            print(f"[API] Error in _handle_pipeline_start: {e}")
            print(traceback.format_exc())
            with _pipeline_progress_lock:
                _pipeline_progress = {
                    "stage": "error",
                    "message": f"Pipeline failed: {str(e)}",
                    "progress": 0,
                }
            self._send_error(f"Error: {str(e)}", 500)

    def _handle_pipeline_restore(self):
        """Restore validator/graph from persisted cache if available."""
        print("[API] Starting _handle_pipeline_restore")
        try:
            global _pipeline_progress

            # If validator already exists, nothing to restore
            if validator is not None:
                self._send_json({
                    "success": True,
                    "restored": False,
                    "message": "Validator already initialized",
                })
                return

            persisted = _load_persisted_validator()
            if not persisted:
                self._send_error("No persisted validator state found", 404)
                return

            global _source_patent_id, _source_filename, _source_type
            sm = persisted.get("source_metadata") or {}
            _source_patent_id = sm.get("patent_id")
            _source_filename = sm.get("filename")
            _source_type = sm.get("source")

            with _pipeline_lock:
                set_progress = {
                    "stage": "validator_init",
                    "message": "Restoring validator from cache",
                    "progress": 98,
                }
                with _pipeline_progress_lock:
                    _pipeline_progress = set_progress

                initialize_validator(
                    graph=persisted.get("graph"),
                    triples=persisted.get("triples"),
                    id_to_name=persisted.get("id_to_name"),
                    sentence_split=persisted.get("sentence_split"),
                )

                with _pipeline_progress_lock:
                    _pipeline_progress = {
                        "stage": "complete",
                        "message": "Cache restore complete",
                        "progress": 100,
                    }

            graph = persisted.get("graph")
            triples = persisted.get("triples") or []
            id_to_name = persisted.get("id_to_name") or {}

            self._send_json({
                "success": True,
                "restored": True,
                "num_nodes": graph.number_of_nodes() if graph else 0,
                "num_edges": graph.number_of_edges() if graph else 0,
                "num_triples": len(triples),
                "id_to_name": id_to_name,
            })
        except Exception as e:
            import traceback
            print(f"[API] Error in _handle_pipeline_restore: {e}")
            print(traceback.format_exc())
            with _pipeline_progress_lock:
                _pipeline_progress = {
                    "stage": "error",
                    "message": f"Restore failed: {str(e)}",
                    "progress": 0,
                }
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

    def _get_analyze_data(self) -> Dict[str, Any]:
        """Get sentence text with entity spans for analysis view."""
        if not _sentence_split:
            return {
                "error": "No sentence data available. Please run the pipeline first.",
                "text": "",
                "sentences": [],
                "triples": [],
            }

        sentences_data = []
        parts = []
        offset = 0
        for idx, sentence in enumerate(_sentence_split):
            text = getattr(sentence, "text", "") or ""
            sentence_start = offset
            sentence_end = sentence_start + len(text)
            parts.append(text)

            entities_data = []
            for entity in getattr(sentence, "entities", []) or []:
                ref_value = (
                    getattr(entity, "ref", None)
                    or getattr(entity, "id", None)
                    or getattr(entity, "ref_short", None)
                    or getattr(entity, "name", "")
                )
                entities_data.append({
                    "name": getattr(entity, "name", ""),
                    "label": getattr(entity, "label", ""),
                    "ref": ref_value,
                    "ref_short": getattr(entity, "ref_short", "") or "",
                    "start": int(getattr(entity, "start", 0) or 0),
                    "end": int(getattr(entity, "end", 0) or 0),
                    "sentence_id": getattr(entity, "sentence_id", "") or f"s{idx}",
                    "sentence_index": idx,
                    "entity_type": getattr(entity, "entity_type", None),
                })

            sentences_data.append({
                "id": getattr(sentence, "id", None) or f"s{idx}",
                "index": idx,
                "text": text,
                "start": sentence_start,
                "end": sentence_end,
                "entities": entities_data,
            })

            offset = sentence_end + 1

        full_text = " ".join(parts)
        triples_data = []
        if validator:
            triples_payload = self._get_triples()
            triples_data = triples_payload.get("triples", [])

        return {
            "text": full_text,
            "sentences": sentences_data,
            "triples": triples_data,
        }
    
    def _get_graph_status(self) -> Dict[str, Any]:
        """Return whether the cached graph is current. Used by graph page to show 'outdated' tag."""
        global _cached_graph_html, _cached_graph_triples_count
        v = validator
        triples_count = len(v.triples) if v else 0
        has_graph = _cached_graph_html is not None and len((_cached_graph_html or "").strip()) > 0
        outdated = not has_graph or _cached_graph_triples_count is None or _cached_graph_triples_count != triples_count
        return {
            "hasGraph": has_graph,
            "outdated": outdated,
            "triplesCount": triples_count,
            "graphTriplesCount": _cached_graph_triples_count,
        }

    def _get_graph_html(self, *, layout: str = "", refresh: bool = False) -> Dict[str, Any]:
        """Return cached graph HTML, or build and cache when refresh=True. Never auto-build on first visit."""
        global _cached_graph_html, _cached_graph_triples_count, _cached_graph_layout
        if not validator:
            return {"error": "Validator not initialized", "html": ""}

        triples_count = len(validator.triples)
        layout_n = (layout or "").strip().lower()

        if refresh:
            # Explicit Refresh: build and cache
            result = self._build_graph_html(layout=layout_n)
            if "error" not in result or result.get("html"):
                _cached_graph_html = result.get("html") or ""
                _cached_graph_triples_count = triples_count
                _cached_graph_layout = layout_n
            return result

        # No refresh: return cache only if current and same layout
        if (
            _cached_graph_html
            and _cached_graph_triples_count == triples_count
            and _cached_graph_layout == layout_n
        ):
            return {"html": _cached_graph_html}
        # No cache or outdated
        if not _cached_graph_html:
            return {
                "html": "",
                "error": "Graph not generated. Press Refresh to generate.",
            }
        return {
            "html": _cached_graph_html,
            "error": "Graph is outdated (triples changed). Press Refresh to update.",
            "outdated": True,
        }

    def _build_graph_html(self, *, layout: str = "") -> Dict[str, Any]:
        """Build graph HTML from current triples. Does not read or set cache."""
        if not validator:
            return {"error": "Validator not initialized", "html": ""}
        try:
            from tools.graph.visualizer import GraphVisualizer
            import tempfile
            import os

            triples = validator.triples
            id_to_name = validator.id_to_name

            if not triples:
                return {"error": "No graph or triples available", "html": ""}

            visualizer = GraphVisualizer()
            graph = visualizer.build_graph(triples)
            validator.graph = graph

            print(f"[API] Generating graph HTML (nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()})...")
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False)
            temp_path = temp_file.name
            temp_file.close()

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
                hierarchical = layout in {"tree", "hierarchical", "hierarchy"}
                visualizer.visualize_pyvis(
                    graph,
                    out_file=temp_path,
                    id_to_name=id_to_name,
                    hierarchical=hierarchical,
                )

            with open(temp_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            os.unlink(temp_path)
            print(f"[API] Graph HTML generated successfully ({len(html_content)} chars)")

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

    def _get_graph_neo4j_html(self) -> Dict[str, Any]:
        """Return an HTML page that renders the Neo4j graph with Neovis."""
        manager = _get_neo4j_manager()
        if not manager:
            return {"error": "Neo4j is not configured (NEO4J_URI/USERNAME/PASSWORD).", "html": ""}

        env_cypher = (os.getenv("NEO4J_GRAPH_CYPHER") or "").strip()
        if not env_cypher:
            env_cypher = "MATCH (n)-[r:INTERACTS]->(m) RETURN *"
        initial_cypher = env_cypher
        database = manager.database or ""
        database_line = f'            database: {json.dumps(database)},\n' if database else ""

        html = f"""
<!doctype html>
<html>
<head>
    <title>Neovis.js Simple Example</title>
    <style type="text/css">
        html, body {{
            font: 16pt arial;
        }}

        #viz {{
            width: 900px;
            height: 700px;
            border: 1px solid lightgray;
            font: 22pt arial;
        }}
    </style>
    <script src="https://unpkg.com/neovis.js@2.0.2"></script>
</head>
<body onload="draw()">
<div id="viz"></div>
<script type="text/javascript">
    let neoViz;

    function draw() {{
        const config = {{
            containerId: "viz",
            neo4j: {{
                serverUrl: {json.dumps(manager.uri)},
                serverUser: {json.dumps(manager.user)},
                serverPassword: {json.dumps(manager.password)},
            }},
{database_line}            labels: {{
                Character: {{
                    label: "name",
                    value: "pagerank",
                    group: "community",
                    [NeoVis.NEOVIS_ADVANCED_CONFIG]: {{
                        function: {{
                            title: (node) => neoViz.nodeToHtml(node, [
                                "name",
                                "pagerank"
                            ])
                        }}
                    }}
                }}
            }},
            relationships: {{
                INTERACTS: {{
                    value: "weight"
                }}
            }},
            initialCypher: {json.dumps(initial_cypher)}
        }};

        neoViz = new NeoVis.default(config);
        neoViz.render();
    }}
</script>
</body>
</html>
""".strip()

        return {"html": html}

    def _get_neo4j_stats(self) -> Dict[str, Any]:
        """Return node/edge counts directly from Neo4j."""
        manager = _get_neo4j_manager()
        if not manager:
            return {"error": "Neo4j is not configured (NEO4J_URI/USERNAME/PASSWORD)."}

        try:
            node_result = manager.run_cypher("MATCH (n) RETURN count(n) AS nodes")
            edge_result = manager.run_cypher("MATCH ()-[r]->() RETURN count(r) AS edges")
            nodes = 0
            edges = 0
            if node_result.get("records"):
                nodes = node_result["records"][0].get("nodes", 0) or 0
            if edge_result.get("records"):
                edges = edge_result["records"][0].get("edges", 0) or 0

            return {
                "nodes": nodes,
                "edges": edges,
                "database": manager.database or "",
            }
        except Exception as e:
            return {"error": f"Neo4j stats error: {str(e)}"}
    
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
            patent_description = ""
            num_independent = 3
            num_dependent_per_independent = 2
            similarity_threshold = 0.3
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                try:
                    raw = self.rfile.read(content_length)
                    body = raw.decode('utf-8') if isinstance(raw, bytes) else raw
                    data = json.loads(body) if body and body.strip() else {}
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"[API] _handle_generate_claims: Body parse failed (using defaults): {e}")
                    data = {}
                patent_description = data.get("patent_description") or ""
                num_independent = data.get("num_independent", 3)
                num_dependent_per_independent = data.get("num_dependent_per_independent", 2)
                similarity_threshold = data.get("similarity_threshold", 0.3)
            
            patent_description = (patent_description or "").strip() if patent_description is not None else ""
            # Get patent description from validator state if not provided
            print(f"[DEBUG] _handle_generate_claims: Initial patent_description: '{patent_description[:100] if patent_description else 'EMPTY'}...'")
            print(f"[DEBUG] _handle_generate_claims: patent_description is None: {patent_description is None}")
            print(f"[DEBUG] _handle_generate_claims: patent_description.strip() == '': {(patent_description or '').strip() == ''}")
            
            if not patent_description:
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
                if not patent_description:
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
            print(f"[DEBUG] _handle_generate_claims: Final patent_description preview: {patent_description[:200] if patent_description else 'EMPTY'}...")
            print(f"[DEBUG] _handle_generate_claims: num_independent={num_independent}, num_dependent_per_independent={num_dependent_per_independent}")
            
            if not validator.graph or not validator.triples:
                self._send_error(
                    "Graph or triples not ready. Run the pipeline (upload/analyze) first, then generate claims.",
                    400,
                )
                return
            
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
            
            # Convert to JSON-serializable format (ensure used_triples and prompt always present)
            claims_data = []
            for claim in generated_claims:
                raw_triples = getattr(claim, "used_triples", None)
                if not isinstance(raw_triples, list):
                    raw_triples = []
                # Ensure each triple is a plain dict for JSON
                used_triples = []
                for t in raw_triples:
                    if isinstance(t, dict):
                        used_triples.append({
                            "head": str(t.get("head", "")),
                            "relation": str(t.get("relation", "")),
                            "tail": str(t.get("tail", "")),
                            "similarity": float(t.get("similarity", 0.0)) if t.get("similarity") is not None else 0.0,
                        })
                raw_prompt = getattr(claim, "prompt", None)
                prompt = str(raw_prompt) if raw_prompt else ""
                claims_data.append({
                    "claim_number": claim.claim_number,
                    "claim_text": claim.claim_text,
                    "claim_type": claim.claim_type,
                    "parent_claim_number": getattr(claim, "parent_claim_number", None),
                    "focus": getattr(claim, "focus", "") or "",
                    "used_triples": used_triples,
                    "prompt": prompt,
                    "refinement_iterations": getattr(claim, "refinement_iterations", 0) or 0,
                    "final_score": float(getattr(claim, "final_score", 0.0) or 0.0),
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

    def _get_source(self) -> Dict[str, Any]:
        """Get source metadata for PDF header: patent ID or file used, and short abstract.
        When source is an EPO patent ID, abstract comes from PatentProvider.getAbstract only.
        When source is a file upload, abstract is first 4 sentences from pipeline text."""
        global _source_patent_id, _source_filename, _source_type, _sentence_split
        short_abstract = ""
        if _source_type == "patent_id" and _source_patent_id:
            try:
                short_abstract = PatentProvider().getAbstract(_source_patent_id) or ""
            except Exception:
                short_abstract = ""
        # Only use pipeline text for file uploads (pdf/text), not when patent_id was used
        if not short_abstract and _source_type != "patent_id" and _sentence_split and len(_sentence_split) > 0:
            sentences = []
            for s in _sentence_split[:4]:
                t = getattr(s, "text", None) or (s if isinstance(s, str) else str(s))
                if t and str(t).strip():
                    sentences.append(str(t).strip())
            short_abstract = " ".join(sentences) if sentences else ""
        source_label = None
        if _source_type == "patent_id" and _source_patent_id:
            source_label = f"Patent ID: {_source_patent_id}"
        elif _source_filename:
            source_label = f"File: {_source_filename}"
        elif _source_type:
            source_label = f"Source: {_source_type}"
        return {
            "success": True,
            "patent_id": _source_patent_id,
            "filename": _source_filename,
            "source": _source_type,
            "source_label": source_label,
            "short_abstract": short_abstract,
        }
    
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

    def _get_pipeline_progress(self) -> Dict[str, Any]:
        """Get current pipeline progress."""
        global _pipeline_progress
        try:
            with _pipeline_progress_lock:
                progress = dict(_pipeline_progress) if isinstance(_pipeline_progress, dict) else {}

            if progress.get("stage"):
                return {"success": True, "progress": progress}

            return {
                "success": True,
                "progress": {
                    "stage": "idle",
                    "message": "No pipeline running",
                    "progress": 0,
                },
            }
        except Exception as e:
            print(f"[API] Error getting pipeline progress: {e}")
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
                
                _save_persisted_validator(
                    graph=validator.graph,
                    triples=validator.triples,
                    id_to_name=validator.id_to_name,
                    sentence_split=_sentence_split,
                )
                _invalidate_graph_cache()
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
                
                _save_persisted_validator(
                    graph=validator.graph,
                    triples=validator.triples,
                    id_to_name=validator.id_to_name,
                    sentence_split=_sentence_split,
                )
                _invalidate_graph_cache()
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

    def _handle_triple_update(self):
        """Handle head/tail replacement for a single triple."""
        if not validator:
            self._send_error("Validator not initialized")
            return

        try:
            content_length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(content_length).decode('utf-8'))

            triple_index = data.get("index")
            if triple_index is None or not isinstance(triple_index, int):
                self._send_error("Triple index is required", 400)
                return

            if triple_index < 0 or triple_index >= len(validator.triples):
                self._send_error("Invalid triple index", 400)
                return

            triple = validator.triples[triple_index]

            create_head = bool(data.get("create_head"))
            create_tail = bool(data.get("create_tail"))
            head_id = data.get("head_id")
            tail_id = data.get("tail_id")

            head_name = (data.get("head_name") or "").strip()
            head_label = (data.get("head_label") or "").strip() or "unknown_entity"
            tail_name = (data.get("tail_name") or "").strip()
            tail_label = (data.get("tail_label") or "").strip() or "unknown_entity"
            relation = (data.get("relation") or "").strip()

            def find_entity(entity_id: str):
                for item in validator.triples:
                    if get_triple_head_id(item) == entity_id:
                        return item.head
                    if get_triple_tail_id(item) == entity_id:
                        return item.tail
                return None

            def create_entity(name: str, label: str):
                from tools.sentence.entity import Entity
                entity_id = str(uuid.uuid4())
                return Entity(
                    id=entity_id,
                    ref=entity_id,
                    ref_short=entity_id[-4:],
                    name=name,
                    label=label,
                    start=0,
                    end=len(name),
                    sentence_id="manual",
                    entity_type=label,
                )

            updated = False

            if create_head:
                if not head_name:
                    self._send_error("Head name is required to create a new entity", 400)
                    return
                new_head = create_entity(head_name, head_label)
                triple.head = new_head
                validator.id_to_name[new_head.ref] = new_head.name
                updated = True
            elif head_id:
                existing_head = find_entity(head_id)
                if not existing_head:
                    self._send_error("Head entity not found", 404)
                    return
                triple.head = existing_head
                if head_id not in validator.id_to_name:
                    validator.id_to_name[head_id] = existing_head.name
                updated = True

            if create_tail:
                if not tail_name:
                    self._send_error("Tail name is required to create a new entity", 400)
                    return
                new_tail = create_entity(tail_name, tail_label)
                triple.tail = new_tail
                validator.id_to_name[new_tail.ref] = new_tail.name
                updated = True
            elif tail_id:
                existing_tail = find_entity(tail_id)
                if not existing_tail:
                    self._send_error("Tail entity not found", 404)
                    return
                triple.tail = existing_tail
                if tail_id not in validator.id_to_name:
                    validator.id_to_name[tail_id] = existing_tail.name
                updated = True

            if relation:
                triple.relation = relation
                updated = True

            if validator.graph is not None and updated:
                visualizer = GraphVisualizer()
                validator.graph = visualizer.build_graph(validator.triples)

            _save_persisted_validator(
                graph=validator.graph,
                triples=validator.triples,
                id_to_name=validator.id_to_name,
                sentence_split=_sentence_split,
            )
            _invalidate_graph_cache()

            self._send_json({
                "success": True,
                "message": "Triple updated successfully",
                "updated": updated,
                "index": triple_index,
            })
        except json.JSONDecodeError:
            self._send_error("Invalid JSON", 400)
        except Exception as e:
            import traceback
            print(f"[API] Error in _handle_triple_update: {e}")
            print(traceback.format_exc())
            self._send_error(f"Error: {str(e)}", 500)

    def _handle_triple_add(self):
        """Add a new triple. Payload: head_id or (create_head, head_name, head_label); tail_id or (create_tail, tail_name, tail_label); relation (required)."""
        if not validator:
            self._send_error("Validator not initialized")
            return

        try:
            content_length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(content_length).decode('utf-8'))

            create_head = bool(data.get("create_head"))
            create_tail = bool(data.get("create_tail"))
            head_id = data.get("head_id")
            tail_id = data.get("tail_id")

            head_name = (data.get("head_name") or "").strip()
            head_label = (data.get("head_label") or "").strip() or "unknown_entity"
            tail_name = (data.get("tail_name") or "").strip()
            tail_label = (data.get("tail_label") or "").strip() or "unknown_entity"
            relation = (data.get("relation") or "").strip()

            if not relation:
                self._send_error("Relation is required", 400)
                return

            def find_entity(entity_id: str):
                for item in validator.triples:
                    if get_triple_head_id(item) == entity_id:
                        return item.head
                    if get_triple_tail_id(item) == entity_id:
                        return item.tail
                return None

            def create_entity(name: str, label: str):
                from tools.sentence.entity import Entity
                entity_id = str(uuid.uuid4())
                return Entity(
                    id=entity_id,
                    ref=entity_id,
                    ref_short=entity_id[-4:],
                    name=name,
                    label=label,
                    start=0,
                    end=len(name),
                    sentence_id="manual",
                    entity_type=label,
                )

            if create_head:
                if not head_name:
                    self._send_error("Head name is required to create a new entity", 400)
                    return
                head = create_entity(head_name, head_label)
                validator.id_to_name[head.ref] = head.name
            elif head_id:
                head = find_entity(head_id)
                if not head:
                    self._send_error("Head entity not found", 404)
                    return
                if head_id not in validator.id_to_name:
                    validator.id_to_name[head_id] = head.name
            else:
                self._send_error("Provide head_id or create_head with head_name and head_label", 400)
                return

            if create_tail:
                if not tail_name:
                    self._send_error("Tail name is required to create a new entity", 400)
                    return
                tail = create_entity(tail_name, tail_label)
                validator.id_to_name[tail.ref] = tail.name
            elif tail_id:
                tail = find_entity(tail_id)
                if not tail:
                    self._send_error("Tail entity not found", 404)
                    return
                if tail_id not in validator.id_to_name:
                    validator.id_to_name[tail_id] = tail.name
            else:
                self._send_error("Provide tail_id or create_tail with tail_name and tail_label", 400)
                return

            new_triple = Triple(head=head, relation=relation, tail=tail)
            validator.triples.append(new_triple)

            if validator.graph is not None:
                visualizer = GraphVisualizer()
                validator.graph = visualizer.build_graph(validator.triples)

            _save_persisted_validator(
                graph=validator.graph,
                triples=validator.triples,
                id_to_name=validator.id_to_name,
                sentence_split=_sentence_split,
            )
            _invalidate_graph_cache()

            new_index = len(validator.triples) - 1
            self._send_json({
                "success": True,
                "message": "Triple added",
                "index": new_index,
            })
        except json.JSONDecodeError:
            self._send_error("Invalid JSON", 400)
        except Exception as e:
            import traceback
            print(f"[API] Error in _handle_triple_add: {e}")
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

            _save_persisted_validator(
                graph=validator.graph,
                triples=validator.triples,
                id_to_name=validator.id_to_name,
                sentence_split=_sentence_split,
            )
            _invalidate_graph_cache()

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

            _save_persisted_validator(
                graph=validator.graph,
                triples=validator.triples,
                id_to_name=validator.id_to_name,
                sentence_split=_sentence_split,
            )
            _invalidate_graph_cache()

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

            _save_persisted_validator(
                graph=validator.graph,
                triples=validator.triples,
                id_to_name=validator.id_to_name,
                sentence_split=_sentence_split,
            )
            _invalidate_graph_cache()

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
    use_neo4j: bool = True,
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
        use_neo4j: Whether the graph page should use the Neo4j visualization
        port: Port number for the API server
        open_browser: Whether to automatically open browser
        debug: Enable debug mode
    """
    global validator, _server_running, _api_server, _api_thread, _nextjs_process, _nextjs_thread
    global _source_patent_id, _source_filename, _source_type
    if graph is None and triples is None and id_to_name is None and sentence_split is None:
        persisted = _load_persisted_validator()
        if persisted:
            graph = persisted.get("graph")
            triples = persisted.get("triples")
            id_to_name = persisted.get("id_to_name")
            sentence_split = persisted.get("sentence_split")
            sm = persisted.get("source_metadata") or {}
            _source_patent_id = sm.get("patent_id")
            _source_filename = sm.get("filename")
            _source_type = sm.get("source")
            print("[Server] Loaded persisted validator state")
    api_host = (os.getenv("API_HOST") or "127.0.0.1").strip() or "127.0.0.1"
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
        _api_server = ThreadedHTTPServer((api_host, api_port), GraphValidatorHandler)
        print(f"✓ API Server running on http://{api_host}:{api_port}")
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
        graph_endpoint = "/api/graph/neo4j" if use_neo4j else "/api/graph/html"
        
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
            env = os.environ.copy()
            env["NEXT_PUBLIC_GRAPH_ENDPOINT"] = graph_endpoint
            _nextjs_process = subprocess.Popen(
                ["npm", "run", "dev", "--", "-p", str(nextjs_port)],
                cwd=str(nextjs_dir),
                shell=shell_flag,
                stdout=None,
                stderr=None,
                env=env,
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
