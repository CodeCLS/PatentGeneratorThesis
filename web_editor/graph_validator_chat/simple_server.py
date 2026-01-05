"""
Simple HTTP server for Graph Validator Chat Interface.
Uses Python's built-in http.server - NO Flask, NO Jinja2 dependencies!
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import json
import threading
import webbrowser
import time
from typing import Optional, Dict, Any, List
import networkx as nx

from tools.graph.Triple import Triple
from tools.graph.graph_validator import GraphValidator

# Try to import LangGraph adapter (optional)
try:
    from tools.graph.graph_validator_langgraph_adapter import GraphValidatorLangGraphAdapter
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    GraphValidatorLangGraphAdapter = None


# Global validator instance
validator: Optional[GraphValidator] = None


def initialize_validator(
    graph: Optional[nx.MultiDiGraph] = None,
    triples: Optional[List[Triple]] = None,
    id_to_name: Optional[Dict[str, str]] = None,
    use_langgraph: bool = True,  # Use LangGraph by default if available
) -> Optional[GraphValidator]:
    """
    Initialize the validator with graph and/or triples.
    
    Args:
        graph: Optional NetworkX graph
        triples: Optional list of Triple objects
        id_to_name: Optional mapping from entity ID to display name
        use_langgraph: Whether to use LangGraph-based validator (default: True if available)
    
    Returns:
        GraphValidator instance (or GraphValidatorLangGraphAdapter if use_langgraph=True)
    """
    global validator
    
    # Use LangGraph adapter if requested and available
    if use_langgraph and LANGGRAPH_AVAILABLE and GraphValidatorLangGraphAdapter:
        validator = GraphValidatorLangGraphAdapter(
            graph=graph,
            triples=triples,
            id_to_name=id_to_name,
        )
        validator.analyze(graph=graph, triples=triples, id_to_name=id_to_name)
    else:
        # Fallback to original validator
        validator = GraphValidator()
        validator.analyze(graph=graph, triples=triples, id_to_name=id_to_name)
    
    return validator


def get_validator() -> Optional[GraphValidator]:
    """Get the current validator instance."""
    return validator


class GraphValidatorHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the graph validator chat interface."""
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/' or path == '/index.html':
            self.serve_html()
        elif path == '/api/status':
            self.serve_json(self.get_status())
        elif path == '/api/questions/first':
            self.serve_json(self.get_first_question())
        elif path == '/api/chat/start':
            self.serve_json(self.get_chat_start())
        elif path == '/api/state':
            self.serve_json(self.get_state())
        elif path == '/api/export':
            self.serve_json(self.export_data())
        elif path.startswith('/static/'):
            self.serve_static(path)
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path.startswith('/api/questions/') and path.endswith('/answer'):
            question_id = path.split('/')[-2]
            self.handle_answer(question_id)
        elif path == '/api/chat':
            self.handle_chat()
        else:
            self.send_error(404)
    
    def serve_html(self):
        """Serve the main HTML page."""
        html_content = self.get_html_content()
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))
    
    def serve_json(self, data: Dict[str, Any]):
        """Serve JSON response."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def serve_static(self, path: str):
        """Serve static files (CSS, JS)."""
        import os
        file_path = path[1:]  # Remove leading /
        
        # Map paths to actual files
        if path == '/static/style.css':
            content = self.get_css_content()
            content_type = 'text/css'
        elif path == '/static/app.js':
            content = self.get_js_content()
            content_type = 'application/javascript'
        else:
            self.send_error(404)
            return
        
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))
    
    def get_status(self) -> Dict[str, Any]:
        """Get current validator status."""
        if not validator:
            return {"initialized": False, "message": "Validator not initialized"}
        
        # Handle both original validator and LangGraph adapter
        num_questions = 0
        num_unanswered = 0
        num_responses = 0
        
        if hasattr(validator, 'questions'):
            num_questions = len(validator.questions)
        
        if hasattr(validator, 'getUnansweredQuestions'):
            try:
                unanswered = validator.getUnansweredQuestions()
                num_unanswered = len(unanswered) if unanswered else 0
            except:
                num_unanswered = num_questions  # Fallback
        else:
            num_unanswered = num_questions  # Fallback
        
        if hasattr(validator, 'responses'):
            num_responses = len(validator.responses)
        
        return {
            "initialized": True,
            "num_questions": num_questions,
            "num_unanswered": num_unanswered,
            "num_responses": num_responses,
            "has_graph": validator.graph is not None if hasattr(validator, 'graph') else False,
            "num_triples": len(validator.triples) if hasattr(validator, 'triples') else 0,
        }
    
    def get_first_question(self) -> Dict[str, Any]:
        """Get the first unanswered question."""
        if not validator:
            return {"error": "Validator not initialized"}
        
        try:
            question = validator.getFirstQuestion()
        except Exception as e:
            print(f"Error calling getFirstQuestion: {e}")
            import traceback
            traceback.print_exc()
            return {"question": None}
        
        if not question:
            # Check if all questions are answered
            if hasattr(validator, 'getUnansweredQuestions'):
                try:
                    unanswered = validator.getUnansweredQuestions()
                    if not unanswered:
                        return {
                            "question": None,
                            "message": "All questions have been answered!",
                            "all_completed": True,
                        }
                except:
                    pass
            return {"question": None}
        
        # Handle both Question object and dict
        # ALWAYS check isinstance(dict) first - this is the most reliable
        if isinstance(question, dict):
            # Dict format - use .get() for all fields
            return {
                "question": {
                    "id": question.get("id", ""),
                    "text": question.get("text", ""),
                    "category": question.get("category", "unclear"),
                    "priority": question.get("priority", 5),
                    "show_widget": question.get("show_widget", False),
                    "widget_type": question.get("widget_type", None),
                    "widget_parameters": question.get("widget_parameters", {}),
                    "answered": question.get("answered", False),
                    "num_responses": question.get("num_responses", 0),
                }
            }
        else:
            # Question object - use getattr for safety
            try:
                return {
                    "question": {
                        "id": getattr(question, 'id', ''),
                        "text": getattr(question, 'text', ''),
                        "category": getattr(question, 'category', 'unclear'),
                        "priority": getattr(question, 'priority', 5),
                        "show_widget": getattr(question, 'show_widget', False),
                        "widget_type": getattr(question, 'widget_type', None),
                        "widget_parameters": getattr(question, 'widget_parameters', {}),
                        "answered": getattr(question, 'answered', False),
                        "num_responses": getattr(question, 'num_responses', 0),
                    }
                }
            except Exception as e:
                print(f"Error accessing question attributes: {e}")
                import traceback
                traceback.print_exc()
                return {"question": None}
    
    def get_chat_start(self) -> Dict[str, Any]:
        """Get the initial chat message."""
        if not validator:
            return {"error": "Validator not initialized"}
        
        # For LangGraph adapter, trigger initial question generation via chat
        if hasattr(validator, 'validator') and hasattr(validator.validator, 'chat'):
            # This is the LangGraph adapter - trigger initial state
            try:
                # Call chat with empty message to trigger initial question generation
                result = validator.chat("", generate_next_question=True)
                initial_text = result.get("text", "")
                next_q = result.get("next_question")
                
                # If we got a question, use it as the initial message
                if next_q:
                    initial_text = f"Let me start by asking: {next_q}"
                elif not initial_text or "ready to chat" in initial_text.lower():
                    # Fallback: generate question directly
                    if hasattr(validator, 'questions') and validator.questions and len(validator.questions) > 0:
                        first_q = validator.questions[0]
                        if isinstance(first_q, dict):
                            next_q = first_q.get("text", "")
                        else:
                            next_q = getattr(first_q, "text", "")
                        if next_q:
                            initial_text = f"Let me start by asking: {next_q}"
                
                return {
                    "text": initial_text or "I'm analyzing your graph and will ask you questions shortly.",
                    "next_question": next_q,
                    "validation_complete": False,
                }
            except Exception as e:
                print(f"Error in LangGraph initial chat: {e}")
                import traceback
                traceback.print_exc()
                # Fall through to original validator logic
        
        # Fallback for original validator
        # Get the first message from global conversation history, or generate one
        if hasattr(validator, 'global_conversation_history') and validator.global_conversation_history:
            first_msg = validator.global_conversation_history[0]
            initial_text = first_msg.get("content", "") if isinstance(first_msg, dict) else str(first_msg)
            
            # Ensure we have a question in the initial message
            next_q = None
            if hasattr(validator, 'questions') and validator.questions and len(validator.questions) > 0:
                first_question = validator.questions[0]
                # Handle both Question object and dict - check dict first
                if isinstance(first_question, dict):
                    question_text = first_question.get("text", "")
                else:
                    # Use getattr to safely access attribute (works for both objects and won't fail on dicts)
                    question_text = getattr(first_question, 'text', str(first_question))
                
                if question_text and question_text.lower() not in initial_text.lower():
                    initial_text = f"Let me start by asking: {question_text}"
                next_q = question_text
            
            return {
                "text": initial_text,
                "next_question": next_q,
                "validation_complete": False,
            }
        else:
            # Generate initial question if available
            initial_text = "I'm ready to help you validate and improve your knowledge graph."
            next_q = None
            
            if hasattr(validator, 'questions') and validator.questions and len(validator.questions) > 0:
                first_q = validator.questions[0]
                if isinstance(first_q, dict):
                    next_q = first_q.get("text", "")
                else:
                    next_q = getattr(first_q, "text", "")
                if next_q:
                    initial_text = f"Let me start by asking: {next_q}"
            
            return {
                "text": initial_text,
                "next_question": next_q,
                "validation_complete": False,
            }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current validator state."""
        if not validator:
            return {"error": "Validator not initialized"}
        
        updated_graph = validator.getUpdatedGraph()
        updated_triples = validator.getUpdatedTriples()
        changes = validator.getChanges()
        
        graph_info = None
        if updated_graph:
            graph_info = {
                "num_nodes": updated_graph.number_of_nodes(),
                "num_edges": updated_graph.number_of_edges(),
            }
        
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
        
        return {
            "graph": graph_info,
            "num_triples": len(updated_triples),
            "num_entities": len(entities),
            "entities": entities[:100],
            "id_to_name": validator.id_to_name,
            "changes": changes,
        }
    
    def handle_chat(self):
        """Handle flexible chat mode."""
        try:
            if not validator:
                self.serve_json({"error": "Validator not initialized"})
                return
            
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.serve_json({"error": "No message provided"})
                return
            
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            user_message = data.get("message", "")
            generate_next_question = data.get("generate_next_question", True)
            
            if not user_message:
                self.serve_json({"error": "Message text is required"})
                return
            
            chat_response = validator.chat(user_message, generate_next_question=generate_next_question)
            
            # Include changes_summary and stats in response
            response_data = {
                **chat_response,
                "changes_summary": chat_response.get("changes_summary", []),
                "stats": chat_response.get("stats", {}),
            }
            
            self.serve_json(response_data)
        except json.JSONDecodeError as e:
            self.serve_json({"error": f"Invalid JSON: {str(e)}"})
        except Exception as e:
            import traceback
            error_msg = f"Error in chat handler: {str(e)}"
            print(f"ERROR in handle_chat: {error_msg}")
            print(traceback.format_exc())
            self.serve_json({"error": error_msg})
    
    def handle_answer(self, question_id: str):
        """Handle answering a question."""
        if not validator:
            self.serve_json({"error": "Validator not initialized"})
            return
        
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        answer_text = data.get("answer", "")
        
        if not answer_text:
            self.serve_json({"error": "Answer text is required"})
            return
        
        response = validator.answerQuestion(question_id, answer_text)
        
        # Handle both Response object and dict (for adapter compatibility)
        if isinstance(response, dict):
            response_data = {
                "question_id": response.get("question_id", question_id),
                "text": response.get("text", ""),
                "show_widget": response.get("show_widget", False),
                "widget_type": response.get("widget_type", None),
                "question_completed": response.get("question_completed", False),
                "actions": response.get("actions", []),
                "hidden_actions": response.get("hidden_actions", []),
            }
        else:
            # Response object - use getattr for safety
            actions = getattr(response, "actions", [])
            response_data = {
                "question_id": getattr(response, "question_id", question_id),
                "text": getattr(response, "text", ""),
                "show_widget": getattr(response, "show_widget", False),
                "widget_type": getattr(response, "widget_type", None),
                "question_completed": getattr(response, "question_completed", False),
                "actions": [
                    {
                        "type": getattr(action, "type", type(action)).value if hasattr(getattr(action, "type", None), "value") else str(getattr(action, "type", "")),
                        "parameters": getattr(action, "parameters", {}),
                        "description": getattr(action, "description", ""),
                    }
                    for action in actions
                ] if actions else [],
                "hidden_actions": [
                    {
                        "type": getattr(action, "type", type(action)).value if hasattr(getattr(action, "type", None), "value") else str(getattr(action, "type", "")),
                        "parameters": getattr(action, "parameters", {}),
                        "description": getattr(action, "description", ""),
                    }
                    for action in getattr(response, "hidden_actions", [])
                ],
            }
        
        self.serve_json(response_data)
    
    def export_data(self) -> Dict[str, Any]:
        """Export full graph, triples, and entities."""
        if not validator:
            return {"error": "Validator not initialized"}
        
        import pickle
        import base64
        
        updated_graph = validator.getUpdatedGraph()
        updated_triples = validator.getUpdatedTriples()
        
        graph_data = None
        if updated_graph:
            graph_bytes = pickle.dumps(updated_graph)
            graph_data = base64.b64encode(graph_bytes).decode('utf-8')
        
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
        
        return {
            "graph": graph_data,
            "triples": triples_data,
            "entities": entities,
            "id_to_name": validator.id_to_name,
            "changes": validator.getChanges(),
        }
    
    def get_html_content(self) -> str:
        """Get the HTML content for the chat interface."""
        return """<!DOCTYPE html>
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
                <input type="text" id="answerInput" placeholder="Type your answer here..." disabled>
                <button id="sendButton" disabled>Send</button>
            </div>
        </div>
        
        <div class="sidebar">
            <h3>Current Question</h3>
            <div id="currentQuestion" class="question-display">
                <p>Loading question...</p>
            </div>
            
            <h3>Recent Changes</h3>
            <div id="changesDisplay" class="state-display">
                <p>No changes yet</p>
            </div>
            
            <h3>Graph Statistics</h3>
            <div id="graphStats" class="state-display">
                <p>Loading...</p>
            </div>
        </div>
    </div>
    
    <script src="/static/app.js"></script>
</body>
</html>"""
    
    def get_css_content(self) -> str:
        """Get the CSS content."""
        return """* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    background: #f5f5f5;
    color: #333;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    display: grid;
    grid-template-columns: 1fr 300px;
    gap: 20px;
    height: 100vh;
}

header {
    grid-column: 1 / -1;
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

header h1 {
    font-size: 24px;
    color: #2c3e50;
}

.status {
    padding: 8px 16px;
    background: #e8f5e9;
    color: #2e7d32;
    border-radius: 4px;
    font-size: 14px;
}

.chat-container {
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    display: flex;
    flex-direction: column;
    height: calc(100vh - 120px);
}

.messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
}

.message {
    margin-bottom: 16px;
    display: flex;
}

.message.bot {
    justify-content: flex-start;
}

.message.user {
    justify-content: flex-end;
}

.message-content {
    max-width: 70%;
    padding: 12px 16px;
    border-radius: 12px;
    line-height: 1.5;
}

.message.bot .message-content {
    background: #e3f2fd;
    color: #1565c0;
}

.message.user .message-content {
    background: #4caf50;
    color: white;
}

.input-area {
    display: flex;
    gap: 10px;
    padding: 20px;
    border-top: 1px solid #e0e0e0;
}

#answerInput {
    flex: 1;
    padding: 12px;
    border: 2px solid #e0e0e0;
    border-radius: 6px;
    font-size: 14px;
}

#answerInput:focus {
    outline: none;
    border-color: #4caf50;
}

#answerInput:disabled {
    background: #f5f5f5;
    cursor: not-allowed;
}

#sendButton {
    padding: 12px 24px;
    background: #4caf50;
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
}

#sendButton:hover:not(:disabled) {
    background: #45a049;
}

#sendButton:disabled {
    background: #ccc;
    cursor: not-allowed;
}

.sidebar > div {
    background: white;
    padding: 16px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.sidebar h3 {
    font-size: 16px;
    margin-bottom: 12px;
    color: #2c3e50;
}

.question-display, .state-display {
    font-size: 14px;
    line-height: 1.6;
}

.question-display p {
    margin-bottom: 8px;
}

.state-display p {
    margin: 4px 0;
    font-family: monospace;
    font-size: 12px;
}"""
    
    def get_js_content(self) -> str:
        """Get the JavaScript content."""
        # Read from the actual JS file
        import os
        js_path = os.path.join(os.path.dirname(__file__), 'static', 'app.js')
        if os.path.exists(js_path):
            with open(js_path, 'r', encoding='utf-8') as f:
                return f.read()
        # Fallback to inline JS
        return """// Graph Validator Chat Interface
let currentQuestion = null;

document.addEventListener('DOMContentLoaded', () => {
    checkStatus();
    loadFirstQuestion();
    updateGraphState();
    
    document.getElementById('sendButton').addEventListener('click', sendAnswer);
    document.getElementById('answerInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendAnswer();
        }
    });
    
    setInterval(updateGraphState, 5000);
});

async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        const statusEl = document.getElementById('status');
        if (data.initialized) {
            statusEl.textContent = `Ready | ${data.num_questions} questions | ${data.num_triples} triples`;
            statusEl.style.background = '#e8f5e9';
            statusEl.style.color = '#2e7d32';
        } else {
            statusEl.textContent = 'Not initialized';
            statusEl.style.background = '#ffebee';
            statusEl.style.color = '#c62828';
        }
    } catch (error) {
        console.error('Error checking status:', error);
    }
}

async function loadFirstQuestion() {
    try {
        const response = await fetch('/api/questions/first');
        const data = await response.json();
        if (data.question) {
            currentQuestion = data.question;
            displayQuestion(data.question);
            enableInput();
        } else {
            addMessage('bot', 'No more questions! The graph validation is complete.');
            disableInput();
        }
    } catch (error) {
        console.error('Error loading question:', error);
        addMessage('bot', 'Error loading question. Please refresh the page.');
    }
}

function displayQuestion(question) {
    const questionEl = document.getElementById('currentQuestion');
    questionEl.innerHTML = `<p><strong>${question.category.toUpperCase()}</strong> (Priority: ${question.priority})</p><p>${question.text}</p>`;
    addMessage('bot', question.text);
}

async function sendAnswer() {
    const input = document.getElementById('answerInput');
    const answer = input.value.trim();
    if (!answer || !currentQuestion) return;
    
    disableInput();
    addMessage('user', answer);
    input.value = '';
    
    try {
        const response = await fetch(`/api/questions/${currentQuestion.id}/answer`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ answer: answer }),
        });
        
        const data = await response.json();
        if (data.error) {
            addMessage('bot', `Error: ${data.error}`);
        } else {
            addMessage('bot', data.text);
            updateGraphState();
            setTimeout(() => { loadFirstQuestion(); }, 1000);
        }
    } catch (error) {
        console.error('Error sending answer:', error);
        addMessage('bot', 'Error processing answer. Please try again.');
        enableInput();
    }
}

function addMessage(sender, text) {
    const messagesEl = document.getElementById('messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    messageDiv.innerHTML = `<div class="message-content"><strong>${sender === 'bot' ? 'Bot' : 'You'}:</strong> ${text}</div>`;
    messagesEl.appendChild(messageDiv);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function updateGraphState() {
    try {
        const response = await fetch('/api/state');
        const data = await response.json();
        if (data.error) return;
        
        const stateEl = document.getElementById('graphState');
        let html = '';
        if (data.graph) {
            html += `<p><strong>Graph:</strong> ${data.graph.num_nodes} nodes, ${data.graph.num_edges} edges</p>`;
        }
        html += `<p><strong>Triples:</strong> ${data.num_triples}</p>`;
        html += `<p><strong>Entities:</strong> ${data.num_entities}</p>`;
        if (data.changes && Object.keys(data.changes).length > 0) {
            html += `<p><strong>Changes:</strong></p><ul style="margin-left: 20px; font-size: 12px;">`;
            for (const [key, value] of Object.entries(data.changes)) {
                html += `<li>${key}: ${value}</li>`;
            }
            html += `</ul>`;
        }
        stateEl.innerHTML = html;
    } catch (error) {
        console.error('Error updating state:', error);
    }
}

function enableInput() {
    document.getElementById('answerInput').disabled = false;
    document.getElementById('sendButton').disabled = false;
    document.getElementById('answerInput').focus();
}

function disableInput() {
    document.getElementById('answerInput').disabled = true;
    document.getElementById('sendButton').disabled = true;
}"""
    
    def log_message(self, format, *args):
        """Override to suppress server logs."""
        pass


def run_server(port: int = 5001, open_browser: bool = True):
    """Run the simple HTTP server."""
    import socket
    
    # Check if port is available
    def is_port_available(port: int) -> bool:
        """Check if a port is available."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return True
            except OSError:
                return False
    
    # Try to find an available port if the default is not available
    original_port = port
    if not is_port_available(port):
        print(f"⚠️  Port {port} is not available. Trying alternative ports...")
        for alt_port in range(5001, 5010):
            if is_port_available(alt_port):
                port = alt_port
                print(f"✓ Using port {port} instead")
                break
        else:
            # If no port found in range, try a few more
            for alt_port in [8000, 8001, 8080, 8888]:
                if is_port_available(alt_port):
                    port = alt_port
                    print(f"✓ Using port {port} instead")
                    break
            else:
                print(f"❌ ERROR: Could not find an available port.")
                print(f"   Port {original_port} and alternatives are all in use.")
                print(f"   Please close other applications using these ports or specify a different port.")
                return
    
    try:
        server = HTTPServer(('127.0.0.1', port), GraphValidatorHandler)
        print(f"✓ Graph Validator Chat starting on http://localhost:{port}")
        
        if open_browser:
            def open_browser_delayed():
                time.sleep(1.5)
                webbrowser.open(f"http://localhost:{port}")
            threading.Thread(target=open_browser_delayed, daemon=True).start()
        
        print(f"✓ Server running. Close the browser window when done.")
        server.serve_forever()
    except OSError as e:
        if "10013" in str(e) or "PermissionError" in str(type(e)):
            print(f"❌ ERROR: Permission denied for port {port}.")
            print(f"   This usually means:")
            print(f"   1. The port is already in use by another application")
            print(f"   2. Windows Firewall is blocking the port")
            print(f"   3. The port requires administrator privileges")
            print(f"\n   Try:")
            print(f"   - Closing other applications that might be using port {port}")
            print(f"   - Using a different port: start_validator_chat(..., port=8000)")
            print(f"   - Running as administrator (if needed)")
        else:
            print(f"❌ ERROR starting server: {e}")
    except KeyboardInterrupt:
        print("\n✓ Server stopped.")


def start_validator_chat(
    graph: Optional[nx.MultiDiGraph] = None,
    triples: Optional[List[Triple]] = None,
    id_to_name: Optional[Dict[str, str]] = None,
    port: int = 5001,
    open_browser: bool = True,
    use_langgraph: bool = True,  # Use LangGraph by default if available
) -> None:
    """Start the graph validator chat interface using simple HTTP server."""
    # Initialize validator
    validator = initialize_validator(graph, triples, id_to_name, use_langgraph=use_langgraph)
    
    if validator:
        print(f"✓ Initialized validator")
        if graph:
            print(f"✓ Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        if triples:
            print(f"✓ Triples: {len(triples)} triples")
    
    # Start server in a separate thread
    server_thread = threading.Thread(
        target=run_server,
        args=(port, open_browser),
        daemon=True
    )
    server_thread.start()

