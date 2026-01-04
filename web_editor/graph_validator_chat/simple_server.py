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


# Global validator instance
validator: Optional[GraphValidator] = None


def initialize_validator(
    graph: Optional[nx.MultiDiGraph] = None,
    triples: Optional[List[Triple]] = None,
    id_to_name: Optional[Dict[str, str]] = None,
) -> Optional[GraphValidator]:
    """Initialize the validator with graph and/or triples."""
    global validator
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
        
        unanswered = validator.getUnansweredQuestions()
        return {
            "initialized": True,
            "num_questions": len(validator.questions),
            "num_unanswered": len(unanswered),
            "num_responses": len(validator.responses),
            "has_graph": validator.graph is not None,
            "num_triples": len(validator.triples),
        }
    
    def get_first_question(self) -> Dict[str, Any]:
        """Get the first unanswered question."""
        if not validator:
            return {"error": "Validator not initialized"}
        
        question = validator.getFirstQuestion()
        if not question:
            # Check if all questions are answered
            unanswered = validator.getUnansweredQuestions()
            if not unanswered:
                return {
                    "question": None,
                    "message": "All questions have been answered!",
                    "all_completed": True,
                }
            return {"question": None}
        
        return {
            "question": {
                "id": question.id,
                "text": question.text,
                "category": question.category,
                "priority": question.priority,
                "show_widget": question.show_widget,
                "widget_type": question.widget_type,
                "widget_parameters": question.widget_parameters,
                "answered": question.answered,
                "num_responses": question.num_responses,
            }
        }
    
    def get_chat_start(self) -> Dict[str, Any]:
        """Get the initial chat message."""
        if not validator:
            return {"error": "Validator not initialized"}
        
        # Get the first message from global conversation history, or generate one
        if validator.global_conversation_history:
            first_msg = validator.global_conversation_history[0]
            initial_text = first_msg["content"]
            
            # Ensure we have a question in the initial message
            next_q = None
            if validator.questions and len(validator.questions) > 0:
                first_question = validator.questions[0]
                # Check if question is already in the message
                if first_question.text.lower() not in initial_text.lower():
                    initial_text += f"\n\n{first_question.text}"
                next_q = first_question.text
            else:
                # No questions yet - try to generate them if we have graph/triples
                if validator.graph or validator.triples:
                    try:
                        context = validator._build_context()
                        validator.questions = validator._generate_questions(context)
                        if validator.questions and len(validator.questions) > 0:
                            first_question = validator.questions[0]
                            if first_question.text.lower() not in initial_text.lower():
                                initial_text += f"\n\n{first_question.text}"
                            next_q = first_question.text
                    except Exception as e:
                        print(f"Error generating questions in get_chat_start: {e}")
            
            return {
                "text": initial_text,
                "next_question": next_q,
                "validation_complete": False,
            }
        else:
            # Generate initial question if available
            initial_text = "I'm ready to help you validate and improve your knowledge graph."
            next_q = None
            
            if validator.questions and len(validator.questions) > 0:
                next_q = validator.questions[0].text
                initial_text += f"\n\nLet me start by asking: {next_q}"
            
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
        
        response_data = {
            "question_id": response.question_id,
            "text": response.text,
            "show_widget": response.show_widget,
            "widget_type": response.widget_type,
            "question_completed": getattr(response, "question_completed", False),  # Safe access with default
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
    server = HTTPServer(('127.0.0.1', port), GraphValidatorHandler)
    print(f"✓ Graph Validator Chat starting on http://localhost:{port}")
    
    if open_browser:
        def open_browser_delayed():
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{port}")
        threading.Thread(target=open_browser_delayed, daemon=True).start()
    
    print(f"✓ Server running. Close the browser window when done.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n✓ Server stopped.")


def start_validator_chat(
    graph: Optional[nx.MultiDiGraph] = None,
    triples: Optional[List[Triple]] = None,
    id_to_name: Optional[Dict[str, str]] = None,
    port: int = 5001,
    open_browser: bool = True,
) -> None:
    """Start the graph validator chat interface using simple HTTP server."""
    # Initialize validator
    validator = initialize_validator(graph, triples, id_to_name)
    
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

