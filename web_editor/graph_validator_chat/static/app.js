// Graph Validator Chat Interface
let currentQuestion = null;
let questionHistory = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkStatus();
    loadFirstQuestion();
    updateGraphState();
    
    // Set up send button
    document.getElementById('sendButton').addEventListener('click', sendAnswer);
    document.getElementById('answerInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendAnswer();
        }
    });
    
    // Auto-refresh state and status every 5 seconds
    setInterval(() => {
        updateGraphState();
        checkStatus();
    }, 5000);
});

async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        const statusEl = document.getElementById('status');
        if (data.initialized) {
            const unanswered = data.num_unanswered !== undefined ? data.num_unanswered : data.num_questions;
            statusEl.textContent = `Ready | ${unanswered} question${unanswered !== 1 ? 's' : ''} remaining | ${data.num_triples} triples`;
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
        
        // Update status when loading questions
        checkStatus();
        
        if (data.all_completed) {
            addMessage('bot', '✅ All questions have been answered! Graph validation is complete.');
            disableInput();
            return;
        }
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
    questionEl.innerHTML = `
        <p><strong>${question.category.toUpperCase()}</strong> (Priority: ${question.priority})</p>
        <p>${question.text}</p>
        ${question.show_widget ? `<div class="widget-indicator">Widget: ${question.widget_type || 'default'}</div>` : ''}
    `;
    
    addMessage('bot', question.text);
}

async function sendAnswer() {
    const input = document.getElementById('answerInput');
    const answer = input.value.trim();
    
    if (!answer || !currentQuestion) {
        return;
    }
    
    // Disable input while processing
    disableInput();
    
    // Show user message
    addMessage('user', answer);
    input.value = '';
    
    try {
        const response = await fetch(`/api/questions/${currentQuestion.id}/answer`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ answer: answer }),
        });
        
        const data = await response.json();
        
        if (data.error) {
            addMessage('bot', `Error: ${data.error}`);
        } else {
            // Show bot response
            addMessage('bot', data.text);
            
            // Show widget if needed
            if (data.show_widget) {
                addMessage('bot', `[Widget: ${data.widget_type}]`);
            }
            
            // Check if question is completed
            if (data.question_completed) {
                addMessage('bot', '✓ Question completed. Moving to next question...');
            }
            
            // Log hidden actions (for debugging)
            if (data.hidden_actions && data.hidden_actions.length > 0) {
                console.log('Hidden actions applied:', data.hidden_actions);
            }
            
            // Update graph state and status
            updateGraphState();
            checkStatus();
            
            // Load next question only if current question is completed
            if (data.question_completed) {
                setTimeout(() => {
                    loadFirstQuestion();
                    checkStatus(); // Update status after loading next question
                }, 1500);
            } else {
                // Question not complete, re-enable input for follow-up
                // Only show the annoying message if no actions were taken
                if (!data.hidden_actions || data.hidden_actions.length === 0) {
                    enableInput();
                    addMessage('bot', 'Please provide more information or clarify your answer.');
                } else {
                    // Actions were taken but question not marked complete - enable input silently
                    enableInput();
                }
            }
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
    messageDiv.innerHTML = `
        <div class="message-content">
            <strong>${sender === 'bot' ? 'Bot' : 'You'}:</strong> ${text}
        </div>
    `;
    messagesEl.appendChild(messageDiv);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function updateGraphState() {
    try {
        const response = await fetch('/api/state');
        const data = await response.json();
        
        if (data.error) {
            return;
        }
        
        const stateEl = document.getElementById('graphState');
        let html = '';
        
        if (data.graph) {
            html += `<p><strong>Graph:</strong> ${data.graph.num_nodes} nodes, ${data.graph.num_edges} edges</p>`;
        }
        html += `<p><strong>Triples:</strong> ${data.num_triples}</p>`;
        html += `<p><strong>Entities:</strong> ${data.num_entities}</p>`;
        
        if (data.changes && Object.keys(data.changes).length > 0) {
            html += `<p><strong>Changes:</strong></p>`;
            html += `<ul style="margin-left: 20px; font-size: 12px;">`;
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
}

