// Graph Validator Chat Interface
let currentQuestion = null;
let questionHistory = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkStatus();
    startChat();
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

async function startChat() {
    try {
        const response = await fetch('/api/questions/first');
        const data = await response.json();
        
        if (data.error) {
            addMessage('bot', `Error: ${data.error}`);
        } else if (data.question) {
            currentQuestion = data.question;
            addMessage('bot', data.question.text);
            // Don't add message here - let communicator handle it
            // Just update the UI display
            document.getElementById('currentQuestion').innerHTML = `<p>${data.question.text}</p>`;
            enableInput();
        } else {
            addMessage('bot', 'No questions available. The graph validation is complete.');
            enableInput();
        }
    } catch (error) {
        console.error('Error starting chat:', error);
        addMessage('bot', 'Error loading questions. Please refresh the page.');
        enableInput();
    }
}

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
    
    if (!answer) {
        return;
    }
    
    // Disable input while processing
    disableInput();
    
    // Show user message
    addMessage('user', answer);
    input.value = '';
    
    try {
        // Use chat endpoint for flexible conversation
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: answer }),
        });
        
        let data;
        try {
            data = await response.json();
        } catch (e) {
            console.error('Failed to parse JSON response:', e);
            addMessage('bot', 'Error: Invalid response from server. Please try again.');
            enableInput();
            return;
        }
        
        if (data.error) {
            addMessage('bot', `Error: ${data.error}`);
            enableInput();
        } else {
            // Show bot response
            addMessage('bot', data.text);
            
            // Display changes made
            displayChanges(data.changes_summary || []);
            
            // Display stats
            displayStats(data.stats || {});
            
            // Show widget if needed
            if (data.show_widget) {
                addMessage('bot', `[Widget: ${data.widget_type}]`);
            }
            
            // Check if validation is complete
            if (data.validation_complete) {
                addMessage('bot', '🎉 Graph validation complete!');
                disableInput();
            } else {
                enableInput();
            }
            
            // Update graph state and status
            updateGraphState();
            checkStatus();
        }
    } catch (error) {
        console.error('Error sending answer:', error);
        let errorMsg = 'Error processing answer. Please try again.';
        if (error.message) {
            errorMsg += ` (${error.message})`;
        }
        addMessage('bot', errorMsg);
        enableInput();
    }
}

function displayChanges(changes) {
    const changesEl = document.getElementById('changesDisplay');
    
    if (!changes || changes.length === 0) {
        changesEl.innerHTML = '<p style="color: #999; font-style: italic;">No changes made</p>';
        return;
    }
    
    let html = '<ul style="margin: 0; padding-left: 20px; font-size: 13px;">';
    for (const change of changes.slice(0, 5)) { // Show max 5 changes
        html += `<li style="margin-bottom: 6px;">${change}</li>`;
    }
    if (changes.length > 5) {
        html += `<li style="color: #999; font-style: italic;">... and ${changes.length - 5} more</li>`;
    }
    html += '</ul>';
    
    changesEl.innerHTML = html;
}

function displayStats(stats) {
    const statsEl = document.getElementById('graphStats');
    
    if (!stats || Object.keys(stats).length === 0) {
        statsEl.innerHTML = '<p style="color: #999;">Loading stats...</p>';
        return;
    }
    
    let html = '<div style="font-size: 13px; line-height: 1.8;">';
    
    // Show only key stats
    if (stats.total_triples !== undefined) {
        html += `<p><strong>Total Triples:</strong> ${stats.total_triples}</p>`;
    }
    if (stats.total_entities !== undefined) {
        html += `<p><strong>Total Entities:</strong> ${stats.total_entities}</p>`;
    }
    if (stats.triples_changed !== undefined && stats.triples_changed !== 0) {
        const sign = stats.triples_changed > 0 ? '+' : '';
        html += `<p><strong>Triples Changed:</strong> <span style="color: ${stats.triples_changed > 0 ? '#2e7d32' : '#c62828'}">${sign}${stats.triples_changed}</span></p>`;
    }
    if (stats.entities_changed !== undefined && stats.entities_changed !== 0) {
        const sign = stats.entities_changed > 0 ? '+' : '';
        html += `<p><strong>Entities Changed:</strong> <span style="color: ${stats.entities_changed > 0 ? '#2e7d32' : '#c62828'}">${sign}${stats.entities_changed}</span></p>`;
    }
    
    html += '</div>';
    statsEl.innerHTML = html;
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
        
        // Update stats display
        const statsEl = document.getElementById('graphStats');
        let html = '<div style="font-size: 13px; line-height: 1.8;">';
        
        if (data.graph) {
            html += `<p><strong>Nodes:</strong> ${data.graph.num_nodes}</p>`;
            html += `<p><strong>Edges:</strong> ${data.graph.num_edges}</p>`;
        }
        html += `<p><strong>Triples:</strong> ${data.num_triples}</p>`;
        html += `<p><strong>Entities:</strong> ${data.num_entities}</p>`;
        html += '</div>';
        
        statsEl.innerHTML = html;
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

