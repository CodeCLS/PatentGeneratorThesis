// Graph Validator Chat Interface
let currentQuestion = null;
let questionHistory = [];

// Initialize
let allTriples = [];

document.addEventListener('DOMContentLoaded', () => {
    checkStatus();
    startChat();
    updateGraphState();
    loadTriples();
    
    // Set up send button
    document.getElementById('sendButton').addEventListener('click', sendAnswer);
    document.getElementById('answerInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendAnswer();
        }
    });
    
    // Set up triple search
    const searchInput = document.getElementById('tripleSearch');
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            filterTriples(e.target.value);
        });
    }
    
    // Auto-refresh state and status every 5 seconds
    setInterval(() => {
        updateGraphState();
        checkStatus();
        loadTriples();
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
                displayWidget(data.widget_type, data.widget_data || {});
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

async function loadTriples() {
    try {
        const response = await fetch('/api/triples');
        const data = await response.json();
        
        if (data.error) {
            document.getElementById('triplesList').innerHTML = `<p style="color: #999;">${data.error}</p>`;
            return;
        }
        
        allTriples = data.triples || [];
        filterTriples(document.getElementById('tripleSearch').value || '');
    } catch (error) {
        console.error('Error loading triples:', error);
        document.getElementById('triplesList').innerHTML = '<p style="color: #999;">Error loading triples</p>';
    }
}

function filterTriples(searchTerm) {
    const term = searchTerm.toLowerCase().trim();
    const filtered = term 
        ? allTriples.filter(t => 
            t.head.name.toLowerCase().includes(term) ||
            t.tail.name.toLowerCase().includes(term) ||
            t.relation.toLowerCase().includes(term) ||
            (t.head.label && t.head.label.toLowerCase().includes(term)) ||
            (t.tail.label && t.tail.label.toLowerCase().includes(term))
        )
        : allTriples;
    
    displayTriples(filtered);
}

function displayTriples(triples) {
    const container = document.getElementById('triplesList');
    
    if (!triples || triples.length === 0) {
        container.innerHTML = '<p style="color: #999; font-style: italic;">No triples found</p>';
        return;
    }
    
    let html = '<div class="triples-scroll">';
    triples.forEach(triple => {
        html += `
            <div class="triple-widget">
                <div class="triple-index">#${triple.index}</div>
                <div class="triple-content">
                    <div class="triple-head">
                        <span class="entity-name">${escapeHtml(triple.head.name)}</span>
                        ${triple.head.label ? `<span class="entity-label">${escapeHtml(triple.head.label)}</span>` : ''}
                    </div>
                    <div class="triple-relation">${escapeHtml(triple.relation)}</div>
                    <div class="triple-tail">
                        <span class="entity-name">${escapeHtml(triple.tail.name)}</span>
                        ${triple.tail.label ? `<span class="entity-label">${escapeHtml(triple.tail.label)}</span>` : ''}
                    </div>
                </div>
            </div>
        `;
    });
    html += '</div>';
    
    container.innerHTML = html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function displayWidget(widgetType, widgetData) {
    const messagesEl = document.getElementById('messages');
    const widgetDiv = document.createElement('div');
    widgetDiv.className = 'widget-container';
    
    let widgetContent = '';
    
    switch(widgetType) {
        case 'edges_widget':
            const edges = widgetData.triples || [];
            const showCount = 5;
            const hasMore = edges.length > showCount;
            const listId = 'edges-list-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
            widgetContent = '<div class="widget edges-widget"><h4>Triples</h4><ul id="' + listId + '">';
            edges.slice(0, showCount).forEach((edge, idx) => {
                const index = edge.index !== undefined ? edge.index : idx;
                widgetContent += `<li>${index}. ${escapeHtml(edge.head || '')} --[${escapeHtml(edge.relation || '')}]--> ${escapeHtml(edge.tail || '')}</li>`;
            });
            widgetContent += '</ul>';
            if (hasMore) {
                const edgesJson = JSON.stringify(edges).replace(/"/g, '&quot;');
                widgetContent += `<button class="widget-button" onclick="showMoreEdges('${listId}', ${showCount}, '${edgesJson}')">Show More (${edges.length - showCount} remaining)</button>`;
            }
            widgetContent += '</div>';
            break;
            
        case 'graph_widget':
        case 'graph_subsection_widget':
            widgetContent = `<div class="widget graph-widget"><h4>Graph Visualization</h4><div class="graph-placeholder">Graph visualization would appear here</div></div>`;
            break;
            
        case 'question_widget_general':
            widgetContent = `<div class="widget question-widget"><h4>${escapeHtml(widgetData.question || 'Question')}</h4><textarea class="widget-input" placeholder="Your answer..."></textarea><button class="widget-button" onclick="submitWidgetAnswer('${widgetType}')">Submit</button></div>`;
            break;
            
        case 'question_widget_triple':
            const triple = widgetData.triple || {};
            widgetContent = `<div class="widget question-widget"><h4>Confirm or correct this triple:</h4><p>${escapeHtml(triple.head || '')} --[${escapeHtml(triple.relation || '')}]--> ${escapeHtml(triple.tail || '')}</p><textarea class="widget-input" placeholder="Corrections or confirm..."></textarea><button class="widget-button" onclick="submitWidgetAnswer('${widgetType}')">Submit</button></div>`;
            break;
            
        case 'question_widget_entity':
            widgetContent = `<div class="widget question-widget"><h4>Validate or explain: ${escapeHtml(widgetData.entity_name || 'Entity')}</h4><textarea class="widget-input" placeholder="Your explanation..."></textarea><button class="widget-button" onclick="submitWidgetAnswer('${widgetType}')">Submit</button></div>`;
            break;
            
        case 'question_widget_cluster_triple':
            widgetContent = `<div class="widget question-widget"><h4>Rate importance in cluster</h4><p>${escapeHtml(widgetData.triple?.head || '')} --[${escapeHtml(widgetData.triple?.relation || '')}]--> ${escapeHtml(widgetData.triple?.tail || '')}</p><input type="range" min="1" max="5" value="3" class="widget-slider"><button class="widget-button" onclick="submitWidgetAnswer('${widgetType}')">Submit</button></div>`;
            break;
            
        case 'validation_summary_widget':
            const stats = widgetData.stats || {};
            widgetContent = `<div class="widget summary-widget"><h4>Validation Summary</h4><p>Success Rate: ${stats.success_rate || 'N/A'}%</p><p>Total Validated: ${stats.total || 0}</p><p>Passed: ${stats.passed || 0}</p><p>Failed: ${stats.failed || 0}</p></div>`;
            break;
            
        case 'patent_analysis_widget':
            const patent = widgetData.patent || {};
            widgetContent = `<div class="widget patent-widget"><h4>Patent Analysis</h4><p>Status: ${escapeHtml(patent.status || 'N/A')}</p><p>Risk: ${escapeHtml(patent.risk || 'N/A')}</p><p>Key Metadata: ${escapeHtml(patent.metadata || 'N/A')}</p></div>`;
            break;
            
        case 'connection_check_widget':
            const issues = widgetData.issues || [];
            widgetContent = '<div class="widget connection-widget"><h4>Connection Check</h4><ul>';
            issues.forEach(issue => {
                widgetContent += `<li class="${issue.severity || 'info'}">${escapeHtml(issue.message || '')}</li>`;
            });
            widgetContent += '</ul></div>';
            break;
            
        case 'suggestion_widget':
            const suggestions = widgetData.suggestions || [];
            widgetContent = '<div class="widget suggestion-widget"><h4>Suggestions</h4><ul>';
            suggestions.forEach(suggestion => {
                widgetContent += `<li>${escapeHtml(suggestion.text || '')} <button class="widget-button-small" onclick="acceptSuggestion('${suggestion.id || ''}')">Accept</button> <button class="widget-button-small" onclick="dismissSuggestion('${suggestion.id || ''}')">Dismiss</button></li>`;
            });
            widgetContent += '</ul></div>';
            break;
            
        default:
            widgetContent = `<div class="widget"><p>Widget: ${escapeHtml(widgetType)}</p></div>`;
    }
    
    widgetDiv.innerHTML = widgetContent;
    messagesEl.appendChild(widgetDiv);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function submitWidgetAnswer(widgetType) {
    const widgetContainer = event.target.closest('.widget-container');
    const input = widgetContainer.querySelector('.widget-input') || widgetContainer.querySelector('.widget-slider');
    const answer = input ? input.value : '';
    if (answer) {
        document.getElementById('answerInput').value = answer;
        sendAnswer();
    }
}

function acceptSuggestion(suggestionId) {
    document.getElementById('answerInput').value = `Accept suggestion ${suggestionId}`;
    sendAnswer();
}

function dismissSuggestion(suggestionId) {
    document.getElementById('answerInput').value = `Dismiss suggestion ${suggestionId}`;
    sendAnswer();
}

function showMoreEdges(listId, currentCount, allEdgesJson) {
    const listEl = document.getElementById(listId);
    const buttonEl = event.target;
    const allEdges = JSON.parse(allEdgesJson.replace(/&quot;/g, '"'));
    const remaining = allEdges.slice(currentCount);
    
    remaining.forEach((edge, idx) => {
        const index = edge.index !== undefined ? edge.index : (currentCount + idx);
        const li = document.createElement('li');
        li.textContent = `${index}. ${edge.head || ''} --[${edge.relation || ''}]--> ${edge.tail || ''}`;
        listEl.appendChild(li);
    });
    
    buttonEl.remove();
}

function showMoreEdges(listId, currentCount, allEdgesJson) {
    const listEl = document.getElementById(listId);
    const buttonEl = event.target;
    const allEdges = JSON.parse(allEdgesJson.replace(/&quot;/g, '"'));
    const remaining = allEdges.slice(currentCount);
    
    remaining.forEach((edge, idx) => {
        const index = edge.index !== undefined ? edge.index : (currentCount + idx);
        const li = document.createElement('li');
        li.textContent = `${index}. ${edge.head || ''} --[${edge.relation || ''}]--> ${edge.tail || ''}`;
        listEl.appendChild(li);
    });
    
    buttonEl.remove();
}

