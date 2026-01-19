// Global state for Triple Editor
let triples = [];
let entities = [];
let currentTriple = null;
let allLabels = new Set();
let searchQuery = '';
let labelFilter = '';

// Global state for Graph Validator Chat Interface
let currentQuestion = null;
let questionHistory = [];
let allTriples = []; // This is distinct from the 'triples' used by the editor

// Helper for HTML escaping (used by both)
function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// --- Triple Editor Functions (from web_editor/static/app.js) ---

async function loadEntities() {
    try {
        const response = await fetch('/api/entities');
        if (!response.ok) throw new Error('Failed to load entities');
        
        entities = await response.json();
        console.log(`Loaded ${entities.length} entities`);
    } catch (error) {
        console.error('Error loading entities:', error);
        showError('Failed to load entities: ' + error.message);
    }
}

async function loadTriples() {
    try {
        const response = await fetch('/api/triples');
        if (!response.ok) throw new Error('Failed to load triples');
        
        // IMPORTANT: Completely replace the triples array with fresh data from server
        // This ensures all entity labels are up-to-date
        const freshTriples = await response.json();
        triples = freshTriples; // Replace entire array
        
        // Collect all unique labels from the fresh data
        allLabels.clear();
        triples.forEach(t => {
            allLabels.add(t.head.label);
            allLabels.add(t.tail.label);
        });
        
        // Preserve current filter state
        const labelFilterSelect = document.getElementById('label-filter');
        const currentFilterValue = labelFilterSelect ? labelFilterSelect.value : '';
        
        // Populate label filter with updated labels
        if (labelFilterSelect) {
            labelFilterSelect.innerHTML = '<option value="">All Labels</option>';
            Array.from(allLabels).sort().forEach(label => {
                const option = document.createElement('option');
                option.value = label;
                option.textContent = label;
                // Restore previous filter selection if it still exists
                if (label === currentFilterValue) {
                    option.selected = true;
                }
                labelFilterSelect.appendChild(option);
            });
        }
        
        // Re-apply filters to show updated triple list
        applyFilters();
    } catch (error) {
        console.error('Error loading triples:', error);
        showError('Failed to load triples: ' + error.message);
    }
}

async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        if (!response.ok) throw new Error('Failed to load stats');
        
        const stats = await response.json();
        const tripleCountEl = document.getElementById('triple-count');
        const entityCountEl = document.getElementById('entity-count');

        if (tripleCountEl) tripleCountEl.textContent = stats.triple_count;
        if (entityCountEl) entityCountEl.textContent = stats.entity_count;
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

function applyFilters() {
    // Always use the current triples array (which should be fresh from server)
    let filtered = [...triples]; // Create a copy to avoid mutating the original
    
    // Apply label filter first (strict match)
    if (labelFilter) {
        filtered = filtered.filter(t => 
            t.head.label === labelFilter || t.tail.label === labelFilter
        );
    }
    
    // Then apply search query
    if (searchQuery) {
        filtered = filtered.filter(t => 
            t.head.name.toLowerCase().includes(searchQuery) ||
            t.tail.name.toLowerCase().includes(searchQuery) ||
            t.relation.toLowerCase().includes(searchQuery) ||
            t.head.label.toLowerCase().includes(searchQuery) ||
            t.tail.label.toLowerCase().includes(searchQuery)
        );
    }
    
    // Always re-render with the filtered list
    renderTripleList(filtered);
}

function renderTripleList(filteredTriples = null) {
    const list = document.getElementById('triple-list');
    if (!list) return; // Added null check

    const triplesToShow = filteredTriples || triples;
    
    if (triplesToShow.length === 0) {
        list.innerHTML = '<div class="loading">No triples found</div>';
        return;
    }
    
    list.innerHTML = triplesToShow.map(triple => {
        const isActive = currentTriple && currentTriple.id === triple.id;
        return `
            <div class="triple-item ${isActive ? 'active' : ''}" onclick="selectTriple('${triple.id}')">
                <div class="triple-preview">
                    <span class="head-name">${escapeHtml(triple.head.name)}</span>
                    <span class="relation">${escapeHtml(triple.relation)}</span>
                    <span class="tail-name">${escapeHtml(triple.tail.name)}</span>
                </div>
                <div class="labels">
                    <span>H: ${escapeHtml(triple.head.label)}</span>
                    <span>T: ${escapeHtml(triple.tail.label)}</span>
                </div>
            </div>
        `;
    }).join('');
}

async function selectTriple(tripleId) {
    try {
        const response = await fetch(`/api/triples/${tripleId}`);
        if (!response.ok) throw new Error('Failed to load triple');
        
        currentTriple = await response.json();
        await loadEntities(); // Reload entities in case they changed
        renderEditor();
        applyFilters(); // Re-apply filters to update active state
    } catch (error) {
        console.error('Error loading triple:', error);
        showError('Failed to load triple: ' + error.message);
    }
}

function renderEditor() {
    const panel = document.getElementById('editor-panel');
    if (!panel) return; // Added null check
    
    if (!currentTriple) {
        panel.innerHTML = '<div class="empty-state"><p>Select a triple to edit</p></div>';
        return;
    }
    
    // Ensure entities are loaded
    if (entities.length === 0) {
        loadEntities().then(() => {
            renderEditor(); // Retry after loading
        });
        return;
    }
    
    // Create entity options for dropdowns with data attributes for search
    const entityOptions = entities.map(e => 
        `<option value="${escapeHtml(e.id)}" data-name="${escapeHtml(e.name)}" data-label="${escapeHtml(e.label)}">
            ${escapeHtml(e.name)} (${escapeHtml(e.label)}) - ${escapeHtml(e.id.substring(0, 8))}...
        </option>`
    ).join('');
    
    // Create merge options (exclude current entity) with data attributes
    const mergeOptions = entities
        .filter(e => e.id !== currentTriple.head.id)
        .map(e => 
            `<option value="${escapeHtml(e.id)}" data-name="${escapeHtml(e.name)}" data-label="${escapeHtml(e.label)}">
                ${escapeHtml(e.name)} (${escapeHtml(e.label)}) - ${escapeHtml(e.id.substring(0, 8))}...
            </option>`
        ).join('');
    
    const tailMergeOptions = entities
        .filter(e => e.id !== currentTriple.tail.id)
        .map(e => 
            `<option value="${escapeHtml(e.id)}" data-name="${escapeHtml(e.name)}" data-label="${escapeHtml(e.label)}">
                ${escapeHtml(e.name)} (${escapeHtml(e.label)}) - ${escapeHtml(e.id.substring(0, 8))}...
            </option>`
        ).join('');
    
    panel.innerHTML = `
        <div class="editor-form active">
            <div id="message-area"></div>
            
            <div class="form-section">
                <h3>Head Entity</h3>
                <div class="form-group">
                    <label>Replace with Entity</label>
                    <div class="searchable-dropdown">
                        <input type="text" class="dropdown-search" id="head-entity-search" placeholder="Search entities..." oninput="filterDropdown('head-entity-select', 'head-entity-search')">
                        <select id="head-entity-select" onchange="handleHeadEntityReplace()">
                            <option value="">-- Keep Current Entity --</option>
                            ${entityOptions}
                        </select>
                    </div>
                </div>
                <div class="form-group">
                    <label>Name</label>
                    <input type="text" id="head-name" value="${escapeHtml(currentTriple.head.name)}">
                </div>
                <div class="form-group">
                    <label>Label</label>
                    <input type="text" id="head-label" value="${escapeHtml(currentTriple.head.label)}" list="label-suggestions">
                    <datalist id="label-suggestions">
                        ${Array.from(allLabels).map(l => `<option value="${escapeHtml(l)}">`).join('')}
                    </datalist>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label>Ref Short</label>
                        <input type="text" id="head-ref-short" value="${escapeHtml(currentTriple.head.ref_short || '')}">
                    </div>
                    <div class="form-group">
                        <label>Ref</label>
                        <input type="text" id="head-ref" value="${escapeHtml(currentTriple.head.ref || '')}">
                    </div>
                </div>
                <div class="form-group">
                    <label>Entity Type</label>
                    <input type="text" id="head-entity-type" value="${escapeHtml(currentTriple.head.entity_type || '')}">
                </div>
                <div class="form-group">
                    <label>ID (read-only)</label>
                    <input type="text" id="head-id" value="${escapeHtml(currentTriple.head.id)}" readonly>
                </div>
                <div class="form-group">
                    <label>Merge Head Entity Into</label>
                    <div class="searchable-dropdown">
                        <input type="text" class="dropdown-search" id="head-merge-search" placeholder="Search entities..." oninput="filterDropdown('head-merge-select', 'head-merge-search')">
                        <select id="head-merge-select">
                            <option value="">-- Select Target Entity --</option>
                            ${mergeOptions}
                        </select>
                    </div>
                    <button class="merge-button" onclick="mergeEntity('${escapeHtml(currentTriple.head.id)}', 'head-merge-select', 'head')">Merge Head Entity</button>
                </div>
                <button class="delete-button" onclick="deleteEntity('${escapeHtml(currentTriple.head.id)}', 'head')">Delete Head Entity</button>
            </div>
            
            <div class="form-section">
                <h3>Relation</h3>
                <div class="form-group">
                    <label>Relation</label>
                    <input type="text" id="relation" value="${escapeHtml(currentTriple.relation)}">
                </div>
                <button class="delete-relation-button" onclick="deleteTriple('${escapeHtml(currentTriple.id)}')">Delete This Triple (Keep Entities)</button>
            </div>
            
            <div class="form-section">
                <h3>Tail Entity</h3>
                <div class="form-group">
                    <label>Replace with Entity</label>
                    <div class="searchable-dropdown">
                        <input type="text" class="dropdown-search" id="tail-entity-search" placeholder="Search entities..." oninput="filterDropdown('tail-entity-select', 'tail-entity-search')">
                        <select id="tail-entity-select" onchange="handleTailEntityReplace()">
                            <option value="">-- Keep Current Entity --</option>
                            ${entityOptions}
                        </select>
                    </div>
                </div>
                <div class="form-group">
                    <label>Name</label>
                    <input type="text" id="tail-name" value="${escapeHtml(currentTriple.tail.name)}">
                </div>
                <div class="form-group">
                    <label>Label</label>
                    <input type="text" id="tail-label" value="${escapeHtml(currentTriple.tail.label)}" list="label-suggestions">
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label>Ref Short</label>
                        <input type="text" id="tail-ref-short" value="${escapeHtml(currentTriple.tail.ref_short || '')}">
                    </div>
                    <div class="form-group">
                        <label>Ref</label>
                        <input type="text" id="tail-ref" value="${escapeHtml(currentTriple.tail.ref || '')}">
                    </div>
                </div>
                <div class="form-group">
                    <label>Entity Type</label>
                    <input type="text" id="tail-entity-type" value="${escapeHtml(currentTriple.tail.entity_type || '')}">
                </div>
                <div class="form-group">
                    <label>ID (read-only)</label>
                    <input type="text" id="tail-id" value="${escapeHtml(currentTriple.tail.id)}" readonly>
                </div>
                <div class="form-group">
                    <label>Merge Tail Entity Into</label>
                    <div class="searchable-dropdown">
                        <input type="text" class="dropdown-search" id="tail-merge-search" placeholder="Search entities..." oninput="filterDropdown('tail-merge-select', 'tail-merge-search')">
                        <select id="tail-merge-select">
                            <option value="">-- Select Target Entity --</option>
                            ${tailMergeOptions}
                        </select>
                    </div>
                    <button class="merge-button" onclick="mergeEntity('${escapeHtml(currentTriple.tail.id)}', 'tail-merge-select', 'tail')">Merge Tail Entity</button>
                </div>
                <button class="delete-button" onclick="deleteEntity('${escapeHtml(currentTriple.tail.id)}', 'tail')">Delete Tail Entity</button>
            </div>
            
            <button class="save-button" onclick="saveTriple()">Save Changes</button>
        </div>
    `;
}

async function handleHeadEntityReplace() {
    const select = document.getElementById('head-entity-select');
    const selectedId = select.value;
    
    if (!selectedId) return;
    
    const selectedEntity = entities.find(e => e.id === selectedId);
    if (!selectedEntity) return;
    
    // Update form fields with selected entity
    document.getElementById('head-id').value = selectedEntity.id;
    document.getElementById('head-name').value = selectedEntity.name;
    document.getElementById('head-label').value = selectedEntity.label;
    document.getElementById('head-ref-short').value = selectedEntity.ref_short || '';
    document.getElementById('head-ref').value = selectedEntity.ref || '';
    document.getElementById('head-entity-type').value = selectedEntity.entity_type || '';
}

async function handleTailEntityReplace() {
    const select = document.getElementById('tail-entity-select');
    const selectedId = select.value;
    
    if (!selectedId) return;
    
    const selectedEntity = entities.find(e => e.id === selectedId);
    if (!selectedEntity) return;
    
    // Update form fields with selected entity
    document.getElementById('tail-id').value = selectedEntity.id;
    document.getElementById('tail-name').value = selectedEntity.name;
    document.getElementById('tail-label').value = selectedEntity.label;
    document.getElementById('tail-ref-short').value = selectedEntity.ref_short || '';
    document.getElementById('tail-ref').value = selectedEntity.ref || '';
    document.getElementById('tail-entity-type').value = selectedEntity.entity_type || '';
}

async function deleteEntity(entityId, position) {
    if (!confirm(`Are you sure you want to delete this entity? This will also delete all triples connected to it.`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/entities/${entityId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to delete entity');
        }
        
        const result = await response.json();
        
        // Reload everything
        await loadTriples();
        await loadEntities();
        await loadStats();
        
        // Clear current selection if the deleted entity was part of it
        if (currentTriple && (currentTriple.head.id === entityId || currentTriple.tail.id === entityId)) {
            currentTriple = null;
            renderEditor();
        }
        
        showSuccess(result.message || 'Entity deleted successfully');
    } catch (error) {
        console.error('Error deleting entity:', error);
        showError('Failed to delete entity: ' + error.message);
    }
}

async function saveTriple() {
    if (!currentTriple) return;
    
    try {
        // Check if entity replacement was selected
        const headReplaceSelect = document.getElementById('head-entity-select');
        const tailReplaceSelect = document.getElementById('tail-entity-select');
        
        const headData = {
            id: document.getElementById('head-id').value,
        };
        
        // If replacement was selected, use replace_with_id
        if (headReplaceSelect && headReplaceSelect.value) {
            headData.replace_with_id = headReplaceSelect.value;
        } else {
            // Otherwise, update properties
            headData.name = document.getElementById('head-name').value;
            headData.label = document.getElementById('head-label').value;
            headData.ref_short = document.getElementById('head-ref-short').value;
            headData.ref = document.getElementById('head-ref').value || null;
            headData.entity_type = document.getElementById('head-entity-type').value || null;
        }
        
        const tailData = {
            id: document.getElementById('tail-id').value,
        };
        
        // If replacement was selected, use replace_with_id
        if (tailReplaceSelect && tailReplaceSelect.value) {
            tailData.replace_with_id = tailReplaceSelect.value;
        } else {
            // Otherwise, update properties
            tailData.name = document.getElementById('tail-name').value;
            tailData.label = document.getElementById('tail-label').value;
            tailData.ref_short = document.getElementById('tail-ref-short').value;
            tailData.ref = document.getElementById('tail-ref').value || null;
            tailData.entity_type = document.getElementById('tail-entity-type').value || null;
        }
        
        const relation = document.getElementById('relation').value;
        
        // Update via API
        const response = await fetch(`/api/triples/${currentTriple.id}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                head: headData,
                tail: tailData,
                relation: relation,
            }),
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to update triple');
        }
        
        const updated = await response.json();
        
        // IMPORTANT: Reload ALL triples from server to get updated entity labels
        // This is critical because when an entity label changes, ALL triples
        // that reference that entity need to show the updated label
        await loadTriples(); // This replaces the entire triples array with fresh data
        
        // Also reload entities and stats
        await loadEntities();
        await loadStats();
        
        // Update current triple from the freshly loaded triples array
        const reloadedTriple = triples.find(t => t.id === currentTriple.id);
        if (reloadedTriple) {
            currentTriple = reloadedTriple;
        } else {
            // If triple was deleted or not found, use the updated response
            currentTriple = updated;
        }
        
        // Re-render everything - this will show updated labels in ALL triples
        renderEditor();
        // applyFilters() is called inside loadTriples(), so the list is already updated
        // But we call it again to ensure the current filter is maintained
        applyFilters();
        
        showSuccess('Triple updated successfully! All triples with the same entities have been updated.');
    } catch (error) {
        console.error('Error saving triple:', error);
        showError('Failed to save: ' + error.message);
    }
}

function showError(message) {
    const area = document.getElementById('message-area');
    if (area) {
        area.innerHTML = `<div class="error">${escapeHtml(message)}</div>`;
        setTimeout(() => {
            area.innerHTML = '';
        }, 5000);
    } else {
        alert('Error: ' + message);
    }
}

function showSuccess(message) {
    const area = document.getElementById('message-area');
    if (area) {
        area.innerHTML = `<div class="success">${escapeHtml(message)}</div>`;
        setTimeout(() => {
            area.innerHTML = '';
        }, 3000);
    }
}

async function deleteTriple(tripleId) {
    if (!confirm('Are you sure you want to delete this triple? The entities will be kept, but the relation will be removed.')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/triples/${tripleId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to delete triple');
        }
        
        const result = await response.json();
        
        // IMPORTANT: Reload everything to update the list
        await loadTriples(); // This will refresh the entire triples array and re-render the list
        await loadEntities();
        await loadStats();
        
        // Clear current selection since the triple is deleted
        currentTriple = null;
        renderEditor();
        // applyFilters() is already called in loadTriples(), but call it again to ensure
        applyFilters();
        
        showSuccess(result.message || 'Triple deleted successfully. Entities were preserved.');
    } catch (error) {
        console.error('Error deleting triple:', error);
        showError('Failed to delete triple: ' + error.message);
    }
}

async function mergeEntity(sourceId, selectId, position) {
    const select = document.getElementById(selectId);
    const targetId = select.value;
    
    if (!targetId) {
        showError('Please select a target entity to merge into');
        return;
    }
    
    if (sourceId === targetId) {
        showError('Cannot merge entity with itself');
        return;
    }
    
    if (!confirm(`Are you sure you want to merge this entity into the target? All triples pointing to this entity will be updated, and this entity will be deleted.`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/entities/${sourceId}/merge`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                target_id: targetId
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to merge entities');
        }
        
        const result = await response.json();
        
        // Reload everything
        await loadTriples();
        await loadEntities();
        await loadStats();
        
        // Clear current selection if the merged entity was part of it
        if (currentTriple && (currentTriple.head.id === sourceId || currentTriple.tail.id === sourceId)) {
            currentTriple = null;
            renderEditor();
        } else {
            // Reload current triple if it still exists
            if (currentTriple) {
                await selectTriple(currentTriple.id);
            }
        }
        
        showSuccess(result.message || 'Entities merged successfully');
    } catch (error) {
        console.error('Error merging entities:', error);
        showError('Failed to merge entities: ' + error.message);
    }
}

function filterDropdown(selectId, searchId) {
    const searchInput = document.getElementById(searchId);
    const select = document.getElementById(selectId);
    const filter = searchInput.value.toLowerCase();
    
    // Get all options
    const options = Array.from(select.options);
    
    // Show/hide options based on search
    options.forEach(option => {
        if (option.value === '') {
            // Always show the first option (placeholder)
            option.style.display = '';
            return;
        }
        
        const name = option.getAttribute('data-name') || '';
        const label = option.getAttribute('data-label') || '';
        const text = option.textContent || '';
        
        if (filter === '' || 
            name.toLowerCase().includes(filter) || 
            label.toLowerCase().includes(filter) ||
            text.toLowerCase().includes(filter)) {
            option.style.display = '';
        } else {
            option.style.display = 'none';
        }
    });
}

// --- Graph Validator Chat Interface Functions (from web_editor/graph_validator_chat/static/app.js) ---

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
        if (statusEl) { // Added null check
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
    if (!changesEl) return; // Added null check
    
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
    if (!statsEl) return; // Added null check
    
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
    if (!messagesEl) return; // Added null check

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
        if (!statsEl) return; // Added null check
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
    const answerInput = document.getElementById('answerInput');
    const sendButton = document.getElementById('sendButton');
    if (answerInput) {
        answerInput.disabled = false;
        answerInput.focus();
    }
    if (sendButton) sendButton.disabled = false;
}

function disableInput() {
    const answerInput = document.getElementById('answerInput');
    const sendButton = document.getElementById('sendButton');
    if (answerInput) answerInput.disabled = true;
    if (sendButton) sendButton.disabled = true;
}

async function loadChatTriples() { // Renamed to avoid conflict with loadTriples from editor
    try {
        const response = await fetch('/api/triples');
        const data = await response.json();
        
        if (data.error) {
            const triplesListEl = document.getElementById('triplesList');
            if (triplesListEl) triplesListEl.innerHTML = `<p style="color: #999;">${data.error}</p>`;
            return;
        }
        
        allTriples = data.triples || [];
        const tripleSearchEl = document.getElementById('tripleSearch');
        filterTriples(tripleSearchEl ? tripleSearchEl.value : '');
    } catch (error) {
        console.error('Error loading triples:', error);
        const triplesListEl = document.getElementById('triplesList');
        if (triplesListEl) triplesListEl.innerHTML = '<p style="color: #999;">Error loading triples</p>';
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
    if (!container) return; // Added null check
    
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

function displayWidget(widgetType, widgetData) {
    const messagesEl = document.getElementById('messages');
    if (!messagesEl) return; // Added null check

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


// --- Combined DOMContentLoaded listeners and global exports ---
document.addEventListener('DOMContentLoaded', async () => {
    // Editor initialization
    await loadEntities();
    await loadTriples();
    await loadStats();
    
    // Search functionality
    const searchInput = document.getElementById('search-input');
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            searchQuery = e.target.value.toLowerCase();
            applyFilters();
        });
    }

    const labelFilterSelect = document.getElementById('label-filter');
    if (labelFilterSelect) {
        labelFilterSelect.addEventListener('change', (e) => {
            labelFilter = e.target.value;
            applyFilters();
        });
    }

    // Chat interface initialization
    checkStatus();
    startChat();
    updateGraphState();
    loadChatTriples(); // Renamed to avoid conflict
    
    // Set up back button to navigate to widget showcase
    const backButton = document.getElementById('backButton');
    if (backButton) {
        backButton.disabled = false;
        backButton.removeAttribute('disabled');
        backButton.style.pointerEvents = 'auto';
        backButton.style.cursor = 'pointer';
        backButton.style.opacity = '1';
        
        backButton.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('Back button clicked!');
            window.location.href = '/widget-showcase';
        });
        
        backButton.addEventListener('mousedown', (e) => {
            e.stopPropagation();
        });
        
        let longPressTimer = null;
        const LONG_PRESS_DURATION = 1500;
        
        backButton.addEventListener('mousedown', (e) => {
            longPressTimer = setTimeout(() => {
                backButton.style.opacity = '0.5';
                backButton.style.transform = 'scale(0.95)';
                setTimeout(() => {
                    window.location.href = '/widget-showcase';
                }, 100);
            }, LONG_PRESS_DURATION);
        });
        
        backButton.addEventListener('mouseup', () => {
            if (longPressTimer) {
                clearTimeout(longPressTimer);
                longPressTimer = null;
            }
            backButton.style.opacity = '';
            backButton.style.transform = '';
        });
        
        backButton.addEventListener('mouseleave', () => {
            if (longPressTimer) {
                clearTimeout(longPressTimer);
                longPressTimer = null;
            }
            backButton.style.opacity = '';
            backButton.style.transform = '';
        });
        
        backButton.addEventListener('touchstart', (e) => {
            longPressTimer = setTimeout(() => {
                backButton.style.opacity = '0.5';
                backButton.style.transform = 'scale(0.95)';
                setTimeout(() => {
                    window.location.href = '/widget-showcase';
                }, 100);
            }, LONG_PRESS_DURATION);
        }, { passive: false });
        
        backButton.addEventListener('touchend', () => {
            if (longPressTimer) {
                clearTimeout(longPressTimer);
                longPressTimer = null;
            }
            backButton.style.opacity = '';
            backButton.style.transform = '';
        });
    }
    
    // Set up send button
    const sendButton = document.getElementById('sendButton');
    if (sendButton) { // Added null check
        sendButton.addEventListener('click', sendAnswer);
    }
    const answerInput = document.getElementById('answerInput');
    if (answerInput) { // Added null check
        answerInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendAnswer();
            }
        });
    }
    
    // Set up triple search
    const tripleSearchInput = document.getElementById('tripleSearch');
    if (tripleSearchInput) { // Added null check
        tripleSearchInput.addEventListener('input', (e) => {
            filterTriples(e.target.value);
        });
    }
    
    // Auto-refresh state and status every 5 seconds
    setInterval(() => {
        updateGraphState();
        checkStatus();
        loadChatTriples(); // Renamed to avoid conflict
    }, 5000);
});

// Make functions available globally
window.selectTriple = selectTriple;
window.saveTriple = saveTriple;
window.handleHeadEntityReplace = handleHeadEntityReplace;
window.handleTailEntityReplace = handleTailEntityReplace;
window.deleteEntity = deleteEntity;
window.mergeEntity = mergeEntity;
window.deleteTriple = deleteTriple;
window.filterDropdown = filterDropdown;
window.displayWidget = displayWidget; // Expose displayWidget globally
window.submitWidgetAnswer = submitWidgetAnswer;
window.acceptSuggestion = acceptSuggestion;
window.dismissSuggestion = dismissSuggestion;
window.showMoreEdges = showMoreEdges;
window.startChat = startChat; // Expose startChat globally for potential external calls
window.checkStatus = checkStatus; // Expose checkStatus globally
window.updateGraphState = updateGraphState; // Expose updateGraphState globally
window.loadChatTriples = loadChatTriples; // Expose loadChatTriples globally
window.filterTriples = filterTriples; // Expose filterTriples globally
window.displayTriples = displayTriples; // Expose displayTriples globally
window.enableInput = enableInput; // Expose enableInput globally
window.disableInput = disableInput; // Expose disableInput globally
window.addMessage = addMessage; // Expose addMessage globally
window.displayChanges = displayChanges; // Expose displayChanges globally
window.displayStats = displayStats; // Expose displayStats globally
window.loadStats = loadStats; // Expose loadStats globally
window.loadEntities = loadEntities; // Expose loadEntities globally
window.renderEditor = renderEditor; // Expose renderEditor globally
window.applyFilters = applyFilters; // Expose applyFilters globally
window.renderTripleList = renderTripleList; // Expose renderTripleList globally
window.showError = showError; // Expose showError globally
window.showSuccess = showSuccess; // Expose showSuccess globally

