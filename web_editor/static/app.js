// Global state
let triples = [];
let entities = [];
let currentTriple = null;
let allLabels = new Set();
let searchQuery = '';
let labelFilter = '';

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    await loadEntities();
    await loadTriples();
    await loadStats();
    
    // Search functionality
    document.getElementById('search-input').addEventListener('input', (e) => {
        searchQuery = e.target.value.toLowerCase();
        applyFilters();
    });
    document.getElementById('label-filter').addEventListener('change', (e) => {
        labelFilter = e.target.value;
        applyFilters();
    });
});

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
        document.getElementById('triple-count').textContent = stats.triple_count;
        document.getElementById('entity-count').textContent = stats.entity_count;
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

function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
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

// Make functions available globally
window.selectTriple = selectTriple;
window.saveTriple = saveTriple;
window.handleHeadEntityReplace = handleHeadEntityReplace;
window.handleTailEntityReplace = handleTailEntityReplace;
window.deleteEntity = deleteEntity;
window.mergeEntity = mergeEntity;
window.deleteTriple = deleteTriple;
window.filterDropdown = filterDropdown;
