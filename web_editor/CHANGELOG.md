# Web Editor Updates

## New Features

### 1. Entity Replacement Dropdowns
- Added dropdown menus to replace head/tail entities with existing entities
- When an entity is selected from the dropdown, the form fields are automatically populated
- Option to keep the current entity or replace it with another

### 2. Improved Filtering
- Label filter now strictly filters triples - only shows triples where head OR tail has the selected label
- Search query works on top of the label filter (applied sequentially)
- No more showing triples with different labels when a label filter is active

### 3. Entity Deletion
- Added "Delete Entity" buttons for both head and tail entities
- Deleting an entity automatically deletes all triples connected to it
- Confirmation dialog before deletion
- Shows count of deleted triples in success message

### 4. Label Propagation
- When an entity label is changed, it automatically updates in ALL triples that reference that entity
- Changes are reflected immediately across the entire knowledge graph
- Frontend reloads all data after saves to show updated labels everywhere

## Technical Details

### Backend Changes
- Added `replace_with_id` parameter to triple update endpoint
- Added DELETE endpoint for entities (`/api/entities/<id>`)
- Entity deletion cascades to all connected triples
- Label updates automatically propagate through shared entity objects

### Frontend Changes
- Entity dropdowns populated from all available entities
- Improved filter logic: label filter → search query (sequential)
- Delete buttons with confirmation dialogs
- Automatic data reload after saves to show label changes everywhere

## Usage

1. **Replace Entity**: Select an entity from the "Replace with Entity" dropdown
2. **Filter by Label**: Select a label from the filter dropdown - only triples with that label will show
3. **Delete Entity**: Click "Delete Entity" button - confirms and deletes entity + all connected triples
4. **Update Label**: Change an entity label - it updates in all triples automatically

