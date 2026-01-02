# Triple & Entity Web Editor

A simple web interface for editing triples and entities in real-time. Perfect for refining your knowledge graph before running claim clustering algorithms.

## Installation

First, install the required dependencies:

```bash
pip install flask flask-cors
```

Or install from the requirements file:

```bash
pip install -r web_editor/requirements.txt
```

## Usage in Jupyter Notebook

### Basic Usage

```python
from web_editor import start_triple_editor, get_updated_triples

# Assuming you have a list of triples (e.g., from your pipeline)
# Start the editor
start_triple_editor(triples, port=5000)

# The browser will open automatically
# Edit your triples and entities in the web interface
# Changes are saved in real-time

# When you're done editing, close the browser window
# Then retrieve the updated triples:
updated_triples = get_updated_triples()

# Now use updated_triples for your ClaimCluster algorithms
```

### Complete Example

```python
# 1. Generate or load your triples
# (your existing triple generation code here)
triples = [...]  # Your list of Triple objects

# 2. Start the web editor
from web_editor import start_triple_editor
start_triple_editor(triples, port=5000, open_browser=True)

# 3. Edit in the browser:
#    - Click on any triple to edit it
#    - Change entity names, labels, ref_short, ref, entity_type
#    - Change relations
#    - All changes are saved automatically

# 4. After editing, get the updated triples
from web_editor import get_updated_triples
updated_triples = get_updated_triples()

# 5. Continue with your pipeline
from tools.graph.claim_clusterers import HybridClaimClusterer
# ... use updated_triples for clustering
```

## Features

### Triple Editing
- View all triples in a searchable list
- Filter triples by entity label
- Edit triple relations
- Edit head and tail entities

### Entity Editing
- Change entity labels (most important feature)
- Update entity names
- Modify ref_short and ref
- Change entity_type
- All entity changes are reflected in real-time across all triples

### Real-time Updates
- Changes are immediately saved to the repository
- No need to manually save - everything is automatic
- Close the browser when done - your changes are preserved

## API Endpoints

The web server provides the following REST API endpoints:

- `GET /api/triples` - Get all triples
- `GET /api/triples/<id>` - Get a specific triple
- `PUT /api/triples/<id>` - Update a triple
- `GET /api/entities` - Get all entities
- `GET /api/entities/<id>` - Get a specific entity
- `PUT /api/entities/<id>` - Update an entity
- `GET /api/stats` - Get statistics (triple count, entity count)

## Architecture

- **Backend**: Flask server with REST API
- **Frontend**: Vanilla JavaScript with modern CSS
- **Repository**: Uses `EnhancedEntityTripleRepository` for data management
- **Threading**: Server runs in a separate thread, keeping notebook responsive

## Notes

- The server runs in a daemon thread, so it will stop when the notebook kernel is restarted
- Changes are stored in memory in the repository
- If you restart the kernel, you'll need to call `start_triple_editor()` again
- The repository maintains consistency - updating an entity updates all triples that reference it

