# Graph Validator Chat Interface

A simple chat-based web interface for validating and modifying knowledge graphs using LLM-powered questions and answers.

## Features

- **Interactive Chat**: Answer LLM-generated questions about your graph
- **Graph Modification**: LLM can automatically modify the graph based on your answers
- **Real-time Updates**: See graph state changes in real-time
- **Export to Notebook**: Get updated graph, triples, and entities back in your Jupyter notebook

## Installation

The chat interface uses Flask. Make sure you have the required dependencies:

```bash
pip install flask flask-cors requests
```

## Quick Start

### 1. Start the Chat Interface

In your Jupyter notebook:

```python
from web_editor.graph_validator_chat import start_validator_chat
from tools.graph.visualizer import GraphVisualizer
from tools.graph.kg_gen_converter import build_id_to_name_map

# Assuming you have:
# - triples: List of Triple objects
# - G: NetworkX graph (optional)

# Build id_to_name mapping
id_to_name = build_id_to_name_map(triples)

# Start the chat interface
start_validator_chat(
    graph=G,  # Optional: NetworkX graph
    triples=triples,  # Optional: List of Triple objects
    id_to_name=id_to_name,  # Optional: Entity ID to name mapping
    port=5001,  # Port for the web interface
    open_browser=True,  # Automatically open browser
)
```

The browser will open at `http://localhost:5001` with the chat interface.

### 2. Use the Chat Interface

1. The LLM will analyze your graph and ask questions
2. Answer questions in the chat input
3. The LLM will process your answers and may modify the graph automatically
4. Watch the graph state update in real-time in the sidebar

### 3. Get Updated Data

After you're done chatting, close the browser and run:

```python
from web_editor.graph_validator_chat.helper import get_all_updates

# Get all updated data
updates = get_all_updates()

# Extract components
G_updated = updates['graph']
triples_updated = updates['triples']
entities_updated = updates['entities']
id_to_name_updated = updates['id_to_name']
changes = updates['changes']

print(f"✅ Updated graph: {G_updated.number_of_nodes()} nodes")
print(f"✅ Updated triples: {len(triples_updated)} triples")
print(f"✅ Changes: {changes}")
```

## Available Functions

### Starting the Chat

- `start_validator_chat(graph, triples, id_to_name, port=5001, open_browser=True)` - Start the chat interface

### Getting Updated Data

- `get_all_updates()` - Get everything: graph, triples, entities, changes, id_to_name
- `get_updated_graph()` - Get updated NetworkX graph
- `get_updated_triples()` - Get updated list of Triple objects
- `get_updated_entities()` - Get list of all entities
- `get_changes_summary()` - Get summary of changes made

## Graph Modification Actions

The LLM can automatically perform these actions based on your answers:

- **Add/Delete Triples**: Add new triples or remove existing ones
- **Modify Triples**: Change relations, head, or tail entities
- **Merge Entities**: Combine multiple entities into one
- **Delete Entities**: Remove entities and all connected triples
- **Rename Entities**: Change entity names
- **Change Entity Labels**: Update entity types/labels
- **Add/Remove Relations**: Add or remove specific relation edges
- **Split Entities**: Split one entity into multiple entities
- **Create Entities**: Create new entities

## Example Workflow

```python
# 1. Generate your graph and triples
from tools.graph.visualizer import GraphVisualizer
visualizer = GraphVisualizer()
G = visualizer.build_graph(triples)
id_to_name = build_id_to_name_map(triples)

# 2. Start chat interface
from web_editor.graph_validator_chat import start_validator_chat
start_validator_chat(graph=G, triples=triples, id_to_name=id_to_name)

# 3. Chat with the LLM in the browser
# Answer questions, let the LLM modify the graph

# 4. Get updated data
from web_editor.graph_validator_chat.helper import get_all_updates
updates = get_all_updates()

# 5. Use updated data in your pipeline
G_updated = updates['graph']
triples_updated = updates['triples']

# Continue with your claim generation pipeline...
```

## Notes

- The server runs in a background thread, so your notebook remains responsive
- Changes are applied in real-time to both the graph and triples
- The chat interface shows the current graph state in the sidebar
- All modifications are tracked and can be viewed in the changes summary

