# Graph Validator with LangGraph

A multi-agent graph validation system using LangGraph for better modularity and readability.

## Architecture

The system uses **LangGraph** to create a state graph with multiple specialized agent nodes:

### Agent Nodes

1. **Communicator** (Entry Point)
   - Main communication agent that handles user messages
   - Coordinates other agents based on conversation needs
   - Decides which agent to route to next

2. **Retriever**
   - Retrieves detailed information about entities and triples
   - Tools: `get_entity_info`, `get_triple_info`, `get_related_triples`, `search_entities`
   - Returns to communicator with retrieved data

3. **Visualizer**
   - Decides which widgets to show and with what data
   - Analyzes conversation context to determine visualization needs
   - Widget types: `triple_editor`, `entity_selector`, `importance_selector`, `graph_viewer`, `confirmation_dialog`

4. **Analyzer**
   - Generates new validation questions
   - Analyzes graph state to find issues
   - Updates question list in state

5. **Modifier**
   - Applies graph modifications based on hidden actions
   - Actions: `add_triples`, `delete_triples`, `merge_entities`, `rename_entity`, `modify_triple`
   - Updates graph, triples, and calculates statistics

### State Structure

The `GraphValidatorState` TypedDict contains:

- **Conversation**: `messages`, `current_question_id`, `current_question_text`, `questions`
- **Graph Data**: `graph`, `triples`, `id_to_name`
- **Agent Decisions**: `next_agent`, `validation_complete`
- **Actions**: `hidden_actions`, `display_actions`
- **Widgets**: `show_widget`, `widget_type`, `widget_data`
- **Metadata**: `conversation_turn`, `changes_summary`, `stats`

### Routing

Each agent node has a routing function that determines the next node:
- Agents typically return to `communicator` after completing their task
- `communicator` routes to specialized agents based on needs
- `END` is reached when `validation_complete` is True

## Usage

### Basic Usage

```python
from tools.graph.graph_validator_langgraph import GraphValidatorLangGraph
from tools.graph.visualizer import GraphVisualizer
from tools.graph.kg_gen_converter import build_id_to_name_map

# Build graph and get triples
visualizer = GraphVisualizer()
G = visualizer.build_graph(triples)
id_to_name = build_id_to_name_map(triples)

# Initialize validator
validator = GraphValidatorLangGraph(
    graph=G,
    triples=triples,
    id_to_name=id_to_name,
)

# Analyze graph (generates questions)
validator.analyze(graph=G, triples=triples, id_to_name=id_to_name)

# Chat with the validator
response = validator.chat("Please ask me questions about the graph")
print(response["text"])
print(response["next_question"])
```

### Using the Adapter (Backward Compatible)

```python
from tools.graph.graph_validator_langgraph_adapter import GraphValidatorLangGraphAdapter

# Use adapter for compatibility with existing server
validator = GraphValidatorLangGraphAdapter(
    graph=G,
    triples=triples,
    id_to_name=id_to_name,
)

validator.analyze(graph=G, triples=triples, id_to_name=id_to_name)
response = validator.chat("Hello")
```

### Integration with Server

To use LangGraph validator in the server, update `simple_server.py`:

```python
# Option 1: Use LangGraph adapter
from tools.graph.graph_validator_langgraph_adapter import GraphValidatorLangGraphAdapter

def initialize_validator(...):
    global validator
    validator = GraphValidatorLangGraphAdapter(
        graph=graph,
        triples=triples,
        id_to_name=id_to_name,
    )
    validator.analyze(graph=graph, triples=triples, id_to_name=id_to_name)
    return validator
```

## Benefits

1. **Modularity**: Each agent has a single, clear responsibility
2. **Readability**: Easy to understand the flow and add new agents
3. **Extensibility**: Add new agents by creating a node and routing function
4. **State Management**: Centralized state makes it easy to track conversation and graph changes
5. **Tool Integration**: Tools are cleanly separated from agent logic

## Adding New Agents

To add a new agent:

1. Create a node function:
```python
def _my_agent_node(self, state: GraphValidatorState) -> GraphValidatorState:
    # Agent logic here
    return {
        **state,
        "next_agent": "communicator",  # Or another agent
    }
```

2. Add routing function:
```python
def _route_from_my_agent(self, state: GraphValidatorState) -> str:
    return "communicator"  # Or "end"
```

3. Register in `_build_graph()`:
```python
workflow.add_node("my_agent", self._my_agent_node)
workflow.add_conditional_edges(
    "my_agent",
    self._route_from_my_agent,
    {"communicator": "communicator", "end": END}
)
```

4. Update communicator to route to your agent when needed

## Tools

The `GraphValidatorTools` class provides tools for agents:

- `get_entity_info(entity_name)`: Get detailed entity information
- `get_triple_info(triple_index)`: Get detailed triple information
- `get_related_triples(entity_name, max_depth)`: Get related triples
- `search_entities(query, limit)`: Search for entities by name
- `calculate_stats()`: Calculate graph statistics

## Installation

```bash
pip install langgraph
```

## Notes

- The LangGraph version is designed to be a drop-in replacement for the original `GraphValidator`
- Use the adapter class for backward compatibility
- State is managed by LangGraph's checkpointing system (MemorySaver by default)
- Each conversation thread has its own state via `config["configurable"]["thread_id"]`

