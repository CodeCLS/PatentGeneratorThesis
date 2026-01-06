"""
Modifier node - applies graph modifications based on hidden actions.
"""

from typing import TYPE_CHECKING

from tools.graph.Triple import Triple
from tools.sentence.entity import Entity

# Import GraphValidatorState at runtime (not just TYPE_CHECKING)
# This is needed because LangGraph might inspect type hints at runtime
from tools.graph.langgraph.state import GraphValidatorState

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def modifier_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState") -> "GraphValidatorState":
    """
    Modification agent - applies graph modifications based on hidden actions.
    """
    hidden_actions = state.get("hidden_actions", [])
    changes_summary = []
    # Get graph/triples from instance, not state (to avoid serialization issues)
    graph = validator.graph
    triples = validator.triples.copy()  # Work with a copy
    id_to_name = validator.id_to_name.copy()  # Work with a copy
    
    # Apply each action
    for action in hidden_actions:
        action_type = action.get("type")
        params = action.get("parameters", {})
        
        try:
            if action_type == "add_triples":
                # Add new triples
                new_triples_data = params.get("triples", [])
                added_count = 0
                for triple_data in new_triples_data:
                    # Create Triple objects from data
                    head_name = triple_data.get("head", "")
                    tail_name = triple_data.get("tail", "")
                    relation = triple_data.get("relation", "")
                    
                    # Find or create entities
                    head_id = None
                    tail_id = None
                    for eid, name in id_to_name.items():
                        if name == head_name:
                            head_id = eid
                        if name == tail_name:
                            tail_id = eid
                    
                    if head_id and tail_id:
                        # Create entities and triple
                        head_entity = Entity(
                            id=head_id,
                            name=head_name,
                            label=graph.nodes[head_id].get("node_type", "UNKNOWN") if graph and graph.has_node(head_id) else "UNKNOWN"
                        )
                        tail_entity = Entity(
                            id=tail_id,
                            name=tail_name,
                            label=graph.nodes[tail_id].get("node_type", "UNKNOWN") if graph and graph.has_node(tail_id) else "UNKNOWN"
                        )
                        new_triple = Triple(head=head_entity, relation=relation, tail=tail_entity)
                        triples.append(new_triple)
                        added_count += 1
                
                if added_count > 0:
                    changes_summary.append(f"Added {added_count} triples")
            
            elif action_type == "delete_triples":
                # Delete triples by index (in reverse order to maintain indices)
                indices = sorted(params.get("triple_indices", []), reverse=True)
                deleted_count = 0
                for idx in indices:
                    if 0 <= idx < len(triples):
                        triples.pop(idx)
                        deleted_count += 1
                if deleted_count > 0:
                    changes_summary.append(f"Deleted {deleted_count} triples")
            
            elif action_type == "merge_entities":
                # Merge entities
                entity_names = params.get("entity_names", [])
                if len(entity_names) >= 2:
                    # Keep first, merge others into it
                    target_name = entity_names[0]
                    source_names = entity_names[1:]
                    
                    # Find IDs
                    target_id = None
                    source_ids = []
                    for eid, name in id_to_name.items():
                        if name == target_name:
                            target_id = eid
                        elif name in source_names:
                            source_ids.append(eid)
                    
                    if target_id and source_ids:
                        # Update triples to point to target
                        for triple in triples:
                            head_id = getattr(triple.head, "ref", None) or getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None) or str(triple.head)
                            tail_id = getattr(triple.tail, "ref", None) or getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None) or str(triple.tail)
                            
                            if head_id in source_ids:
                                triple.head.ref = target_id
                            if tail_id in source_ids:
                                triple.tail.ref = target_id
                        
                        # Remove source entities from id_to_name
                        for sid in source_ids:
                            if sid in id_to_name:
                                del id_to_name[sid]
                        
                        changes_summary.append(f"Merged {len(source_names)} entities into '{target_name}'")
            
            elif action_type == "rename_entity":
                old_name = params.get("old_name")
                new_name = params.get("new_name")
                if old_name and new_name:
                    # Find entity ID
                    for eid, name in list(id_to_name.items()):
                        if name == old_name:
                            id_to_name[eid] = new_name
                            changes_summary.append(f"Renamed '{old_name}' to '{new_name}'")
                            break
            
            elif action_type == "modify_triple":
                triple_index = params.get("triple_index")
                new_relation = params.get("new_relation")
                if triple_index is not None and 0 <= triple_index < len(triples):
                    triple = triples[triple_index]
                    if new_relation:
                        triple.relation = new_relation
                        changes_summary.append(f"Modified triple {triple_index}")
            
        except Exception as e:
            changes_summary.append(f"Error applying action {action_type}: {str(e)}")
    
    # Update instance data (not state, to avoid serialization issues)
    validator.triples = triples
    validator.id_to_name = id_to_name
    if graph:
        validator.graph = graph
    
    # Update tools with new data
    validator.tools.triples = triples
    validator.tools.id_to_name = id_to_name
    if graph:
        validator.tools.graph = graph
    
    # Update state metadata only (for tracking)
    state["triples_count"] = len(triples)
    state["entities_count"] = len(id_to_name)
    if graph:
        state["graph_nodes_count"] = graph.number_of_nodes()
        state["graph_edges_count"] = graph.number_of_edges()
    
    # Calculate stats
    stats = validator.tools.calculate_stats()
    
    return {
        **state,
        "hidden_actions": [],  # Clear after processing
        "changes_summary": state.get("changes_summary", []) + changes_summary,
        "stats": stats,
        "next_agent": "communicator",
    }

