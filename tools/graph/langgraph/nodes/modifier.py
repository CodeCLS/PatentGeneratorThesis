"""
Modifier node - applies graph modifications based on hidden actions.
"""

from typing import TYPE_CHECKING
from tools.graph.Triple import Triple
from tools.sentence.entity import Entity
from tools.graph.langgraph.state import GraphValidatorState
from tools.graph.langgraph.helpers import get_triple_head_id, get_triple_tail_id

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def modifier_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState") -> "GraphValidatorState":
    """Modification agent - applies graph modifications based on hidden actions."""
    hidden_actions = state.get("hidden_actions", [])
    changes_summary = []
    graph = validator.graph
    triples = validator.triples.copy()
    id_to_name = validator.id_to_name.copy()
    
    for action in hidden_actions:
        action_type = action.get("type")
        params = action.get("parameters", {})
        
        if action_type == "add_triples":
            new_triples_data = params.get("triples", [])
            added_count = 0
            for triple_data in new_triples_data:
                head_name = triple_data.get("head", "")
                tail_name = triple_data.get("tail", "")
                relation = triple_data.get("relation", "")
                
                head_id = None
                tail_id = None
                for eid, name in id_to_name.items():
                    if name == head_name:
                        head_id = eid
                    if name == tail_name:
                        tail_id = eid
                
                if head_id and tail_id:
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
                    triples.append(Triple(head=head_entity, relation=relation, tail=tail_entity))
                    added_count += 1
            
            if added_count > 0:
                changes_summary.append(f"Added {added_count} triples")
        
        elif action_type == "delete_triples":
            indices = sorted(params.get("triple_indices", []), reverse=True)
            deleted_count = 0
            for idx in indices:
                if 0 <= idx < len(triples):
                    triples.pop(idx)
                    deleted_count += 1
            if deleted_count > 0:
                changes_summary.append(f"Deleted {deleted_count} triples")
        
        elif action_type == "merge_entities":
            entity_names = params.get("entity_names", [])
            if len(entity_names) >= 2:
                target_name = entity_names[0]
                source_names = entity_names[1:]
                
                target_id = None
                source_ids = []
                for eid, name in id_to_name.items():
                    if name == target_name:
                        target_id = eid
                    elif name in source_names:
                        source_ids.append(eid)
                
                if target_id and source_ids:
                    for triple in triples:
                        head_id = get_triple_head_id(triple)
                        tail_id = get_triple_tail_id(triple)
                        
                        if head_id in source_ids:
                            triple.head.ref = target_id
                        if tail_id in source_ids:
                            triple.tail.ref = target_id
                    
                    for sid in source_ids:
                        id_to_name.pop(sid, None)
                    
                    changes_summary.append(f"Merged {len(source_names)} entities into '{target_name}'")
        
        elif action_type == "rename_entity":
            old_name = params.get("old_name")
            new_name = params.get("new_name")
            if old_name and new_name:
                for eid, name in list(id_to_name.items()):
                    if name == old_name:
                        id_to_name[eid] = new_name
                        changes_summary.append(f"Renamed '{old_name}' to '{new_name}'")
                        break
        
        elif action_type == "modify_triple":
            triple_index = params.get("triple_index")
            new_relation = params.get("new_relation")
            if triple_index is not None and 0 <= triple_index < len(triples):
                if new_relation:
                    triples[triple_index].relation = new_relation
                    changes_summary.append(f"Modified triple {triple_index}")
    
    # Update instance data
    validator.triples = triples
    validator.id_to_name = id_to_name
    if graph:
        validator.graph = graph
    
    validator.tools.triples = triples
    validator.tools.id_to_name = id_to_name
    if graph:
        validator.tools.graph = graph
    
    stats = validator.tools.calculate_stats()
    
    return {
        **state,
        "hidden_actions": [],
        "changes_summary": state.get("changes_summary", []) + changes_summary,
        "stats": stats,
        "next_agent": "communicator",
    }
