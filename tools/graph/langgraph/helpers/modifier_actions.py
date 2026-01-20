from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from tools.graph.constants_graph import *
from tools.graph.Triple import Triple
from tools.sentence.entity import Entity
class ModifierActions:

    def __init__(self):
        pass

    """
    Applies modifications to a list of triples and an id->name mapping.

    You call:
        changes = []
        actions = ModifierActions()
        actions.apply(action_type, params, triples, id_to_name, changes, graph=graph)
    """

    def apply(
        self,
        action_type: str,
        params: Dict[str, Any],
        triples: List[Triple],
        id_to_name: Dict[str, str],
        changes_summary: List[str],
        graph: Any = None,
    ) -> None:
        """Dispatch an action."""
        handlers = {
            ACTION_ADD_TRIPLES: self._add_triples,
            ACTION_DELETE_TRIPLES: self._delete_triples,
            ACTION_MERGE_ENTITIES: self._merge_entities,
            ACTION_RENAME_ENTITY: self._rename_entity,
            ACTION_UPDATE_ENTITY_LABEL: self._update_entity_label,
            ACTION_MODIFY_TRIPLE: self._modify_triple,
        }

        handler = handlers.get(action_type)
        if not handler:
            changes_summary.append(f"Unknown action: {action_type}")
            return

        handler(params, triples, id_to_name, changes_summary, graph)

    # -----------------------
    # Helpers
    # -----------------------
    @staticmethod
    def _find_id_by_name(id_to_name: Dict[str, str], target_name: str) -> Optional[str]:
        """Find entity ID by name (case-insensitive)."""
        target_lower = target_name.lower().strip()
        for eid, name in id_to_name.items():
            if name.lower().strip() == target_lower:
                return eid
        return None

    @staticmethod
    def _entity_label(graph: Any, entity_id: str) -> str:
        """
        Safely fetch a label from a NetworkX-like graph if provided.
        """
        if graph is None:
            return DEFAULT_UNKNOWN
        try:
            if graph.has_node(entity_id):
                return graph.nodes[entity_id].get(STATE_INTERNAL_NODE_TYPE, DEFAULT_UNKNOWN)
        except Exception:
            pass
        return DEFAULT_UNKNOWN

    @staticmethod
    def _triple_head_id(triple: Triple) -> str:
        return triple.head.id

    @staticmethod
    def _triple_tail_id(triple: Triple) -> str:
        return triple.tail.id

    # -----------------------
    # Action implementations
    # -----------------------
    def _add_triples(
        self,
        params: Dict[str, Any],
        triples: List[Triple],
        id_to_name: Dict[str, str],
        changes_summary: List[str],
        graph: Any,
    ) -> None:
        import uuid
        new_triples_data = params.get(KEY_TRIPLES, [])
        added_count = 0
        skipped_count = 0

        for triple_data in new_triples_data:
            head_name = triple_data.get(KEY_HEAD, "").strip()
            tail_name = triple_data.get(KEY_TAIL, "").strip()
            relation = triple_data.get(KEY_RELATION, "").strip()

            if not head_name or not tail_name or not relation:
                continue

            # Find or create head entity (case-insensitive)
            head_id = self._find_id_by_name(id_to_name, head_name)
            if not head_id:
                # Create new entity
                head_id = str(uuid.uuid4())
                id_to_name[head_id] = head_name
                if graph:
                    graph.add_node(head_id, name=head_name, label=DEFAULT_UNKNOWN)
            
            # Find or create tail entity (case-insensitive)
            tail_id = self._find_id_by_name(id_to_name, tail_name)
            if not tail_id:
                # Create new entity
                tail_id = str(uuid.uuid4())
                id_to_name[tail_id] = tail_name
                if graph:
                    graph.add_node(tail_id, name=tail_name, label=DEFAULT_UNKNOWN)

            # Check for duplicates
            is_duplicate = False
            for existing_triple in triples:
                existing_head_id = existing_triple.head.ref or existing_triple.head.id or existing_triple.head.ref_short
                existing_tail_id = existing_triple.tail.ref or existing_triple.tail.id or existing_triple.tail.ref_short
                if (existing_head_id == head_id and 
                    existing_tail_id == tail_id and 
                    existing_triple.relation == relation):
                    is_duplicate = True
                    break
            
            if is_duplicate:
                skipped_count += 1
                continue

            # Get entity names from id_to_name
            head_name_final = id_to_name.get(head_id, head_name)
            tail_name_final = id_to_name.get(tail_id, tail_name)
            
            # Get labels from graph or use default
            head_label = self._entity_label(graph, head_id)
            tail_label = self._entity_label(graph, tail_id)
            
            # Create entities with all required fields
            head_entity = Entity(
                name=head_name_final,
                label=head_label,
                ref_short=head_id[:8] if len(head_id) >= 8 else head_id,  # Use first 8 chars of UUID as ref_short
                id=head_id,
                ref=head_id,
            )
            tail_entity = Entity(
                name=tail_name_final,
                label=tail_label,
                ref_short=tail_id[:8] if len(tail_id) >= 8 else tail_id,
                id=tail_id,
                ref=tail_id,
            )
            
            triple_obj = Triple(head=head_entity, relation=relation, tail=tail_entity)
            triples.append(triple_obj)
            
            # Update graph if provided
            if graph:
                graph.add_edge(head_id, tail_id, key=triple_obj.id, relation=relation)
            
            added_count += 1

        if added_count > 0:
            changes_summary.append(f"Added {added_count} triples")
        if skipped_count > 0:
            changes_summary.append(f"Skipped {skipped_count} duplicate triples")

    def _delete_triples(
        self,
        params: Dict[str, Any],
        triples: List[Triple],
        id_to_name: Dict[str, str],
        changes_summary: List[str],
        graph: Any,
    ) -> None:
        indices = sorted(params.get("triple_indices", []), reverse=True)
        deleted_count = 0

        for idx in indices:
            if 0 <= idx < len(triples):
                triple = triples.pop(idx)
                deleted_count += 1
                
                # Update graph if provided
                if graph:
                    head_id = self._triple_head_id(triple)
                    tail_id = self._triple_tail_id(triple)
                    relation = triple.relation
                    
                    # Try to find and remove the specific edge
                    if graph.has_edge(head_id, tail_id):
                        # MultiDiGraph can have multiple edges between same nodes
                        # We try to find the one with the same relation
                        edges = graph.get_edge_data(head_id, tail_id)
                        if isinstance(edges, dict):
                            for key, data in list(edges.items()):
                                if data.get("relation") == relation:
                                    graph.remove_edge(head_id, tail_id, key=key)
                                    break

        if deleted_count > 0:
            changes_summary.append(f"Deleted {deleted_count} triples")

    def _merge_entities(
        self,
        params: Dict[str, Any],
        triples: List[Triple],
        id_to_name: Dict[str, str],
        changes_summary: List[str],
        graph: Any,
    ) -> None:
        entity_names = params.get("entity_names", [])
        if len(entity_names) < 2:
            return

        target_name = entity_names[0]
        source_names = entity_names[1:]

        target_id = self._find_id_by_name(id_to_name, target_name)
        if not target_id:
            return

        source_ids = []
        for name in source_names:
            sid = self._find_id_by_name(id_to_name, name)
            if sid:
                source_ids.append(sid)

        if not source_ids:
            return

        # Update triples
        for t in triples:
            if self._triple_head_id(t) in source_ids:
                t.head.id = target_id
                t.head.name = target_name
            if self._triple_tail_id(t) in source_ids:
                t.tail.id = target_id
                t.tail.name = target_name

        # Update graph if provided
        if graph:
            for source_id in source_ids:
                if not graph.has_node(source_id):
                    continue
                    
                # Move outgoing edges
                for _, tail, data in list(graph.out_edges(source_id, data=True)):
                    if tail == source_id: # Self-loop
                        graph.add_edge(target_id, target_id, **data)
                    else:
                        graph.add_edge(target_id, tail, **data)
                
                # Move incoming edges
                for head, _, data in list(graph.in_edges(source_id, data=True)):
                    if head != source_id: # Already handled self-loops
                        graph.add_edge(head, target_id, **data)
                
                # Remove source node
                graph.remove_node(source_id)

        # Remove merged ids from mapping
        for sid in source_ids:
            id_to_name.pop(sid, None)

        changes_summary.append(f"Merged {len(source_ids)} entities into '{target_name}'")

    def _rename_entity(
        self,
        params: Dict[str, Any],
        triples: List[Triple],
        id_to_name: Dict[str, str],
        changes_summary: List[str],
        graph: Any,
    ) -> None:
        old_name = params.get("old_name", "")
        new_name = params.get("new_name", "")
        if not old_name or not new_name:
            return

        entity_id = self._find_id_by_name(id_to_name, old_name)
        if not entity_id:
            return

        id_to_name[entity_id] = new_name

        # Update graph node if provided
        if graph and graph.has_node(entity_id):
            graph.nodes[entity_id]["name"] = new_name

        # Update any triples that contain this entity id
        for t in triples:
            if self._triple_head_id(t) == entity_id:
                t.head.name = new_name
            if self._triple_tail_id(t) == entity_id:
                t.tail.name = new_name

        changes_summary.append(f"Renamed '{old_name}' to '{new_name}'")

    def _update_entity_label(
        self,
        params: Dict[str, Any],
        triples: List[Triple],
        id_to_name: Dict[str, str],
        changes_summary: List[str],
        graph: Any,
    ) -> None:
        entity_name = params.get(KEY_NAME, "")
        new_label = params.get("new_label", "")
        if not entity_name or not new_label:
            return

        entity_id = self._find_id_by_name(id_to_name, entity_name)
        if not entity_id:
            return

        # Update graph if provided
        if graph and graph.has_node(entity_id):
            graph.nodes[entity_id][STATE_INTERNAL_NODE_TYPE] = new_label

        updated_count = 0
        old_label: Optional[str] = None

        # Update graph node label if provided
        if graph and graph.has_node(entity_id):
            graph.nodes[entity_id][STATE_INTERNAL_NODE_TYPE] = new_label

        for t in triples:
            if self._triple_head_id(t) == entity_id:
                old_label = old_label or t.head.label
                t.head.label = new_label
                updated_count += 1
            if self._triple_tail_id(t) == entity_id:
                old_label = old_label or t.tail.label
                t.tail.label = new_label
                updated_count += 1

        if updated_count > 0:
            if old_label is not None:
                changes_summary.append(
                    f"Updated label of '{entity_name}' from '{old_label}' to '{new_label}'"
                )
            else:
                changes_summary.append(f"Updated label of '{entity_name}' to '{new_label}'")

    def _modify_triple(
        self,
        params: Dict[str, Any],
        triples: List[Triple],
        id_to_name: Dict[str, str],
        changes_summary: List[str],
        graph: Any,
    ) -> None:
        triple_index = params.get("triple_index")
        new_relation = params.get("new_relation")

        if triple_index is None or not (0 <= triple_index < len(triples)):
            return
        if not new_relation:
            return

        t = triples[triple_index]
        t.relation = new_relation
        
        # Update graph if provided
        if graph:
            head_id = self._triple_head_id(t)
            tail_id = self._triple_tail_id(t)
            if graph.has_edge(head_id, tail_id, key=t.id):
                graph.edges[head_id, tail_id, t.id]["relation"] = new_relation

        changes_summary.append(f"Modified triple {triple_index}")
