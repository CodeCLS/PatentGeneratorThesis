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
        for eid, name in id_to_name.items():
            if name == target_name:
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
        new_triples_data = params.get(KEY_TRIPLES, [])
        added_count = 0

        for triple_data in new_triples_data:
            head_name = triple_data.get(KEY_HEAD, "")
            tail_name = triple_data.get(KEY_TAIL, "")
            relation = triple_data.get(KEY_RELATION, "")

            if not head_name or not tail_name or not relation:
                continue

            head_id = self._find_id_by_name(id_to_name, head_name)
            tail_id = self._find_id_by_name(id_to_name, tail_name)

            if head_id and tail_id:
                head_entity = Entity(
                    id=head_id,
                    name=head_name,
                    label=self._entity_label(graph, head_id),
                )
                tail_entity = Entity(
                    id=tail_id,
                    name=tail_name,
                    label=self._entity_label(graph, tail_id),
                )
                triples.append(Triple(head=head_entity, relation=relation, tail=tail_entity))
                added_count += 1

        if added_count > 0:
            changes_summary.append(f"Added {added_count} triples")

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
                triples.pop(idx)
                deleted_count += 1

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

        updated_count = 0
        old_label: Optional[str] = None

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

        triples[triple_index].relation = new_relation
        changes_summary.append(f"Modified triple {triple_index}")
