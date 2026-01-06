"""
Converter to convert kg_gen Graph objects to Triple objects for use with the claim pipeline.
"""
from __future__ import annotations

# Fix Jinja2 compatibility - patch before any Flask imports
import jinja2
if not hasattr(jinja2, 'escape'):
    try:
        from markupsafe import escape
        jinja2.escape = escape
    except ImportError:
        def escape(s):
            if s is None:
                return ''
            s = str(s)
            return (s.replace('&', '&amp;')
                    .replace('<', '&lt;')
                    .replace('>', '&gt;')
                    .replace('"', '&quot;')
                    .replace("'", '&#x27;'))
        jinja2.escape = escape

from typing import List, Dict, Set, Tuple
import uuid

from tools.graph.Triple import Triple
from tools.sentence.entity import Entity


def kg_gen_graph_to_triples(kg_graph) -> List[Triple]:
    """
    Convert a kg_gen Graph object to a list of Triple objects.
    
    Args:
        kg_graph: Graph object from kg_gen with attributes:
            - entities: Set of entity names (or dict with entity info)
            - relations: Set of tuples (subject, predicate, object) or Relation objects
    
    Returns:
        List of Triple objects with Entity objects as head and tail
    """
    triples = []
    
    # Create entity cache to avoid duplicates
    entity_cache: Dict[str, Entity] = {}
    
    def get_or_create_entity(name: str, label: str = "UNCLASSIFIED_ENTITY") -> Entity:
        """Get or create an Entity object for a given name."""
        if name in entity_cache:
            return entity_cache[name]
        
        entity_id = str(uuid.uuid4())
        entity = Entity(
            id=entity_id,
            name=name,
            label=label,
            ref_short=entity_id[-4:],
            ref=entity_id,
            start=0,  # kg_gen doesn't provide spans
            end=len(name),
            sentence_id="kg_gen",
            entity_type=label,
        )
        entity_cache[name] = entity
        return entity
    
    # Extract relations from kg_gen graph
    # kg_gen relations can be tuples or Relation objects
    relations = getattr(kg_graph, "relations", set())
    
    for relation in relations:
        # Handle tuple format: (subject, predicate, object)
        if isinstance(relation, (tuple, list)) and len(relation) >= 3:
            subject = str(relation[0]).strip()
            predicate = str(relation[1]).strip()
            obj = str(relation[2]).strip()
        # Handle Relation object with attributes
        elif hasattr(relation, "subject") and hasattr(relation, "predicate") and hasattr(relation, "object"):
            subject = str(relation.subject).strip()
            predicate = str(relation.predicate).strip()
            obj = str(relation.object).strip()
        else:
            continue
        
        if not subject or not predicate or not obj:
            continue
        
        # Create entities
        head_entity = get_or_create_entity(subject)
        tail_entity = get_or_create_entity(obj)
        
        # Create triple
        triple = Triple(
            head=head_entity,
            relation=predicate,
            tail=tail_entity
        )
        triples.append(triple)
    
    print(f"✅ Converted kg_gen graph to {len(triples)} triples")
    return triples


def build_id_to_name_map(triples: List[Triple]) -> Dict[str, str]:
    """
    Build a mapping from entity ref (identifier) to display name from triples.
    Note: Function name kept as build_id_to_name_map for backward compatibility,
    but it now uses ref instead of id.
    
    Args:
        triples: List of Triple objects
    
    Returns:
        Dict mapping entity ref to entity name
    """
    id_to_name = {}
    for triple in triples:
        # Use ref as primary identifier, fallback to ref_short if ref not available
        head_ref = getattr(triple.head, "ref", None) or getattr(triple.head, "ref_short", None) or getattr(triple.head, "id", None)
        tail_ref = getattr(triple.tail, "ref", None) or getattr(triple.tail, "ref_short", None) or getattr(triple.tail, "id", None)
        
        if head_ref:
            id_to_name[head_ref] = getattr(triple.head, "name", str(triple.head))
        if tail_ref:
            id_to_name[tail_ref] = getattr(triple.tail, "name", str(triple.tail))
    
    return id_to_name

