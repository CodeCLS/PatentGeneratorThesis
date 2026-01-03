"""
Flask server for editing triples and entities in real-time.
"""
# Fix Jinja2 compatibility issue - must be imported BEFORE Flask
import web_editor.jinja2_compat  # noqa: F401

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
from typing import List, Dict, Any, Optional
import threading
import webbrowser
import time

# Import the repository and Triple class
from tools.sentence.entity import EnhancedEntityTripleRepository
from tools.graph.Triple import Triple

app = Flask(__name__)
CORS(app)

# Global repository instance
repo: Optional[EnhancedEntityTripleRepository] = None


def initialize_repository(triples: List[Triple]) -> EnhancedEntityTripleRepository:
    """
    Initialize the repository with triples.
    
    Args:
        triples: List of Triple objects
    
    Returns:
        EnhancedEntityTripleRepository instance
    """
    global repo
    
    # Extract all unique entities from triples
    entities = {}
    for triple in triples:
        if triple.head.id not in entities:
            entities[triple.head.id] = triple.head
        if triple.tail.id not in entities:
            entities[triple.tail.id] = triple.tail
    
    repo = EnhancedEntityTripleRepository(
        entities=list(entities.values()),
        triples=triples
    )
    
    return repo


def serialize_triple(triple: Triple) -> Dict[str, Any]:
    """Convert Triple to JSON-serializable dict."""
    return {
        "id": triple.id,
        "head": {
            "id": triple.head.id,
            "name": triple.head.name,
            "label": triple.head.label,
            "ref_short": triple.head.ref_short,
            "ref": triple.head.ref,
            "entity_type": triple.head.entity_type,
            "sentence_id": triple.head.sentence_id,
        },
        "relation": triple.relation,
        "tail": {
            "id": triple.tail.id,
            "name": triple.tail.name,
            "label": triple.tail.label,
            "ref_short": triple.tail.ref_short,
            "ref": triple.tail.ref,
            "entity_type": triple.tail.entity_type,
            "sentence_id": triple.tail.sentence_id,
        },
        "importance": triple.importance,
        "info_quality": triple.info_quality,
        "novelty": triple.novelty,
        "tags": triple.tags,
    }


def serialize_entity(entity) -> Dict[str, Any]:
    """Convert Entity to JSON-serializable dict."""
    return {
        "id": entity.id,
        "name": entity.name,
        "label": entity.label,
        "ref_short": entity.ref_short,
        "ref": entity.ref,
        "entity_type": entity.entity_type,
        "sentence_id": entity.sentence_id,
        "start": entity.start,
        "end": entity.end,
    }


@app.route('/')
def index():
    """Serve the main editor page."""
    return render_template('index.html')


@app.route('/api/triples', methods=['GET'])
def get_triples():
    """Get all triples."""
    if repo is None:
        return jsonify({"error": "Repository not initialized"}), 400
    
    triples = list(repo.get_all_triples().values())
    return jsonify([serialize_triple(t) for t in triples])


@app.route('/api/triples/<triple_id>', methods=['GET'])
def get_triple(triple_id: str):
    """Get a specific triple by ID."""
    if repo is None:
        return jsonify({"error": "Repository not initialized"}), 400
    
    try:
        triple = repo.get_triple(triple_id)
        return jsonify(serialize_triple(triple))
    except KeyError:
        return jsonify({"error": "Triple not found"}), 404


@app.route('/api/triples/<triple_id>', methods=['PUT'])
def update_triple(triple_id: str):
    """Update a triple."""
    if repo is None:
        return jsonify({"error": "Repository not initialized"}), 400
    
    try:
        data = request.json
        
        # Get current triple to access entities
        triple = repo.get_triple(triple_id)
        
        # Handle head entity replacement or update
        head = None
        if "head" in data:
            head_data = data["head"]
            if "replace_with_id" in head_data:
                # Replace with a different existing entity
                try:
                    head = repo.get_by_id(head_data["replace_with_id"])
                except KeyError:
                    return jsonify({"error": f"Entity {head_data['replace_with_id']} not found"}), 404
            elif "id" in head_data:
                # Update existing entity properties
                try:
                    head = repo.get_by_id(head_data["id"])
                    # Update entity properties
                    if "label" in head_data:
                        head.label = head_data["label"]
                    if "name" in head_data:
                        head.name = head_data["name"]
                    if "ref_short" in head_data:
                        head.ref_short = head_data["ref_short"]
                    if "ref" in head_data:
                        head.ref = head_data.get("ref")
                    if "entity_type" in head_data:
                        head.entity_type = head_data.get("entity_type")
                    repo.save(head)
                except KeyError:
                    # Create new entity
                    from tools.sentence.entity import Entity
                    head = Entity(
                        id=head_data["id"],
                        name=head_data.get("name", ""),
                        label=head_data.get("label", ""),
                        ref_short=head_data.get("ref_short", ""),
                        ref=head_data.get("ref"),
                        entity_type=head_data.get("entity_type"),
                        sentence_id=head_data.get("sentence_id"),
                        start=head_data.get("start", 0),
                        end=head_data.get("end", 0),
                    )
                    repo.save(head)
            else:
                head = triple.head
        
        # Handle tail entity replacement or update
        tail = None
        if "tail" in data:
            tail_data = data["tail"]
            if "replace_with_id" in tail_data:
                # Replace with a different existing entity
                try:
                    tail = repo.get_by_id(tail_data["replace_with_id"])
                except KeyError:
                    return jsonify({"error": f"Entity {tail_data['replace_with_id']} not found"}), 404
            elif "id" in tail_data:
                # Update existing entity properties
                try:
                    tail = repo.get_by_id(tail_data["id"])
                    # Update entity properties
                    if "label" in tail_data:
                        tail.label = tail_data["label"]
                    if "name" in tail_data:
                        tail.name = tail_data["name"]
                    if "ref_short" in tail_data:
                        tail.ref_short = tail_data["ref_short"]
                    if "ref" in tail_data:
                        tail.ref = tail_data.get("ref")
                    if "entity_type" in tail_data:
                        tail.entity_type = tail_data.get("entity_type")
                    repo.save(tail)
                except KeyError:
                    # Create new entity
                    from tools.sentence.entity import Entity
                    tail = Entity(
                        id=tail_data["id"],
                        name=tail_data.get("name", ""),
                        label=tail_data.get("label", ""),
                        ref_short=tail_data.get("ref_short", ""),
                        ref=tail_data.get("ref"),
                        entity_type=tail_data.get("entity_type"),
                        sentence_id=tail_data.get("sentence_id"),
                        start=tail_data.get("start", 0),
                        end=tail_data.get("end", 0),
                    )
                    repo.save(tail)
            else:
                tail = triple.tail
        
        # Update triple
        update_kwargs = {}
        if head is not None:
            update_kwargs["head"] = head
        if tail is not None:
            update_kwargs["tail"] = tail
        if "relation" in data:
            update_kwargs["relation"] = data["relation"]
        if "importance" in data:
            update_kwargs["importance"] = data["importance"]
        if "info_quality" in data:
            update_kwargs["info_quality"] = data["info_quality"]
        if "novelty" in data:
            update_kwargs["novelty"] = data["novelty"]
        if "tags" in data:
            update_kwargs["tags"] = data["tags"]
        
        updated_triple = repo.update_triple(triple_id, **update_kwargs)
        
        return jsonify(serialize_triple(updated_triple))
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/triples/<triple_id>', methods=['DELETE'])
def delete_triple(triple_id: str):
    """Delete a triple (relation/edge) without deleting the entities."""
    if repo is None:
        return jsonify({"error": "Repository not initialized"}), 400
    
    try:
        # Get triple info before deletion
        triple = repo.get_triple(triple_id)
        head_name = triple.head.name
        tail_name = triple.tail.name
        relation = triple.relation
        
        # Delete the triple (this doesn't delete entities)
        deleted = repo.delete_triple(triple_id)
        
        if deleted:
            return jsonify({
                "success": True,
                "message": f"Triple deleted: {head_name} --[{relation}]--> {tail_name}",
                "deleted_triple_id": triple_id
            })
        else:
            return jsonify({"error": "Triple not found"}), 404
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/entities', methods=['GET'])
def get_entities():
    """Get all entities."""
    if repo is None:
        return jsonify({"error": "Repository not initialized"}), 400
    
    entities = list(repo.getAll().values())
    return jsonify([serialize_entity(e) for e in entities])


@app.route('/api/entities/<entity_id>', methods=['GET'])
def get_entity(entity_id: str):
    """Get a specific entity by ID."""
    if repo is None:
        return jsonify({"error": "Repository not initialized"}), 400
    
    try:
        entity = repo.get_by_id(entity_id)
        return jsonify(serialize_entity(entity))
    except KeyError:
        return jsonify({"error": "Entity not found"}), 404


@app.route('/api/entities/<entity_id>', methods=['PUT'])
def update_entity(entity_id: str):
    """Update an entity. When label is changed, all triples referencing this entity are updated."""
    if repo is None:
        return jsonify({"error": "Repository not initialized"}), 400
    
    try:
        data = request.json
        
        # Get the entity before update to check if label changed
        old_entity = repo.get_by_id(entity_id)
        old_label = old_entity.label
        
        # Update the entity
        updated_entity = repo.update_entity(
            entity_id,
            name=data.get("name"),
            label=data.get("label"),
            ref_short=data.get("ref_short"),
            ref=data.get("ref"),
            entity_type=data.get("entity_type"),
            sentence_id=data.get("sentence_id"),
            start=data.get("start"),
            end=data.get("end"),
        )
        
        # If label changed, update all triples that reference this entity
        # The repository's update_entity already handles this through the save() method,
        # but we need to ensure all triples are refreshed
        if data.get("label") is not None and data.get("label") != old_label:
            # Get all triples that reference this entity
            affected_triples = repo.get_triples_by_entity(entity_id)
            # The entities in these triples are already updated since they reference the same object
            # But we should reload to ensure consistency
        
        return jsonify(serialize_entity(updated_entity))
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/entities/<entity_id>', methods=['DELETE'])
def delete_entity(entity_id: str):
    """Delete an entity and all triples connected to it."""
    if repo is None:
        return jsonify({"error": "Repository not initialized"}), 400
    
    try:
        # Get count of triples that will be deleted
        affected_triples = repo.get_triples_by_entity(entity_id)
        triple_count = len(affected_triples)
        
        # Delete the entity (this also deletes all connected triples via delete_entity)
        deleted = repo.delete_entity(entity_id)
        
        if deleted:
            return jsonify({
                "success": True,
                "message": f"Entity deleted. {triple_count} triples were also deleted.",
                "deleted_triple_count": triple_count
            })
        else:
            return jsonify({"error": "Entity not found"}), 404
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/entities/<source_id>/merge', methods=['POST'])
def merge_entities(source_id: str):
    """
    Merge source entity into target entity.
    All triples pointing to source_id will be updated to point to target_id,
    then source_id will be deleted.
    """
    if repo is None:
        return jsonify({"error": "Repository not initialized"}), 400
    
    try:
        data = request.json
        target_id = data.get("target_id")
        
        if not target_id:
            return jsonify({"error": "target_id is required"}), 400
        
        if source_id == target_id:
            return jsonify({"error": "Cannot merge entity with itself"}), 400
        
        # Verify both entities exist
        try:
            source_entity = repo.get_by_id(source_id)
            target_entity = repo.get_by_id(target_id)
        except KeyError as e:
            return jsonify({"error": f"Entity not found: {str(e)}"}), 404
        
        # Get all triples that reference the source entity (before updates)
        source_triples = repo.get_triples_by_entity(source_id)
        affected_triple_ids = set([t.id for t in source_triples])
        affected_count = len(affected_triple_ids)
        
        # Update all triples that have source as head
        head_triples = repo.get_triples_by_head(source_id)
        for triple in head_triples:
            repo.update_triple(triple.id, head=target_entity)
        
        # Update all triples that have source as tail
        tail_triples = repo.get_triples_by_tail(source_id)
        for triple in tail_triples:
            repo.update_triple(triple.id, tail=target_entity)
        
        # Now delete the source entity
        # Since we've moved all references, delete_entity should only delete the entity itself
        # But delete_entity also deletes triples, so we need to be careful
        # Actually, we should manually delete the entity without deleting triples
        # since we've already moved all references
        
        # Manually remove entity from repository (don't use delete_entity as it deletes triples)
        if source_id in repo._entities:
            del repo._entities[source_id]
        
        # Clean up indices
        if source_id in repo._triples_by_head:
            del repo._triples_by_head[source_id]
        if source_id in repo._triples_by_tail:
            del repo._triples_by_tail[source_id]
        
        return jsonify({
            "success": True,
            "message": f"Entity merged successfully. {affected_count} triples were updated.",
            "merged_triple_count": affected_count,
            "source_id": source_id,
            "target_id": target_id
        })
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about triples and entities."""
    if repo is None:
        return jsonify({"error": "Repository not initialized"}), 400
    
    return jsonify({
        "triple_count": repo.get_triple_count(),
        "entity_count": repo.get_entity_count(),
    })


def run_server(port: int = 5000, debug: bool = False, open_browser: bool = True):
    """
    Run the Flask server.
    
    Args:
        port: Port to run the server on
        debug: Enable debug mode
        open_browser: Whether to open browser automatically
    """
    if open_browser:
        def open_browser_delayed():
            time.sleep(1.5)
            webbrowser.open(f'http://localhost:{port}')
        
        threading.Thread(target=open_browser_delayed, daemon=True).start()
    
    app.run(port=port, debug=debug, use_reloader=False)


if __name__ == '__main__':
    run_server()

