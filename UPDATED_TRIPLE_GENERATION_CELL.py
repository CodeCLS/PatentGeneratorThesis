# ============================================================================
# UPDATED TRIPLE GENERATION CELL
# Copy this code into your existing cell that has RateLimiter and process_sentence
# ============================================================================

from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import threading
import time
from typing import Any
from tools.graph.Triple import Triple

triples = []  # Accumulate all triples as we process sentences

# ---------------- Helper: Find existing triples for entities ----------------
def get_existing_triples_for_entities(entities, all_triples):
    """
    Find all existing triples connected to entities in the current sentence.
    
    Args:
        entities: List of entity dicts with 'id' field
        all_triples: List of existing Triple objects
    
    Returns:
        List of Triple objects connected to any entity in entities
    """
    if not all_triples or not entities:
        return []
    
    # Get entity IDs from current sentence
    entity_ids = {ent.get("id") for ent in entities if ent.get("id")}
    entity_ref_shorts = {ent.get("ref_short") for ent in entities if ent.get("ref_short")}
    
    # Also check entity names for matching (in case IDs differ)
    entity_names = {ent.get("text", "").lower().strip() for ent in entities if ent.get("text")}
    
    connected_triples = []
    for triple in all_triples:
        # Get head and tail IDs/refs
        head_id = getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None) or ""
        tail_id = getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None) or ""
        head_name = getattr(triple.head, "name", None) or getattr(triple.head, "text", None) or ""
        tail_name = getattr(triple.tail, "name", None) or getattr(triple.tail, "text", None) or ""
        
        # Check if head or tail matches any entity in current sentence
        head_matches = (
            head_id in entity_ids or 
            head_id in entity_ref_shorts or
            (head_name and head_name.lower().strip() in entity_names)
        )
        tail_matches = (
            tail_id in entity_ids or 
            tail_id in entity_ref_shorts or
            (tail_name and tail_name.lower().strip() in entity_names)
        )
        
        if head_matches or tail_matches:
            connected_triples.append(triple)
    
    return connected_triples

# ---------------- Rate limiter ----------------
class RateLimiter:
    """
    Simple token-bucket limiter.
    max_per_minute: allowed calls per minute (e.g., 900 to stay below 1000)
    """
    def __init__(self, max_per_minute: int):
        self.capacity = max_per_minute
        self.tokens = float(max_per_minute)
        self.fill_rate = max_per_minute / 60.0  # tokens per second
        self.timestamp = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self, tokens: float = 1.0):
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.timestamp
                # refill
                self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
                self.timestamp = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return  # allowed

                # need to wait for more tokens
                missing = tokens - self.tokens
                wait_time = missing / self.fill_rate

            time.sleep(wait_time)

# set to e.g. 900 to keep headroom under 1000/min
rate_limiter = RateLimiter(max_per_minute=900)

# ---------------- Your helpers ----------------
def sentence_entities_to_llm_inventory(sentence_obj) -> list[dict]:
    inv = []
    for ent in sorted(sentence_obj.entities, key=lambda e: e.start):
        inv.append({
            "id": ent.id,
            "label": ent.label,
            "span": [ent.start, ent.end],
            "text": ent.name,
            "ref_short": ent.ref_short,
        })
    return inv

_thread_local = threading.local()

def get_node_gen():
    if not hasattr(_thread_local, "node_gen"):
        _thread_local.node_gen = NodeGenerator()
    return _thread_local.node_gen

# ---------------- Run: SEQUENTIAL PROCESSING WITH CONTEXT ----------------
# NOTE: Changed from parallel to sequential to build context from existing triples
# Each sentence now sees all previously generated triples connected to its entities

print(f"Processing {len(sentence_split)} sentences sequentially with existing triples context...")
print("=" * 80)

node_gen = NodeGenerator()

# Process sentences SEQUENTIALLY (to build context from existing triples)
for i, sent in enumerate(sentence_split, 1):
    # Rate limit
    rate_limiter.acquire()
    
    # Get entities for this sentence
    entities = sentence_entities_to_llm_inventory(sent)
    
    if len(entities) < 2:
        continue  # Skip sentences with < 2 entities
    
    # Find existing triples connected to entities in this sentence
    existing_triples = get_existing_triples_for_entities(entities, triples)
    
    print(f"\n[{i}/{len(sentence_split)}] {sent.text[:70]}...")
    print(f"  Entities: {len(entities)}, Existing triples: {len(existing_triples)}")
    
    # Generate new triples with context
    try:
        new_triple_dicts = node_gen.run(
            sentence=sent.text,
            entities=entities,
            repo=repo,
            existing_triples=existing_triples  # Pass existing triples for context
        )
        
        # Handle both Triple objects and dicts
        from tools.graph.Triple import Triple
        
        for item in new_triple_dicts:
            # If it's already a Triple object, append directly
            if isinstance(item, Triple):
                triples.append(item)
            # If it's a dict, convert to Triple object
            elif isinstance(item, dict):
                head_id = item.get("head")
                tail_id = item.get("tail")
                relation = item.get("relation")
                
                if not head_id or not tail_id or not relation:
                    continue
                
                # Get entities from repo
                try:
                    head_ent = repo.get_by_id(head_id)
                    tail_ent = repo.get_by_id(tail_id)
                    
                    if head_ent and tail_ent:
                        triple_obj = Triple(
                            head=head_ent,
                            relation=relation,
                            tail=tail_ent
                        )
                        triples.append(triple_obj)
                except KeyError:
                    # Entity not found - skip this triple
                    continue
        
        print(f"  ✅ Generated {len(new_triple_dicts)} new triples (total: {len(triples)})")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        continue

print("\n" + "=" * 80)
print(f"✅ Triple generation complete! Total triples: {len(triples)}")
print("=" * 80)

