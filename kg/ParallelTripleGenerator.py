"""
Parallel triple generation with rate limiting and thread-safe shared context.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from tools.graph.data.Triple import Triple
from tools.sentence.entity import InMemoryEntityRepository
from kg.NodeGenerator import NodeGenerator


class RateLimiter:
    """
    Simple token-bucket rate limiter for API calls.
    
    Args:
        max_per_minute: Maximum number of calls allowed per minute
    """
    def __init__(self, max_per_minute: int):
        self.capacity = max_per_minute
        self.tokens = float(max_per_minute)
        self.fill_rate = max_per_minute / 60.0
        self.timestamp = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self, tokens: float = 1.0):
        """Acquire tokens, blocking if necessary."""
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.timestamp
                self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
                self.timestamp = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return

                missing = tokens - self.tokens
                wait_time = missing / self.fill_rate

            time.sleep(wait_time)


class ParallelTripleGenerator:
    """
    Parallel triple generation from sentences with rate limiting and shared context.
    
    Features:
    - Thread-safe shared triples list
    - Rate limiting for API calls
    - Parallel processing with configurable workers
    - Context-aware triple generation (uses existing triples)
    """
    
    def __init__(
        self,
        repo: Optional[InMemoryEntityRepository] = None,
        max_workers: int = 10,
        rate_limit_per_minute: int = 900,
        verbose: bool = True,
        build_repo_from_sentences: bool = False
    ):
        """
        Initialize parallel triple generator.
        
        Args:
            repo: Entity repository containing all entities. If None and build_repo_from_sentences=True,
                  will be built from sentences during generate() call.
            max_workers: Number of parallel workers. Set to 1 for sequential processing (default: 10)
            rate_limit_per_minute: Maximum API calls per minute (default: 900, ignored if sequential)
            verbose: Whether to print progress (default: True)
            build_repo_from_sentences: If True and repo is None, build repo from sentences (default: False)
        """
        self.repo = repo
        self.max_workers = max_workers
        self.sequential = max_workers == 1
        self.rate_limiter = RateLimiter(rate_limit_per_minute) if not self.sequential else None
        self.verbose = verbose
        self.build_repo_from_sentences = build_repo_from_sentences
        
        # Thread-safe shared state
        self.triples: List[Triple] = []
        self.triples_lock = threading.Lock() if not self.sequential else None
        
        # Thread-local storage for NodeGenerator (one per worker thread)
        self._thread_local = threading.local()
        
        # Single NodeGenerator for sequential mode
        self._node_gen = None if not self.sequential else NodeGenerator()
    
    def _get_node_gen(self) -> NodeGenerator:
        """Get or create NodeGenerator for current thread."""
        if self.sequential:
            return self._node_gen
        if not hasattr(self._thread_local, "node_gen"):
            self._thread_local.node_gen = NodeGenerator()
        return self._thread_local.node_gen
    
    def _sentence_entities_to_llm_inventory(self, sentence_obj) -> List[Dict[str, Any]]:
        """Convert sentence entities to LLM inventory format."""
        inv = []
        for ent in sorted(sentence_obj.entities, key=lambda e: e.start):
            # Use ref as primary identifier (fallback to id or ref_short)
            entity_id = ent.ref or ent.id or ent.ref_short
            inv.append({
                "id": entity_id,  # Note: field name kept as "id" for LLM compatibility, but value is now ref
                "label": ent.label,
                "span": [ent.start, ent.end],
                "text": ent.name,
                "ref_short": ent.ref_short,
            })
        return inv
    
    def _get_existing_triples_for_entities(
        self, 
        entities: List[Dict[str, Any]], 
        all_triples: List[Triple]
    ) -> List[Triple]:
        """Find existing triples connected to the given entities."""
        if not all_triples or not entities:
            return []

        entity_ids = {ent.get("id") for ent in entities if ent.get("id")}  # "id" field now contains ref value
        entity_ref_shorts = {ent.get("ref_short") for ent in entities if ent.get("ref_short")}
        entity_names = {ent.get("text", "").lower().strip() for ent in entities if ent.get("text")}

        connected_triples = []
        for triple in all_triples:
            # Use ref as primary identifier (fallback to id or ref_short)
            head_id = getattr(triple.head, "ref", None) or getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None) or ""
            tail_id = getattr(triple.tail, "ref", None) or getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None) or ""
            head_name = getattr(triple.head, "name", None) or getattr(triple.head, "text", None) or ""
            tail_name = getattr(triple.tail, "name", None) or getattr(triple.tail, "text", None) or ""

            head_matches = (
                head_id in entity_ids
                or head_id in entity_ref_shorts
                or (head_name and head_name.lower().strip() in entity_names)
            )
            tail_matches = (
                tail_id in entity_ids
                or tail_id in entity_ref_shorts
                or (tail_name and tail_name.lower().strip() in entity_names)
            )

            if head_matches or tail_matches:
                connected_triples.append(triple)

        return connected_triples
    
    def _process_sentence(self, index: int, sentence) -> Dict[str, Any]:
        """
        Process a single sentence to generate triples.
        
        Returns:
            Dictionary with processing results
        """
        entities = self._sentence_entities_to_llm_inventory(sentence)
        if len(entities) < 2:
            return {
                "index": index,
                "skipped": True,
                "reason": "<2 entities",
                "new": 0,
                "total": None
            }
        
        # Debug: Check if entities exist in repository
        if self.verbose and index <= 3:  # Only print for first few sentences
            missing_entities = []
            for ent_dict in entities:
                entity_id = ent_dict.get("id")
                if entity_id:
                    try:
                        self.repo.get_by_id(entity_id)
                    except KeyError:
                        missing_entities.append(entity_id)
            if missing_entities:
                print(f"  ⚠️  Warning: {len(missing_entities)} entities not found in repository: {missing_entities[:3]}")

        # Rate limit the API call (thread-safe, only for parallel mode)
        if self.rate_limiter:
            self.rate_limiter.acquire()

        # Get current triples for context
        if self.sequential:
            triples_snapshot = list(self.triples)
        else:
            # Snapshot current triples for context (minimize lock hold time)
            with self.triples_lock:
                triples_snapshot = list(self.triples)

        existing_triples = self._get_existing_triples_for_entities(entities, triples_snapshot)

        node_gen = self._get_node_gen()

        # Generate triples with error handling
        try:
            new_triple_items = node_gen.run(
                sentence=sentence.text,
                entities=entities,
                repo=self.repo,
                existing_triples=existing_triples,
            )
        except Exception as e:
            return {
                "index": index,
                "skipped": False,
                "error": str(e),
                "text": sentence.text,
                "entities": len(entities),
                "existing": len(existing_triples),
                "new": 0,
                "total": len(self.triples),
            }

        # Convert results to Triple objects
        to_add: List[Triple] = []
        for item in new_triple_items:
            if isinstance(item, Triple):
                to_add.append(item)
                continue

            if isinstance(item, dict):
                head_id = item.get("head")
                tail_id = item.get("tail")
                relation = item.get("relation")
                if not head_id or not tail_id or not relation:
                    continue

                try:
                    head_ent = self.repo.get_by_id(head_id)
                    tail_ent = self.repo.get_by_id(tail_id)
                    if head_ent and tail_ent:
                        to_add.append(Triple(head=head_ent, relation=relation, tail=tail_ent))
                except KeyError as e:
                    # Entity not found - skip this triple (matching notebook behavior)
                    if self.verbose:
                        print(f"    ⚠️  Skipping triple: entity not found in repository")
                        print(f"       Head ID: {head_id}, Tail ID: {tail_id}")
                        print(f"       Error: {e}")
                        # Debug: Show available entity IDs in repo
                        if hasattr(self.repo, 'getAll'):
                            all_entities = self.repo.getAll()
                            if all_entities:
                                sample_ids = list(all_entities.keys())[:3]
                                print(f"       Sample repo entity IDs: {sample_ids}")
                    continue

        # Append to shared list (thread-safe for parallel, direct for sequential)
        if self.sequential:
            self.triples.extend(to_add)
            total_now = len(self.triples)
        else:
            with self.triples_lock:
                self.triples.extend(to_add)
                total_now = len(self.triples)

        return {
            "index": index,
            "skipped": False,
            "text": sentence.text,
            "entities": len(entities),
            "existing": len(existing_triples),
            "new": len(to_add),
            "total": total_now,
        }
    
    def generate(
        self, 
        sentences: List[Any],
        progress_callback: Optional[callable] = None
    ) -> List[Triple]:
        """
        Generate triples from sentences (parallel or sequential based on max_workers).
        
        Args:
            sentences: List of sentence objects with entities
            progress_callback: Optional callback function(index, result) for progress updates
            
        Returns:
            List of generated Triple objects
        """
        if not sentences:
            return []
        
        # Build entity repository from sentences if requested
        if self.build_repo_from_sentences and self.repo is None:
            self.repo = InMemoryEntityRepository()
            for sent in sentences:
                for ent in sent.entities:
                    self.repo.save(ent)
        
        if self.repo is None:
            raise ValueError("Entity repository is required. Either pass repo to __init__ or set build_repo_from_sentences=True")
        
        # Ensure all entities from sentences are in the repository
        # This is critical: entities must be in repo before being used in inventory
        if not self.build_repo_from_sentences:
            # If repo was provided, ensure all sentence entities are in it
            for sent in sentences:
                for ent in sent.entities:
                    # Get the key that would be used for this entity
                    entity_key = ent.ref or ent.id or ent.ref_short
                    if entity_key:
                        try:
                            # Check if entity exists in repo
                            self.repo.get_by_id(entity_key)
                        except KeyError:
                            # Entity not in repo, add it
                            self.repo.save(ent)
        
        # Reset triples list
        if self.sequential:
            self.triples.clear()
        else:
            with self.triples_lock:
                self.triples.clear()
        
        if self.verbose:
            mode = "sequentially" if self.sequential else f"in parallel with {self.max_workers} workers"
            context_note = " with context" if not self.sequential else ""
            print(f"Processing {len(sentences)} sentences {mode}{context_note}...")
            print("=" * 80)
        
        errors = 0
        skipped = 0
        
        if self.sequential:
            # Sequential processing (matching notebook behavior)
            for i, sent in enumerate(sentences, 1):
                result = self._process_sentence(i, sent)
                
                if result.get("skipped"):
                    skipped += 1
                    if progress_callback:
                        progress_callback(result["index"], result)
                    continue
                
                if result.get("error"):
                    errors += 1
                    if self.verbose:
                        print(f"\n[{i}/{len(sentences)}] Processing: {result['text'][:60]}...")
                        print(f"  Entities: {result['entities']}, Existing triples found: {result['existing']}")
                        print(f"  ❌ Error processing sentence: {result['error']}")
                    if progress_callback:
                        progress_callback(result["index"], result)
                    continue
                
                if self.verbose:
                    text_preview = (
                        result["text"][:60] + "..." 
                        if len(result["text"]) > 60 
                        else result["text"]
                    )
                    print(f"\n[{i}/{len(sentences)}] Processing: {text_preview}")
                    print(f"  Entities: {result['entities']}, Existing triples found: {result['existing']}")
                    print(f"  ✅ Generated {result['new']} new triples (total: {result['total']})")
                
                if progress_callback:
                    progress_callback(result["index"], result)
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(self._process_sentence, i, sent): i 
                    for i, sent in enumerate(sentences, 1)
                }
                
                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        
                        if result.get("skipped"):
                            skipped += 1
                            if progress_callback:
                                progress_callback(result["index"], result)
                            continue
                        
                        if result.get("error"):
                            errors += 1
                            if self.verbose:
                                i = result["index"]
                                text_preview = (
                                    (result["text"][:70] + "...") 
                                    if len(result["text"]) > 70 
                                    else result["text"]
                                )
                                print(f"\n[{i}/{len(sentences)}] {text_preview}")
                                print(f"  ❌ Error processing sentence: {result['error']}")
                            if progress_callback:
                                progress_callback(result["index"], result)
                            continue
                        
                        if self.verbose:
                            i = result["index"]
                            text_preview = (
                                (result["text"][:70] + "...") 
                                if len(result["text"]) > 70 
                                else result["text"]
                            )
                            print(f"\n[{i}/{len(sentences)}] {text_preview}")
                            print(f"  Entities: {result['entities']}, Existing triples: {result['existing']}")
                            print(f"  ✅ Generated {result['new']} new triples (total: {result['total']})")
                        
                        if progress_callback:
                            progress_callback(result["index"], result)
                            
                    except Exception as e:
                        errors += 1
                        if self.verbose:
                            print(f"  ❌ Error in worker: {e}")
        
        if self.verbose:
            print("\n" + "=" * 80)
            print(f"✅ Triple generation complete!")
            print(f"   Total triples generated: {len(self.triples)}")
            if skipped > 0 or errors > 0:
                print(f"   Skipped: {skipped}, Errors: {errors}")
            print("=" * 80)
        
        return self.triples
