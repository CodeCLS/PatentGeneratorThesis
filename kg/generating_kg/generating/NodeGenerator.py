# kg/formatting/splitting/triple_generator.py
import json
from typing import List, Dict, Any

from kg.generating_kg.generating.NodeGeneratorAgent import NodeGeneratorAgent
from tools.sentence.entity import Entity,InMemoryEntityRepository
import json
import re
from typing import Any, Dict, List
class NodeGenerator:
    """
    Extract KG triples from a sentence + entity inventory (no inline markers required).

    Input:
      - sentence: plain text
      - entities: list of dicts with:
          id: str
          label: str
          span: [start,end]  (sentence-local char offsets)
          text: str

    Output:
      JSON array of triples:
        [{"head":"<entity_id>", "relation":"...", "tail":"<entity_id>"}]
    """

    def __init__(self, max_iter: int = 3):
        """
        Initialize NodeGenerator.
        
        Args:
            max_iter: Maximum number of API calls for recursive prompting (default 3 to limit API usage)
        """
        prompt_text = (
            "Task: Extract knowledge graph triples from a sentence.\n"
            "You are given:\n"
            "1) The raw sentence text.\n"
            "2) A JSON list of entities with ids and spans.\n"
            "3) Existing triples connected to entities in this sentence (for context).\n\n"
            "CRITICAL RULES - BE EXTREMELY STRICT:\n"
            "- ONLY extract triples that are EXPLICITLY, CLEARLY, and UNEQUIVOCALLY stated in the sentence.\n"
            "- The relation MUST be directly and unambiguously supported by the sentence wording.\n"
            "- DO NOT infer, assume, or guess relationships.\n"
            "- DO NOT create triples based on common sense or domain knowledge - ONLY what is explicitly stated.\n"
            "- If you have ANY doubt or uncertainty about a relation, DO NOT include it - return [] instead.\n"
            "- If the relation is vague, ambiguous, or could be interpreted multiple ways, DO NOT include it.\n"
            "- Use ONLY entity ids from the provided entity list.\n"
            "- 'head' and 'tail' must be EXACT ids from the list.\n"
            "- 'relation' should be a concise phrase DIRECTLY from the sentence text, not paraphrased.\n"
            "- Do NOT invent entities.\n"
            "- Do NOT invent relations not explicitly stated in the sentence.\n"
            "- Do NOT create triples that are redundant with existing triples (check the existing triples list).\n"
            "- Do NOT create triples that contradict existing triples.\n"
            "- If no meaningful triple can be formed with ABSOLUTE certainty, return [] (empty array).\n"
            "- It is BETTER to return [] than to include uncertain or incorrect triples.\n"
            "- Return ONLY a JSON array with keys: head, relation, tail.\n\n"
            "Quality Standards:\n"
            "- Each triple must be 100% correct and explicitly supported by the sentence.\n"
            "- Relations must be specific and directly stated, not generic or inferred.\n"
            "- Prefer returning [] over creating uncertain triples.\n"
            "- Quality over quantity: fewer correct triples is better than many uncertain ones.\n\n"
            "When to skip a triple:\n"
            "- If the relation is implied but not explicitly stated\n"
            "- If you need to infer or guess the relationship\n"
            "- If the sentence is ambiguous about the relationship\n"
            "- If the entities' relationship is unclear from the sentence\n"
            "- If you're not completely certain the triple is correct\n\n"
            "Important:\n"
            "- Entities may overlap. Prefer the most specific span if needed.\n"
            "- Pronouns may appear; if a pronoun has an entity id (coref), you may use it ONLY if the relation is explicit.\n"
            "- Review existing triples to avoid duplicates and contradictions.\n"
            "- When in doubt, leave it out - return [] rather than risk incorrect triples.\n\n"
            "Example:\n"
            "Sentence: The display device is for appreciation regarding pseudo space.\n"
            'Entities: [{"id":"E1","text":"display device"},{"id":"E2","text":"pseudo space"}]\n'
            'Existing triples: []\n'
            'Output: [{"head":"E1","relation":"is for appreciation regarding","tail":"E2"}]\n'
        )
        
        self.agent = NodeGeneratorAgent(prompt_text, max_iter=max_iter)


    def run(self, sentence: str, entities: List[Dict[str, Any]], repo, existing_triples: List = None):
        """
        Extract triples from a sentence with context from existing triples.
        
        Args:
            sentence: The sentence text
            entities: List of entity dicts with id, text, etc.
            repo: Entity repository
            existing_triples: List of existing Triple objects connected to entities in this sentence
        
        Returns:
            List of triple dicts: [{"head": "...", "relation": "...", "tail": "..."}]
        """
        print("sentence:", sentence, entities)

        # Minimal guard: need at least 2 entities to form head+tail
        if not sentence or not sentence.strip() or not entities or len(entities) < 2:
            return []

        # Get entity IDs in this sentence
        entity_ids = {ent.get("id") for ent in entities if ent.get("id")}
        
        # Format existing triples for context
        existing_triples_context = []
        if existing_triples:
            for triple in existing_triples:
                # Check if triple involves any entity in this sentence
                head_id = getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None) or str(triple.head)
                tail_id = getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None) or str(triple.tail)
                
                # Check if head or tail matches any entity in current sentence
                if head_id in entity_ids or tail_id in entity_ids:
                    head_name = getattr(triple.head, "name", None) or getattr(triple.head, "text", None) or str(triple.head)
                    tail_name = getattr(triple.tail, "name", None) or getattr(triple.tail, "text", None) or str(triple.tail)
                    existing_triples_context.append({
                        "head": head_id,
                        "head_name": head_name,
                        "relation": triple.relation,
                        "tail": tail_id,
                        "tail_name": tail_name,
                    })

        payload = {
            "sentence": sentence, 
            "entities": entities,
            "existing_triples": existing_triples_context
        }

        prompt = (
            "Extract entity-to-entity triples from the sentence.\n\n"
            "CRITICAL: ONLY extract triples that are EXPLICITLY and UNEQUIVOCALLY stated.\n"
            "- If you have ANY doubt or uncertainty, DO NOT create the triple - return [] instead.\n"
            "- DO NOT infer, assume, or guess relationships.\n"
            "- DO NOT create triples based on common sense or domain knowledge.\n"
            "- If the relation is vague, ambiguous, or unclear, DO NOT include it.\n"
            "- It is BETTER to return [] than to include uncertain or incorrect triples.\n\n"
            "You MUST use ONLY the provided entity 'id' values for head/tail.\n"
            "IMPORTANT: Review the existing_triples list to avoid duplicates and contradictions.\n"
            "Only create NEW triples that are explicitly stated and not already present.\n"
            "When in doubt, leave it out - return [] rather than risk incorrect triples.\n\n"
            "Return ONLY JSON (no text, no code fences) as a JSON array like:\n"
            '[{"head":"<entity_id>","relation":"<relation>","tail":"<entity_id>"}]\n'
            'If no certain triples exist, return: []\n\n'
            "INPUT:\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

        raw = self.agent.run(prompt, repo)
        print("raw:", type(raw), str(raw)[:300])

        # If agent already returns Python list
        if isinstance(raw, list):
            return raw

        # If agent returns a string, parse JSON
        if isinstance(raw, str):
            txt = raw.strip()
            # strip ```json ... ``` if the model adds it anyway
            txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt)
            try:
                parsed = json.loads(txt)
                return parsed if isinstance(parsed, list) else []
            except Exception as e:
                print("json parse error:", e, "text:", txt[:300])
                return []

        return []
