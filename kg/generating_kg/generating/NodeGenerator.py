# kg/formatting/splitting/triple_generator.py
import json
from typing import List, Dict, Any

from kg.generating_kg.generating.NodeGeneratorAgent import NodeGeneratorAgent


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

    def __init__(self):
        self.agent = NodeGeneratorAgent(
            "Task: Extract knowledge graph triples from a sentence.\n"
            "You are given:\n"
            "1) The raw sentence text.\n"
            "2) A JSON list of entities with ids and spans.\n\n"
            "Rules:\n"
            "- Identify relations explicitly supported by the sentence.\n"
            "- Use ONLY entity ids from the provided entity list.\n"
            "- 'head' and 'tail' must be EXACT ids from the list.\n"
            "- 'relation' should be a short phrase grounded in the sentence wording.\n"
            "- Do NOT invent entities.\n"
            "- Do NOT invent relations not supported by the sentence.\n"
            "- If no meaningful triple can be formed, return [].\n"
            "- Return ONLY a JSON array with keys: head, relation, tail.\n\n"
            "Important:\n"
            "- Entities may overlap. Prefer the most specific span if needed.\n"
            "- Pronouns may appear; if a pronoun has an entity id (coref), you may use it.\n\n"
            "Example:\n"
            "Sentence: The display device is for appreciation regarding pseudo space.\n"
            'Entities: [{"id":"E1","text":"display device"},{"id":"E2","text":"pseudo space"}]\n'
            'Output: [{"head":"E1","relation":"is for appreciation regarding","tail":"E2"}]\n'
        )

    def run(self, sentence: str, entities: List[Dict[str, Any]]):
        print("sentence: " + sentence  + " " + str(entities))
        # ---- Cheap short-circuits (saves tokens + avoids guaranteed []) ----
        if not sentence or not sentence.strip():
            return []
        if not entities or len(entities) < 2:
            # Can't form a head+tail triple with <2 entities (given your "IDs only" rule)
            return []

        # Optional: dedupe entities by id to reduce prompt size
        seen = set()
        deduped_entities = []
        for e in entities:
            eid = e.get("id")
            if not eid or eid in seen:
                continue
            seen.add(eid)
            deduped_entities.append(e)

        if len(deduped_entities) < 2:
            return []

        # Structured input is easier for LLMs to follow than freeform blocks
        payload = {"sentence": sentence, "entities": deduped_entities}

        prompt = (
            "INPUT JSON (use ONLY these entity ids for head/tail):\n"
            f"{json.dumps(payload, ensure_ascii=False)}\n\n"
            "Return ONLY a valid JSON array of triples like:\n"
            '[{"head":"<entity_id>","relation":"<relation phrase>","tail":"<entity_id>"}]\n'
        )

        return self.agent.run(prompt)
