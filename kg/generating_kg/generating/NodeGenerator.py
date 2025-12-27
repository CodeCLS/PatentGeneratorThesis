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


    def run(self, sentence: str, entities: List[Dict[str, Any]], repo):
        print("sentence:", sentence, entities)

        # Minimal guard: need at least 2 entities to form head+tail
        if not sentence or not sentence.strip() or not entities or len(entities) < 2:
            return []

        payload = {"sentence": sentence, "entities": entities}

        prompt = (
            "Extract entity-to-entity triples from the sentence.\n"
            "You MUST use ONLY the provided entity 'id' values for head/tail.\n"
            "Return ONLY JSON (no text, no code fences) as a JSON array like:\n"
            '[{"head":"<entity_id>","relation":"<relation>","tail":"<entity_id>"}]\n\n'
            "INPUT:\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
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
