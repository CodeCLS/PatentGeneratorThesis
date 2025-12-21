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
            "Your job:\n"
            "- Identify semantic relations explicitly supported by the sentence.\n"
            "- Use ONLY entity ids from the provided entity list.\n"
            "- 'head' and 'tail' must be EXACT ids from the list.\n"
            "- 'relation' should be a short phrase grounded in the sentence wording.\n"
            "- Do NOT invent entities.\n"
            "- Do NOT invent relations not supported by the sentence.\n"
            "- If no meaningful triple can be formed, return [].\n"
            "- Return ONLY a JSON array with keys: head, relation, tail.\n\n"
            "Important:\n"
            "- Entities may overlap. Prefer the most specific span if needed.\n"
            "- Pronouns may appear; if a pronoun has an entity id (coref), you may use it.\n"
        )

    def run(self, sentence: str, entities: List[Dict[str, Any]]):
        payload = {
            "sentence": sentence,
            "entities": entities,
        }
        prompt = (
            "SENTENCE:\n"
            f"{sentence}\n\n"
            "ENTITIES (JSON):\n"
            f"{json.dumps(entities, ensure_ascii=False)}\n"
        )

        triples = self.agent.run(prompt)
        return triples
