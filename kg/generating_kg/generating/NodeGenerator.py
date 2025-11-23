# kg/formatting/splitting/triple_generator.py
from kg.generating_kg.generating.NodeGeneratorAgent import NodeGeneratorAgent
import json


class NodeGenerator:
    """
    Extracts knowledge graph triples from a sentence with inline entity markers.

    Input format:
      - Entities are annotated as: [_entityId_label]
      - Example: "[_caleb123_person] is going to [school123_building]"

    Output format:
      - A JSON array of triples, each as an object:
        { "head": "...", "relation": "...", "tail": "..." }

      - Example output for the above sentence:
        [
          {
            "head": "_caleb123_person",
            "relation": "going to",
            "tail": "school123_building"
          }
        ]
    """

    def __init__(self):
        self.agent = NodeGeneratorAgent(
            "Task: Read the following sentence and extract knowledge graph triples.\n"
            "The sentence contains entities marked as: [_entityId_label].\n"
            "Entities can be persons, devices, components, actions, etc.\n\n"
            "Your job is to detect semantic relations between these entities and\n"
            "represent them as triples (head, relation, tail).\n\n"
            "Rules:\n"
            "- Use ONLY entities that appear in the text as [_entityId_label].\n"
            "- 'head' and 'tail' must be the entity identifiers WITHOUT brackets,\n"
            "  exactly as written inside, e.g. \"_caleb123_person\" or \"school123_building\".\n"
            "  between head and tail, using words from the sentence.\n"
            "- Do NOT invent entities or relations that are not supported by the text.\n"
            "- If multiple clear relations exist, output multiple triples.\n"
            "- If no meaningful triple can be formed, return an empty list [].\n"
            "- Return ONLY a JSON array of objects with keys 'head', 'relation', 'tail'.\n\n"
            "Input example:\n"
            "  \"[_caleb123_person is going to [school123_building]]\"\n"
            "Valid output:\n"
            "[\n"
            "  {\"head\": \"_caleb123_person\", \"relation\": \"going to\", \"tail\": \"school123_building\"}\n"
            "]\n"
        )


    def run(self, sentence: str):
        """
        Generate KG triples from a single annotated sentence,
        parse the LLM JSON response, and return readable triples.
        """
        raw = self.agent.run(sentence)

        print("RAW TRIPLE: " + raw)

        # 1. Parse JSON returned by the LLM
        try:
            triples_json = json.loads(raw)
        except Exception:
            raise ValueError(f"TripleGenerator: LLM did not return valid JSON: {raw}")

        # 2. Convert each triple object to a readable tuple format
        readable = []
        for t in triples_json:
            head = t.get("head", "")
            rel = t.get("relation", "")
            tail = t.get("tail", "")
            readable.append(f"({head}, {rel}, {tail})")

        return readable
