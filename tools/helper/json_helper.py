# tools/util/json_helper.py
import json, ast, re
from typing import List, Any

FENCE_RE = re.compile(r"^```(?:json|JSON)?\s*(.*?)\s*```$", re.DOTALL)

class JsonHelper:
    @staticmethod
    def _unfence(text: str) -> str:
        m = FENCE_RE.match(text.strip())
        return m.group(1) if m else text

    @staticmethod
    def parse_string_list(data: str) -> List[str]:
        if not data:
            print("NOT DATA")
            return []

        text = data.strip()

        # --- Remove Markdown code fences if present ---
        # ```json ... ```
        if text.startswith("```"):
            # remove starting fence: ```json or ```
            first_line_end = text.find("\n")
            if first_line_end != -1:
                text = text[first_line_end + 1:].strip()

            # remove trailing ```
            if text.endswith("```"):
                text = text[:-3].strip()

        # Attempt to parse
        try:
            obj = json.loads(text)
        except Exception as e:
            print("[parse_string_list] JSON parse exception:", e)
            print("[parse_string_list] RAW:", repr(data))
            print("[parse_string_list] CLEANED:", repr(text))
            return []

        # Must be list
        if not isinstance(obj, list):
            print("NOT INSTANCE")
            return []

        # Keep only strings
        return [s.strip() for s in obj if isinstance(s, str) and s.strip()]
    @staticmethod
    def parse_triple_list(data: str) -> list[dict]:
        """
        Accepts:
          - JSON array of triples: [{"head": "...", "relation": "...", "tail": "..."}]
          - fenced JSON: ```json [ {...}, {...} ] ```
          - python-literal list of dicts
        Returns:
          - list of normalized dicts with keys: head, relation, tail
        """
        text = JsonHelper._unfence(data).strip()
        if not text:
            return []

        # 1) strict JSON first
        try:
            val: Any = json.loads(text)
        except json.JSONDecodeError:
            # 2) safe Python literal
            try:
                val = ast.literal_eval(text)
            except Exception:
                return []

        if not isinstance(val, list):
            return []

        triples: list[dict] = []
        for item in val:
            if not isinstance(item, dict):
                continue
            head = str(item.get("head", "")).strip()
            rel = str(item.get("relation", "")).strip()
            tail = str(item.get("tail", "")).strip()
            if head and rel and tail:
                triples.append({"head": head, "relation": rel, "tail": tail})

        return triples
