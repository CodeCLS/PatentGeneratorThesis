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
        """
        Accepts:
          - JSON array: ["a","b"]
          - fenced JSON: ```json [ "a", "b" ] ```
          - python-literal list: ['a','b']
          - dict wrapper: {"sentences": [...]}
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

        # normalize shapes
        if isinstance(val, dict) and "sentences" in val:
            val = val["sentences"]

        if isinstance(val, list):
            out = [s.strip() for s in val if isinstance(s, str) and s.strip()]
            return out

        return []
