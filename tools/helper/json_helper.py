# tools/util/json_helper.py
import ast
import json
import re
from typing import Any, List

FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$", re.DOTALL)


class JsonHelper:
    @staticmethod
    def _unfence(text: str) -> str:
        """Remove markdown code fences from text."""
        if not isinstance(text, str):
            return ""
        stripped = text.strip()
        m = FENCE_RE.match(stripped)
        return m.group(1).strip() if m else stripped

    @staticmethod
    def _extract_json_from_text(text: str) -> str:
        """
        Extract the first JSON object or array found in text.

        This is more reliable than regex for nested JSON because it uses a
        bracket/brace balancing scan and ignores braces inside quoted strings.
        """
        if not isinstance(text, str):
            return ""

        s = text.strip()
        if not s:
            return s

        # Find first candidate start: '{' or '['
        start = None
        start_ch = None
        for i, ch in enumerate(s):
            if ch == "{":
                start = i
                start_ch = "{"
                break
            if ch == "[":
                start = i
                start_ch = "["
                break

        if start is None:
            return s

        end_ch = "}" if start_ch == "{" else "]"
        stack = [start_ch]
        in_string = False
        escape = False

        for j in range(start + 1, len(s)):
            ch = s[j]

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue

            if ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    break
                top = stack[-1]
                if (top == "{" and ch == "}") or (top == "[" and ch == "]"):
                    stack.pop()
                    if not stack:
                        return s[start : j + 1].strip()
                else:
                    # Mismatched brackets; stop trying to extract
                    break

        # If we fail to find a balanced end, return original
        return s

    @staticmethod
    def parse_string_list(data: str) -> List[str]:
        if not isinstance(data, str) or not data.strip():
            return []

        text = JsonHelper._unfence(data)

        # Try JSON first, then python literal
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            try:
                obj = ast.literal_eval(text)
            except Exception:
                return []
        except Exception:
            return []

        if not isinstance(obj, list):
            return []

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
        if not isinstance(data, str):
            return []

        text = JsonHelper._unfence(data).strip()
        if not text:
            return []

        # strict JSON first
        try:
            val: Any = json.loads(text)
        except json.JSONDecodeError:
            try:
                val = ast.literal_eval(text)
            except Exception:
                return []
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

    @staticmethod
    def parse_json(data: str) -> Any:
        """
        Parse JSON from a string, handling markdown fences, Python literals, and extraction.

        Accepts:
          - JSON object or array: {"key": "value"} or [1, 2, 3]
          - fenced JSON: ```json {...} ```
          - Text with JSON embedded in it (extracts JSON automatically)
          - Python-literal dict or list

        Returns:
          - Parsed object (dict, list, etc.) or None if parsing fails
        """
        if not isinstance(data, str) or not data.strip():
            return None

        # Step 1: Remove markdown fences
        text = JsonHelper._unfence(data).strip()
        if not text:
            return None

        # Step 2: Extract JSON from surrounding text
        extracted = JsonHelper._extract_json_from_text(text).strip()
        if not extracted:
            return None

        # Step 3: strict JSON first
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            # Step 4: safe Python literal
            try:
                return ast.literal_eval(extracted)
            except Exception:
                return None
        except Exception:
            return None
