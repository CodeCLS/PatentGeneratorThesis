# models/AnthropicApiModel.py
import anthropic
import json
import re
import langsmith
from langsmith import traceable

from langsmith.wrappers import wrap_anthropic
FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
                      re.DOTALL | re.IGNORECASE)

class AnthropicModel:
    def __init__(self, model: str = "claude-haiku-4-5", max_tokens: int = 500, api_key: str | None = None):
        self.client = wrap_anthropic(anthropic.Anthropic(api_key=api_key))
        self.model = model
        self.max_tokens = max_tokens

    def _extract_text(self, resp) -> str:
        parts = []
        for b in getattr(resp, "content", []) or []:
            if getattr(b, "type", None) == "text" and hasattr(b, "text"):
                parts.append(b.text)
        return "".join(parts).strip()

    def _parse_json(self, raw: str) -> dict:
        if not raw:
            raise RuntimeError("LLM returned empty text.")
        m = FENCE_RE.match(raw)      # strip ```json fences if present
        if m:
            raw = m.group(1).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)  # fallback: first {...}
            if m:
                return json.loads(m.group(0))
            raise
    @traceable
    def send(self, message: str) -> dict:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            system="Respond only with a single JSON object. No explanations, no markdown. IT MUST BE A SINGLE JSON OBJECT",
            messages=[{"role": "user", "content": message}],
        )
        raw = self._extract_text(resp)
        obj = self._parse_json(raw)

        # validate expected schema; adjust to your needs
        if not isinstance(obj, dict) or not {"sentence1", "sentence2"} <= set(obj.keys()):
            raise ValueError(f"Unexpected JSON schema: {obj}")
        return obj
