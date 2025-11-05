# models/AnthropicApiModel.py
import anthropic
import json
import re
from typing import Any, List
from langsmith import traceable
from langsmith.wrappers import wrap_anthropic
from tools.api.base.base_llm_model import LLMModel

# Non-greedy capture inside fences:
FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)

# Non-greedy fallbacks:
FIRST_ARRAY_RE  = re.compile(r"\[(?:.|\n)*?\]")
FIRST_OBJECT_RE = re.compile(r"\{(?:.|\n)*?\}")

class AnthropicModel(LLMModel):
    def __init__(self, model: str = "claude-haiku-4-5",system_prompt :str = "Respond only with a JSON array of strings. No explanations or markdown.", max_tokens: int = 500, api_key: str | None = None):
        self.client = wrap_anthropic(anthropic.Anthropic(api_key=api_key))
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    def _extract_text(self, resp) -> str:
        parts = []
        for b in getattr(resp, "content", []) or []:
            if getattr(b, "type", None) == "text" and hasattr(b, "text"):
                parts.append(b.text)
        return "".join(parts).strip()

    @traceable
    def send(self, message: str) -> List[str]:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            system=self.system_prompt,
            messages=[{"role": "user", "content": message}],
        )

        raw_text = self._extract_text(resp)
        return raw_text
