# models/AnthropicApiModel.py
import anthropic
import json
import re
from google import genai
from typing import Any, List
from tools.api.base.base_llm_model import LLMModel

# Non-greedy capture inside fences:
FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)

# Non-greedy fallbacks:
FIRST_ARRAY_RE  = re.compile(r"\[(?:.|\n)*?\]")
FIRST_OBJECT_RE = re.compile(r"\{(?:.|\n)*?\}")

class GeminiModel(LLMModel):
    def __init__(self, model: str = "gemini-2.0-flash-lite",system_prompt :str = "Respond only with a JSON array of strings. No explanations or markdown.", max_tokens: int = 500, api_key: str | None = None):
        self.client = genai.Client()
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

    @property
    def name(self) -> str:
        return f"gemini:{self.model}"


    def send(self, message: str) -> List[str]:

        response = self.client.models.generate_content(
        model=self.model,
        contents=message)



        return response.text
