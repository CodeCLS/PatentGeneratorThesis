# tools/api/LLMApi_Repo.py
from __future__ import annotations

from uuid import uuid4

from tools.api.llm_models.anthropic_model import AnthropicModel  # adjust import if your paths differ
from tools.api.llm_models.gemini_model import GeminiModel  # adjust import if your paths differ
from tools.api.base.base_llm_model import LLMModel


class ChatResult:
    def __init__(self, data: dict):
        self._data = data


class LLmApi_Repo:
        def __init__(self, llm_client: LLMModel = GeminiModel()):
            self.client = llm_client
        def chat(self, message: str, **kwargs) -> dict:
            return self.client.send(message)

if __name__ == "__main__":
    repo = LLmApi_Repo()
    print(repo.chat("I am a dog and I am a cat."))
