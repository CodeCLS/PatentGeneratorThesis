# tools/api/LLMApi_Repo.py
from __future__ import annotations
import os
from uuid import uuid4
from langfuse import Langfuse
from langfuse import observe
from tools.api.llm_models.anthropic_model import AnthropicModel  # adjust import if your paths differ
from tools.api.llm_models.gemini_model import GeminiModel  # adjust import if your paths differ
from tools.api.llm_models.gpt_oss_model import GPTOSSModel  # adjust import if your paths differ

from tools.api.base.base_llm_model import LLMModel

from tools.api.llm_models.deepseek_model import DeepSeekModel
class ChatResult:
    def __init__(self, data: dict):
        self._data = data

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host="https://cloud.langfuse.com",  # Optional, default shown
)

class LLmApi_Repo:
        def __init__(self, llm_client: LLMModel = GPTOSSModel(), test_mode: bool = False):
            self.test_mode = test_mode
            self.client = llm_client
        @observe
        def chat(self, message: str, **kwargs) -> dict:
            if self.test_mode:
                return {"response": "Dummy response for testing.", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
            return self.client.send(message)

if __name__ == "__main__":
    repo = LLmApi_Repo()
    print(repo.chat("I am a dog and I am a cat."))
