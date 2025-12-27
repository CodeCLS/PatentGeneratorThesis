import os

from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of fast language models",
        }
    ],
    model="llama-3.3-70b-versatile",
)

print(chat_completion.choices[0].message.content)

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

class GPTOSSModel(LLMModel):
    def __init__(self, model: str = "openai/gpt-oss-20b",system_prompt :str = "Respond only with a JSON array of strings. No explanations or markdown.", max_tokens: int = 500, api_key: str | None = None):
        self.client =Groq(
            api_key=os.environ.get("GROQ_API_KEY"
                                   ))
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

    @property
    def name(self) -> str:
        return f"gemini:{self.model}"


    def send(self, message: str) -> List[str]:

        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": message,
                }
            ],
            model=self.model,
        )



        return chat_completion.choices[0].message.content
