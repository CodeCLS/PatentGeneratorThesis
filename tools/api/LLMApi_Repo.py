import models.AnthropicApiModel as AnthropicModel
import langchain
class ChatResult:
    def __init__(self, input : str):
        self._input = input
class LLmApi_Repo:
   def __init__(self, llm_client: LLMModel = AnthropicModel()):
        self.client = llm_client
   def chat(self, message: str, **kwargs) -> ChatResult:
        self.client.send(message)
