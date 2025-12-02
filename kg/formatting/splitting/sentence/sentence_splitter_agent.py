# tools/agents/sentence_splitter_agent.py
from typing import Any, Dict, List
from tools.api.base.base_recursive_prompter import RecursivePromptingAgent
from tools.api.llm_api_repo import LLmApi_Repo
from tools.sentence.sentence import Sentence
from tools.helper.json_helper import JsonHelper


class SentenceSplitterAgent(RecursivePromptingAgent):
    def __init__(self, task: str):
        super().__init__(task)              # keep base init simple
        self.task = task
        self.api_repo = LLmApi_Repo()
        self.max_iter = 5

    @property
    def name(self) -> str:
        return "SentenceSplitter"

    def initial_state(self, seed: str) -> Dict[str, Any]:
        return {
            "text": seed,
            "sentences": [],        # accepted sentences
            "improvement": None,    # feedback for next turn
            "done": False
        }

    def build_prompt(self, state: Dict[str, Any]) -> str:
     

        # Ask STRICTLY for a JSON array of NEW sentences (no prose)
        return (
            f"{self.task}\n\n"
            f'text = """{state["text"]}"""\n\n'
            "Return:\n"
            '["Sentence one.", "Sentence two.", "..."]')


    def handle_response(self, state: Dict[str, Any], response: str) -> Dict[str, Any]:
        new_parts = JsonHelper.parse_string_list(response)
        state["done"] = True
        state["sentences"].extend(new_parts)

    


        return state

    def should_stop(self, state: Dict[str, Any], iteration: int) -> bool:
        return state.get("done", False) or iteration >= self.max_iter

    def result(self, state: Dict[str, Any]) -> List[Sentence]:
        return [Sentence(text=s, index=i) for i, s in enumerate(state["sentences"])]

    def run(self, seed: str) -> List[Sentence]:
        state = self.initial_state(seed)
        iteration = 0
        while not self.should_stop(state, iteration):
            prompt = self.build_prompt(state)
            raw = self.api_repo.chat(prompt)          # LLM returns text
            state = self.handle_response(state, raw)  # robust JSON parse
            iteration += 1

   
        return self.result(state)
