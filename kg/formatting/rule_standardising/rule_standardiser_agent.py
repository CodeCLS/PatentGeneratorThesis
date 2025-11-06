# tools/agents/rule_enforcer_agent.py
from typing import List, Dict, Any
from tools.api.llm_api_repo import LLmApi_Repo
from tools.helper.json_helper import JsonHelper
from tools.sentence.sentence import Sentence


class RuleStandardiserAgent:
    """
    Enforces basic text rules (e.g., splitting, formatting) using an LLM.
    Keeps a simple iterative structure for future multi-step enforcement.
    """

    def __init__(self, task: str):
        self.task = task
        self.api_repo = LLmApi_Repo()
        self.max_len = 120
        self.max_iter = 3

    @property
    def name(self) -> str:
        return "RuleEnforcerAgent"

    def initial_state(self, text: str) -> Dict[str, Any]:
        return {
            "text": text,
            "sentences": [],
            "done": False
        }

    def build_prompt(self, text: str) -> str:
        return (
            f"{self.task}\n\n"
            f'text = """{text}"""\n\n'
            '["Sentence"]'
        )

    def handle_response(self, state: Dict[str, Any], raw: str) -> Dict[str, Any]:
        sentences = JsonHelper.parse_string_list(raw)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Enforce simple rule: max length
        long_sentences = [s for s in sentences if len(s) > self.max_len]
        if long_sentences:
            state["text"] = " ".join(long_sentences)
            state["done"] = False
        else:
            state["done"] = True

        state["sentences"].extend(sentences)
        return state

    def should_stop(self, state: Dict[str, Any], iteration: int) -> bool:
        return state["done"] or iteration >= self.max_iter

    def result(self, state: Dict[str, Any]) -> List[Sentence]:
        return [Sentence(text=s, index=i) for i, s in enumerate(state["sentences"])]

    def run(self, text: str) -> List[Sentence]:
        state = self.initial_state(text)
        iteration = 0

        while not self.should_stop(state, iteration):
            prompt = self.build_prompt(state["text"])
            raw = self.api_repo.chat(prompt)
            state = self.handle_response(state, raw)
            iteration += 1

        return self.result(state)
