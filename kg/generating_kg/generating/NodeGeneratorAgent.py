# tools/agents/sentence_splitter_agent.py
from typing import Any, Dict, List, Tuple
from tools.api.base.base_recursive_prompter import RecursivePromptingAgent
from tools.api.llm_api_repo import LLmApi_Repo
from tools.sentence.sentence import Sentence
from tools.helper.json_helper import JsonHelper


class NodeGeneratorAgent(RecursivePromptingAgent):
    def __init__(self, task: str):
        super().__init__(task)
        self.task = task
        self.api_repo = LLmApi_Repo()
        self.max_iter = 5

    @property
    def name(self) -> str:
        return "NodeGeneratorAgent"

    def initial_state(self, seed: str) -> Dict[str, Any]:
        return {
            "text": seed,
            "triples": [],      # will be: List[Dict[str, str]]
            "improvement": None,
            "done": False
        }

    def build_prompt(self, state: Dict[str, Any]) -> str:
        note = f"\nNote: {state['improvement']}" if state.get("improvement") else ""

        # IMPORTANT: Ask the LLM for a JSON array   of triple objects
        return (
            f"{self.task}\n"
            f"{note}\n\n"
            f'text = """{state["text"]}"""\n\n'
            "Return **only** a valid JSON array of objects in this exact format:\n"
            '[\n'
            '  {"head": "ID_of_head", "relation": "relation phrase", "tail": "ID_of_tail"}\n'
            ']\n'
        )

    def handle_response(self, state: Dict[str, Any], response: str) -> Dict[str, Any]:

        # parse_triple_list should return List[Dict[str, str]]
        content = JsonHelper.parse_triple_list(response)

        if content:
            # content is already a list of triples → extend our list
            state["triples"].extend(content)
            state["done"] = True   # stop after first successful parse
        # if content is empty, you might want to set an improvement message here
        return state

    def should_stop(self, state: Dict[str, Any], iteration: int) -> bool:
        return state.get("done", False) or iteration >= self.max_iter

   
    def result(self, state: Dict[str, Any]) -> List[tuple]:
        """
        Converts the internal list of triple dicts into
        a list of (head, relation, tail) tuples.
        """
        triples: List[tuple] = []

        for t in state["triples"]:
            head = t.get("head")
            rel = t.get("relation")
            tail = t.get("tail")

            if head and rel and tail:
                triples.append((head, rel, tail))

        return triples

    def run(self, seed: str) -> List[Sentence]:
        state = self.initial_state(seed)
        iteration = 0
        while not self.should_stop(state, iteration):
            prompt = self.build_prompt(state)
            raw = self.api_repo.chat(prompt)
            state = self.handle_response(state, raw)
            iteration += 1
        return self.result(state)
