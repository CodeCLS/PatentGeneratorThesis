# tools/agents/sentence_splitter_agent.py
from typing import Any, Dict, List
from tools.api.base.base_recursive_prompter import RecursivePromptingAgent
from tools.api.llm_api_repo import LLmApi_Repo
from tools.sentence.sentence import Sentence
from tools.helper.json_helper import JsonHelper


class NodeGeneratorAgent(RecursivePromptingAgent):
    def __init__(self, task: str):
        super().__init__(task)              # keep base init simple
        self.task = task
        self.api_repo = LLmApi_Repo()
        self.max_iter = 5

    @property
    def name(self) -> str:
        return "NodeGeneratorAgent"

    def initial_state(self, seed: str) -> Dict[str, Any]:
        return {
            "text": seed,
            "nodes": [],        # accepted sentences
            "improvement": None,    # feedback for next turn
            "done": False
        }

    def build_prompt(self, state: Dict[str, Any]) -> str:
        prev = state["nodes"]
        note = f"\nNote: {state['improvement']}" if state.get("improvement") else ""
 
        # Ask STRICTLY for a JSON array of NEW sentences (no prose)
        return (
            f"{self.task}\n\n"
            f"Note: {note}\n\n"
            f'text = """{state["text"]}"""\n\n'
            "Return:\n"
            '["Triple one.", "Triple two.", "..."]')


    def handle_response(self, state: Dict[str, Any], response: str) -> Dict[str, Any]:
        new_parts = JsonHelper.parse_string_list(response)

        if not new_parts:
            # No parse or no content -> likely done
            state["done"] = True
            return state

        # Deduplicate against existing
        existing = set(state["sentences"])
        new_parts = [s for s in new_parts if s not in existing]

        state["sentences"].extend(new_parts)

        # Feedback if any too long
        if any(len(s) > 5000 for s in new_parts):
            state["improvement"] = "One or more triples is too long"
            long = [s for s in new_parts if len(s) > 5000]
            state["text"] = " ".join(long)
            state["done"] = False
        else:
            state["improvement"] = None
            state["done"] = True


        return state

    def should_stop(self, state: Dict[str, Any], iteration: int) -> bool:
        return state.get("done", False) or iteration >= self.max_iter

    def result(self, state: Dict[str, Any]) -> List[Sentence]:
        return [Sentence(text=s, index=i) for i, s in enumerate(state["nodes"])]

    def run(self, seed: str) -> List[Sentence]:
        state = self.initial_state(seed)
        iteration = 0
        print("State")
        while not self.should_stop(state, iteration):
            prompt = self.build_prompt(state)
            raw = self.api_repo.chat(prompt)          # LLM returns text
            state = self.handle_response(state, raw)  # robust JSON parse
            iteration += 1
            print("Raw: " + str(raw) + " State: " + str(state) + " " + "iteration: " + str(iteration))
        return self.result(state)
