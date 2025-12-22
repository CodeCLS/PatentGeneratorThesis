from typing import Any, Dict, List, Tuple
from tools.api.base.base_recursive_prompter import RecursivePromptingAgent
from tools.api.llm_api_repo import LLmApi_Repo
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
        return {"text": seed, "triples": [], "improvement": None, "done": False}

    def build_prompt(self, state: Dict[str, Any]) -> str:
        note = f"\nNote: {state['improvement']}" if state.get("improvement") else ""
        return (
            f"{self.task}\n"
            f"{note}\n\n"
            f"{state['text']}\n\n"
            "Return ONLY a valid JSON array of objects in this exact format:\n"
            '[{"head":"<entity_id>","relation":"<relation phrase>","tail":"<entity_id>"}]\n'
        )

    def handle_response(self, state: Dict[str, Any], response: str) -> Dict[str, Any]:
        print("RAW:", response)

        content = JsonHelper.parse_triple_list(response)
        print("CONTENT:", content)

        if content and len(content) > 0:
            state["triples"].extend(content)
            state["done"] = True
        else:
            # keep trying
            state["improvement"] = (
                "Your previous response was empty or not valid JSON. "
                "Return ONLY JSON (no markdown fences, no commentary). "
                "If truly no relation exists, return []."
            )
            state["done"] = False

        return state

    def should_stop(self, state: Dict[str, Any], iteration: int) -> bool:
        return state.get("done", False) or iteration >= self.max_iter

    from typing import Any, Dict, List

    def result(self, state: Dict[str, Any]) -> List[Dict[str, str]]:
        # Return dicts, not tuples
        out: List[Dict[str, str]] = []
        for t in state["triples"]:
            head = t.get("head")
            rel = t.get("relation")
            tail = t.get("tail")
            if head and rel and tail:
                out.append({"head": head, "relation": rel, "tail": tail})
        return out



    def run(self, seed: str) -> List[Tuple[str, str, str]]:
        state = self.initial_state(seed)
        iteration = 0
        while not self.should_stop(state, iteration):
            prompt = self.build_prompt(state)
            raw = self.api_repo.chat(prompt)
            state = self.handle_response(state, raw)
            iteration += 1
        return self.result(state)
