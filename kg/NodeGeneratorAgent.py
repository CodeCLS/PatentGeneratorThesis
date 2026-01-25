from __future__ import annotations

from typing import Any, Dict, List
from tools.api.base.base_recursive_prompter import RecursivePromptingAgent
from tools.api.llm_api_repo import LLmApi_Repo
from tools.helper.json_helper import JsonHelper

from tools.graph.data.Triple import Triple
from tools.sentence.entity import Entity,InMemoryEntityRepository


class NodeGeneratorAgent(RecursivePromptingAgent):
    def __init__(self, task: str, max_iter: int = 3):
        super().__init__(task)
        self.task = task
        self.api_repo = LLmApi_Repo()
        self.max_iter = max_iter  # Reduced from 5 to 3 to limit API calls

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

    def _to_triple(self, obj, repo: InMemoryEntityRepository):
        head_id = obj.get("head"); rel = obj.get("relation"); tail_id = obj.get("tail")
        if not head_id or not rel or not tail_id:
            return None

        try:
            head = repo.get_by_id(head_id)
            tail = repo.get_by_id(tail_id)
            return Triple(head=head, relation=str(rel), tail=tail)
        except KeyError as e:
            print("TOTRIPLERROR" + str(e))
            # fallback: keep raw ids so you don't lose the triple
            return Triple(head=head_id, relation=str(rel), tail=tail_id)



    def handle_response(self, state: Dict[str, Any], response: str,repo: InMemoryEntityRepository) -> Dict[str, Any]:

        parsed = JsonHelper.parse_triple_list(response)  # expected: List[Dict[str,str]]
        print("Parsed" + str(parsed))

        triples: List[Triple] = []
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    t = self._to_triple(item,repo)
                    if t is not None:
                        triples.append(t)

        if triples:
            state["triples"].extend(triples)  # store Triple objects
            state["done"] = True
        else:
            state["improvement"] = (
                "Your previous response was empty or not valid JSON. "
                "Return ONLY JSON (no markdown fences, no commentary). "
                "If no certain triples exist, return [] (empty array). "
                "Remember: It is better to return [] than to create uncertain triples."
            )
            state["done"] = False

        return state

    def should_stop(self, state: Dict[str, Any], iteration: int) -> bool:
        return state.get("done", False) or iteration >= self.max_iter

    def result(self, state: Dict[str, Any]) -> List[Triple]:
        out: List[Triple] = []
        for t in state.get("triples", []):
            if isinstance(t, Triple):
                out.append(t)
        return out

    def run(self, seed: str,repo: InMemoryEntityRepository) -> List[Triple]:
        state = self.initial_state(seed)
        iteration = 0
        while not self.should_stop(state, iteration):
            prompt = self.build_prompt(state)
            raw = self.api_repo.chat(prompt)
            state = self.handle_response(state, raw,repo)
            iteration += 1
        return self.result(state)