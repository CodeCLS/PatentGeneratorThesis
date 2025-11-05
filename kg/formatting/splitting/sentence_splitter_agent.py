from typing import Any, Dict, List
from tools.api.base.base_recursive_prompter import RecursivePromptingAgent
from tools.api.llm_api_repo import LLmApi_Repo
from tools.sentence.sentence import Sentence


class SentenceSplitterAgent(RecursivePromptingAgent):


    def __init__(self,seed : str):
        super().__init__(seed)
        self.seed = seed
        self.api_repo = LLmApi_Repo()

    @property
    def name(self) -> str:
        return "SentenceSplitter"

    def initial_state(self, seed: str) -> Dict[str, Any]:
        return {
            "text": seed,
            "sentences": [],          # already accepted sentences
            "improvement": None,      # last feedback string (if any)
            "done": False
        }

    def build_prompt(self, state: Dict[str, Any]) -> str:
        prev = state["sentences"]
        note = f"\nNOTE: {state['improvement']}" if state.get("improvement") else ""
        already = "\n".join(f"- {s}" for s in prev[-10:])  # show only the last few to keep prompt short

        return (
            self.seed + 
            f"{note}\n\n"
            "Already extracted sentences (do NOT repeat these; only return NEW ones):\n"
            f"{already if already else '(none)'}\n\n"
            "Text:\n"
            f"{state['text']}\n\n"
            "Output:\n"
            "- Return one sentence per line.\n"
            "- Only include sentences that are NOT already listed above."
        )

    def handle_response(self, state: Dict[str, Any], response: str) -> Dict[str, Any]:
        lines = [s.strip() for s in response.split("\n") if s.strip()]
        existing = set(state["sentences"])
        new_parts = [s for s in lines if s not in existing]

        state["sentences"].extend(new_parts)

        # Simple length check feedback
        if any(len(s) > 120 for s in new_parts):
            state["improvement"] = "One or more returned sentences are still > 120 chars. Split further."
        else:
            # If nothing new was added, we’re likely done
            if not new_parts:
                state["done"] = True
            else:
                state["improvement"] = None

        return state

    def should_stop(self, state: Dict[str, Any], iteration: int) -> bool:
        # Stop if marked done or after a few tries (baseline safety)
        return state.get("done", False) or iteration >= 3

    def result(self, state: Dict[str, Any]) -> List[Sentence]:
        return [Sentence(text=s) for s in state["sentences"]]

    def run(self) -> List[Sentence]:
        state = self.initial_state(self.seed)
        iteration = 0
        while not self.should_stop(state, iteration):
            prompt = self.build_prompt(state)
            raw = self.api_repo.chat(prompt)
            state = self.handle_response(state, raw)
            iteration += 1
        return self.result(state)
