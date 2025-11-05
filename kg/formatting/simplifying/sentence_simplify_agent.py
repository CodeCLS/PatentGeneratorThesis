# tools/agents/sentence_simplify_agent.py
import re
from typing import Any, Dict, List
from tools.api.base.base_recursive_prompter import RecursivePromptingAgent
from tools.api.llm_api_repo import LLmApi_Repo
from tools.sentence.sentence import Sentence
from tools.helper.json_helper import JsonHelper

# --- helpers -------------------------------------------------
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_/]{1,}")

def _extract_must_keep_terms(text: str) -> List[str]:
    """
    Keep a light-touch set of domain-ish tokens:
    - hyphenated
    - contains '/' or '_'
    - Capitalized/CamelCase
    """
    terms = set()
    for tok in _WORD.findall(text):
        if "-" in tok or "/" in tok or "_" in tok or tok[0].isupper():
            terms.add(tok)
    return sorted(terms)

def _is_one_sentence(s: str) -> bool:
    return len(re.findall(r"[.!?]", s)) <= 1

def _includes_all(hay: str, needles: List[str]) -> bool:
    low = hay.lower()
    return all(n.lower() in low for n in needles) if needles else True

def _same(a: str, b: str) -> bool:
    return a.strip().lower() == b.strip().lower()
# --- agent ---------------------------------------------------
class SentenceSimplifyAgent(RecursivePromptingAgent):
    """
    Simplify a SINGLE sentence’s wording without losing information.
    Ultra-minimal prompting; minimal post-checks.
    Output must be a JSON array with one string: ["..."].
    """
    def __init__(self, task: str):
        super().__init__(task)
        self.task = task
        self.api_repo = LLmApi_Repo()
        self.max_iter = 3

    @property
    def name(self) -> str:
        return "SentenceSimplify"

    def initial_state(self, seed: str) -> Dict[str, Any]:
        return {
            "task": self.task,
            "text": seed,
            "best": None,
            "improvement": None,
            "done": False,
            "must_keep": _extract_must_keep_terms(seed),
        }

    def build_prompt(self, state: Dict[str, Any]) -> str:
        note = f"\nNote: {state['improvement']}" if state.get("improvement") else ""

        # IMPORTANT: no mention of length; focus on simplification while preserving info
        return (
            f"{self.task}{note}\n\n"
            f'text = """{state["text"]}"""\n\n'
            "Return:\n"
            '["One simplified sentence."]'
        )

    def handle_response(self, state: Dict[str, Any], response: str) -> Dict[str, Any]:
        cands = JsonHelper.parse_string_list(response) or []
        cand = cands[0].strip() if cands and isinstance(cands[0], str) else ""

        if not cand:
            state["improvement"] = 'Return a JSON array with one simplified sentence, e.g. ["..."].'
            return state
            
            # NEW: reject identical output so the model must simplify something
        if _same(cand, state["text"]):
            state["text"] = cand
            state["improvement"] = "Replace at least 2 non-domain words with simpler synonyms; do not change key terms."
            return state

        state["best"] = cand
        state["improvement"] = None
        state["done"] = True
        return state


    def should_stop(self, state: Dict[str, Any], iteration: int) -> bool:
        return state.get("done", False) or iteration >= self.max_iter

    def result(self, state: Dict[str, Any]) -> List[Sentence]:
        final_text = state["best"] or state["original"]
        return [Sentence(text=final_text, index=0)]

    def run(self, seed: str) -> List[Sentence]:
        state = self.initial_state(seed)
        iteration = 0
        while not self.should_stop(state, iteration):
            prompt = self.build_prompt(state)
            raw = self.api_repo.chat(prompt)
            state = self.handle_response(state, raw)
            iteration += 1
            print("Raw: " + str(raw) + " State: " + str(state) + " " + "iteration: " + str(iteration))

        return self.result(state)
