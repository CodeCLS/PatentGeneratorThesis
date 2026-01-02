"""
Extracts only sentences that refer to the invention from patent descriptions.
Excludes introduction, examples, background, prior art, etc.
"""
from __future__ import annotations

from typing import Any, Dict, List
from tools.api.base.base_recursive_prompter import RecursivePromptingAgent
from tools.api.llm_api_repo import LLmApi_Repo
from tools.helper.json_helper import JsonHelper
from tools.sentence.sentence import Sentence


class InventionSentenceExtractor(RecursivePromptingAgent):
    """
    Extracts sentences that describe the invention itself, excluding:
    - Introduction/background sections
    - Prior art descriptions
    - Examples and use cases
    - General statements
    - References to other documents
    """
    
    def __init__(self, task: str = None):
        default_task = (
            "You are an expert patent analyst. Your task is to extract ONLY sentences "
            "that directly describe the INVENTION itself from a patent description.\n\n"
            "INCLUDE sentences that:\n"
            "- Describe the structure, components, or features of the invention\n"
            "- Explain how the invention works or functions\n"
            "- Describe the technical implementation or configuration\n"
            "- Specify relationships between components\n"
            "- Describe the method or process steps of the invention\n\n"
            "EXCLUDE sentences that:\n"
            "- Are introductory or background information\n"
            "- Describe prior art or existing solutions\n"
            "- Give examples or use cases (e.g., 'For example...', 'In one embodiment...')\n"
            "- Are general statements about the field\n"
            "- Reference other documents or patents\n"
            "- Describe advantages or benefits without technical details\n"
            "- Are headings, titles, or section markers\n"
            "- Are too vague or general (e.g., 'The present invention relates to...')\n\n"
            "Return ONLY a valid JSON array of sentence strings that describe the invention.\n"
            "Do not include any commentary, explanations, or markdown formatting.\n"
            "Example format: [\"The device comprises a first component.\", \"The method includes processing the data.\"]"
        )
        super().__init__(task or default_task)
        self.task = task or default_task
        self.api_repo = LLmApi_Repo()
        self.max_iter = 5
    
    @property
    def name(self) -> str:
        return "InventionSentenceExtractor"
    
    def initial_state(self, seed: str) -> Dict[str, Any]:
        return {
            "text": seed,
            "sentences": [],
            "improvement": None,
            "done": False
        }
    
    def build_prompt(self, state: Dict[str, Any]) -> str:
        note = f"\n\nNote: {state['improvement']}" if state.get("improvement") else ""
        return (
            f"{self.task}\n\n"
            f"Patent Description:\n"
            f'"""\n{state["text"]}\n"""\n\n'
            f"{note}\n\n"
            "Return ONLY a valid JSON array of strings (sentences), nothing else.\n"
            "No markdown fences, no commentary, just the JSON array.\n"
            'Example: ["Sentence one about the invention.", "Sentence two about the invention."]'
        )
    
    def handle_response(self, state: Dict[str, Any], response: str) -> Dict[str, Any]:
        # Parse JSON array of sentences
        parsed = JsonHelper.parse_string_list(response)
        
        if isinstance(parsed, list) and len(parsed) > 0:
            # Filter out empty strings and very short fragments
            valid_sentences = [
                s.strip() for s in parsed 
                if isinstance(s, str) and len(s.strip()) > 10
            ]
            
            if valid_sentences:
                state["sentences"] = valid_sentences
                state["done"] = True
            else:
                state["improvement"] = (
                    "Your response contained no valid sentences. "
                    "Return a JSON array with at least one sentence that describes the invention."
                )
                state["done"] = False
        else:
            state["improvement"] = (
                "Your response was not a valid JSON array of strings. "
                "Return ONLY a JSON array like: [\"sentence 1\", \"sentence 2\"] "
                "with no markdown fences or commentary."
            )
            state["done"] = False
        
        return state
    
    def should_stop(self, state: Dict[str, Any], iteration: int) -> bool:
        return state.get("done", False) or iteration >= self.max_iter
    
    def result(self, state: Dict[str, Any]) -> List[Sentence]:
        """Return list of Sentence objects."""
        return [
            Sentence(text=s, index=i) 
            for i, s in enumerate(state.get("sentences", []))
        ]
    
    def run(self, seed: str) -> List[Sentence]:
        """
        Extract invention-related sentences from patent description.
        
        Args:
            seed: Full patent description text
            
        Returns:
            List of Sentence objects containing only invention-related sentences
        """
        state = self.initial_state(seed)
        iteration = 0
        
        while not self.should_stop(state, iteration):
            prompt = self.build_prompt(state)
            raw = self.api_repo.chat(prompt)
            
            # Handle different response formats (string or dict)
            if isinstance(raw, dict):
                # Extract text from dict response
                response_text = raw.get("content", raw.get("text", raw.get("message", "")))
                if not response_text and "choices" in raw:
                    # OpenAI-style format
                    response_text = raw["choices"][0].get("message", {}).get("content", "")
            else:
                response_text = str(raw) if raw else ""
            
            state = self.handle_response(state, response_text)
            iteration += 1
        
        return self.result(state)

