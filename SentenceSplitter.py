

from __future__ import annotations
from typing import List, Tuple
import re
import nltk
from nltk.tokenize import sent_tokenize
from SentenceSplitterLLMTool import SentenceSplitterLLM
import spacy
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage


SPLIT_TOKENS = {"wherein", "thereby", "thereafter", "therefore"}
MAX_LEN = 80  # characters; tweak per your tolerance
class SentenceSplitter:

    def __init__(self,text: str, nlp: spacy.language.Language, llm: BaseChatModel):
        self.text = text
        self.nlp = nlp
        self.llm = llm
        pass
    def split(self):
        pass
    def shorten(self):
        pass
    def llm_splitter(self, post_alg=None):
        splitter = SentenceSplitterLLM(self.llm, max_len=MAX_LEN)
        return splitter.shorten(self.text)

    def algo_splitter(self):
        doc = self.nlp(self.text)
        pre = [s.text.strip() for s in doc.sents if s.text.strip()]

            # Stage B: Clause-level splitting (algorithmic) for patent-y run-ons
        post_alg: List[str] = []
        for s in pre:
            post_alg.extend(self._algo_clause_split(self.nlp, s))

        # If everything looks short enough, we're done
        if all(len(s) <= MAX_LEN for s in post_alg):
            return self._normalize_final_periods(post_alg)
        return post_alg
    def hybrid_splitter(self):
        return self._hybrid_shorten_sentences()

    def _hybrid_shorten_sentences(self) -> List[str]:
        """
        Algorithmic first. If any sentence remains very long, use LLM *indices* fallback.
        The LLM never rewrites text; it only tells us where to split.
        """
        post_alg = self.algo_splitter()

        # Stage C: LLM fallback only for stubborn long sentences
        final = self.llm(post_alg)

        return self._normalize_final_periods(final)
    

    def _algo_clause_split(self, nlp, sentence: str) -> List[str]:
        """
        Heuristics:
        - Split at ';' and ':' if both sides look like clauses.
        - Split at 'wherein/thereby/thereafter/therefore' tokens.
        - Split coordinated clauses when we have clear conj at the root level.
        """
        segs = self._split_on_punct(sentence, seps=(";", ":"))
        out = []
        for seg in segs:
            # Split on discourse markers (preserve markers at start of new clause)
            seg = seg.strip()
            parts = self._split_on_markers(seg, SPLIT_TOKENS)

            for p in parts:
                p = p.strip()
                if not p:
                    continue

                # Attempt dep-based conj split
                doc = nlp(p)
                root = self._get_root(doc)
                if not root:
                    out.append(p)
                    continue

                # Collect top-level conj heads
                cut_positions = []
                for tok in doc:
                    if tok.dep_ == "conj" and tok.head == root:
                        cut_positions.append(tok.idx - doc[0].idx)  # local start

                # Build slices
                if cut_positions:
                    start = 0
                    for cut in sorted(cut_positions):
                        out.append(p[start:cut].strip())
                        start = cut
                    tail = p[start:].strip()
                    if tail:
                        out.append(tail)
                else:
                    out.append(p)

        # Clean & ensure each ends with a period (downstream normalizer will fix)
        return [s.strip() for s in out if s.strip()]
    def _split_on_punct(self, s: str, seps: Tuple[str, ...]) -> List[str]:
        # Split only where both sides are non-empty and next char looks like a letter/number
        pattern = "(" + "|".join(map(re.escape, seps)) + ")"
        parts = []
        last = 0
        for m in re.finditer(pattern, s):
            i = m.start()
            left = s[last:i].strip()
            right = s[m.end():].lstrip()
            if left and right and re.match(r"[A-Za-z0-9<\(]", right):
                parts.append(left)
                last = m.end()
        parts.append(s[last:].strip())
        return parts

    def _split_on_markers(self, s: str, markers: set) -> List[str]:
        # Split at markers as standalone tokens, transfer marker to the beginning of next clause
        tokens = s.split()
        indices = [i for i, t in enumerate(tokens) if t.lower() in markers]
        if not indices:
            return [s]

        parts, start = [], 0
        for idx in indices:
            left = " ".join(tokens[start:idx]).strip()
            if left:
                parts.append(left)
            start = idx
        tail = " ".join(tokens[start:]).strip()
        if tail:
            parts.append(tail)
        return parts

    def _get_root(self,doc):
        #
        for t in doc:
            if t.head == t:  # ROOT
                return t
        return None

    def _normalize_final_periods(self, spans: List[str]) -> List[str]:
        res = []
        for s in spans:
            s = s.strip()
            if not s:
                continue
            if not s.endswith("."):
                s += "."
            res.append(s)
        return res