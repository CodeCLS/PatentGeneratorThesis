

from __future__ import annotations
from typing import List, Tuple
import re
from SentenceSplitterAgent import SentenceSplitterAgent
import spacy
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from BoilerplateRemoverAgent import BoilerplateRemoverAgent
from SentenceStandardiserAgent import SentenceStandardiserAgent

MAX_LEN = 80  # characters; tweak per your tolerance
class SentenceStandardiser:

    def __init__(self, nlp: spacy.language.Language, llm: BaseChatModel):
        self.nlp = nlp
        self.llm = llm
        pass
    def standardise(self, sentences):
        return SentenceStandardiserAgent(self.llm).standardise(sentences)
        

