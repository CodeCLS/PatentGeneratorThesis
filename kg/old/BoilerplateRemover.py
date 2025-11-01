

from __future__ import annotations
from typing import List, Tuple
import re
from SentenceSplitterAgent import SentenceSplitterAgent
import spacy
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from BoilerplateRemoverAgent import BoilerplateRemoverAgent

MAX_LEN = 80  # characters; tweak per your tolerance
class BoilerplateRemover:

    def __init__(self,sentences: str, nlp: spacy.language.Language, llm: BaseChatModel):
        self.sentences = sentences
        self.nlp = nlp
        self.llm = llm
        pass
    def remove(self, ):
        return BoilerplateRemoverAgent(self.llm).removeBoilerplate(self.sentences)
        

