
from __future__ import annotations
import os

from typing import List, Tuple
import re
from SentenceSplitter import SentenceSplitter
import spacy
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage

class PatentTextFormatter:
    def __init__(self,text,spacy,client):
        # Initialize with the raw patent description text
        self.text = text
        self.sentence_sequence_pre_formatted = text.split('.')
        self.sentence_sequence_post_formatted = []
        self.spacy = spacy
        self.llm_client = client


    # Split long compound sentences into shorter, self-contained ones
    def shorten_sentences(self):
        return SentenceSplitter(text = self.text, nlp=self.spacy, llm = self.llm_client).llm_splitter()

    # Convert passive voice structures into active voice for clarity
    def convert_passive_to_active(self):
        pass

    # Remove generic boilerplate phrases like “The present invention relates to…”
    def remove_boilerplate_sentences(self):
        pass

    # Standardize figure and reference notations (e.g., FIG.1 → <FIG_1>)
    def standardise_references(self):
        pass

    # Isolate enumerations such as (i), (ii), (iii) into separate sentences
    def isolate_enumerations(self):
        pass

    # Remove redundant connectors such as “furthermore”, “moreover”, “in addition”
    def remove_redundant_connectors(self):
        pass

    # Normalize units and measurement expressions (e.g., 5mm → 5 mm)
    def normalize_units(self):
        pass

    # Remove subjective adjectives that add no technical meaning (e.g., “efficient”, “simple”)
    def remove_subjective_adjectives(self):
        pass

    # Keep or discard quantitative adjectives based on whether they add functional value
    def keep_or_remove_quantitative_adjectives(self):
        pass

    # Normalize synonym variants (e.g., “controller” and “control unit” → “<CONTROL_UNIT>”)
    def normalize_synonym_variants(self):
        pass

    # Standardize spatial and prepositional phrases (e.g., “on top of” → “above”)
    def standardize_prepositional_structures(self):
        pass

    def start_standard_conversion(self):
        self.shorten_sentences()
        self.convert_passive_to_active()
        self.remove_boilerplate_sentences()
        self.standardise_references()
        self.isolate_enumerations()
        self.remove_redundant_connectors()
        self.normalize_units()
        self.remove_subjective_adjectives()
        self.keep_or_remove_quantitative_adjectives()
        self.normalize_synonym_variants()
        self.standardize_prepositional_structures()


    
        
