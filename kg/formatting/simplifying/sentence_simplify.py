# kg/formatting/simplifying/sentence_simplifier.py
import spacy

class SentenceSimplifier:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")


    def run(self, sentence: str):
        doc = self.nlp(sentence) 
        lemmas = [token.lemma_ for token in doc]
        return lemmas
