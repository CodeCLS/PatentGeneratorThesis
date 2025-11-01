import spacy
import claucy

class SentenceSplitter():
    def __init__(self,nlp = spacy.load("en_core_web_sm")):
        self.nlp = nlp
        claucy.add_to_pipe(nlp)

    def commit(self,text):
        doc = self.nlp("AE died in Princeton in 1955.")
        propositions = doc._.clauses[0].to_propositions(as_text=True)   
        return propositions   


if __name__ == "__main__":
    splitter = SentenceSplitter()
    print(splitter.commit("""
A cat, hearing that the birds in a certain aviary were ailing dressed himself up as a physician, 
and, taking his cane and a bag of instruments becoming his profession, went to call on them.


                    """))
