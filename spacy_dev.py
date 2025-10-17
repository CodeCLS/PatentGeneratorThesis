from __future__ import annotations
import itertools
from collections import defaultdict
from typing import List, Dict, Set, Tuple

import spacy
import networkx as nx
import matplotlib.pyplot as plt

# pip install -U spacy
# python -m spacy download en_core_web_sm

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")

# Process whole documents
text = ("""I am a doctor. I am a dog. I am stupid. I am ok.
        
        
        """)
doc = nlp(text)

sentences = doc.sents

# Analyze syntax
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Nouns:", [token.lemma_ for token in doc if token.pos_ in ["NOUN", "PROPN"]])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])
print("Adjectives:", [token.lemma_ for token in doc if token.pos_ == "ADJ"])
print("Adverbs:", [token.lemma_ for token in doc if token.pos_ == "ADV"])
print("Prepositions:", [token.lemma_ for token in doc if token.pos_ == "ADP"])
print("Conjunctions:", [token.lemma_ for token in doc if token.pos_ in ["CCONJ", "SCONJ"]])
print("Determiners:", [token.lemma_ for token in doc if token.pos_ == "DET"])
print("Quantifiers:", [token.text for token in doc if token.tag_ in ["CD", "PDT"]])  # numbers & pre-determiners

doc = nlp(text)

# Split into sentences
sentences = list(doc.sents)

# Create a list to store separate graphs
graphs = []
G = nx.DiGraph()  # directed graph (dependency direction)

for i, sentence in enumerate(sentences, 1):
    
    for token in sentence:
        # Add each token as a node with part-of-speech and dependency info
        G.add_node(token.i, label= token.text, pos=token.pos_, dep=token.dep_)
        
        # Add edges only within this sentence (token.head in same sentence)
        if token.head in sentence and token.head != token:
            G.add_edge(token.head.i, token.i, dep=token.dep_)
    
    graphs.append(G)
    
# Optional: visualize each sentence graph
plt.figure(figsize=(6, 4))
labels = nx.get_node_attributes(G, "label")
nx.draw(
    G,
    with_labels=True,
    node_size=500,
    labels=labels,
    node_color="lightblue",
    font_size=10,
    arrows=True,
)
plt.title(f"Sentence {i} Dependency Graph")
plt.show()

print(f"Generated {len(graphs)} separate sentence graphs.")


# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)