from langchain.text_splitter import RecursiveCharacterTextSplitter
import spacy

class KR_generator():

    def __init__(self, input =  """
    LangChain is a framework for developing applications powered by large language models.
    It provides utilities for connecting LLMs with data sources and managing context windows effectively.
    The RecursiveCharacterTextSplitter helps split long texts into chunks suitable for LLM input.
    """):
        self.nlp = spacy.load("en_core_web_sm")

        doc = self.nlp(input)
        print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
        print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

        # Find named entities, phrases and concepts
        for entity in doc.ents:
            print(entity.text, entity.label_)



KR_generator()

# pip install spacy
# python -m spacy download en_core_web_sm

import spacy
nlp = spacy.load("en_core_web_sm")

def collapse_phrasal_verb(verb):
    # include particles (e.g., "pick up") and negation ("do not")
    parts = [verb.lemma_]
    neg = None
    for child in verb.children:
        if child.dep_ == "prt":       # particle
            parts.append(child.text)
        if child.dep_ == "neg":       # negation
            neg = child.text
    verb_text = " ".join(parts)
    if neg:
        verb_text = f"{neg} {verb_text}"
    return verb_text

def complement_for_copula(head):
    # For "He is alive", spaCy parses "alive" as head with cop "is"
    # Return predicate adjective or attribute tokens attached to head
    # e.g., acomp/attr, plus prepositional complements if present
    comp_tokens = []
    for child in head.children:
        if child.dep_ in ("acomp", "attr", "xcomp"):
            comp_tokens.append(child)
    # if none, the head itself is the complement (e.g., "alive")
    if not comp_tokens:
        comp_tokens = [head]
    # include prepositional phrases hanging off the complement
    span = []
    for tok in comp_tokens:
        subtree = list(tok.subtree)
        subtree.sort(key=lambda t: t.i)
        span.extend(subtree)
    span = [t.text for t in sorted(set(span), key=lambda t: t.i)]
    return " ".join(span)

def extract_triples(text):
    doc = nlp(text)
    triples = []

    for sent in doc.sents:
        # Case 1: normal verb predicates
        for token in sent:
            if token.pos_ == "VERB" and token.dep_ != "aux":
                # subject (explicit or implicit)
                subj = None
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass", "csubj"):
                        subj = " ".join([t.text for t in child.subtree])
                        break
                # Imperative: no subject → underscore
                if subj is None:
                    # heuristic: if sentence root is verb and no subject, treat as imperative
                    if token == sent.root:
                        subj = "_"

                # object or complement
                comp = None
                # direct/indirect/prepositional object
                objs = []
                for child in token.children:
                    if child.dep_ in ("dobj", "obj", "iobj"):
                        objs.append(" ".join([t.text for t in child.subtree]))
                    if child.dep_ == "prep":
                        # include prep phrase like "walk home" (home often as advmod/pobj)
                        pobj = [t for t in child.children if t.dep_ == "pobj"]
                        if pobj:
                            objs.append(child.text + " " + " ".join([t.text for t in pobj[0].subtree]))
                # adverbial place like "home"
                adv_places = [c for c in token.children if c.dep_ in ("advmod", "obl") and c.pos_ in ("NOUN","PROPN","ADV")]
                for ap in adv_places:
                    objs.append(" ".join([t.text for t in ap.subtree]))

                if objs:
                    comp = "; ".join(objs)

                verb = collapse_phrasal_verb(token)
                if subj or comp:
                    triples.append([subj or "_", verb, comp or "_"])

        # Case 2: copular predicates (e.g., "He is alive")
        head = sent.root
        # copula pattern: head has a child with dep_ == "cop"
        if any(child.dep_ == "cop" for child in head.children):
            subj = None
            for child in head.children:
                if child.dep_ in ("nsubj", "nsubjpass", "csubj"):
                    subj = " ".join([t.text for t in child.subtree])
                    break
            cop = [c for c in head.children if c.dep_ == "cop"]
            if cop:
                verb = collapse_phrasal_verb(cop[0])
            else:
                verb = "is"
            comp = complement_for_copula(head)
            triples.append([subj or "_", verb, comp or "_"])

    # Deduplicate while preserving order
    seen = set()
    dedup = []
    for t in triples:
        tup = tuple(t)
        if tup not in seen:
            seen.add(tup)
            dedup.append(t)
    return dedup

text = """A water tank for appreciation provided with a
water storage, a water pipe, an air bubble generating
member, wherein said water storage is provided with a
water inlet port and an opening portion and said opening
portion is so provided that the convection can be generated in said water tank for appreciation by the liquid
current from a water tank through said opening portion,
and said water storage is installed upwardly of a water
tank for appreciation and one end of said water pipe is
connected to an inlet water port of a water storage and
the other end is so installed that it is in the liquid when
the liquid is filled in a water tank for appreciation and an
air bubble generating member is so installed that it can
lead the air bubble inside of a water pipe in the peripheral portion of said other end of said water pipe and a
display device for appreciation using said water tank for
appreciation are used."""
print(extract_triples(text))



        

