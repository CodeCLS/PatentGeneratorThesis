"""
SpaCy extensions for patent processing pipeline.
"""
from spacy.tokens import Doc, Span

# Span extensions
if not Span.has_extension("norm_label"):
    Span.set_extension("norm_label", default=None)

if not Span.has_extension("kb_id"):
    Span.set_extension("kb_id", default=None)

# Doc extensions
if not Doc.has_extension("coref_clusters"):
    Doc.set_extension("coref_clusters", default=[])