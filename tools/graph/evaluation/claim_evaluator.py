import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np
import os

class ClaimEvaluator:
    def __init__(self):
        # Ensure nltk data is downloaded
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')
        
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1

    def evaluate(self, reference_claims, generated_claims):
        """
        Compares reference claims to generated claims using BLEU and ROUGE metrics.
        reference_claims: list of original patent claims (strings)
        generated_claims: list of generated claims (strings)
        """
        if not reference_claims or not generated_claims:
            return {
                "bleu": 0.0,
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0
            }

        # Join all claims into a single block of text for aggregate evaluation
        ref_text = " ".join(reference_claims)
        gen_text = " ".join(generated_claims)

        # BLEU Score
        # reference is a list of lists of tokens
        # hypothesis is a list of tokens
        ref_tokens = [nltk.word_tokenize(ref_text.lower())]
        gen_tokens = nltk.word_tokenize(gen_text.lower())
        
        bleu_score = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=self.smoothing)

        # ROUGE Scores
        rouge_scores = self.rouge_scorer.score(ref_text, gen_text)
        
        return {
            "bleu": float(bleu_score),
            "rouge1": float(rouge_scores['rouge1'].fmeasure),
            "rouge2": float(rouge_scores['rouge2'].fmeasure),
            "rougeL": float(rouge_scores['rougeL'].fmeasure)
        }
