import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np
import os
from typing import List, Dict, Any, Optional

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class ClaimEvaluator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Ensure nltk data is downloaded
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')
        
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
        self._model = None
        self.model_name = model_name

    def _get_model(self):
        if self._model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"Error loading SentenceTransformer: {e}")
                return None
        return self._model

    def calculate_cosine_similarity(self, ref_text: str, gen_text: str) -> float:
        """Calculates cosine similarity between two text blocks."""
        model = self._get_model()
        if not model:
            return 0.0
        
        try:
            embeddings = model.encode([ref_text, gen_text])
            # Cosine similarity formula: (A . B) / (||A|| * ||B||)
            # sentence-transformers encode returns normalized vectors usually, 
            # but let's be explicit
            v1 = embeddings[0]
            v2 = embeddings[1]
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = np.dot(v1, v2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return 0.0

    def evaluate(self, reference_claims: List[str], generated_claims: List[str]) -> Dict[str, float]:
        """
        Compares reference claims to generated claims using BLEU, ROUGE, and Cosine metrics.
        reference_claims: list of original patent claims (strings)
        generated_claims: list of generated claims (strings)
        """
        if not reference_claims or not generated_claims:
            return {
                "bleu": 0.0,
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
                "cosine": 0.0
            }

        # Join all claims into a single block of text for aggregate evaluation
        ref_text = " ".join(reference_claims)
        gen_text = " ".join(generated_claims)

        # BLEU Score
        ref_tokens = [nltk.word_tokenize(ref_text.lower())]
        gen_tokens = nltk.word_tokenize(gen_text.lower())
        
        bleu_score = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=self.smoothing)

        # ROUGE Scores
        rouge_scores = self.rouge_scorer.score(ref_text, gen_text)
        
        # Cosine Similarity
        cosine_sim = self.calculate_cosine_similarity(ref_text, gen_text)
        
        return {
            "bleu": float(bleu_score),
            "rouge1": float(rouge_scores['rouge1'].fmeasure),
            "rouge2": float(rouge_scores['rouge2'].fmeasure),
            "rougeL": float(rouge_scores['rougeL'].fmeasure),
            "cosine": float(cosine_sim)
        }
