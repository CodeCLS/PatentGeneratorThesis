'use client';

import { useState } from 'react';
import SideNav from '@/components/SideNav';
import styles from '../page.module.css';
import evalStyles from './evaluation.module.css';

interface EvaluationResults {
  bleu: number;
  rouge1: number;
  rouge2: number;
  rougeL: number;
  cosine: number;
}

export default function EvaluationPage() {
  const [referenceText, setReferenceText] = useState('');
  const [generatedText, setGeneratedText] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<EvaluationResults | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleEvaluate = async () => {
    if (!referenceText.trim() || !generatedText.trim()) {
      setError('Both reference and generated text are required');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await fetch('/api/evaluate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          reference_text: referenceText,
          generated_text: generatedText,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to evaluate text');
      }

      const data = await response.json();
      setResults(data);
    } catch (err: any) {
      setError(err.message || 'An error occurred during evaluation');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.layout}>
      <SideNav current="evaluation" />
      <main className={styles.content}>
        <div className={evalStyles.container}>
          <header className={styles.header}>
            <h1>Text Evaluation</h1>
            <div className={styles.status}>
              Metrics: BLEU, ROUGE, Cosine Similarity
            </div>
          </header>

          <div className={evalStyles.inputSection}>
            <div className={evalStyles.textAreaContainer}>
              <h3>Reference Text</h3>
              <textarea
                className={evalStyles.textarea}
                placeholder="Paste the reference patent claim or description here..."
                value={referenceText}
                onChange={(e) => setReferenceText(e.target.value)}
                rows={10}
              />
            </div>

            <div className={evalStyles.textAreaContainer}>
              <h3>Generated Text</h3>
              <textarea
                className={evalStyles.textarea}
                placeholder="Paste the AI-generated claim or description here..."
                value={generatedText}
                onChange={(e) => setGeneratedText(e.target.value)}
                rows={10}
              />
            </div>
          </div>

          <div className={evalStyles.actionSection}>
            <button
              className={styles.sendButton}
              onClick={handleEvaluate}
              disabled={loading}
            >
              {loading ? 'Evaluating...' : 'Run Evaluation'}
            </button>
          </div>

          {error && (
            <div className={evalStyles.error}>
              {error}
            </div>
          )}

          {results && (
            <div className={evalStyles.resultsGrid}>
              <div className={evalStyles.resultCard}>
                <span className={evalStyles.resultLabel}>BLEU Score</span>
                <span className={evalStyles.resultValue}>{(results.bleu * 100).toFixed(2)}%</span>
                <div className={evalStyles.progressBar}>
                  <div className={evalStyles.progressFill} style={{ width: `${results.bleu * 100}%` }}></div>
                </div>
              </div>

              <div className={evalStyles.resultCard}>
                <span className={evalStyles.resultLabel}>ROUGE-L</span>
                <span className={evalStyles.resultValue}>{(results.rougeL * 100).toFixed(2)}%</span>
                <div className={evalStyles.progressBar}>
                  <div className={evalStyles.progressFill} style={{ width: `${results.rougeL * 100}%` }}></div>
                </div>
              </div>

              <div className={evalStyles.resultCard}>
                <span className={evalStyles.resultLabel}>Cosine Similarity</span>
                <span className={evalStyles.resultValue}>{(results.cosine * 100).toFixed(2)}%</span>
                <div className={evalStyles.progressBar}>
                  <div className={evalStyles.progressFill} style={{ width: `${results.cosine * 100}%` }}></div>
                </div>
              </div>

              <div className={evalStyles.resultCard}>
                <span className={evalStyles.resultLabel}>ROUGE-1</span>
                <span className={evalStyles.resultValue}>{(results.rouge1 * 100).toFixed(2)}%</span>
                <div className={evalStyles.progressBar}>
                  <div className={evalStyles.progressFill} style={{ width: `${results.rouge1 * 100}%` }}></div>
                </div>
              </div>

              <div className={evalStyles.resultCard}>
                <span className={evalStyles.resultLabel}>ROUGE-2</span>
                <span className={evalStyles.resultValue}>{(results.rouge2 * 100).toFixed(2)}%</span>
                <div className={evalStyles.progressBar}>
                  <div className={evalStyles.progressFill} style={{ width: `${results.rouge2 * 100}%` }}></div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
