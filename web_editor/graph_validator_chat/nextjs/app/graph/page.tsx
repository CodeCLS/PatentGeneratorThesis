'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import styles from './page.module.css';

const STORAGE_KEY_GRAPH_HTML = 'graph_html';
const STORAGE_KEY_GRAPH_TIMESTAMP = 'graph_timestamp';
const GRAPH_CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

export default function GraphPage() {
  const router = useRouter();
  const [mounted, setMounted] = useState(false);
  const [graphHtml, setGraphHtml] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>('');
  const hasLoadedRef = useRef(false);

  const loadGraph = useCallback(async (force = false) => {
    // Check cache first unless forcing a reload
    if (!force && typeof window !== 'undefined') {
      try {
        const storedHtml = localStorage.getItem(STORAGE_KEY_GRAPH_HTML);
        const storedTimestamp = localStorage.getItem(STORAGE_KEY_GRAPH_TIMESTAMP);
        if (storedHtml && storedTimestamp && storedHtml.length > 0) {
          const timestamp = parseInt(storedTimestamp, 10);
          const now = Date.now();
          if (!isNaN(timestamp) && now - timestamp < GRAPH_CACHE_DURATION) {
            console.log('Using cached graph');
            setGraphHtml(storedHtml);
            setLoading(false);
            return; // Use cached version - don't fetch
          } else {
            console.log('Cache expired, clearing');
            localStorage.removeItem(STORAGE_KEY_GRAPH_HTML);
            localStorage.removeItem(STORAGE_KEY_GRAPH_TIMESTAMP);
          }
        }
      } catch (e) {
        console.error('Failed to check cache:', e);
      }
    }
    
    // Only fetch if cache check failed or force is true
    console.log('Fetching graph from API', force ? '(forced)' : '');
    try {
      setLoading(true);
      setError('');
      const response = await fetch('/api/graph/html');
      if (!response.ok) {
        throw new Error('Failed to load graph');
      }
      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        const html = data.html || '';
        if (html) {
          setGraphHtml(html);
          if (typeof window !== 'undefined') {
            localStorage.setItem(STORAGE_KEY_GRAPH_HTML, html);
            localStorage.setItem(STORAGE_KEY_GRAPH_TIMESTAMP, Date.now().toString());
            console.log('Graph cached');
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load graph');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (hasLoadedRef.current) return; // Prevent multiple loads
    hasLoadedRef.current = true;
    setMounted(true);
    
    // Try to load from cache first - loadGraph will check cache
    loadGraph(false);
  }, [loadGraph]);

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <button
          className={styles.navButton}
          onClick={() => router.push('/')}
          title="Go to Chat"
        >
          ←
        </button>
        <div className={styles.headerContent}>
          <h1>Knowledge Graph Visualization</h1>
          <button className={styles.refreshButton} onClick={() => loadGraph(true)} disabled={loading}>
            {loading ? 'Loading...' : 'Refresh'}
          </button>
        </div>
        <button
          className={styles.navButton}
          onClick={async () => {
            // Generate claims before navigating to editor
            try {
              setLoading(true);
              // Store initial progress BEFORE starting generation
              const initialProgress = {
                stage: 'planning',
                message: 'Planning claim structure...',
                progress: 0,
              };
              localStorage.setItem('claim_generation_progress', JSON.stringify(initialProgress));
              console.log('[Graph] Set initial progress:', initialProgress);
              
              // Navigate to editor immediately so user can see progress
              router.push('/editor');
              
              // Get stored similarity threshold or use default
              const storedThreshold = localStorage.getItem('similarity_threshold');
              const similarityThreshold = storedThreshold ? parseFloat(storedThreshold) : 0.3;
              
              // Start generation in background
              const response = await fetch('/api/claims/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  num_independent: 3,
                  num_dependent_per_independent: 2,
                  similarity_threshold: similarityThreshold,
                }),
              });
              
              if (response.ok) {
                const data = await response.json();
                console.log(`[Graph] Generated ${data.num_claims} claims`);
                console.log(`[Graph] Response data:`, data);
                // Store claims in localStorage for editor page
                if (data.claims && data.claims.length > 0) {
                  localStorage.setItem('generated_claims', JSON.stringify(data.claims));
                  console.log(`[Graph] Stored ${data.claims.length} claims in localStorage`);
                  console.log(`[Graph] First claim preview:`, data.claims[0]);
                } else {
                  console.warn(`[Graph] No claims in response! data.claims:`, data.claims);
                }
                // Store final progress
                if (data.progress) {
                  localStorage.setItem('claim_generation_progress', JSON.stringify(data.progress));
                  console.log('[Graph] Set final progress:', data.progress);
                }
              } else {
                console.error('[Graph] Failed to generate claims');
                const errorData = await response.json().catch(() => ({}));
                // Store error progress
                localStorage.setItem('claim_generation_progress', JSON.stringify({
                  stage: 'error',
                  message: errorData.error || 'Failed to generate claims',
                  progress: 0,
                }));
              }
            } catch (error) {
              console.error('[Graph] Error generating claims:', error);
              localStorage.setItem('claim_generation_progress', JSON.stringify({
                stage: 'error',
                message: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
                progress: 0,
              }));
            } finally {
              setLoading(false);
              // Navigate to editor after generation completes (or fails)
              router.push('/editor');
            }
          }}
          title="Generate Claims & Go to Editor"
          disabled={loading}
        >
          →
        </button>
      </div>

      <div className={styles.graphContainer}>
        {loading && (
          <div className={styles.loading}>
            <p>Loading graph...</p>
          </div>
        )}
        
        {error && (
          <div className={styles.error}>
            <p>Error: {error}</p>
            <button onClick={loadGraph}>Retry</button>
          </div>
        )}
        
        {!loading && !error && graphHtml && (
          <iframe
            srcDoc={graphHtml}
            className={styles.graphIframe}
            title="Knowledge Graph"
            sandbox="allow-scripts allow-same-origin"
            onLoad={(e) => {
              // Prevent iframe from navigating parent window
              try {
                const iframe = e.target as HTMLIFrameElement;
                if (iframe.contentWindow) {
                  iframe.contentWindow.addEventListener('beforeunload', (event) => {
                    event.preventDefault();
                  });
                }
              } catch (err) {
                // Cross-origin restrictions may prevent this
              }
            }}
          />
        )}
        
        {!loading && !error && !graphHtml && (
          <div className={styles.empty}>
            <p>No graph data available. Please initialize the graph validator first.</p>
            <p className={styles.hint}>Go back to the chat page to initialize the graph.</p>
          </div>
        )}
      </div>
    </div>
  );
}

