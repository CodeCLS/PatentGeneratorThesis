'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import styles from './page.module.css';
import EditPanel from '@/components/EditPanel';
import { injectGraphClickHandler } from '@/lib/graph-interactions';
import { bootstrapPipelineIfNeeded } from '@/lib/pipeline-bootstrap';
import SideNav from '@/components/SideNav';

const STORAGE_KEY_GRAPH_HTML = 'graph_html';
const STORAGE_KEY_GRAPH_TIMESTAMP = 'graph_timestamp';
const STORAGE_KEY_GRAPH_TRIPLES_COUNT = 'graph_triples_count';
const GRAPH_CACHE_DURATION = 5 * 60 * 1000; // 5 minutes
const GRAPH_HTML_ENDPOINT = process.env.NEXT_PUBLIC_GRAPH_ENDPOINT || '/api/graph/html';

interface Entity {
  id: string;
  name: string;
  label: string;
}

interface Triple {
  index: number;
  head: Entity;
  relation: string;
  tail: Entity;
}

export default function GraphPage() {
  const router = useRouter();
  const [mounted, setMounted] = useState(false);
  const [graphHtml, setGraphHtml] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>('');
  const [selectedEntity, setSelectedEntity] = useState<Entity | null>(null);
  const [selectedTriple, setSelectedTriple] = useState<Triple | null>(null);
  const [allEntities, setAllEntities] = useState<Entity[]>([]);
  const [editPanelOpen, setEditPanelOpen] = useState(false);
  const [graphKey, setGraphKey] = useState(0);
  const [chatValue, setChatValue] = useState('');
  const [chatDisabled, setChatDisabled] = useState(false);
  const [chatStatus, setChatStatus] = useState<{ type: 'success' | 'error' | 'info'; message: string } | null>(null);
  const [cypherOutput, setCypherOutput] = useState<string>('');
  const [cypherSummary, setCypherSummary] = useState<Record<string, unknown> | null>(null);
  const [neo4jStats, setNeo4jStats] = useState<{ nodes: number; edges: number; database?: string } | null>(null);
  const [neo4jStatsError, setNeo4jStatsError] = useState<string>('');
  const [neo4jStatsLoading, setNeo4jStatsLoading] = useState(false);
  const [pipelineBootstrapping, setPipelineBootstrapping] = useState(false);
  const [pipelineBootstrapError, setPipelineBootstrapError] = useState<string>('');
  const hasLoadedRef = useRef(false);
  const iframeRef = useRef<HTMLIFrameElement>(null);

  const loadTriples = useCallback(async (): Promise<number | null> => {
    try {
      const response = await fetch('/api/triples', { cache: 'no-store' });
      if (response.ok) {
        const data = await response.json();
        if (data.triples) {
          // Extract unique entities
          const entities: Entity[] = [];
          const entityIds = new Set<string>();
          
          for (const triple of data.triples) {
            if (triple.head && triple.head.id && !entityIds.has(triple.head.id)) {
              entities.push(triple.head);
              entityIds.add(triple.head.id);
            }
            if (triple.tail && triple.tail.id && !entityIds.has(triple.tail.id)) {
              entities.push(triple.tail);
              entityIds.add(triple.tail.id);
            }
          }
          setAllEntities(entities);
          return data.triples.length;
        }
      }
    } catch (err) {
      console.error('Failed to load triples:', err);
    }
    return null;
  }, []);

  const loadGraph = useCallback(async (force = false) => {
    // Check cache first unless forcing a reload
    if (!force && typeof window !== 'undefined') {
      try {
        const storedHtml = localStorage.getItem(STORAGE_KEY_GRAPH_HTML);
        const storedTimestamp = localStorage.getItem(STORAGE_KEY_GRAPH_TIMESTAMP);
        const storedTriplesCount = localStorage.getItem(STORAGE_KEY_GRAPH_TRIPLES_COUNT);
        if (storedHtml && storedTimestamp && storedHtml.length > 0) {
          const timestamp = parseInt(storedTimestamp, 10);
          const now = Date.now();
          if (!isNaN(timestamp) && now - timestamp < GRAPH_CACHE_DURATION) {
            console.log('Using cached graph');
            setGraphHtml(storedHtml);
            setLoading(false);
            const latestCount = await loadTriples();
            const cachedCount = storedTriplesCount ? parseInt(storedTriplesCount, 10) : null;
            if (latestCount !== null && cachedCount !== null && latestCount !== cachedCount) {
              console.log('Cached graph is stale, refetching');
              localStorage.removeItem(STORAGE_KEY_GRAPH_HTML);
              localStorage.removeItem(STORAGE_KEY_GRAPH_TIMESTAMP);
              localStorage.removeItem(STORAGE_KEY_GRAPH_TRIPLES_COUNT);
              await loadGraph(true);
            }
            return; // Use cached version - don't fetch
          } else {
            console.log('Cache expired, clearing');
            localStorage.removeItem(STORAGE_KEY_GRAPH_HTML);
            localStorage.removeItem(STORAGE_KEY_GRAPH_TIMESTAMP);
            localStorage.removeItem(STORAGE_KEY_GRAPH_TRIPLES_COUNT);
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
      const response = await fetch(GRAPH_HTML_ENDPOINT, { cache: 'no-store' });
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
          setGraphKey(prev => prev + 1);
          if (typeof window !== 'undefined') {
            localStorage.setItem(STORAGE_KEY_GRAPH_HTML, html);
            localStorage.setItem(STORAGE_KEY_GRAPH_TIMESTAMP, Date.now().toString());
            console.log('Graph cached');
          }
          const latestCount = await loadTriples();
          if (typeof window !== 'undefined' && latestCount !== null) {
            localStorage.setItem(STORAGE_KEY_GRAPH_TRIPLES_COUNT, latestCount.toString());
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load graph');
    } finally {
      setLoading(false);
    }
  }, [loadTriples]);

  const loadNeo4jStats = useCallback(async () => {
    try {
      setNeo4jStatsLoading(true);
      setNeo4jStatsError('');
      const response = await fetch('/api/neo4j/stats', { cache: 'no-store' });
      const data = await response.json();
      if (!response.ok || data.error) {
        throw new Error(data.error || 'Failed to load Neo4j stats');
      }
      setNeo4jStats({
        nodes: typeof data.nodes === 'number' ? data.nodes : 0,
        edges: typeof data.edges === 'number' ? data.edges : 0,
        database: data.database || '',
      });
    } catch (err) {
      setNeo4jStats(null);
      setNeo4jStatsError(err instanceof Error ? err.message : 'Failed to load Neo4j stats');
    } finally {
      setNeo4jStatsLoading(false);
    }
  }, []);

  const performUpdate = async (data: any) => {
    try {
      const response = await fetch('/api/entities/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to update');
      }

      const payload = await response.json();
      // Refresh the graph after update without blocking the UI
      void loadGraph(true).catch((err) => {
        console.error('Failed to refresh graph after update:', err);
      });
      return payload;
    } catch (err) {
      throw err;
    }
  };

  const handleMergeEntities = async (sourceId: string, targetId: string) => {
    try {
      const response = await fetch('/api/entities/merge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source_id: sourceId, target_id: targetId }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to merge entities');
      }

      const payload = await response.json();
      // Refresh the graph after merge without blocking the UI
      void loadGraph(true).catch((err) => {
        console.error('Failed to refresh graph after merge:', err);
      });
      return payload;
    } catch (err) {
      throw err;
    }
  };

  const handleDelete = async (data: any) => {
    try {
      let endpoint = '';
      let body = {};

      if (data.type === 'entity') {
        endpoint = '/api/entities/delete';
        body = { id: data.id };
      } else if (data.type === 'triple') {
        endpoint = '/api/triples/delete';
        body = { index: data.index };
      }

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Failed to delete ${data.type}`);
      }

      const payload = await response.json();
      // Refresh the graph after deletion without blocking the UI
      void loadGraph(true).catch((err) => {
        console.error('Failed to refresh graph after delete:', err);
      });
      return payload;
    } catch (err) {
      throw err;
    }
  };

  const handleUpdate = async (data: any) => {
    return performUpdate(data);
  };

  const handleCypherSubmit = async () => {
    if (!chatValue.trim() || chatDisabled) {
      return;
    }

    setChatDisabled(true);
    setChatStatus(null);
    setCypherOutput('');
    setCypherSummary(null);
    try {
      const response = await fetch('/api/cypher/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: chatValue }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Failed to process Cypher request');
      }

      if (data.query) {
        setChatValue(data.query);
      }

      if (data.results) {
        setCypherOutput(JSON.stringify(data.results, null, 2));
      }
      if (data.summary) {
        setCypherSummary(data.summary);
      }

      setChatStatus({
        type: data.ran ? 'success' : 'info',
        message: data.message || (data.ran ? 'Cypher query run' : 'Cypher query ready'),
      });
    } catch (err) {
      setChatStatus({
        type: 'error',
        message: err instanceof Error ? err.message : 'Failed to process Cypher request',
      });
    } finally {
      setChatDisabled(false);
    }
  };

  const ensurePipeline = useCallback(async () => {
    setPipelineBootstrapError('');
    setPipelineBootstrapping(true);
    const result = await bootstrapPipelineIfNeeded();
    if (!result.initialized && result.error) {
      setPipelineBootstrapError(result.error);
    }
    setPipelineBootstrapping(false);
    return result.initialized;
  }, []);

  useEffect(() => {
    const messageHandler = (event: MessageEvent) => {
      try {
        const data = event.data;
        if (data.type === 'selectEntity') {
          setSelectedEntity({
            id: data.id,
            name: data.name,
            label: data.label,
          });
          setSelectedTriple(null);
          setEditPanelOpen(true);
        } else if (data.type === 'selectTriple') {
          setSelectedTriple({
            index: data.index,
            head: data.head,
            relation: data.relation,
            tail: data.tail,
          });
          setSelectedEntity(null);
          setEditPanelOpen(true);
        }
      } catch (err) {
        console.error('Failed to process message from iframe:', err);
      }
    };

    window.addEventListener('message', messageHandler);
    return () => window.removeEventListener('message', messageHandler);
  }, []);

  useEffect(() => {
    if (hasLoadedRef.current) return; // Prevent multiple loads
    hasLoadedRef.current = true;
    setMounted(true);

    const initialize = async () => {
      const ready = await ensurePipeline();
      if (ready) {
        // Try to load from cache first - loadGraph will check cache
        loadGraph(false);
        loadNeo4jStats();
      } else {
        setLoading(false);
      }
    };

    initialize();
  }, [ensurePipeline, loadGraph, loadNeo4jStats]);

  const handleGenerateClaims = async () => {
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
        localStorage.setItem(
          'claim_generation_progress',
          JSON.stringify({
            stage: 'error',
            message: errorData.error || 'Failed to generate claims',
            progress: 0,
          })
        );
      }
    } catch (error) {
      console.error('[Graph] Error generating claims:', error);
      localStorage.setItem(
        'claim_generation_progress',
        JSON.stringify({
          stage: 'error',
          message: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
          progress: 0,
        })
      );
    } finally {
      setLoading(false);
      // Navigate to editor after generation completes (or fails)
      router.push('/editor');
    }
  };

  return (
    <div className={styles.layout}>
      <SideNav current="graph" />
      <div className={styles.container}>
        <div className={styles.header}>
          <div className={styles.headerContent}>
            <div className={styles.headerTitle}>
              <h1>Knowledge Graph Visualization</h1>
            </div>
            <div className={styles.headerActions}>
              <div
                className={`${styles.neo4jStatus} ${
                  neo4jStatsError
                    ? styles.neo4jStatusError
                    : neo4jStatsLoading
                    ? styles.neo4jStatusLoading
                    : styles.neo4jStatusOk
                }`}
                role="status"
                aria-live="polite"
              >
                {neo4jStatsLoading && 'Neo4j: checking connection...'}
                {!neo4jStatsLoading && neo4jStatsError && `Neo4j: ${neo4jStatsError}`}
                {!neo4jStatsLoading && !neo4jStatsError && neo4jStats && (
                  <>
                    Neo4j: {neo4jStats.nodes} nodes, {neo4jStats.edges} edges
                    {neo4jStats.database ? ` (${neo4jStats.database})` : ''}
                  </>
                )}
                {!neo4jStatsLoading && !neo4jStatsError && !neo4jStats && 'Neo4j: no data'}
              </div>
              <button
                className={styles.refreshButton}
                onClick={() => {
                  loadGraph(true);
                  loadNeo4jStats();
                }}
                disabled={loading}
              >
                {loading ? 'Loading...' : 'Refresh'}
              </button>
              <button
                className={styles.editButton}
                onClick={() => router.push('/edit')}
                disabled={loading}
              >
                Edit Triples
              </button>
              <button
                className={styles.primaryButton}
                onClick={() => {
                  void handleGenerateClaims();
                }}
                disabled={loading}
              >
                Generate Claims
              </button>
            </div>
        </div>
      </div>

      <div className={styles.graphContainer}>
        {(loading || pipelineBootstrapping) && (
          <div className={styles.loading}>
            <p>{pipelineBootstrapping ? 'Initializing pipeline...' : 'Loading graph...'}</p>
          </div>
        )}
        
        {error && (
          <div className={styles.error}>
            <p>Error: {error}</p>
            <button onClick={() => loadGraph()}>Retry</button>
          </div>
        )}
        
        {!loading && !error && graphHtml && (
          <>
            <iframe
              key={graphKey}
              ref={iframeRef}
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

                    // Inject click handler
                    setTimeout(() => {
                      if (iframe.contentWindow) {
                        injectGraphClickHandler(iframe.contentWindow);
                      }
                    }, 500);
                  }
                } catch (err) {
                  // Cross-origin restrictions may prevent this
                  console.error('Error setting up iframe:', err);
                }
              }}
            />
            {editPanelOpen && (
              <EditPanel
                entity={selectedEntity || undefined}
                triple={selectedTriple || undefined}
                allEntities={allEntities}
                onClose={() => {
                  setEditPanelOpen(false);
                  setSelectedEntity(null);
                  setSelectedTriple(null);
                }}
                onUpdate={handleUpdate}
                onMerge={handleMergeEntities}
                onDelete={handleDelete}
              />
            )}
            
            <div className={styles.chatBarContainer}>
              <div className={styles.chatBarRow}>
                <input
                  type="text"
                  className={styles.chatInput}
                  placeholder="Ask a question about the graph or suggest a change..."
                  value={chatValue}
                  onChange={(e) => setChatValue(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      void handleCypherSubmit();
                    }
                  }}
                  disabled={chatDisabled}
                />
                <button
                  className={styles.chatSendButton}
                  disabled={chatDisabled || !chatValue.trim()}
                  onClick={() => {
                    void handleCypherSubmit();
                  }}
                >
                  Enter
                </button>
              </div>
              {chatStatus && (
                <div
                  className={`${styles.chatStatus} ${
                    chatStatus.type === 'success'
                      ? styles.chatStatusSuccess
                      : chatStatus.type === 'error'
                      ? styles.chatStatusError
                      : styles.chatStatusInfo
                  }`}
                >
                  {chatStatus.message}
                </div>
              )}
              {cypherSummary && (
                <div className={styles.cypherSummary}>
                  {JSON.stringify(cypherSummary)}
                </div>
              )}
              {cypherOutput && (
                <pre className={styles.cypherOutput}>{cypherOutput}</pre>
              )}
            </div>
          </>
        )}
        
        {!loading && !pipelineBootstrapping && !error && !graphHtml && (
          <div className={styles.empty}>
            <p>
              {pipelineBootstrapError
                ? pipelineBootstrapError
                : 'No graph data available. Please initialize the graph validator first.'}
            </p>
            <p className={styles.hint}>
              Upload a file to build the graph, or start the pipeline again.
            </p>
            <button
              className={styles.refreshButton}
              onClick={() => router.push('/upload')}
            >
              Go to Upload
            </button>
          </div>
        )}
      </div>
      </div>
    </div>
  );
}

