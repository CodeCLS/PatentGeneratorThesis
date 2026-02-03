'use client';

import { useState, useRef, useEffect } from 'react';
import styles from './page.module.css';
import { bootstrapPipelineIfNeeded } from '@/lib/pipeline-bootstrap';
import SideNav from '@/components/SideNav';

const STORAGE_KEY_EDITOR_CONTENT = 'editor_content';
const STORAGE_KEY_CLAIMS = 'generated_claims';
const STORAGE_KEY_CLAIM_PROGRESS = 'claim_generation_progress';
const STORAGE_KEY_SIMILARITY_THRESHOLD = 'similarity_threshold';

interface ProgressUpdate {
  stage: string;
  message: string;
  progress: number;
  current_claim?: number;
  total_claims?: number;
  num_claims?: number;
}

export default function EditorPage() {
  const editorRef = useRef<HTMLDivElement>(null);
  const [mounted, setMounted] = useState(false);
  const [content, setContent] = useState<string>('');
  const [claims, setClaims] = useState<any[]>([]);
  const [progress, setProgress] = useState<ProgressUpdate | null>(null);
  const [selectedClaimNumber, setSelectedClaimNumber] = useState<number | null>(null);
  const [showTriplesPanel, setShowTriplesPanel] = useState<boolean>(true);
  const [expandedPromptClaim, setExpandedPromptClaim] = useState<number | null>(null);
  const [similarityThreshold, setSimilarityThreshold] = useState<number>(0.3);
  const [pipelineBootstrapError, setPipelineBootstrapError] = useState<string>('');
  const progressCheckIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const isRegeneratingRef = useRef<boolean>(false);

  useEffect(() => {
    const ensurePipeline = async () => {
      const result = await bootstrapPipelineIfNeeded();
      if (!result.initialized && result.error) {
        setPipelineBootstrapError(result.error);
      }
    };

    ensurePipeline();

    setMounted(true);
    
    // Load similarity threshold from localStorage
    const storedThreshold = localStorage.getItem(STORAGE_KEY_SIMILARITY_THRESHOLD);
    if (storedThreshold) {
      const threshold = parseFloat(storedThreshold);
      // Only use stored value if it's valid and greater than 0 (0 means not set)
      if (!isNaN(threshold) && threshold > 0 && threshold <= 1) {
        setSimilarityThreshold(threshold);
      } else {
        // Invalid or 0 value - set default and save it
        setSimilarityThreshold(0.3);
        localStorage.setItem(STORAGE_KEY_SIMILARITY_THRESHOLD, '0.3');
      }
    } else {
      // No stored value - set default and save it
      setSimilarityThreshold(0.3);
      localStorage.setItem(STORAGE_KEY_SIMILARITY_THRESHOLD, '0.3');
    }
    
    // Check for progress updates
    const checkProgress = async () => {
      try {
        const response = await fetch('/api/claims/progress');
        if (response.ok) {
          const data = await response.json();
          console.log('[Editor] Progress check:', data);
          if (data.success && data.progress) {
            const progressUpdate = data.progress;
            console.log('[Editor] Progress check result:', progressUpdate);
            
            // Only show notification if there's active progress (not idle)
            // Don't overwrite progress if we're in the middle of a regeneration
            if (progressUpdate.stage && progressUpdate.stage !== 'idle') {
              console.log('[Editor] Setting active progress:', progressUpdate);
              setProgress(progressUpdate);
              // Store in localStorage
              localStorage.setItem(STORAGE_KEY_CLAIM_PROGRESS, JSON.stringify(progressUpdate));
              
              // If complete, stop checking and load claims immediately
              if (progressUpdate.stage === 'complete') {
                if (progressCheckIntervalRef.current) {
                  clearInterval(progressCheckIntervalRef.current);
                  progressCheckIntervalRef.current = null;
                }
                // Load claims immediately when complete
                console.log('[Editor] Generation complete, loading claims...');
                // Fetch claims from API first, then load from localStorage
                setTimeout(async () => {
                  try {
                    const response = await fetch('/api/claims');
                    if (response.ok) {
                      const data = await response.json();
                      if (data.success && data.claims && data.claims.length > 0) {
                        console.log(`[Editor] Fetched ${data.claims.length} claims from API`);
                        localStorage.setItem(STORAGE_KEY_CLAIMS, JSON.stringify(data.claims));
                        loadGeneratedClaims();
                      } else {
                        console.log('[Editor] No claims in API response, trying localStorage...');
                        loadGeneratedClaims();
                      }
                    } else {
                      console.log('[Editor] Failed to fetch claims from API, trying localStorage...');
                      loadGeneratedClaims();
                    }
                  } catch (e) {
                    console.error('[Editor] Error fetching claims from API:', e);
                    loadGeneratedClaims();
                  }
                }, 500);
              }
            } else {
              // Idle state - don't show notification, but keep checking if we have stored progress
              console.log('[Editor] Progress is idle, clearing notification');
              const storedProgress = localStorage.getItem(STORAGE_KEY_CLAIM_PROGRESS);
              if (!storedProgress || JSON.parse(storedProgress).stage === 'idle') {
                setProgress(null);
              }
              // Keep polling so we don't miss a new generation start
            }
          } else {
            console.log('[Editor] No progress data in response:', data);
            // If response has error, show it
            if (data.error) {
              setProgress({
                stage: 'error',
                message: data.error,
                progress: 0,
              });
            }
          }
        } else {
          const errorText = await response.text().catch(() => 'Unknown error');
          console.error('[Editor] Progress check failed:', response.status, errorText);
          // Don't show error for 404s - just means no generation yet
          if (response.status === 404) {
            console.log('[Editor] Progress endpoint not found (404) - this is normal if no generation has started');
            setProgress(null);
          } else {
            // Show error for other status codes
            setProgress({
              stage: 'error',
              message: `Server error: ${response.status}`,
              progress: 0,
            });
          }
        }
      } catch (e) {
        console.error('[Editor] Failed to check progress:', e);
      }
    };
    
    // Check progress immediately
    checkProgress();
    
    // Set up interval to check progress every 1 second
    progressCheckIntervalRef.current = setInterval(checkProgress, 1000);
    
    // Load stored progress FIRST (before API check)
    try {
      const storedProgress = localStorage.getItem(STORAGE_KEY_CLAIM_PROGRESS);
      if (storedProgress) {
        const progressData = JSON.parse(storedProgress);
        const isStaleGenerateMessage =
          !isRegeneratingRef.current &&
          typeof progressData?.message === 'string' &&
          progressData.message.toLowerCase().includes('generating claims with similarity');
        if (isStaleGenerateMessage) {
          localStorage.removeItem(STORAGE_KEY_CLAIM_PROGRESS);
        } else {
          console.log('[Editor] Loaded stored progress:', progressData);
          setProgress(progressData);
        }
        // If complete, load claims immediately
        if (!isStaleGenerateMessage && progressData.stage === 'complete') {
          console.log('[Editor] Stored progress shows complete, loading claims...');
          // Small delay to ensure editor is ready
          setTimeout(async () => {
            // Try to fetch from API first
            try {
              const response = await fetch('/api/claims');
              if (response.ok) {
                const data = await response.json();
                if (data.success && data.claims && data.claims.length > 0) {
                  console.log(`[Editor] Fetched ${data.claims.length} claims from API`);
                  localStorage.setItem(STORAGE_KEY_CLAIMS, JSON.stringify(data.claims));
                  loadGeneratedClaims();
                  return;
                }
              }
            } catch (e) {
              console.error('[Editor] Error fetching claims from API:', e);
            }
            // Fallback to localStorage
            loadGeneratedClaims();
          }, 200);
        }
        // If not complete, the interval will keep checking
      } else {
        console.log('[Editor] No stored progress found');
        // Try to load claims anyway (might have been generated previously)
        setTimeout(() => {
          loadGeneratedClaims();
        }, 200);
      }
    } catch (e) {
      console.error('[Editor] Failed to load progress:', e);
      loadGeneratedClaims();
    }
    
    // Cleanup interval on unmount
    return () => {
      if (progressCheckIntervalRef.current) {
        clearInterval(progressCheckIntervalRef.current);
      }
    };
  }, []);
  
  const escapeHtml = (text: string): string => {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  };
  
  const loadGeneratedClaims = () => {
    try {
      // Wait for editor ref to be ready
      if (!editorRef.current) {
        console.log('[Editor] Editor ref not ready, retrying...');
        setTimeout(() => loadGeneratedClaims(), 100);
        return;
      }
      
      const storedClaims = localStorage.getItem(STORAGE_KEY_CLAIMS);
      if (storedClaims) {
        const claimsData = JSON.parse(storedClaims);
        console.log('[Editor] Loading claims:', claimsData.length, 'claims');
        console.log('[Editor] Claims data:', claimsData);
        
        if (claimsData && Array.isArray(claimsData) && claimsData.length > 0) {
          setClaims(claimsData);
          // Format claims for display - this will set the editor content
          formatClaimsForEditor(claimsData);
        } else {
          console.warn('[Editor] Claims data is empty or invalid:', claimsData);
        }
      } else {
        console.log('[Editor] No claims found in localStorage');
        // If no claims, try to load stored content
        if (editorRef.current) {
          try {
            const stored = localStorage.getItem(STORAGE_KEY_EDITOR_CONTENT);
            if (stored) {
              editorRef.current.innerHTML = stored;
              setContent(stored);
            }
          } catch (e) {
            console.error('Failed to load stored editor content:', e);
          }
        }
      }
      
      // Focus editor after loading
      if (editorRef.current) {
        editorRef.current.focus();
      }
    } catch (e) {
      console.error('[Editor] Failed to load claims:', e);
      console.error('[Editor] Error details:', e);
    }
  };
  
  const formatClaimsForEditor = (claimsData: any[]) => {
    if (!claimsData || claimsData.length === 0) {
      console.log('[Editor] Cannot format claims - empty claims data');
      return;
    }
    
    if (!editorRef.current) {
      console.log('[Editor] Editor ref not ready, waiting...');
      setTimeout(() => formatClaimsForEditor(claimsData), 100);
      return;
    }
    
    console.log('[Editor] Formatting', claimsData.length, 'claims for display');
    console.log('[Editor] Claims to format:', claimsData);
    
    // Sort claims by claim_number to ensure correct order
    const sortedClaims = [...claimsData].sort((a: any, b: any) => {
      const numA = a.claim_number || 0;
      const numB = b.claim_number || 0;
      return numA - numB;
    });
    
    // Group dependent claims by their parent claim number
    const independentClaims: any[] = [];
    const dependentByParent: { [key: number]: any[] } = {};
    
    sortedClaims.forEach((claim: any) => {
      if (claim.claim_type === 'independent') {
        independentClaims.push(claim);
        // Initialize empty array for this independent claim's dependents
        if (!dependentByParent[claim.claim_number]) {
          dependentByParent[claim.claim_number] = [];
        }
      } else if (claim.claim_type === 'dependent' && claim.parent_claim_number) {
        // Group dependent claim under its parent
        if (!dependentByParent[claim.parent_claim_number]) {
          dependentByParent[claim.parent_claim_number] = [];
        }
        dependentByParent[claim.parent_claim_number].push(claim);
      }
    });
    
    let html = '<div style="font-family: Georgia, serif; line-height: 1.8; padding: 20px; max-width: 100%; background-color: white; padding-bottom: 200px; display: block;">';
    html += '<h1 style="margin-bottom: 30px; border-bottom: 2px solid #333; padding-bottom: 10px; font-size: 28px;">Patent Claims</h1>';
    
    // Display independent claims with their dependent claims grouped underneath
    independentClaims.forEach((independentClaim: any) => {
      // Independent claim - make it clickable
      const claimNum = independentClaim.claim_number;
      const claimText = (independentClaim.claim_text || '').trim();
      // Replace claim number in text with formatted number (e.g., "1." stays "1.")
      const formattedClaimText = claimText.replace(/^\d+\./, `${claimNum}.`);
      
      html += `<div class="claim-item" data-claim-number="${claimNum}" style="margin-bottom: 24px; padding-left: 0; cursor: pointer; padding: 8px; border-radius: 4px; transition: background-color 0.2s;" onmouseover="this.style.backgroundColor='#f5f5f5'" onmouseout="this.style.backgroundColor='transparent'">`;
      html += `<p style="text-indent: 0; margin: 0 0 12px 0; line-height: 1.8; font-size: 16px;">${escapeHtml(formattedClaimText)}</p>`;
      html += `</div>`;
      
      // Dependent claims for this independent claim
      const dependents = dependentByParent[independentClaim.claim_number] || [];
      if (dependents.length > 0) {
        dependents.forEach((dependentClaim: any, depIndex: number) => {
          const depClaimNum = dependentClaim.claim_number;
          const depClaimText = (dependentClaim.claim_text || '').trim();
          // Format as x.1, x.2, etc. where x is the parent claim number
          const displayNumber = `${claimNum}.${depIndex + 1}`;
          // Replace the claim number in the text with the formatted number
          const formattedDepClaimText = depClaimText.replace(/^\d+\./, `${displayNumber}.`);
          
          html += `<div class="claim-item" data-claim-number="${depClaimNum}" style="margin-bottom: 24px; padding-left: 20px; margin-top: 12px; cursor: pointer; padding: 8px; border-radius: 4px; transition: background-color 0.2s;" onmouseover="this.style.backgroundColor='#f5f5f5'" onmouseout="this.style.backgroundColor='transparent'">`;
          html += `<p style="text-indent: 0; margin: 0 0 12px 0; line-height: 1.8; font-size: 16px;">${escapeHtml(formattedDepClaimText)}</p>`;
          html += `</div>`;
        });
      }
    });
    
    // Handle any orphaned dependent claims (no parent found)
    Object.keys(dependentByParent).forEach((parentNumStr) => {
      const parentNum = parseInt(parentNumStr);
      if (!independentClaims.find((ic: any) => ic.claim_number === parentNum)) {
        const orphanedDependents = dependentByParent[parentNum];
        orphanedDependents.forEach((dependentClaim: any) => {
          const depClaimNum = dependentClaim.claim_number;
          html += `<div class="claim-item" data-claim-number="${depClaimNum}" style="margin-bottom: 24px; padding-left: 0; cursor: pointer; padding: 8px; border-radius: 4px; transition: background-color 0.2s;" onmouseover="this.style.backgroundColor='#f5f5f5'" onmouseout="this.style.backgroundColor='transparent'">`;
          const depClaimText = (dependentClaim.claim_text || '').trim();
          html += `<p style="text-indent: 0; margin: 0 0 12px 0; line-height: 1.8; font-size: 16px;">${escapeHtml(depClaimText)}</p>`;
          html += `</div>`;
        });
      }
    });
    
    html += '</div>';
    
    // Set the HTML content
    editorRef.current.innerHTML = html;
    setContent(html);
    localStorage.setItem(STORAGE_KEY_EDITOR_CONTENT, html);
    console.log('[Editor] Claims formatted and displayed');
    
    // Add click handlers to claim items
    setTimeout(() => {
      const claimItems = editorRef.current?.querySelectorAll('.claim-item');
      claimItems?.forEach((item) => {
        item.addEventListener('click', (e) => {
          const claimNum = parseInt(item.getAttribute('data-claim-number') || '0');
          setSelectedClaimNumber(claimNum);
        });
      });
    }, 100);
  };

  const handleInput = () => {
    if (editorRef.current) {
      const newContent = editorRef.current.innerHTML;
      setContent(newContent);
      if (typeof window !== 'undefined') {
        localStorage.setItem(STORAGE_KEY_EDITOR_CONTENT, newContent);
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.ctrlKey || e.metaKey) {
      switch (e.key) {
        case 'b':
          e.preventDefault();
          document.execCommand('bold', false);
          break;
        case 'i':
          e.preventDefault();
          document.execCommand('italic', false);
          break;
        case 'u':
          e.preventDefault();
          document.execCommand('underline', false);
          break;
      }
    }
  };

  const formatText = (command: string, value?: string) => {
    document.execCommand(command, false, value);
    if (editorRef.current) {
      editorRef.current.focus();
    }
  };

  const regenerateClaimsWithThreshold = async (threshold: number) => {
    try {
      // Set flag to prevent progress checker from overwriting our custom message
      isRegeneratingRef.current = true;
      
      const customMessage = `Generating claims with similarity threshold ${(threshold * 100).toFixed(0)}%...`;
      
      setProgress({
        stage: 'planning',
        message: customMessage,
        progress: 0,
      });
      
      // Store initial progress
      localStorage.setItem(STORAGE_KEY_CLAIM_PROGRESS, JSON.stringify({
        stage: 'planning',
        message: customMessage,
        progress: 0,
      }));
      
      // Start generation with new threshold
      const response = await fetch('/api/claims/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          num_independent: 3,
          num_dependent_per_independent: 2,
          similarity_threshold: threshold,
        }),
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log(`[Editor] Regenerated ${data.num_claims} claims with threshold ${threshold}`);
        
        // Wait a moment for server to update progress, then allow progress checker to resume
        setTimeout(() => {
          isRegeneratingRef.current = false;
        }, 1000);
        
        // Update progress from server response
        if (data.progress) {
          setProgress(data.progress);
          localStorage.setItem(STORAGE_KEY_CLAIM_PROGRESS, JSON.stringify(data.progress));
        }
        
        // Load claims if already complete
        if (data.claims && data.claims.length > 0) {
          localStorage.setItem(STORAGE_KEY_CLAIMS, JSON.stringify(data.claims));
          setClaims(data.claims);
          formatClaimsForEditor(data.claims);
        }
      } else {
        isRegeneratingRef.current = false;
        // Try to get error message from response
        let errorMessage = 'Failed to generate claims';
        try {
          const errorData = await response.json();
          if (errorData.error) {
            errorMessage = `Failed to generate claims: ${errorData.error}`;
          } else if (errorData.message) {
            errorMessage = `Failed to generate claims: ${errorData.message}`;
          }
        } catch (e) {
          // If response is not JSON, use status text
          errorMessage = `Failed to generate claims: ${response.status} ${response.statusText}`;
        }
        console.error('[Editor] Failed to generate claims:', errorMessage);
        setProgress({
          stage: 'error',
          message: errorMessage,
          progress: 0,
        });
      }
    } catch (error) {
      isRegeneratingRef.current = false;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      console.error('[Editor] Error generating claims:', error);
      setProgress({
        stage: 'error',
        message: `Error regenerating claims: ${errorMessage}`,
        progress: 0,
      });
    }
  };

  const handleSimilarityThresholdChange = (value: number) => {
    setSimilarityThreshold(value);
    localStorage.setItem(STORAGE_KEY_SIMILARITY_THRESHOLD, value.toString());
    // Don't auto-generate; user must click the button
  };

  return (
    <div className={styles.layout}>
      <SideNav current="editor" />
      <div className={styles.container}>
        <div className={styles.header}>
          <div className={styles.headerContent}>
            <h1>Text Editor</h1>
          </div>
        </div>
        {pipelineBootstrapError && (
          <div className={styles.pipelineNotice}>
            Pipeline not initialized. Please upload a file to build the graph before editing claims.
          </div>
        )}
        <div className={styles.toolbar}>
          <div className={styles.toolbarGroup}>
          <button
            className={styles.toolbarButton}
            onClick={() => formatText('bold')}
            title="Bold (Ctrl+B)"
          >
            <strong>B</strong>
          </button>
          <button
            className={styles.toolbarButton}
            onClick={() => formatText('italic')}
            title="Italic (Ctrl+I)"
          >
            <em>I</em>
          </button>
          <button
            className={styles.toolbarButton}
            onClick={() => formatText('underline')}
            title="Underline (Ctrl+U)"
          >
            <u>U</u>
          </button>
        </div>
        
        <div className={styles.toolbarSeparator} />
        
        <div className={styles.toolbarGroup}>
          <button
            className={styles.toolbarButton}
            onClick={() => formatText('formatBlock', '<h1>')}
            title="Heading 1"
          >
            H1
          </button>
          <button
            className={styles.toolbarButton}
            onClick={() => formatText('formatBlock', '<h2>')}
            title="Heading 2"
          >
            H2
          </button>
          <button
            className={styles.toolbarButton}
            onClick={() => formatText('formatBlock', '<p>')}
            title="Paragraph"
          >
            P
          </button>
        </div>
        
        <div className={styles.toolbarSeparator} />
        
        <div className={styles.toolbarGroup}>
          <button
            className={styles.toolbarButton}
            onClick={() => formatText('insertUnorderedList')}
            title="Bullet List"
          >
            •
          </button>
          <button
            className={styles.toolbarButton}
            onClick={() => formatText('insertOrderedList')}
            title="Numbered List"
          >
            1.
          </button>
        </div>
        
        <div className={styles.toolbarSeparator} />
        
        <div className={styles.toolbarGroup}>
          <button
            className={styles.toolbarButton}
            onClick={() => formatText('justifyLeft')}
            title="Align Left"
          >
            ⬅
          </button>
          <button
            className={styles.toolbarButton}
            onClick={() => formatText('justifyCenter')}
            title="Align Center"
          >
            ⬌
          </button>
          <button
            className={styles.toolbarButton}
            onClick={() => formatText('justifyRight')}
            title="Align Right"
          >
            ➡
          </button>
        </div>
        
        <div className={styles.toolbarSeparator} />
        
        <div className={styles.toolbarGroup}>
          <label className={styles.similarityLabel} title="Cosine Similarity Threshold">
            Similarity: {(similarityThreshold * 100).toFixed(0)}%
          </label>
          <input
            type="range"
            min="0"
            max="100"
            step="5"
            value={similarityThreshold * 100}
            onChange={(e) => handleSimilarityThresholdChange(parseFloat(e.target.value) / 100)}
            className={styles.similaritySlider}
            title={`Cosine similarity threshold: ${(similarityThreshold * 100).toFixed(0)}%`}
          />
          <button
            className={styles.toolbarButton}
            onClick={() => regenerateClaimsWithThreshold(similarityThreshold)}
            title="Generate claims with new similarity threshold"
          >
            Generate
          </button>
        </div>
      </div>

      <div className={styles.editorWrapper}>
        <div className={`${styles.editorContainer} ${showTriplesPanel ? styles.editorWithPanel : ''}`}>
          <div
            ref={editorRef}
            className={styles.editor}
            contentEditable
            onInput={handleInput}
            onKeyDown={handleKeyDown}
            suppressContentEditableWarning
            data-placeholder="Start typing..."
          />
        </div>
        
        {/* Triples Side Panel */}
        {showTriplesPanel && (
          <div className={styles.triplesPanel}>
            <div className={styles.triplesPanelHeader}>
              <h3>Triples Used in Claims</h3>
              <button
                className={styles.closeButton}
                onClick={() => setShowTriplesPanel(false)}
                title="Close panel"
              >
                ×
              </button>
            </div>
            <div className={styles.triplesPanelContent}>
              {selectedClaimNumber ? (
                (() => {
                  const selectedClaim = claims.find((c: any) => c.claim_number === selectedClaimNumber);
                  if (selectedClaim && selectedClaim.used_triples && selectedClaim.used_triples.length > 0) {
                    return (
                      <div>
                        <div className={styles.claimInfo}>
                          <strong>Claim {selectedClaimNumber}</strong>
                          <span className={styles.claimType}>{selectedClaim.claim_type.toUpperCase()}</span>
                        </div>
                        {selectedClaim.focus && (
                          <div className={styles.claimFocus}>
                            <div className={styles.claimFocusLabel}>Planned Focus:</div>
                            <div className={styles.claimFocusText}>{selectedClaim.focus}</div>
                          </div>
                        )}
                        {selectedClaim.prompt && (
                          <div className={styles.promptDropdown}>
                            <button
                              className={styles.promptDropdownButton}
                              onClick={() => setExpandedPromptClaim(
                                expandedPromptClaim === selectedClaimNumber ? null : selectedClaimNumber
                              )}
                            >
                              <span>Prompt</span>
                              <span className={styles.promptDropdownIcon}>
                                {expandedPromptClaim === selectedClaimNumber ? '▼' : '▶'}
                              </span>
                            </button>
                            {expandedPromptClaim === selectedClaimNumber && (
                              <div className={styles.promptContent}>
                                <pre className={styles.promptText}>{selectedClaim.prompt}</pre>
                              </div>
                            )}
                          </div>
                        )}
                        <div className={styles.triplesList}>
                          {selectedClaim.used_triples.map((triple: any, idx: number) => (
                            <div key={idx} className={styles.tripleItem}>
                              <div className={styles.tripleHead}>{triple.head || 'N/A'}</div>
                              <div className={styles.tripleRelation}>— {triple.relation || 'N/A'} —</div>
                              <div className={styles.tripleTail}>{triple.tail || 'N/A'}</div>
                              {triple.similarity !== undefined && (
                                <div className={styles.tripleSimilarity}>
                                  Similarity: {(triple.similarity * 100).toFixed(1)}%
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  } else {
                    return (
                      <div className={styles.noTriples}>
                        <p>No triples recorded for Claim {selectedClaimNumber}</p>
                        <p className={styles.hint}>Click on a claim to see its facts</p>
                      </div>
                    );
                  }
                })()
              ) : (
                <div className={styles.noSelection}>
                  <p>Click on a claim to see which facts were used</p>
                  <div className={styles.claimsList}>
                    {claims.map((claim: any) => (
                      <div
                        key={claim.claim_number}
                        className={styles.claimListItem}
                        onClick={() => setSelectedClaimNumber(claim.claim_number)}
                      >
                        <span className={styles.claimNumber}>Claim {claim.claim_number}</span>
                        <span className={styles.claimTypeBadge}>{claim.claim_type}</span>
                        {claim.used_triples && claim.used_triples.length > 0 && (
                          <span className={styles.tripleCount}>{claim.used_triples.length} facts</span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
        
        {/* Toggle button when panel is closed */}
        {!showTriplesPanel && (
          <button
            className={styles.toggleTriplesButton}
            onClick={() => setShowTriplesPanel(true)}
            title="Show triples panel"
          >
            🔗
          </button>
        )}
      </div>
      
      {/* Progress Notification - Show for all active stages */}
      {progress && progress.stage && progress.stage !== 'idle' && (
        <div className={progress.stage === 'complete' ? `${styles.notification} ${styles.notificationSuccess}` : styles.notification}>
          <div className={styles.notificationContent}>
            <div className={styles.notificationMessage}>
              {progress.stage === 'planning' && '📋 '}
              {progress.stage === 'planning_complete' && '✅ '}
              {progress.stage === 'generating' && '✍️ '}
              {progress.stage === 'refining' && '🔍 '}
              {progress.stage === 'complete' && '✅ '}
              {progress.stage === 'error' && '❌ '}
              {progress.message || 'Processing...'}
            </div>
            {progress.progress !== undefined && progress.progress > 0 && progress.stage !== 'complete' && progress.stage !== 'error' && (
              <div className={styles.progressBar}>
                <div 
                  className={styles.progressFill} 
                  style={{ width: `${Math.min(100, Math.max(0, progress.progress))}%` }}
                />
              </div>
            )}
            {progress.current_claim && progress.total_claims && (
              <div className={styles.progressText}>
                Claim {progress.current_claim} of {progress.total_claims}
              </div>
            )}
          </div>
        </div>
      )}
      </div>
    </div>
  );
}


