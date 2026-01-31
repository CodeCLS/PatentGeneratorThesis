"use client";

import { useEffect, useMemo, useState } from "react";
import styles from "./page.module.css";
import SideNav from "@/components/SideNav";
import { bootstrapPipelineIfNeeded } from "@/lib/pipeline-bootstrap";

interface AnalyzeEntity {
  name: string;
  label: string;
  ref?: string;
  ref_short?: string;
  start: number;
  end: number;
  sentence_id?: string;
  sentence_index?: number;
  entity_type?: string | null;
}

interface AnalyzeSentence {
  id: string;
  index: number;
  text: string;
  start: number;
  end: number;
  entities: AnalyzeEntity[];
}

interface EntitySummary {
  key: string;
  name: string;
  label: string;
  mentions: AnalyzeEntity[];
}

interface TripleEntity {
  id: string;
  name: string;
  label?: string;
}

interface Triple {
  index: number;
  head: TripleEntity;
  relation: string;
  tail: TripleEntity;
}

const LABEL_CLASSES = [
  styles.labelAmber,
  styles.labelBlue,
  styles.labelGreen,
  styles.labelViolet,
  styles.labelRose,
  styles.labelTeal,
];

const getEntityKey = (entity: AnalyzeEntity) =>
  entity.ref || entity.ref_short || entity.name || "";

const hashLabel = (label: string) => {
  let hash = 0;
  for (let i = 0; i < label.length; i += 1) {
    hash = (hash * 31 + label.charCodeAt(i)) % 1000;
  }
  return hash;
};

const getLabelClass = (label: string) => {
  if (!label) return styles.labelDefault;
  const index = hashLabel(label) % LABEL_CLASSES.length;
  return LABEL_CLASSES[index] || styles.labelDefault;
};

const entityMatchesSelection = (entity: AnalyzeEntity, keys: Set<string> | null) => {
  if (!keys) return false;
  return Boolean(
    (entity.ref && keys.has(entity.ref)) ||
      (entity.ref_short && keys.has(entity.ref_short)) ||
      (entity.name && keys.has(entity.name))
  );
};

const normalizeEntities = (text: string, entities: AnalyzeEntity[]) => {
  const normalized: AnalyzeEntity[] = [];
  const lower = text.toLowerCase();

  entities.forEach((entity) => {
    const rawName = entity.name || "";
    if (!rawName) return;
    const name = rawName.trim();
    if (!name) return;

    let start = Number.isFinite(entity.start) ? entity.start : 0;
    let end = Number.isFinite(entity.end) ? entity.end : start + name.length;
    start = Math.max(0, Math.min(text.length, start));
    end = Math.max(start, Math.min(text.length, end));

    const slice = text.slice(start, end);
    if (slice !== name) {
      const nameLower = name.toLowerCase();
      let bestIndex = -1;
      let searchIndex = lower.indexOf(nameLower);
      let bestDistance = Number.POSITIVE_INFINITY;

      while (searchIndex !== -1) {
        const distance = Math.abs(searchIndex - start);
        if (distance < bestDistance) {
          bestIndex = searchIndex;
          bestDistance = distance;
        }
        searchIndex = lower.indexOf(nameLower, searchIndex + 1);
      }

      if (bestIndex !== -1) {
        start = bestIndex;
        end = bestIndex + name.length;
      } else {
        return;
      }
    }

    normalized.push({ ...entity, start, end, name });
  });

  normalized.sort((a, b) => {
    if (a.start !== b.start) return a.start - b.start;
    return b.end - a.end;
  });

  const nonOverlapping: AnalyzeEntity[] = [];
  let cursor = 0;
  normalized.forEach((entity) => {
    if (entity.end <= cursor) return;
    if (entity.start < cursor) {
      return;
    }
    nonOverlapping.push(entity);
    cursor = entity.end;
  });

  return nonOverlapping;
};

export default function AnalyzePage() {
  const [sentences, setSentences] = useState<AnalyzeSentence[]>([]);
  const [triples, setTriples] = useState<Triple[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [pipelineError, setPipelineError] = useState("");
  const [selectedEntity, setSelectedEntity] = useState<AnalyzeEntity | null>(null);

  const sentenceByIndex = useMemo(() => {
    const map = new Map<number, AnalyzeSentence>();
    sentences.forEach((sentence) => map.set(sentence.index, sentence));
    return map;
  }, [sentences]);

  const entitySummaries = useMemo(() => {
    const map = new Map<string, EntitySummary>();
    sentences.forEach((sentence) => {
      sentence.entities?.forEach((entity) => {
        const key = getEntityKey(entity);
        if (!key) return;
        const existing = map.get(key);
        if (existing) {
          existing.mentions.push(entity);
        } else {
          map.set(key, {
            key,
            name: entity.name || key,
            label: entity.label || "Entity",
            mentions: [entity],
          });
        }
      });
    });
    return Array.from(map.values()).sort((a, b) => a.name.localeCompare(b.name));
  }, [sentences]);

  const selectedKeySet = useMemo(() => {
    if (!selectedEntity) return null;
    const keys = new Set<string>();
    if (selectedEntity.ref) keys.add(selectedEntity.ref);
    if (selectedEntity.ref_short) keys.add(selectedEntity.ref_short);
    if (selectedEntity.name) keys.add(selectedEntity.name);
    keys.add(getEntityKey(selectedEntity));
    return keys;
  }, [selectedEntity]);

  const selectedSummary = useMemo(() => {
    if (!selectedEntity) return null;
    const key = getEntityKey(selectedEntity);
    return entitySummaries.find((summary) => summary.key === key) || {
      key,
      name: selectedEntity.name || key,
      label: selectedEntity.label || "Entity",
      mentions: [selectedEntity],
    };
  }, [selectedEntity, entitySummaries]);

  const relatedTriples = useMemo(() => {
    if (!selectedKeySet) return [];
    return triples.filter(
      (triple) =>
        selectedKeySet.has(triple.head?.id) || selectedKeySet.has(triple.tail?.id)
    );
  }, [triples, selectedKeySet]);

  useEffect(() => {
    const initialize = async () => {
      setLoading(true);
      setError("");
      setPipelineError("");
      const result = await bootstrapPipelineIfNeeded();
      if (!result.initialized) {
        setPipelineError(result.error || "Pipeline not initialized.");
        setLoading(false);
        return;
      }

      try {
        const response = await fetch("/api/analyze", { cache: "no-store" });
        const data = await response.json().catch(() => ({}));
        if (!response.ok || data.error) {
          throw new Error(data.error || "Failed to load analysis data.");
        }
        setSentences(data.sentences || []);
        setTriples(data.triples || []);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load analysis data.");
      } finally {
        setLoading(false);
      }
    };

    initialize();
  }, []);

  const renderSentence = (sentence: AnalyzeSentence) => {
    const text = sentence.text || "";
    const entities = normalizeEntities(text, sentence.entities || []);
    if (!entities.length) {
      return text;
    }

    const parts: Array<JSX.Element | string> = [];
    let cursor = 0;

    entities.forEach((entity, index) => {
      const start = Math.max(0, Math.min(text.length, entity.start));
      const end = Math.max(start, Math.min(text.length, entity.end));
      if (end <= cursor) {
        return;
      }
      if (start > cursor) {
        parts.push(text.slice(cursor, start));
      }

      const key = `${entity.ref || entity.ref_short || entity.name}-${start}-${end}-${index}`;
      const match = entityMatchesSelection(entity, selectedKeySet);
      const dimmed = selectedKeySet ? !match : false;
      const className = [
        styles.entitySpan,
        getLabelClass(entity.label || ""),
        match ? styles.entityActive : "",
        dimmed ? styles.entityDim : "",
      ]
        .filter(Boolean)
        .join(" ");

      parts.push(
        <button
          key={key}
          type="button"
          className={className}
          onClick={() => setSelectedEntity(entity)}
        >
          {text.slice(start, end)}
        </button>
      );
      cursor = end;
    });

    if (cursor < text.length) {
      parts.push(text.slice(cursor));
    }

    return parts;
  };

  const renderMentionSnippet = (sentenceText: string, mention: AnalyzeEntity) => {
    const rawStart = Number.isFinite(mention.start) ? mention.start : 0;
    const rawEnd = Number.isFinite(mention.end) ? mention.end : rawStart;
    const start = Math.max(0, Math.min(sentenceText.length, rawStart));
    const end = Math.max(start, Math.min(sentenceText.length, rawEnd));

    const snippetStart = Math.max(0, start - 24);
    const snippetEnd = Math.min(sentenceText.length, end + 24);
    const snippet = sentenceText.slice(snippetStart, snippetEnd);

    const relativeStart = Math.max(0, start - snippetStart);
    const relativeEnd = Math.max(relativeStart, end - snippetStart);

    const prefix = snippet.slice(0, relativeStart);
    const match = snippet.slice(relativeStart, relativeEnd);
    const suffix = snippet.slice(relativeEnd);

    return (
      <>
        {snippetStart > 0 ? "…" : ""}
        {prefix}
        {match ? <span className={styles.mentionHighlight}>{match}</span> : null}
        {suffix}
        {snippetEnd < sentenceText.length ? "…" : ""}
      </>
    );
  };

  return (
    <div className={styles.layout}>
      <SideNav current="analyze" />
      <div className={styles.container}>
        <header className={styles.header}>
          <div className={styles.headerContent}>
            <div className={styles.headerTitle}>
              <h1>Analyze Source Text</h1>
              <span>Review extracted entities and their relationships.</span>
            </div>
            <div className={styles.headerStats}>
              <span className={styles.statPill}>{sentences.length} sentences</span>
              <span className={styles.statPill}>{entitySummaries.length} entities</span>
              <span className={styles.statPill}>{triples.length} relations</span>
            </div>
          </div>
        </header>

        <main className={styles.main}>
          {pipelineError && <div className={styles.notice}>{pipelineError}</div>}
          {error && <div className={styles.notice}>{error}</div>}
          {loading && !error && (
            <div className={styles.loading}>Loading analysis...</div>
          )}

          {!loading && !error && sentences.length === 0 && (
            <div className={styles.empty}>
              <p>No sentence data found.</p>
              <p>Upload a document to generate entities and relations.</p>
            </div>
          )}

          {!loading && !error && sentences.length > 0 && (
            <div className={styles.content}>
              <section className={styles.textPane}>
                <div className={styles.sectionHeader}>
                  <h2>Original Text</h2>
                  <p>Click any highlight to see all references and relations.</p>
                </div>
                <div className={styles.textBody}>
                  {sentences.map((sentence) => (
                    <p key={sentence.id} className={styles.sentence}>
                      <span className={styles.sentenceIndex}>{sentence.index + 1}.</span>
                      <span className={styles.sentenceText}>{renderSentence(sentence)}</span>
                    </p>
                  ))}
                </div>
              </section>

              <aside
                className={`${styles.sidePanel} ${
                  selectedEntity ? styles.sidePanelOpen : ""
                }`}
              >
                {selectedEntity && selectedSummary ? (
                  <div className={styles.sideContent}>
                    <div className={styles.sideHeader}>
                      <div>
                        <h3>{selectedSummary.name}</h3>
                        <span className={styles.sideLabel}>{selectedSummary.label}</span>
                      </div>
                      <button
                        type="button"
                        className={styles.clearButton}
                        onClick={() => setSelectedEntity(null)}
                      >
                        Clear
                      </button>
                    </div>

                    <div className={styles.sideMeta}>
                      <span>{selectedSummary.mentions.length} mentions</span>
                      {selectedEntity.entity_type && (
                        <span>{selectedEntity.entity_type}</span>
                      )}
                    </div>

                    <div className={styles.sideSection}>
                      <h4>Mentions</h4>
                      <div className={styles.mentionList}>
                        {selectedSummary.mentions.map((mention, idx) => {
                          const sentence = sentenceByIndex.get(mention.sentence_index ?? -1);
                          const snippetSource = sentence?.text || "";
                          return (
                            <div key={`${mention.sentence_id}-${idx}`} className={styles.mentionItem}>
                              <span className={styles.mentionMeta}>
                                Sentence {typeof mention.sentence_index === "number" ? mention.sentence_index + 1 : "?"}
                              </span>
                              <span className={styles.mentionSnippet}>
                                {renderMentionSnippet(snippetSource, mention)}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    <div className={styles.sideSection}>
                      <h4>Relations</h4>
                      <div className={styles.relationList}>
                        {relatedTriples.length === 0 && (
                          <p className={styles.relationEmpty}>No relations linked to this entity.</p>
                        )}
                        {relatedTriples.map((triple) => (
                          <div
                            key={`${triple.index}-${triple.head?.id}-${triple.tail?.id}`}
                            className={styles.relationCard}
                          >
                            <span className={styles.relationNode}>{triple.head?.name}</span>
                            <span className={styles.relationVerb}>{triple.relation}</span>
                            <span className={styles.relationNode}>{triple.tail?.name}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className={styles.sideEmpty}>
                    <p>Select a highlighted entity to see details.</p>
                  </div>
                )}
              </aside>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
