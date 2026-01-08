'use client';

import { useState } from 'react';
import styles from './Widget.module.css';

interface WidgetProps {
  type: string;
  data: any;
  onAnswer?: (answer: string) => void;
}

export default function Widget({ type, data, onAnswer }: WidgetProps) {
  const [showAll, setShowAll] = useState(false);

  const handleSubmit = (value: string) => {
    if (onAnswer && value) {
      onAnswer(value);
    }
  };

  switch (type) {
    case 'edges_widget':
      const edges = data.triples || [];
      const showCount = 5;
      const displayedEdges = showAll ? edges : edges.slice(0, showCount);
      
      return (
        <div className={styles.widget}>
          <h4>Triples</h4>
          <ul>
            {displayedEdges.map((edge: any, idx: number) => {
              const index = edge.index !== undefined ? edge.index : idx;
              return (
                <li key={idx}>
                  {index}. {edge.head || ''} --[{edge.relation || ''}]--> {edge.tail || ''}
                </li>
              );
            })}
          </ul>
          {edges.length > showCount && !showAll && (
            <button className={styles.widgetButton} onClick={() => setShowAll(true)}>
              Show More ({edges.length - showCount} remaining)
            </button>
          )}
        </div>
      );

    case 'graph_widget':
    case 'graph_subsection_widget':
      return (
        <div className={styles.widget}>
          <h4>Graph Visualization</h4>
          <div className={styles.graphPlaceholder}>Graph visualization would appear here</div>
        </div>
      );

    case 'question_widget_general':
      return (
        <div className={styles.widget}>
          <h4>{data.question || 'Question'}</h4>
          <textarea
            className={styles.widgetInput}
            placeholder="Your answer..."
            onKeyDown={(e) => {
              if (e.key === 'Enter' && e.ctrlKey) {
                handleSubmit((e.target as HTMLTextAreaElement).value);
              }
            }}
          />
          <button className={styles.widgetButton} onClick={(e) => {
            const input = (e.target as HTMLElement).parentElement?.querySelector('textarea') as HTMLTextAreaElement;
            handleSubmit(input?.value || '');
          }}>
            Submit
          </button>
        </div>
      );

    case 'question_widget_triple':
      return (
        <div className={styles.widget}>
          <h4>Confirm or correct this triple:</h4>
          <p>{data.triple?.head || ''} --[{data.triple?.relation || ''}]--> {data.triple?.tail || ''}</p>
          <textarea
            className={styles.widgetInput}
            placeholder="Corrections or confirm..."
            onKeyDown={(e) => {
              if (e.key === 'Enter' && e.ctrlKey) {
                handleSubmit((e.target as HTMLTextAreaElement).value);
              }
            }}
          />
          <button className={styles.widgetButton} onClick={(e) => {
            const input = (e.target as HTMLElement).parentElement?.querySelector('textarea') as HTMLTextAreaElement;
            handleSubmit(input?.value || '');
          }}>
            Submit
          </button>
        </div>
      );

    case 'question_widget_entity':
      return (
        <div className={styles.widget}>
          <h4>Validate or explain: {data.entity_name || 'Entity'}</h4>
          <textarea
            className={styles.widgetInput}
            placeholder="Your explanation..."
            onKeyDown={(e) => {
              if (e.key === 'Enter' && e.ctrlKey) {
                handleSubmit((e.target as HTMLTextAreaElement).value);
              }
            }}
          />
          <button className={styles.widgetButton} onClick={(e) => {
            const input = (e.target as HTMLElement).parentElement?.querySelector('textarea') as HTMLTextAreaElement;
            handleSubmit(input?.value || '');
          }}>
            Submit
          </button>
        </div>
      );

    case 'question_widget_cluster_triple':
      return (
        <div className={styles.widget}>
          <h4>Rate importance in cluster</h4>
          <p>{data.triple?.head || ''} --[{data.triple?.relation || ''}]--> {data.triple?.tail || ''}</p>
          <input type="range" min="1" max="5" defaultValue="3" className={styles.widgetSlider} />
          <button className={styles.widgetButton} onClick={(e) => {
            const slider = (e.target as HTMLElement).parentElement?.querySelector('input[type="range"]') as HTMLInputElement;
            handleSubmit(slider?.value || '3');
          }}>
            Submit
          </button>
        </div>
      );

    case 'validation_summary_widget':
      const stats = data.stats || {};
      return (
        <div className={styles.widget}>
          <h4>Validation Summary</h4>
          <p>Success Rate: {stats.success_rate || 'N/A'}%</p>
          <p>Total Validated: {stats.total || 0}</p>
          <p>Passed: {stats.passed || 0}</p>
          <p>Failed: {stats.failed || 0}</p>
        </div>
      );

    case 'patent_analysis_widget':
      const patent = data.patent || {};
      return (
        <div className={styles.widget}>
          <h4>Patent Analysis</h4>
          <p>Status: {patent.status || 'N/A'}</p>
          <p>Risk: {patent.risk || 'N/A'}</p>
          <p>Key Metadata: {patent.metadata || 'N/A'}</p>
        </div>
      );

    case 'connection_check_widget':
      const issues = data.issues || [];
      return (
        <div className={styles.widget}>
          <h4>Connection Check</h4>
          <ul>
            {issues.map((issue: any, idx: number) => (
              <li key={idx} className={styles[issue.severity || 'info']}>
                {issue.message || ''}
              </li>
            ))}
          </ul>
        </div>
      );

    case 'suggestion_widget':
      const suggestions = data.suggestions || [];
      return (
        <div className={styles.widget}>
          <h4>Suggestions</h4>
          <ul>
            {suggestions.map((suggestion: any, idx: number) => (
              <li key={idx}>
                {suggestion.text || ''}
                <button
                  className={styles.widgetButtonSmall}
                  onClick={() => onAnswer && onAnswer(`Accept suggestion ${suggestion.id || idx}`)}
                >
                  Accept
                </button>
                <button
                  className={styles.widgetButtonSmall}
                  onClick={() => onAnswer && onAnswer(`Dismiss suggestion ${suggestion.id || idx}`)}
                >
                  Dismiss
                </button>
              </li>
            ))}
          </ul>
        </div>
      );

    default:
      return (
        <div className={styles.widget}>
          <p>Widget: {type}</p>
        </div>
      );
  }
}

