'use client';

import { useState, useEffect } from 'react';
import styles from './EditPanel.module.css';
import { X, Check, AlertCircle } from 'lucide-react';

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

interface EditPanelProps {
  entity?: Entity;
  triple?: Triple;
  allEntities?: Entity[];
  onClose: () => void;
  onUpdate: (data: any) => Promise<void>;
  onMerge?: (sourceId: string, targetId: string) => Promise<void>;
}

export default function EditPanel({
  entity,
  triple,
  allEntities = [],
  onClose,
  onUpdate,
  onMerge,
}: EditPanelProps) {
  const [mode, setMode] = useState<'edit' | 'merge'>('edit');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  // Entity edit state
  const [entityName, setEntityName] = useState(entity?.name || '');
  const [entityLabel, setEntityLabel] = useState(entity?.label || '');

  // Triple edit state
  const [tripleRelation, setTripleRelation] = useState(triple?.relation || '');

  // Merge state
  const [mergeTarget, setMergeTarget] = useState('');

  useEffect(() => {
    if (entity) {
      setEntityName(entity.name);
      setEntityLabel(entity.label);
    }
    if (triple) {
      setTripleRelation(triple.relation);
    }
  }, [entity, triple]);

  const handleEntityUpdate = async () => {
    setLoading(true);
    setError('');
    setSuccess('');
    try {
      await onUpdate({
        type: 'entity',
        id: entity?.id,
        name: entityName,
        label: entityLabel,
      });
      setSuccess('Entity updated successfully!');
      setTimeout(() => onClose(), 1500);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update entity');
    } finally {
      setLoading(false);
    }
  };

  const handleTripleUpdate = async () => {
    setLoading(true);
    setError('');
    setSuccess('');
    try {
      await onUpdate({
        type: 'triple',
        index: triple?.index,
        relation: tripleRelation,
      });
      setSuccess('Triple updated successfully!');
      setTimeout(() => onClose(), 1500);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update triple');
    } finally {
      setLoading(false);
    }
  };

  const handleMerge = async () => {
    if (!mergeTarget || !entity) {
      setError('Please select a target entity to merge with');
      return;
    }
    if (mergeTarget === entity.id) {
      setError('Cannot merge an entity with itself');
      return;
    }

    setLoading(true);
    setError('');
    setSuccess('');
    try {
      if (onMerge) {
        await onMerge(entity.id, mergeTarget);
        setSuccess('Entities merged successfully! All relations transferred.');
        setTimeout(() => onClose(), 1500);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to merge entities');
    } finally {
      setLoading(false);
    }
  };

  const availableMergeTargets = allEntities.filter(
    (e) => e.id !== entity?.id
  );

  const SUGGESTED_LABELS = [
    "INVENTION", "COMPONENT", "SUBSYSTEM", "MATERIAL", "CHEMICAL", "BIOMOLECULE", "COMPOSITION",
    "PROCESS_STEP", "METHOD", "PARAMETER", "MEASUREMENT", "CONDITION", "FUNCTION", "SIGNAL",
    "CONTROL", "SOFTWARE", "HARDWARE", "FIGURE_REF", "CLAIM_ELEMENT", "PRIOR_ART",
    "UNCLASSIFIED_ENTITY", "UNKNOWN",
  ];

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <h2>{entity ? 'Edit Entity' : 'Edit Triple'}</h2>
        <button
          className={styles.closeButton}
          onClick={onClose}
          disabled={loading}
          title="Close panel"
        >
          <X size={20} />
        </button>
      </div>

      {error && (
        <div className={styles.message + ' ' + styles.error}>
          <AlertCircle size={16} />
          <span>{error}</span>
        </div>
      )}

      {success && (
        <div className={styles.message + ' ' + styles.success}>
          <Check size={16} />
          <span>{success}</span>
        </div>
      )}

      {entity && (
        <div className={styles.content}>
          <div className={styles.tabs}>
            <button
              className={`${styles.tab} ${mode === 'edit' ? styles.active : ''}`}
              onClick={() => setMode('edit')}
              disabled={loading}
            >
              Edit
            </button>
            {onMerge && (
              <button
                className={`${styles.tab} ${mode === 'merge' ? styles.active : ''}`}
                onClick={() => setMode('merge')}
                disabled={loading}
              >
                Merge
              </button>
            )}
          </div>

          {mode === 'edit' && (
            <div className={styles.formGroup}>
              <div className={styles.section}>
                <label>Entity ID (read-only)</label>
                <input
                  type="text"
                  value={entity.id}
                  disabled
                  className={styles.input + ' ' + styles.disabled}
                />
              </div>

              <div className={styles.section}>
                <label htmlFor="entityName">Entity Name</label>
                <input
                  id="entityName"
                  type="text"
                  value={entityName}
                  onChange={(e) => setEntityName(e.target.value)}
                  disabled={loading}
                  className={styles.input}
                  placeholder="Enter entity name"
                />
              </div>

              <div className={styles.section}>
                <label htmlFor="entityLabel">Entity Label/Type</label>
                <div className={styles.inputWrapper}>
                  <input
                    id="entityLabel"
                    type="text"
                    list="labelOptions"
                    value={entityLabel}
                    onChange={(e) => setEntityLabel(e.target.value.toUpperCase())}
                    disabled={loading}
                    className={styles.input}
                    placeholder="Select or enter label"
                  />
                  <datalist id="labelOptions">
                    {SUGGESTED_LABELS.map(label => (
                      <option key={label} value={label} />
                    ))}
                  </datalist>
                </div>
              </div>

              <button
                className={styles.primaryButton}
                onClick={handleEntityUpdate}
                disabled={loading}
              >
                {loading ? 'Updating...' : 'Update Entity'}
              </button>
            </div>
          )}

          {mode === 'merge' && (
            <div className={styles.formGroup}>
              <p className={styles.description}>
                Merge <strong>{entity.name}</strong> with another entity. All relations will be transferred to the target entity.
              </p>

              {availableMergeTargets.length > 0 ? (
                <div className={styles.section}>
                  <label htmlFor="mergeTarget">Select Target Entity</label>
                  <select
                    id="mergeTarget"
                    value={mergeTarget}
                    onChange={(e) => setMergeTarget(e.target.value)}
                    disabled={loading}
                    className={styles.select}
                  >
                    <option value="">-- Select an entity --</option>
                    {availableMergeTargets.map((e) => (
                      <option key={e.id} value={e.id}>
                        {e.name} ({e.label}) [{e.id}]
                      </option>
                    ))}
                  </select>
                </div>
              ) : (
                <div className={styles.message + ' ' + styles.warning}>
                  <AlertCircle size={16} />
                  <span>No other entities found.</span>
                </div>
              )}

              <button
                className={styles.primaryButton}
                onClick={handleMerge}
                disabled={loading || !mergeTarget || availableMergeTargets.length === 0}
              >
                {loading ? 'Merging...' : 'Merge Entities'}
              </button>
            </div>
          )}
        </div>
      )}

      {triple && (
        <div className={styles.content}>
          <div className={styles.formGroup}>
            <div className={styles.section}>
              <label>Head Entity</label>
              <div className={styles.entityDisplay}>
                <div className={styles.entityName}>{triple.head.name}</div>
                <div className={styles.entityLabel}>{triple.head.label}</div>
              </div>
            </div>

            <div className={styles.section}>
              <label htmlFor="relation">Relation</label>
              <input
                id="relation"
                type="text"
                value={tripleRelation}
                onChange={(e) => setTripleRelation(e.target.value)}
                disabled={loading}
                className={styles.input}
                placeholder="Enter relation type"
              />
            </div>

            <div className={styles.section}>
              <label>Tail Entity</label>
              <div className={styles.entityDisplay}>
                <div className={styles.entityName}>{triple.tail.name}</div>
                <div className={styles.entityLabel}>{triple.tail.label}</div>
              </div>
            </div>

            <button
              className={styles.primaryButton}
              onClick={handleTripleUpdate}
              disabled={loading}
            >
              {loading ? 'Updating...' : 'Update Triple'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

