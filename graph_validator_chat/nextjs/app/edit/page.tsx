"use client";

import { useEffect, useMemo, useState } from "react";
import styles from "./page.module.css";
import { Check, ChevronsUpDown, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { bootstrapPipelineIfNeeded } from "@/lib/pipeline-bootstrap";
import SideNav from "@/components/SideNav";
import { cn } from "@/lib/utils";

interface Entity {
  id: string;
  name: string;
  label?: string;
}

interface Triple {
  index: number;
  head: Entity;
  relation: string;
  tail: Entity;
}

interface TripleEditState {
  headSelection: string;
  tailSelection: string;
  headLabelSelection: string;
  tailLabelSelection: string;
  newHeadName: string;
  newHeadLabel: string;
  newTailName: string;
  newTailLabel: string;
  relationSelection: string;
  newRelation: string;
  saving: boolean;
  message?: string;
  messageType?: "success" | "error";
}

type ComboOption = { value: string; label: string };

const NEW_ENTITY_VALUE = "__new__";
const NEW_RELATION_VALUE = "__new_relation__";

const Combobox = ({
  valueLabel,
  selectedValue,
  inputValue,
  onInputChange,
  options,
  onSelect,
  onCreate,
  placeholder,
  createLabel,
}: {
  valueLabel: string;
  selectedValue: string;
  inputValue: string;
  onInputChange: (value: string) => void;
  options: ComboOption[];
  onSelect: (value: string) => void;
  onCreate: (value: string) => void;
  placeholder: string;
  createLabel: string;
}) => {
  const [open, setOpen] = useState(false);
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          role="combobox"
          aria-expanded={open}
          className={styles.comboTrigger}
        >
          <span className={styles.comboValue}>{valueLabel}</span>
          <ChevronsUpDown className="ml-auto h-4 w-4 opacity-50" />
        </button>
      </PopoverTrigger>
      <PopoverContent className={styles.comboContent} align="start">
        <Command>
          <CommandInput
            placeholder={placeholder}
            value={inputValue}
            onValueChange={onInputChange}
          />
          <CommandList>
            <CommandEmpty>No results found.</CommandEmpty>
            {inputValue.trim() && (
              <CommandGroup heading="Create">
                <CommandItem
                  value={`create-${inputValue}`}
                  onSelect={() => {
                    const nextValue = inputValue.trim();
                    if (!nextValue) return;
                    onCreate(nextValue);
                    setOpen(false);
                  }}
                >
                  <Plus className="mr-2 h-4 w-4" />
                  {createLabel} "{inputValue.trim()}"
                </CommandItem>
              </CommandGroup>
            )}
            <CommandGroup heading="Options">
              {options.map((option) => (
                <CommandItem
                  key={option.value}
                  value={option.label}
                  onSelect={() => {
                    onSelect(option.value);
                    setOpen(false);
                  }}
                >
                  <Check
                    className={cn(
                      "mr-2 h-4 w-4",
                      selectedValue === option.value ? "opacity-100" : "opacity-0"
                    )}
                  />
                  {option.label}
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
};

const buildEditState = (triple: Triple): TripleEditState => ({
  headSelection: triple.head.id,
  tailSelection: triple.tail.id,
  headLabelSelection: triple.head.label || "unknown_entity",
  tailLabelSelection: triple.tail.label || "unknown_entity",
  newHeadName: "",
  newHeadLabel: "unknown_entity",
  newTailName: "",
  newTailLabel: "unknown_entity",
  relationSelection: triple.relation,
  newRelation: "",
  saving: false,
});

export default function EditPage() {
  const [triples, setTriples] = useState<Triple[]>([]);
  const [editStates, setEditStates] = useState<Record<string, TripleEditState>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>("");
  const [pipelineError, setPipelineError] = useState<string>("");
  const [searchTerm, setSearchTerm] = useState("");

  const entities = useMemo(() => {
    const map = new Map<string, Entity>();
    triples.forEach((triple) => {
      if (triple.head?.id) {
        map.set(triple.head.id, triple.head);
      }
      if (triple.tail?.id) {
        map.set(triple.tail.id, triple.tail);
      }
    });
    return Array.from(map.values()).sort((a, b) => a.name.localeCompare(b.name));
  }, [triples]);

  const entitiesById = useMemo(() => {
    const map = new Map<string, Entity>();
    entities.forEach((entity) => map.set(entity.id, entity));
    return map;
  }, [entities]);

  const relationOptions = useMemo(() => {
    const unique = new Set<string>();
    triples.forEach((triple) => {
      if (triple.relation) {
        unique.add(triple.relation);
      }
    });
    return Array.from(unique).sort((a, b) => a.localeCompare(b));
  }, [triples]);

  const labelOptions = useMemo(() => {
    const unique = new Set<string>(["unknown_entity"]);
    entities.forEach((entity) => {
      if (entity.label) {
        unique.add(entity.label);
      }
    });
    return Array.from(unique).sort((a, b) => a.localeCompare(b));
  }, [entities]);

  const filteredTriples = useMemo(() => {
    if (!searchTerm.trim()) return triples;
    const term = searchTerm.toLowerCase();
    return triples.filter((triple) => {
      return (
        triple.relation.toLowerCase().includes(term) ||
        triple.head.name.toLowerCase().includes(term) ||
        triple.tail.name.toLowerCase().includes(term) ||
        (triple.head.label || "").toLowerCase().includes(term) ||
        (triple.tail.label || "").toLowerCase().includes(term)
      );
    });
  }, [triples, searchTerm]);

  const loadTriples = async () => {
    try {
      setLoading(true);
      setError("");
      const response = await fetch("/api/triples", { cache: "no-store" });
      const data = await response.json();
      if (!response.ok || data.error) {
        throw new Error(data.error || "Failed to load triples");
      }
      const nextTriples: Triple[] = data.triples || [];
      setTriples(nextTriples);
      const nextStates: Record<string, TripleEditState> = {};
      nextTriples.forEach((triple) => {
        nextStates[String(triple.index)] = buildEditState(triple);
      });
      setEditStates(nextStates);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load triples");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const initialize = async () => {
      const result = await bootstrapPipelineIfNeeded();
      if (!result.initialized) {
        setPipelineError(result.error || "Pipeline not initialized.");
        setLoading(false);
        return;
      }
      await loadTriples();
    };

    initialize();
  }, []);

  const updateEditState = (index: number, updates: Partial<TripleEditState>) => {
    setEditStates((prev) => {
      const key = String(index);
      const fallbackTriple = triples.find((t) => t.index === index);
      if (!prev[key] && !fallbackTriple) {
        return prev;
      }
      const current = prev[key] || buildEditState(fallbackTriple as Triple);
      return {
        ...prev,
        [key]: {
          ...current,
          ...updates,
        },
      };
    });
  };

  const handleApply = async (triple: Triple) => {
    const state = editStates[String(triple.index)];
    if (!state) return;

    const payload: Record<string, unknown> = { index: triple.index };
    let hasChanges = false;

    if (state.headSelection === NEW_ENTITY_VALUE) {
      if (!state.newHeadName.trim()) {
        updateEditState(triple.index, {
          message: "Head name is required for a new entity.",
          messageType: "error",
        });
        return;
      }
      payload.create_head = true;
      payload.head_name = state.newHeadName.trim();
      payload.head_label = state.newHeadLabel.trim() || "unknown_entity";
      hasChanges = true;
    } else if (state.headSelection && state.headSelection !== triple.head.id) {
      payload.head_id = state.headSelection;
      hasChanges = true;
    }

    if (state.tailSelection === NEW_ENTITY_VALUE) {
      if (!state.newTailName.trim()) {
        updateEditState(triple.index, {
          message: "Tail name is required for a new entity.",
          messageType: "error",
        });
        return;
      }
      payload.create_tail = true;
      payload.tail_name = state.newTailName.trim();
      payload.tail_label = state.newTailLabel.trim() || "unknown_entity";
      hasChanges = true;
    } else if (state.tailSelection && state.tailSelection !== triple.tail.id) {
      payload.tail_id = state.tailSelection;
      hasChanges = true;
    }

    if (state.relationSelection === NEW_RELATION_VALUE) {
      if (!state.newRelation.trim()) {
        updateEditState(triple.index, {
          message: "Relation is required for a new relation.",
          messageType: "error",
        });
        return;
      }
      payload.relation = state.newRelation.trim();
      hasChanges = true;
    } else if (state.relationSelection && state.relationSelection !== triple.relation) {
      payload.relation = state.relationSelection;
      hasChanges = true;
    }

    if (!hasChanges) {
      updateEditState(triple.index, {
        message: "No changes to apply.",
        messageType: "error",
      });
      return;
    }

    updateEditState(triple.index, { saving: true, message: undefined, messageType: undefined });
    try {
      const response = await fetch("/api/triples/update", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || data.error) {
        throw new Error(data.error || "Failed to update triple");
      }
      updateEditState(triple.index, {
        message: "Triple updated.",
        messageType: "success",
      });
      await loadTriples();
    } catch (err) {
      updateEditState(triple.index, {
        message: err instanceof Error ? err.message : "Failed to update triple",
        messageType: "error",
      });
    } finally {
      updateEditState(triple.index, { saving: false });
    }
  };

  const handleLabelChange = async (entityId: string, label: string, tripleIndex: number) => {
    const entity = entitiesById.get(entityId);
    if (!entity) return;

    updateEditState(tripleIndex, { message: undefined, messageType: undefined });
    try {
      const response = await fetch("/api/entities/update", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "entity",
          id: entityId,
          name: entity.name,
          label,
        }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || data.error) {
        throw new Error(data.error || "Failed to update label");
      }
      updateEditState(tripleIndex, {
        message: "Label updated.",
        messageType: "success",
      });
      await loadTriples();
    } catch (err) {
      updateEditState(tripleIndex, {
        message: err instanceof Error ? err.message : "Failed to update label",
        messageType: "error",
      });
    }
  };

  return (
    <div className={styles.layout}>
      <SideNav current="edit" />
      <div className={styles.container}>
        <header className={styles.header}>
          <div className={styles.headerContent}>
          <div className={styles.headerTitle}>
            <h1>Triple Editor</h1>
            <span>Swap head/tail entities or create new ones.</span>
          </div>
          <div className={styles.statsRow}>
            <Badge variant="secondary">{triples.length} triples</Badge>
            <Badge variant="outline">{entities.length} entities</Badge>
          </div>
          </div>
        </header>

        <main className={styles.main}>
        {pipelineError && <div className={styles.pipelineNotice}>{pipelineError}</div>}
        <div className={styles.controls}>
          <input
            className={styles.searchInput}
            type="text"
            placeholder="Search by entity, label, or relation..."
            value={searchTerm}
            onChange={(event) => setSearchTerm(event.target.value)}
          />
          <Button variant="outline" size="sm" onClick={loadTriples} disabled={loading}>
            {loading ? "Loading..." : "Refresh"}
          </Button>
        </div>

        {error && <div className={styles.pipelineNotice}>{error}</div>}

        {loading && !error && (
          <div className={styles.emptyState}>
            <p>Loading triples...</p>
          </div>
        )}

        {!loading && !error && filteredTriples.length === 0 && (
          <div className={styles.emptyState}>
            <p>No triples found.</p>
            <p>Try adjusting the search or upload a new document.</p>
          </div>
        )}

        {!loading && !error && filteredTriples.length > 0 && (
          <div className={styles.list}>
            {filteredTriples.map((triple) => {
              const state = editStates[String(triple.index)] || buildEditState(triple);
              const selectedHead = entitiesById.get(state.headSelection);
              const selectedTail = entitiesById.get(state.tailSelection);
              const headDisplay =
                state.headSelection === NEW_ENTITY_VALUE
                  ? state.newHeadName || "New entity"
                  : selectedHead?.name || "Select head";
              const tailDisplay =
                state.tailSelection === NEW_ENTITY_VALUE
                  ? state.newTailName || "New entity"
                  : selectedTail?.name || "Select tail";
              const relationDisplay =
                state.relationSelection === NEW_RELATION_VALUE
                  ? state.newRelation || "New relation"
                  : state.relationSelection || "Select relation";

              return (
                <Card key={triple.index} className={styles.tripleCard}>
                  <CardHeader>
                    <CardTitle>Triple #{triple.index + 1}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className={styles.tripleRow}>
                      <div className={styles.entityColumn}>
                        <span className={styles.entityLabel}>Head</span>
                        <Combobox
                          valueLabel={headDisplay}
                          selectedValue={state.headSelection}
                          inputValue={state.newHeadName}
                          onInputChange={(value) =>
                            updateEditState(triple.index, {
                              newHeadName: value,
                              message: undefined,
                              messageType: undefined,
                            })
                          }
                          options={entities.map((entity) => ({
                            value: entity.id,
                            label: entity.name,
                          }))}
                          onSelect={(value) =>
                            updateEditState(triple.index, {
                              headSelection: value,
                              headLabelSelection: entitiesById.get(value)?.label || "unknown_entity",
                              newHeadName: "",
                            })
                          }
                          onCreate={(value) =>
                            updateEditState(triple.index, {
                              headSelection: NEW_ENTITY_VALUE,
                              newHeadName: value,
                            })
                          }
                          placeholder="Search or add entity..."
                          createLabel="Create"
                        />
                        {state.headSelection === NEW_ENTITY_VALUE && (
                          <div className={styles.newEntityRow}>
                            <select
                              className={styles.labelSelect}
                              value={state.newHeadLabel}
                              onChange={(event) =>
                                updateEditState(triple.index, {
                                  newHeadLabel: event.target.value,
                                  message: undefined,
                                  messageType: undefined,
                                })
                              }
                            >
                              {labelOptions.map((label) => (
                                <option key={label} value={label}>
                                  {label}
                                </option>
                              ))}
                            </select>
                          </div>
                        )}
                        {state.headSelection !== NEW_ENTITY_VALUE && selectedHead && (
                          <select
                            className={styles.labelSelect}
                            value={state.headLabelSelection}
                            onChange={(event) => {
                              const nextLabel = event.target.value;
                              updateEditState(triple.index, { headLabelSelection: nextLabel });
                              void handleLabelChange(selectedHead.id, nextLabel, triple.index);
                            }}
                          >
                            {labelOptions.map((label) => (
                              <option key={label} value={label}>
                                {label}
                              </option>
                            ))}
                          </select>
                        )}
                      </div>

                      <div className={styles.relationColumn}>
                        <span className={styles.entityLabel}>Relation</span>
                        <Combobox
                          valueLabel={relationDisplay}
                          selectedValue={state.relationSelection}
                          inputValue={state.newRelation}
                          onInputChange={(value) =>
                            updateEditState(triple.index, {
                              newRelation: value,
                              message: undefined,
                              messageType: undefined,
                            })
                          }
                          options={relationOptions.map((relation) => ({
                            value: relation,
                            label: relation,
                          }))}
                          onSelect={(value) =>
                            updateEditState(triple.index, {
                              relationSelection: value,
                              newRelation: "",
                            })
                          }
                          onCreate={(value) =>
                            updateEditState(triple.index, {
                              relationSelection: NEW_RELATION_VALUE,
                              newRelation: value,
                            })
                          }
                          placeholder="Search or add relation..."
                          createLabel="Create"
                        />
                      </div>

                      <div className={styles.entityColumn}>
                        <span className={styles.entityLabel}>Tail</span>
                        <Combobox
                          valueLabel={tailDisplay}
                          selectedValue={state.tailSelection}
                          inputValue={state.newTailName}
                          onInputChange={(value) =>
                            updateEditState(triple.index, {
                              newTailName: value,
                              message: undefined,
                              messageType: undefined,
                            })
                          }
                          options={entities.map((entity) => ({
                            value: entity.id,
                            label: entity.name,
                          }))}
                          onSelect={(value) =>
                            updateEditState(triple.index, {
                              tailSelection: value,
                              tailLabelSelection: entitiesById.get(value)?.label || "unknown_entity",
                              newTailName: "",
                            })
                          }
                          onCreate={(value) =>
                            updateEditState(triple.index, {
                              tailSelection: NEW_ENTITY_VALUE,
                              newTailName: value,
                            })
                          }
                          placeholder="Search or add entity..."
                          createLabel="Create"
                        />
                        {state.tailSelection === NEW_ENTITY_VALUE && (
                          <div className={styles.newEntityRow}>
                            <select
                              className={styles.labelSelect}
                              value={state.newTailLabel}
                              onChange={(event) =>
                                updateEditState(triple.index, {
                                  newTailLabel: event.target.value,
                                  message: undefined,
                                  messageType: undefined,
                                })
                              }
                            >
                              {labelOptions.map((label) => (
                                <option key={label} value={label}>
                                  {label}
                                </option>
                              ))}
                            </select>
                          </div>
                        )}
                        {state.tailSelection !== NEW_ENTITY_VALUE && selectedTail && (
                          <select
                            className={styles.labelSelect}
                            value={state.tailLabelSelection}
                            onChange={(event) => {
                              const nextLabel = event.target.value;
                              updateEditState(triple.index, { tailLabelSelection: nextLabel });
                              void handleLabelChange(selectedTail.id, nextLabel, triple.index);
                            }}
                          >
                            {labelOptions.map((label) => (
                              <option key={label} value={label}>
                                {label}
                              </option>
                            ))}
                          </select>
                        )}
                      </div>
                    </div>

                    <div className={styles.tripleActions}>
                      <Button size="sm" onClick={() => handleApply(triple)} disabled={state.saving}>
                        {state.saving ? "Saving..." : "Apply changes"}
                      </Button>
                      {state.message && (
                        <span
                          className={`${styles.statusMessage} ${
                            state.messageType === "error"
                              ? styles.statusError
                              : state.messageType === "success"
                              ? styles.statusSuccess
                              : ""
                          }`}
                        >
                          {state.message}
                        </span>
                      )}
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        )}
        </main>
      </div>
    </div>
  );
}
