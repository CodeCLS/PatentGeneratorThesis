# pip install -qU "langchain[anthropic]" langsmith python-dotenv

import os
import re
import json
from typing import List, Dict, Any

import anthropic
from dotenv import load_dotenv
from langsmith import traceable, Client as LSClient
from langsmith.wrappers import wrap_anthropic
from networktest import KnowledgeGraph
load_dotenv()

# --- LangSmith + Anthropic setup ---
# Ensure you have:
#   LANGCHAIN_TRACING_V2=true
#   LANGCHAIN_API_KEY=...
#   LANGCHAIN_PROJECT=...
# (Optional in serverless) set LANGSMITH_TRACING_BACKGROUND=false or flush() at the end.

client = wrap_anthropic(anthropic.Anthropic())
ls_client = LSClient()

# ---------- helpers ----------
def _split_into_sentences(text: str) -> List[str]:
    # Simple, punctuation-aware splitter that keeps things practical without extra deps.
    text = re.sub(r'\s+', ' ', text).strip()
    # Split on . ! ? that are followed by space/line/end, avoid splitting on e.g. "Inc." heuristically
    parts = re.split(r'(?<!\b[A-Z])[.!?](?=\s|$)', text)
    return [s.strip() for s in parts if s.strip()]

def _llm_json(prompt: str, model: str = "claude-sonnet-4-20250514") -> Dict[str, Any]:
    """
    Calls Anthropic and *expects* strict JSON in the response.
    We enforce this via the system prompt and validate on parse.
    """
    resp = client.messages.create(
        model=model,
        max_tokens=1500,
        system=(
            "You are a disciplined JSON generator. "
            "Always respond with a *single* JSON object and nothing else. "
            "No prose, no backticks."
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text if resp.content else "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to salvage JSON if the model emitted stray text
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        return json.loads(m.group(0)) if m else {}

# ---------- Step 1: simplify sentences ----------
@traceable(name="Split into Simpler Sentences", run_type="chain")
def split_sentences(company_text: str) -> Dict[str, Any]:
    sentences = _split_into_sentences(company_text)

    # schema: each original sentence -> many split sentences
    schema_hint = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},          # "s1"
                        "original": {"type": "string"},
                        "splits": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sid": {"type": "string"},  # "s1.1"
                                    "simple": {"type": "string"}
                                },
                                "required": ["sid", "simple"]
                            }
                        }
                    },
                    "required": ["id", "original", "splits"]
                }
            }
        },
        "required": ["items"]
    }

    numbered = [{"id": f"s{i+1}", "original": s} for i, s in enumerate(sentences)]

    user_prompt = (
        "You will receive numbered company sentences.\n"
        "For each original sentence S (e.g., s1), split S into multiple *short* simple sentences "
        "s1.1, s1.2, s1.3, ... so that:\n"
        " - All facts in S are preserved across the splits (do not omit info).\n"
        " - No new info is added.\n"
        " - Use active voice and concrete words.\n"
        " - Keep the order of information.\n"
        " - Prefer 2–6 splits per original sentence depending on complexity.\n\n"
        f"Return a single JSON object following this schema (hint): {json.dumps(schema_hint)}\n\n"
        "Input:\n" + json.dumps(numbered, ensure_ascii=False, indent=2)
    )

    out = _llm_json(user_prompt)
    items = out.get("items", [])
    # Fallback: if model returns nothing, put each original as one split (sid = id + '.1')
    if not items:
        items = [{
            "id": x["id"],
            "original": x["original"],
            "splits": [{"sid": f'{x["id"]}.1', "simple": x["original"]}]
        } for x in numbered]
    return {"items": items}

# ---------- Step 2: reference subjects & relationships ----------
@traceable(name="Reference & Relationship Extraction", run_type="chain")
def reference_and_relate_per_split(split_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    - Builds a global entity dictionary (E1, E2, ...).
    - Annotates each split sentence (sid like s1.1) with [Ref:EX].
    - Extracts triples using entity IDs; each triple cites the split sid as evidence.
    """

    # Flatten splits to a list of {sid, simple, parent_id}
    flat_splits = []
    for item in split_payload.get("items", []):
        for sp in item.get("splits", []):
            flat_splits.append({
                "sid": sp["sid"],
                "simple": sp["simple"],
                "parent": item["id"]
            })

    schema_hint = {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "eid": {"type": "string"},  # E1
                        "name": {"type": "string"},
                        "type": {"type": "string"}
                    },
                    "required": ["eid", "name"]
                }
            },
            "annotated_splits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sid": {"type": "string"},
                        "annotated": {"type": "string"}  # with [Ref:EX] tags
                    },
                    "required": ["sid", "annotated"]
                }
            },
            "triples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "s": {"type": "string"},  # E#
                        "p": {"type": "string"},
                        "o": {"type": "string"},  # E#
                        "justification_sid": {"type": "string"}  # s1.2, etc.
                    },
                    "required": ["s", "p", "o", "justification_sid"]
                }
            }
        },
        "required": ["entities", "annotated_splits", "triples"]
    }

    prompt = (
        "Task: From the split simple sentences, create a global entity list and annotate each split with [Ref:EX] tags.\n"
        "Rules:\n"
        "1) Deduplicate entities across all splits. Assign IDs E1, E2, ...\n"
        "2) In each split sentence, add [Ref:EX] immediately after each mention of a subject/object entity.\n"
        "3) Extract factual triples (s,p,o) using only entity IDs (s/o) and short predicates.\n"
        "4) Use entities only as needed (companies, products, markets, partners, dates, locations, metrics).\n"
        "5) Do not invent facts. Keep the order and meaning of the splits.\n\n"
        f"Return a single JSON object (hint): {json.dumps(schema_hint)}\n\n"
        "Split sentences:\n" + json.dumps(flat_splits, ensure_ascii=False, indent=2)
    )

    out = _llm_json(prompt)
    out.setdefault("entities", [])
    out.setdefault("annotated_splits", [])
    out.setdefault("triples", [])
    return out

# ---------- Step 3: build a knowledge graph structure ----------
@traceable(name="Build KG", run_type="tool")
def build_kg(ref_data: Dict[str, Any]) -> Dict[str, Any]:
    entities = ref_data.get("entities", [])
    triples = ref_data.get("triples", [])

    nodes = [{"id": e["eid"], "label": e["name"], "type": e.get("type", "Entity")} for e in entities]
    edges = [{
        "source": t["s"],
        "predicate": t["p"],
        "target": t["o"],
        "evidence": t.get("justification_sid")
    } for t in triples]

    return {"nodes": nodes, "edges": edges}
# --- New: induce a per-document relation schema ---
@traceable(name="Induce Relation Schema", run_type="chain")
def induce_relation_schema(flat_splits: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Given split sentences, ask the LLM to propose a compact canonical relation set
    tailored to THIS document. Each relation has: name, definition, examples,
    expected domain/range (broad), and allowed argument roles.
    """
    prompt = {
        "task": "Propose a compact relation schema for patent-grade KGs from these sentences.",
        "requirements": [
            "8–20 relations total; no duplicates; short snake_case names.",
            "Each: {name, definition, examples: [surface_phrases], domain_hint, range_hint}.",
            "Definitions must be generic (cross-domain), not document-specific.",
            "Prefer structural (part_of, connected_to, mounted_to, located_adjacent, located_above/below, contains, passes_through) and functional (configured_to, enables, prevents, measures, supplies, receives_from, transmits_to, causes).",
            "If a surface phrase is too niche, allow a custom:* relation (e.g., custom_guides_flow) with a precise definition."
        ],
        "sentences": flat_splits[:120]  # cap to keep prompt bounded
    }
    out = _llm_json(json.dumps(prompt))
    out.setdefault("relations", [])
    # Minimal sanity filter
    rels = []
    seen = set()
    for r in out["relations"]:
        name = r.get("name","").strip()
        if not name or name in seen: 
            continue
        seen.add(name)
        rels.append({
            "name": name,
            "definition": r.get("definition",""),
            "examples": r.get("examples", []),
            "domain_hint": r.get("domain_hint","Entity"),
            "range_hint": r.get("range_hint","Entity"),
        })
    return {"relations": rels}
def _get(o: Dict[str, Any], key: str, default):
    v = o.get(key, default)
    return v if v is not None else default

# --- New: open IE extraction (surface-level, no constraints) ---
@traceable(name="Open Triple Extraction", run_type="chain")
def open_triple_extract(flat_splits: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Extract tuples with SURFACE predicates (+ modality/condition text).
    Returns at least an empty list, never None.
    """
    schema = {
        "type": "object",
        "properties": {
            "triples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sid": {"type": "string"},
                        "subject": {"type": "string"},
                        "surface_pred": {"type": "string"},
                        "object": {"type": "string"},
                        "condition_text": {"type": "string"},
                        "modality": {"type": "string"}  # 'capability'|'requirement'|'optional'|''
                    },
                    "required": ["sid", "subject", "surface_pred", "object"]
                }
            }
        },
        "required": ["triples"]
    }

    prompt = {
        "task": "Extract factual tuples from the sentences. Use the original wording for the predicate.",
        "rules": [
            "Return EXACTLY one JSON object with a key 'triples' (array).",
            "Every item MUST have keys: sid, subject, surface_pred, object.",
            "Resolve pronouns to concrete nouns whenever possible.",
            "If a clause has conditions (when/if/so that), copy the raw text into condition_text.",
            "Map 'may/can'→'capability', 'must/required'→'requirement', 'optional'→'optional'; else ''."
        ],
        "schema_hint": schema,
        "example": {
            "triples": [
                {
                    "sid": "s1.1",
                    "subject": "pump",
                    "surface_pred": "is connected to",
                    "object": "controller",
                    "condition_text": "",
                    "modality": ""
                }
            ]
        },
        "splits": flat_splits[:160]
    }

    out = _llm_json(json.dumps(prompt))
    triples = _get(out, "triples", [])
    # sanitize: keep only well-formed items
    clean = []
    for t in triples:
        sid = t.get("sid"); subj = t.get("subject"); pred = t.get("surface_pred"); obj = t.get("object")
        if sid and subj and pred and obj:
            clean.append({
                "sid": sid,
                "subject": subj.strip(),
                "surface_pred": pred.strip(),
                "object": obj.strip(),
                "condition_text": t.get("condition_text",""),
                "modality": t.get("modality","")
            })
    return {"triples": clean}


# --- New: normalize entities & predicates to the induced schema + self-consistency ---
@traceable(name="Normalize Triples", run_type="chain")
def normalize_triples(triples: List[Dict[str,Any]], rel_schema: Dict[str,Any], votes:int=3) -> Dict[str, Any]:
    """
    Maps surface triples to the induced schema and builds a KG.
    Guarantees non-empty nodes/edges when input triples exist.
    """
    relations = _get(rel_schema, "relations", [])
    rel_names = [r.get("name","") for r in relations]

    if not triples:
        return {"nodes": [], "edges": [], "relations": relations}

    # --- LLM mapper (schema strict) ---
    def ask_llm(batch):
        schema = {
            "type": "object",
            "properties": {
                "normalized": {
                    "type":"array","items":{
                        "type":"object","properties":{
                            "evidence_sid":{"type":"string"},
                            "canonical_subject":{"type":"string"},
                            "canonical_object":{"type":"string"},
                            "canonical_pred":{"type":"string"},
                            "surface_pred":{"type":"string"},
                            "condition":{"type":"string"},
                            "modality":{"type":"string"},
                            "subject_attrs":{"type":"array","items":{"type":"string"}},
                            "object_attrs":{"type":"array","items":{"type":"string"}}
                        },
                        "required":[
                            "evidence_sid","canonical_subject","canonical_object",
                            "canonical_pred","surface_pred"
                        ]
                    }
                }
            },
            "required": ["normalized"]
        }

        prompt = {
            "task": "Map surface triples to canonical schema and canonicalize entities.",
            "schema_relations": relations,
            "mapping_rules": [
                "Choose the most appropriate canonical relation by SEMANTICS.",
                "If none fits, use custom:<short_snake_case> and keep the exact surface_pred.",
                "Canonicalize entity labels to head noun lemmas; remove purpose/marketing adjectives ('for display', 'for appreciation') and legalese ('said', 'the').",
                "Keep functional modifiers as attributes in subject_attrs / object_attrs.",
                "Return EXACTLY one JSON object with key 'normalized' (array)."
            ],
            "example": {
                "normalized":[
                    {
                        "evidence_sid":"s1.1",
                        "canonical_subject":"pump",
                        "canonical_object":"controller",
                        "canonical_pred":"connected_to",
                        "surface_pred":"is connected to",
                        "condition":"",
                        "modality":"",
                        "subject_attrs":[],
                        "object_attrs":[]
                    }
                ]
            },
            "input_triples": batch
        }
        out = _llm_json(json.dumps(prompt))
        return _get(out, "normalized", [])

    # Voting for stability (and also to populate something even if model is flaky)
    vote_buckets: Dict[Tuple[str,str,str,str], Dict[str,int]] = {}
    memo_any_items = []

    chunk = 40
    for i in range(0, len(triples), chunk):
        batch = triples[i:i+chunk]
        round_items = []
        for _ in range(max(1, votes)):
            norm = ask_llm(batch)
            round_items.extend(norm)
            for item in norm:
                key = (
                    item.get("evidence_sid",""),
                    item.get("canonical_subject",""),
                    item.get("canonical_object",""),
                    item.get("surface_pred","")
                )
                pred = item.get("canonical_pred","")
                if not all(key) or not pred:
                    continue
                vote_buckets.setdefault(key, {}).setdefault(pred, 0)
                vote_buckets[key][pred] += 1
        memo_any_items.extend(round_items)

    # If LLM gave absolutely nothing, FALL BACK to direct edges
    if not vote_buckets and not memo_any_items:
        # Deterministic fallback: keep surface_pred as custom_*, simple lemmatization-ish cleanup
        def simple_label(s: str) -> str:
            s = re.sub(r"\b(for|of|the|a|an|said)\b", "", s, flags=re.I).strip()
            return re.sub(r"\s+", " ", s).lower()

        nodes, node_index, edges = [], {}, []
        def nid(label):
            if label not in node_index:
                node_index[label] = f"N{len(node_index)+1}"
                nodes.append({"id": node_index[label], "label": label, "type":"Entity", "attributes":[]})
            return node_index[label]

        for t in triples:
            s = simple_label(t["subject"]); o = simple_label(t["object"])
            p = "custom_" + re.sub(r"[^a-z0-9]+","_", t["surface_pred"].lower()).strip("_")
            sid = t["sid"]
            s_id, o_id = nid(s), nid(o)
            edges.append({
                "source": s_id, "target": o_id,
                "predicate": p, "surface_pred": t["surface_pred"],
                "evidence": sid, "confidence": 0.4
            })
        return {"nodes": nodes, "edges": edges, "relations": relations}

    # Majority selection + build graph
    nodes, node_index, edges = [], {}, []
    def nid(label):
        if label not in node_index:
            node_index[label] = f"N{len(node_index)+1}"
            nodes.append({"id": node_index[label], "label": label, "type":"Entity", "attributes":[]})
        return node_index[label]

    for (sid, subj, obj, surf), pred_counts in vote_buckets.items():
        pred = max(pred_counts.items(), key=lambda kv: kv[1])[0]
        s_id, o_id = nid(subj), nid(obj)
        edges.append({
            "source": s_id,
            "target": o_id,
            "predicate": pred,
            "surface_pred": surf,
            "evidence": sid,
            "confidence": max(pred_counts.values()) / max(1, sum(pred_counts.values()))
        })

    # Keep custom:* or schema relations; drop garbage
    allowed = set(rel_names) | {e["predicate"] for e in edges if e["predicate"].startswith("custom_")}
    edges = [e for e in edges if e["predicate"] in allowed]

    return {"nodes": nodes, "edges": edges, "relations": relations}


# ---------- Orchestrator ----------
@traceable(name="Company → KG Pipeline", run_type="chain")
def company_to_kg(company_text: str) -> Dict[str, Any]:
    step1 = split_sentences(company_text)

    # Flatten splits for downstream prompts
    flat = []
    for item in step1.get("items", []):
        for sp in item.get("splits", []):
            flat.append({"sid": sp["sid"], "text": sp["simple"]})

    # (A) Induce a compact relation schema for THIS text
    schema_doc = induce_relation_schema(flat)

    # (B) Open extraction with surface predicates (no constraints)
    open_ie = open_triple_extract(flat)

    # (C) Normalize to the schema (with self-consistency voting)
    normalized = normalize_triples(open_ie["triples"], schema_doc, votes=3)

    result = {
        "input": company_text,
        "splits": step1,
        "relation_schema": schema_doc,
        "kg": normalized
    }
    return result


# ---------- Example usage ----------
if __name__ == "__main__":
    company_blob = (
        """A water tank for appreciation provided with a
        water storage, a water pipe, an air bubble generating
        member, wherein said water storage is provided with a
        water inlet port and an opening portion and said opening
        portion is so provided that the convection can be generated in said water tank for appreciation by the liquid
        current from a water tank through said opening portion,
        and said water storage is installed upwardly of a water
        tank for appreciation and one end of said water pipe is
        connected to an inlet water port of a water storage and
        the other end is so installed that it is in the liquid when
        the liquid is filled in a water tank for appreciation and an
        air bubble generating member is so installed that it can
        lead the air bubble inside of a water pipe in the peripheral portion of said other end of said water pipe and a
        display device for appreciation using said water tank for
        appreciation are used."""
    )

    # Run your pipeline (split -> reference -> KG)
    result = company_to_kg(company_blob)

    # Build the graph directly from the pipeline output (dict)
    kg = KnowledgeGraph.from_pipeline_output(result)
    print("nodes:", len(kg.graph.nodes), "edges:", len(kg.graph.edges))
    kg.draw()


    # ---- Optional: inspect the structured outputs ----
    print("\n=== Splits ===")
    for item in result["splits"]["items"]:
        print(f'{item["id"]}: {item["original"]}')
        for sp in item["splits"]:
            print(f'  - {sp["sid"]}: {sp["simple"]}')

    print("\n=== Annotated Splits ===")
    for a in result["referenced"]["annotated_splits"]:
        print(f'{a["sid"]}: {a["annotated"]}')

    print("\n=== Entities ===")
    for e in result["referenced"]["entities"]:
        print(f'{e["eid"]}: {e["name"]} ({e.get("type","Entity")})')

    print("\n=== Triples ===")
    for t in result["referenced"]["triples"]:
        print(f'{t["s"]} -[{t["p"]}]-> {t["o"]} (evidence: {t.get("justification_sid")})')
