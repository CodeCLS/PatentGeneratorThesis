"""
Patent claim generator that converts claim clusters into formal patent claims.

This version is tuned for reconstructing claims from knowledge-graph clusters:
- Fidelity-first: do not invent components/capabilities beyond the triples.
- Prefer structural claim drafting (components + connections/placements).
- Avoid "means for" functional claiming unless the triples themselves use "means".
- Independent claim: minimal core configuration.
- Dependent claim: add ONE additional limitation (or two tightly-related).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
import json
import re

from tools.graph.claim_clusterers import ClaimCluster
from tools.api.llm_api_repo import LLmApi_Repo


class PatentClaimGenerator:
    """
    Generates formal patent claims from claim clusters.
    """

    def __init__(self, style: str = "formal"):
        self.style = style
        self.api_repo = LLmApi_Repo()

    @staticmethod
    def _format_cluster_for_prompt(
        cluster: ClaimCluster,
        id_to_name: Dict[str, str],
        claim_number: int,
        is_independent: bool,
    ) -> str:
        claim_type = "INDEPENDENT" if is_independent else "DEPENDENT"

        triples_text: List[str] = []
        for i, (head, tail, relation) in enumerate(sorted(cluster.edges), 1):
            head_name = id_to_name.get(head, head)
            tail_name = id_to_name.get(tail, tail)
            triples_text.append(f"  {i}. {head_name} --[{relation}]--> {tail_name}")

        description = f"""
CLAIM {claim_number} ({claim_type}):
- Number of triples: {cluster.size()}
- Number of entities: {cluster.node_count()}
- Priority: {cluster.priority}
- Metadata: {cluster.metadata if cluster.metadata else 'None'}

Knowledge Graph Triples (authoritative; do not invent beyond these):
{chr(10).join(triples_text)}
""".strip()
        return description

    def _style_text(self) -> str:
        styles = {
            "formal": (
                "Use standard U.S.-style claim drafting: one sentence; 'comprising'; "
                "clear antecedent basis; consistent terminology."
            ),
            "detailed": (
                "Use standard claim drafting but slightly more detail; still one sentence; "
                "avoid optional fluff."
            ),
            "concise": (
                "Use standard claim drafting but keep it compact; still one sentence; "
                "include only essential structure."
            ),
        }
        return styles.get(self.style, styles["formal"])

    @staticmethod
    def _extract_invention_entities(
        id_to_name: Dict[str, str],
        id_to_label: Optional[Dict[str, str]] = None,
        all_clusters: Optional[List[ClaimCluster]] = None,
    ) -> List[str]:
        """
        Extract all entities labeled as "INVENTION" from the knowledge graph.
        
        Args:
            id_to_name: Mapping from entity ID to display name
            id_to_label: Optional mapping from entity ID to label/type
            all_clusters: Optional list of all clusters to search for INVENTION entities
        
        Returns:
            List of INVENTION entity names
        """
        invention_entities = []
        
        if id_to_label:
            # Direct mapping available
            for entity_id, label in id_to_label.items():
                if label and label.upper() == "INVENTION":
                    name = id_to_name.get(entity_id, entity_id)
                    if name:
                        invention_entities.append(name)
        elif all_clusters:
            # Search through all clusters for nodes that might be INVENTION
            # This is a fallback if id_to_label is not available
            seen_ids = set()
            for cluster in all_clusters:
                for node_id in cluster.nodes:
                    if node_id not in seen_ids:
                        seen_ids.add(node_id)
                        # Try to infer from node_id or name
                        name = id_to_name.get(node_id, node_id)
                        # If the name contains "invention" or similar, include it
                        if name and ("invention" in name.lower() or "present invention" in name.lower()):
                            invention_entities.append(name)
        
        return list(set(invention_entities))  # Remove duplicates
    
    def _build_prompt(
        self,
        cluster: ClaimCluster,
        id_to_name: Dict[str, str],
        claim_number: int,
        is_independent: bool,
        previous_claims: Optional[List[str]] = None,
        reference_claim_number: Optional[int] = None,
        id_to_label: Optional[Dict[str, str]] = None,
        all_clusters: Optional[List[ClaimCluster]] = None,
    ) -> str:
        cluster_desc = self._format_cluster_for_prompt(
            cluster, id_to_name, claim_number, is_independent
        )

        ref_claim_no = None
        ref_claim_text = None
        if not is_independent:
            ref_claim_no = reference_claim_number or 1
            if previous_claims:
                if 1 <= ref_claim_no <= len(previous_claims):
                    ref_claim_text = previous_claims[ref_claim_no - 1]
                else:
                    ref_claim_no = 1
                    ref_claim_text = previous_claims[0] if previous_claims else None

        # Word budgets: prompt-level discipline (not word bans)
        indep_budget = "Target ≤ 120 words."
        dep_budget = "Target ≤ 60 words."

        # Extract INVENTION entities for context
        invention_entities = self._extract_invention_entities(
            id_to_name=id_to_name,
            id_to_label=id_to_label,
            all_clusters=all_clusters,
        )
        
        invention_context = ""
        if invention_entities:
            invention_list = ", ".join(invention_entities)
            invention_context = f"""
CRITICAL CONTEXT - THE TRUE INVENTION:
The following entities are labeled as "INVENTION" in the knowledge graph:
{invention_list}

This is the core invention being claimed. Ensure your claim properly reflects this invention and does not confuse it with components or other entities.
"""
        
        claim_rules = f"""
You are drafting ONE patent claim STRICTLY from the provided knowledge-graph triples.
Objective: MAXIMIZE FIDELITY to the triples. Do NOT add new components/capabilities.

{invention_context}
STYLE:
{self._style_text()}

GLOBAL HARD CONSTRAINTS:
1) Use ONLY elements/relationships present in the triples. If a feature is not supported by the triples, omit it.
2) Prefer STRUCTURE terms that appear in the triples (e.g., tank, pipe, inlet port, opening portion, coils, etc.).
3) Avoid "means for" / purely functional claiming UNLESS the triples explicitly use 'means' wording. Use concrete structure whenever available.
4) Avoid subjective results, impressions, user perception, marketing language, or purpose statements unless explicitly present and clearly claimable.
5) Output ONLY the claim sentence. No numbering. No explanation. No markdown.
""".strip()

        if is_independent:
            claim_type_rules = f"""
INDEPENDENT CLAIM RULES:
- Standalone: no reference to other claims.
- Include the MINIMUM core apparatus topology supported by the triples: components + spatial/connection relationships.
- Do NOT pack optional embodiments into claim 1.
- One sentence. {indep_budget}
""".strip()
            anchor = ""
        else:
            claim_type_rules = f"""
DEPENDENT CLAIM RULES:
- Must reference claim {ref_claim_no} (e.g., "The device of claim {ref_claim_no}, wherein ..." or "... further comprising ...").
- Add ONLY ONE additional limitation (or two tightly-related limitations) supported by THIS cluster’s triples.
- One sentence. {dep_budget}
""".strip()
            anchor = ""
            if ref_claim_text:
                anchor = f"""

Referenced Claim {ref_claim_no} (anchor; do not copy blindly):
{ref_claim_text}
""".rstrip()

        # Two-step instruction inside a single response (no extra API calls):
        # Step A: select minimal allowed set from triples
        # Step B: draft claim using only that set
        # We then parse only the final "CLAIM:" line in _parse_llm_response.
        prompt = f"""{cluster_desc}

{claim_rules}

{claim_type_rules}
{anchor}

WORKING METHOD (must follow):
Step A — SELECTION:
- From the triples, select a minimal set of components and relations needed for this claim.
- Independent claim: select ~4–7 core components/relations.
- Dependent claim: select exactly 1 added limitation (or 2 tightly-related) beyond the referenced claim.
- Do NOT introduce any concept not directly supported by the triples.

Step B — DRAFTING:
- Draft the claim using ONLY what you selected in Step A.
- Use concrete structural language when possible (avoid "means for" unless unavoidable and supported by triples).
- Keep within the word budget.

OUTPUT FORMAT (strict):
- First line: "SELECTION: " followed by a compact JSON object with keys:
  - components: [..]
  - relations: [..]
  - added_limitations: [..]   (empty for independent)
- Second line: "CLAIM: " followed by the final claim sentence.

Now produce the output in the required two-line format.
"""
        return prompt

    def _parse_llm_response(self, response: Any) -> str:
        """Parse LLM response to extract ONLY the claim text."""
        text = ""
        if isinstance(response, dict):
            text = response.get("content", response.get("text", response.get("message", ""))) or ""
            if not text and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if isinstance(choice, dict):
                    text = (choice.get("message", {}) or {}).get("content", "") or ""
            if not text:
                for v in response.values():
                    if isinstance(v, str):
                        text = v
                        break
        elif isinstance(response, list):
            text = " ".join(str(x) for x in response) if response else ""
        elif isinstance(response, str):
            text = response
        else:
            text = str(response)

        text = text.strip()
        text = re.sub(r"^```(?:markdown|text)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

        # Prefer extracting the "CLAIM:" line from the enforced two-line format
        m = re.search(r"(?:^|\n)\s*CLAIM:\s*(.+)\s*$", text, flags=re.IGNORECASE)
        if m:
            claim = m.group(1).strip()
        else:
            # Fallback: remove common prefixes and compress whitespace
            claim = re.sub(r"^(?:Claim\s+\d+[:.]?\s*|\d+[.:]\s*)", "", text, flags=re.IGNORECASE).strip()
            claim = re.sub(
                r"^.*?(?:here is|the claim is|claim text|patent claim)[:.]?\s*",
                "",
                claim,
                flags=re.IGNORECASE,
            ).strip()

        claim = re.sub(r"\s+", " ", claim).strip()
        return claim

    def generate_claim(
        self,
        cluster: ClaimCluster,
        id_to_name: Dict[str, str],
        claim_number: int,
        previous_claims: Optional[List[str]] = None,
        reference_claim_number: Optional[int] = None,
        id_to_label: Optional[Dict[str, str]] = None,
        all_clusters: Optional[List[ClaimCluster]] = None,
    ) -> str:
        is_independent = cluster.claim_type == "independent"

        prompt = self._build_prompt(
            cluster=cluster,
            id_to_name=id_to_name,
            claim_number=claim_number,
            is_independent=is_independent,
            previous_claims=previous_claims,
            reference_claim_number=reference_claim_number,
            id_to_label=id_to_label,
            all_clusters=all_clusters,
        )

        try:
            response = self.api_repo.chat(prompt)
            claim_text = self._parse_llm_response(response)

            if not claim_text:
                return "[Error generating claim: empty response]"

            return claim_text

        except Exception as e:
            return f"[Error generating claim: {e}]"

    def generate_claims(
        self,
        clusters: List[ClaimCluster],
        id_to_name: Optional[Dict[str, str]] = None,
        id_to_label: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        id_to_name = id_to_name or {}

        independent_clusters = [c for c in clusters if c.claim_type == "independent"]
        dependent_clusters = [c for c in clusters if c.claim_type == "dependent"]

        generated_claims: List[Dict[str, Any]] = []
        previous_claim_texts: List[str] = []

        # Independent claims first
        for i, cluster in enumerate(independent_clusters, 1):
            claim_text = self.generate_claim(
                cluster=cluster,
                id_to_name=id_to_name,
                claim_number=i,
                previous_claims=None,
                reference_claim_number=None,
                id_to_label=id_to_label,
                all_clusters=clusters,
            )
            generated_claims.append(
                {
                    "claim_number": i,
                    "claim_type": "independent",
                    "claim_text": claim_text,
                    "cluster_id": cluster.cluster_id,
                    "cluster_size": cluster.size(),
                    "priority": cluster.priority,
                }
            )
            previous_claim_texts.append(claim_text)

        # Dependent claims default to referencing claim 1 (common pattern).
        # If you later add metadata to clusters (e.g., depends_on), you can pass that here
        # without changing external callers.
        for j, cluster in enumerate(dependent_clusters, 1):
            claim_number = len(independent_clusters) + j
            claim_text = self.generate_claim(
                cluster=cluster,
                id_to_name=id_to_name,
                claim_number=claim_number,
                previous_claims=previous_claim_texts,
                reference_claim_number=1,
                id_to_label=id_to_label,
                all_clusters=clusters,
            )
            generated_claims.append(
                {
                    "claim_number": claim_number,
                    "claim_type": "dependent",
                    "claim_text": claim_text,
                    "cluster_id": cluster.cluster_id,
                    "cluster_size": cluster.size(),
                    "priority": cluster.priority,
                }
            )
            previous_claim_texts.append(claim_text)

        return generated_claims
