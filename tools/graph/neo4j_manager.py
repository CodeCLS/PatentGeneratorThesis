"""
Neo4j database manager for uploading knowledge graphs.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Any
import networkx as nx
from neo4j import GraphDatabase


class Neo4jManager:
    """
    Manages Neo4j database operations for knowledge graphs.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
    ):
        """
        Initialize the Neo4j manager.

        Args:
            uri: Neo4j database URI
            user: Database username
            password: Database password
        """
        self.uri = uri
        self.user = user
        self.password = password

    @staticmethod
    def sanitize_label(s: str) -> str:
        """Neo4j labels must be identifiers; keep it simple."""
        s = (s or "Entity").strip()
        s = re.sub(r"[^A-Za-z0-9_]", "_", s)
        if re.match(r"^\d", s):
            s = "_" + s
        return s or "Entity"

    @staticmethod
    def sanitize_rel_type(s: str) -> str:
        """Neo4j relationship types must be identifiers."""
        s = (s or "LINKS_TO").upper().strip()
        s = re.sub(r"[^A-Z0-9_]", "_", s)
        if re.match(r"^\d", s):
            s = "_" + s
        return s or "LINKS_TO"

    @staticmethod
    def json_safe(v: Any) -> Any:
        """Convert value to JSON-safe format."""
        if v is None:
            return None
        if isinstance(v, (str, int, float, bool, list, dict)):
            return v
        return str(v)

    def push_multidigraph_to_aura(
        self,
        G: nx.MultiDiGraph,
        node_label: str = "Entity",
        rel_type: str = "LINKS_TO",
        id_to_name: Optional[Dict[str, str]] = None,
        batch_nodes: int = 2000,
        batch_edges: int = 5000,
        dedupe_edges: bool = True,
    ) -> None:
        """
        Upload a MultiDiGraph to Neo4j Aura.

        Creates one relationship per original edge (triple), storing relation phrase
        in the relationship properties.

        Args:
            G: NetworkX MultiDiGraph to upload
            node_label: Neo4j node label
            rel_type: Neo4j relationship type
            id_to_name: Optional mapping from node ID to display name
            batch_nodes: Batch size for node uploads
            batch_edges: Batch size for edge uploads
            dedupe_edges: Whether to deduplicate edges by key
        """
        node_label = self.sanitize_label(node_label)
        rel_type = self.sanitize_rel_type(rel_type)
        id_to_name = id_to_name or {}

        driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

        def chunked(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i+n]

        # Build node rows
        node_rows = []
        for n, attrs in G.nodes(data=True):
            nid = str(n)
            props = {}
            for k, v in (attrs or {}).items():
                props[k] = self.json_safe(v)
            # Friendly name (optional)
            props.setdefault("name", id_to_name.get(nid, nid))
            props.setdefault("id", nid)
            node_rows.append({"id": nid, "props": props})

        # Build edge rows (NO collapsing)
        edge_rows = []
        for u, v, k, attrs in G.edges(keys=True, data=True):
            source = str(u)
            target = str(v)
            relation = ""
            if attrs:
                relation = attrs.get("label", "") or attrs.get("relation", "") or ""

            # Make a stable edge key if you want dedupe / traceability
            edge_key = f"{source}||{target}||{k}||{relation}"

            props = {"relation": str(relation), "key": edge_key}
            # Keep any other edge attrs too (optional)
            for kk, vv in (attrs or {}).items():
                if kk == "label":
                    continue
                props[kk] = self.json_safe(vv)

            edge_rows.append({"source": source, "target": target, "props": props})

        def tx_nodes(tx, rows):
            tx.run(
                f"""
                UNWIND $rows AS row
                MERGE (n:{node_label} {{id: row.id}})
                SET n += row.props
                """,
                rows=rows,
            )

        def tx_edges(tx, rows):
            if dedupe_edges:
                # Distinct edges by key+relation to avoid duplicates across runs
                tx.run(
                    f"""
                    UNWIND $rows AS row
                    MATCH (a:{node_label} {{id: row.source}})
                    MATCH (b:{node_label} {{id: row.target}})
                    MERGE (a)-[r:{rel_type} {{key: row.props.key}}]->(b)
                    SET r += row.props
                    """,
                    rows=rows,
                )
            else:
                # Create every edge (can blow up if you rerun)
                tx.run(
                    f"""
                    UNWIND $rows AS row
                    MATCH (a:{node_label} {{id: row.source}})
                    MATCH (b:{node_label} {{id: row.target}})
                    CREATE (a)-[r:{rel_type}]->(b)
                    SET r += row.props
                    """,
                    rows=rows,
                )

        with driver.session() as session:
            for rows in chunked(node_rows, batch_nodes):
                session.execute_write(tx_nodes, rows)

            for rows in chunked(edge_rows, batch_edges):
                session.execute_write(tx_edges, rows)

        driver.close()
        print(
            f"✅ Uploaded to Aura: nodes={len(node_rows)} edges={len(edge_rows)} "
            f"rel=:{rel_type} label=:{node_label} dedupe_edges={dedupe_edges}"
        )

