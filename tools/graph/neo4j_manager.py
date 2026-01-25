"""
Neo4j database manager for uploading knowledge graphs.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Any, List
import networkx as nx
from neo4j import GraphDatabase


class Neo4jManager:
    """
    Manages Neo4j database operations for knowledge graphs.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        """
        Initialize the Neo4j manager.

        Args:
            uri: Neo4j database URI
            user: Database username
            password: Database password
            database: Optional database name
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver = None

    def connect(self) -> None:
        """Initialize the driver if not already connected."""
        if self._driver is None:
            if not self.uri or not self.user or not self.password:
                raise ValueError("Neo4j connection details are missing.")
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self) -> None:
        """Close the driver if open."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "Neo4jManager":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

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
        database: Optional[str] = None,
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
            database: Optional database name
        """
        node_label = self.sanitize_label(node_label)
        rel_type = self.sanitize_rel_type(rel_type)
        id_to_name = id_to_name or {}

        self.connect()
        driver = self._driver
        db_name = database or self.database

        def chunked(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        node_rows = []
        for n, attrs in G.nodes(data=True):
            nid = str(n)
            props = {}
            for k, v in (attrs or {}).items():
                props[k] = self.json_safe(v)
            props.setdefault("name", id_to_name.get(nid, nid))
            props.setdefault("id", nid)
            node_rows.append({"id": nid, "props": props})

        edge_rows = []
        for u, v, k, attrs in G.edges(keys=True, data=True):
            source = str(u)
            target = str(v)
            relation = ""
            if attrs:
                relation = attrs.get("label", "") or attrs.get("relation", "") or ""

            edge_key = f"{source}||{target}||{k}||{relation}"

            props = {"relation": str(relation), "key": edge_key}
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

        with driver.session(database=db_name) as session:
            for rows in chunked(node_rows, batch_nodes):
                session.execute_write(tx_nodes, rows)

            for rows in chunked(edge_rows, batch_edges):
                session.execute_write(tx_edges, rows)

        print(
            "Uploaded to Neo4j: "
            f"nodes={len(node_rows)} edges={len(edge_rows)} "
            f"rel=:{rel_type} label=:{node_label} dedupe_edges={dedupe_edges}"
        )

    def run_cypher(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run a Cypher query and return JSON-serializable results.

        Args:
            query: Cypher query string
            parameters: Query parameters
            database: Optional database name
            limit: Optional cap on returned records

        Returns:
            Dict with keys: records, keys, summary
        """
        if not query or not query.strip():
            raise ValueError("Cypher query is required.")

        self.connect()
        db_name = database or self.database
        params = parameters or {}

        records: List[Dict[str, Any]] = []
        with self._driver.session(database=db_name) as session:
            result = session.run(query, params)
            keys = list(result.keys())
            for idx, record in enumerate(result):
                if limit is not None and idx >= limit:
                    break
                records.append({k: self.json_safe(v) for k, v in record.data().items()})
            summary = result.consume()

        counters = summary.counters
        summary_data = {
            "nodes_created": counters.nodes_created,
            "nodes_deleted": counters.nodes_deleted,
            "relationships_created": counters.relationships_created,
            "relationships_deleted": counters.relationships_deleted,
            "properties_set": counters.properties_set,
            "labels_added": counters.labels_added,
            "labels_removed": counters.labels_removed,
        }

        return {
            "records": records,
            "keys": keys,
            "summary": summary_data,
        }
