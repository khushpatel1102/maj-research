"""
Neo4j Graph Manager for MAJ

Handles CRUD operations and vector similarity search for:
- Attempt → SATISFIES → Policy
- Attempt → CAUSES → Issue
- Fix → RESOLVES → Issue
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from models import Policy, Attempt, Issue, Fix, Semantic

load_dotenv()

EMBEDDING_DIM = 1536
DEFAULT_POLICY_THRESHOLD = 0.9


class FrozenMemoryViolation(RuntimeError):
    """Raised when a write is attempted on a frozen GraphManager."""


# Methods that mutate the graph; blocked while frozen.
_WRITE_METHODS = (
    "create_policy",
    "create_attempt",
    "create_issue",
    "create_fix",
    "create_semantic",
    "get_or_create_policy",
    "get_or_create_semantic",
    "link_attempt_satisfies_policy",
    "link_attempt_causes_issue",
    "link_fix_resolves_issue",
    "link_issue_abstracts_to_semantic",
    "clear_all",
)


class _FrozenMemoryContext:
    """Context manager that monkey-patches write methods to raise."""

    def __init__(self, gm):
        self.gm = gm
        self._originals = {}

    def __enter__(self):
        def make_blocker(name):
            def _blocker(*args, **kwargs):
                raise FrozenMemoryViolation(
                    f"Write attempted on frozen GraphManager: {name}"
                )
            return _blocker

        for name in _WRITE_METHODS:
            if hasattr(self.gm, name):
                self._originals[name] = getattr(self.gm, name)
                setattr(self.gm, name, make_blocker(name))
        return self.gm

    def __exit__(self, exc_type, exc, tb):
        for name, fn in self._originals.items():
            setattr(self.gm, name, fn)
        self._originals.clear()
        return False


class GraphManager:
    def __init__(self, policy_threshold: float = None):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )
        self.policy_threshold = policy_threshold or float(os.getenv("POLICY_THRESHOLD", DEFAULT_POLICY_THRESHOLD))
        self._ensure_vector_indexes()

    def close(self):
        self.driver.close()

    def _ensure_vector_indexes(self):
        """Create vector indexes for similarity search."""
        with self.driver.session() as session:
            for label in ["Policy", "Attempt", "Issue", "Fix", "Semantic"]:
                session.run(f"""
                    CREATE VECTOR INDEX {label.lower()}_embedding IF NOT EXISTS
                    FOR (n:{label})
                    ON (n.embedding)
                    OPTIONS {{indexConfig: {{
                        `vector.dimensions`: {EMBEDDING_DIM},
                        `vector.similarity_function`: 'cosine'
                    }}}}
                """)

    # --- Create Nodes ---

    def get_or_create_policy(self, policy: Policy, threshold: float = None) -> tuple[str, bool]:
        """
        Get existing policy if similar one exists, otherwise create new.

        Returns: (policy_id, is_new)
        """
        threshold = threshold or self.policy_threshold

        # Check for similar existing policy
        similar = self.find_similar_policies(policy.embedding, top_k=1)
        if similar and similar[0]['score'] >= threshold:
            return similar[0]['id'], False

        # Create new policy
        with self.driver.session() as session:
            session.run(
                "CREATE (p:Policy {id: $id, description: $description, embedding: $embedding})",
                **policy.to_neo4j_props()
            )
        return policy.id, True

    def create_policy(self, policy: Policy) -> str:
        with self.driver.session() as session:
            session.run(
                "CREATE (p:Policy {id: $id, description: $description, embedding: $embedding})",
                **policy.to_neo4j_props()
            )
        return policy.id

    def create_attempt(self, attempt: Attempt) -> str:
        with self.driver.session() as session:
            session.run(
                """CREATE (a:Attempt {
                    id: $id,
                    agent_output: $agent_output,
                    is_successful: $is_successful,
                    reasoning: $reasoning,
                    embedding: $embedding
                })""",
                **attempt.to_neo4j_props()
            )
        return attempt.id

    def create_issue(self, issue: Issue) -> str:
        with self.driver.session() as session:
            session.run(
                "CREATE (i:Issue {id: $id, description: $description, embedding: $embedding})",
                **issue.to_neo4j_props()
            )
        return issue.id

    def create_fix(self, fix: Fix) -> str:
        with self.driver.session() as session:
            session.run(
                "CREATE (f:Fix {id: $id, description: $description, embedding: $embedding})",
                **fix.to_neo4j_props()
            )
        return fix.id

    # --- Create Relationships ---

    def link_attempt_satisfies_policy(self, attempt_id: str, policy_id: str):
        """Attempt → SATISFIES → Policy"""
        with self.driver.session() as session:
            session.run("""
                MATCH (a:Attempt {id: $attempt_id})
                MATCH (p:Policy {id: $policy_id})
                CREATE (a)-[:SATISFIES]->(p)
            """, attempt_id=attempt_id, policy_id=policy_id)

    def link_attempt_causes_issue(self, attempt_id: str, issue_id: str):
        """Attempt → CAUSES → Issue"""
        with self.driver.session() as session:
            session.run("""
                MATCH (a:Attempt {id: $attempt_id})
                MATCH (i:Issue {id: $issue_id})
                CREATE (a)-[:CAUSES]->(i)
            """, attempt_id=attempt_id, issue_id=issue_id)

    def link_fix_resolves_issue(self, fix_id: str, issue_id: str):
        """Fix → RESOLVES → Issue"""
        with self.driver.session() as session:
            session.run("""
                MATCH (f:Fix {id: $fix_id})
                MATCH (i:Issue {id: $issue_id})
                CREATE (f)-[:RESOLVES]->(i)
            """, fix_id=fix_id, issue_id=issue_id)

    # --- Vector Similarity Search ---

    def find_similar_policies(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        """Find similar policies using vector similarity."""
        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('policy_embedding', $top_k, $embedding)
                YIELD node, score
                RETURN node.id as id, node.description as description, score
            """, top_k=top_k, embedding=embedding)
            return [dict(r) for r in result]

    def find_similar_attempts(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        """Find similar attempts using vector similarity."""
        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('attempt_embedding', $top_k, $embedding)
                YIELD node, score
                RETURN node.id as id, node.agent_output as agent_output,
                       node.is_successful as is_successful, node.reasoning as reasoning, score
            """, top_k=top_k, embedding=embedding)
            return [dict(r) for r in result]

    def find_contrastive_attempts(self, embedding: list[float], top_k: int = 3) -> dict:
        """
        Find similar attempts split by success/failure for contrastive learning.

        Returns: {
            "positive": [top_k successful attempts],
            "negative": [top_k failed attempts]
        }
        """
        with self.driver.session() as session:
            # Get more candidates to filter
            result = session.run("""
                CALL db.index.vector.queryNodes('attempt_embedding', $limit, $embedding)
                YIELD node, score
                RETURN node.id as id, node.agent_output as agent_output,
                       node.is_successful as is_successful, node.reasoning as reasoning, score
            """, limit=top_k * 4, embedding=embedding)

            attempts = [dict(r) for r in result]

            positive = [a for a in attempts if a['is_successful']][:top_k]
            negative = [a for a in attempts if not a['is_successful']][:top_k]

            return {"positive": positive, "negative": negative}

    def find_similar_issues(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        """Find similar issues using vector similarity."""
        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('issue_embedding', $top_k, $embedding)
                YIELD node, score
                RETURN node.id as id, node.description as description, score
            """, top_k=top_k, embedding=embedding)
            return [dict(r) for r in result]
 

    # --- Query ---

    def get_attempts_for_policy(self, policy_id: str) -> list[dict]:
        """Get all attempts that satisfy a policy."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:Attempt)-[:SATISFIES]->(p:Policy {id: $policy_id})
                RETURN a.id as id, a.agent_output as agent_output,
                       a.is_successful as is_successful, a.reasoning as reasoning
            """, policy_id=policy_id)
            return [dict(r) for r in result]

    def get_issues_for_attempt(self, attempt_id: str) -> list[dict]:
        """Get all issues caused by an attempt."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:Attempt {id: $attempt_id})-[:CAUSES]->(i:Issue)
                RETURN i.id as id, i.description as description
            """, attempt_id=attempt_id)
            return [dict(r) for r in result]

    def get_fixes_for_issue(self, issue_id: str) -> list[dict]:
        """Get all fixes that resolve an issue."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (f:Fix)-[:RESOLVES]->(i:Issue {id: $issue_id})
                RETURN f.id as id, f.description as description
            """, issue_id=issue_id)
            return [dict(r) for r in result]

    def clear_all(self):
        """Clear all nodes and relationships (for testing)."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    # --- Frozen Memory Audit ---

    def snapshot(self) -> dict:
        """
        Return a deterministic fingerprint of the current memory state.

        Used to verify that no writes occur during evaluation. The snapshot
        contains per-label node counts, per-relationship-type counts, and a
        SHA-256 hash over the sorted IDs of every node and edge.

        Two snapshots taken before and after a frozen-memory evaluation must
        be byte-identical; any divergence is evidence of a write that
        violates the leakage-free protocol.
        """
        import hashlib

        with self.driver.session() as session:
            node_counts = {}
            for r in session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt"
            ):
                label = r["label"] or "Unlabeled"
                node_counts[label] = r["cnt"]

            edge_counts = {}
            for r in session.run(
                "MATCH ()-[e]->() RETURN type(e) AS type, count(e) AS cnt"
            ):
                edge_counts[r["type"]] = r["cnt"]

            node_ids = sorted(
                r["id"] for r in session.run(
                    "MATCH (n) WHERE n.id IS NOT NULL RETURN n.id AS id"
                )
            )
            edge_keys = sorted(
                f"{r['from']}-{r['type']}->{r['to']}" for r in session.run(
                    """MATCH (a)-[e]->(b)
                       WHERE a.id IS NOT NULL AND b.id IS NOT NULL
                       RETURN a.id AS from, type(e) AS type, b.id AS to"""
                )
            )

        h = hashlib.sha256()
        h.update("|".join(node_ids).encode())
        h.update(b"\n")
        h.update("|".join(edge_keys).encode())

        return {
            "node_counts": node_counts,
            "edge_counts": edge_counts,
            "total_nodes": sum(node_counts.values()),
            "total_edges": sum(edge_counts.values()),
            "fingerprint": h.hexdigest(),
        }

    def freeze(self):
        """
        Return a context manager that blocks all writes for its duration.

        Any call to a write method (create_*, link_*, get_or_create_*,
        clear_all) raises ``FrozenMemoryViolation`` while the context is
        active. Pair with ``snapshot``/``diff_snapshots`` to obtain a
        defense-in-depth guarantee that the memory graph does not change
        during evaluation.
        """
        return _FrozenMemoryContext(self)

    @staticmethod
    def diff_snapshots(before: dict, after: dict) -> dict:
        """
        Compute the diff between two snapshots.

        Returns a dict describing whether the snapshots are identical and,
        if not, the per-label and per-relationship deltas. A frozen-memory
        evaluation passes the audit only when ``identical`` is True.
        """
        identical = before["fingerprint"] == after["fingerprint"]

        node_delta = {}
        for label in set(before["node_counts"]) | set(after["node_counts"]):
            d = after["node_counts"].get(label, 0) - before["node_counts"].get(label, 0)
            if d != 0:
                node_delta[label] = d

        edge_delta = {}
        for t in set(before["edge_counts"]) | set(after["edge_counts"]):
            d = after["edge_counts"].get(t, 0) - before["edge_counts"].get(t, 0)
            if d != 0:
                edge_delta[t] = d

        return {
            "identical": identical,
            "node_delta": node_delta,
            "edge_delta": edge_delta,
            "before_total_nodes": before["total_nodes"],
            "after_total_nodes": after["total_nodes"],
            "before_total_edges": before["total_edges"],
            "after_total_edges": after["total_edges"],
            "before_fingerprint": before["fingerprint"],
            "after_fingerprint": after["fingerprint"],
        }

    # --- Semantic Node Operations ---

    def create_semantic(self, semantic: Semantic) -> str:
        """Create a Semantic node."""
        with self.driver.session() as session:
            session.run(
                """CREATE (s:Semantic {
                    id: $id,
                    name: $name,
                    description: $description,
                    embedding: $embedding
                })""",
                **semantic.to_neo4j_props()
            )
        return semantic.id

    def link_issue_abstracts_to_semantic(self, issue_id: str, semantic_id: str):
        """Issue → ABSTRACTS_TO → Semantic"""
        with self.driver.session() as session:
            session.run("""
                MATCH (i:Issue {id: $issue_id})
                MATCH (s:Semantic {id: $semantic_id})
                CREATE (i)-[:ABSTRACTS_TO]->(s)
            """, issue_id=issue_id, semantic_id=semantic_id)

    def find_similar_semantics(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        """Find similar semantic categories using vector similarity."""
        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('semantic_embedding', $top_k, $embedding)
                YIELD node, score
                RETURN node.id as id, node.name as name,
                       node.description as description, score
            """, top_k=top_k, embedding=embedding)
            return [dict(r) for r in result]

    def get_all_semantics(self) -> list[dict]:
        """Get all semantic categories."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Semantic)
                RETURN s.id as id, s.name as name, s.description as description
            """)
            return [dict(r) for r in result]

    def get_or_create_semantic(self, semantic: Semantic, threshold: float = 0.85) -> tuple[str, bool]:
        """
        Get existing semantic if similar one exists, otherwise create new.

        Returns: (semantic_id, is_new)
        """
        # Check for similar existing semantic by embedding
        if semantic.embedding:
            similar = self.find_similar_semantics(semantic.embedding, top_k=1)
            if similar and similar[0]['score'] >= threshold:
                return similar[0]['id'], False

        # Create new semantic
        self.create_semantic(semantic)
        return semantic.id, True

    def get_semantics_for_issue(self, issue_id: str) -> list[dict]:
        """Get semantic categories linked to an issue."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (i:Issue {id: $issue_id})-[:ABSTRACTS_TO]->(s:Semantic)
                RETURN s.id as id, s.name as name, s.description as description
            """, issue_id=issue_id)
            return [dict(r) for r in result]

    def get_issues_for_semantic(self, semantic_id: str) -> list[dict]:
        """Get all issues in a semantic category."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (i:Issue)-[:ABSTRACTS_TO]->(s:Semantic {id: $semantic_id})
                RETURN i.id as id, i.description as description
            """, semantic_id=semantic_id)
            return [dict(r) for r in result]

    # --- Stage 3 Retrieval: Semantic Patterns ---

    def find_semantic_patterns(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        """
        Stage 3 Retrieval: Find semantic patterns from similar issues.

        1. Find similar issues by embedding
        2. Traverse to their connected Semantic nodes
        3. Return aggregated semantic patterns with frequency
        """
        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('issue_embedding', $limit, $embedding)
                YIELD node as issue, score
                WHERE score >= 0.85
                MATCH (issue)-[:ABSTRACTS_TO]->(s:Semantic)
                WITH s, count(issue) as frequency, avg(score) as avg_similarity
                RETURN s.id as id, s.name as name, s.description as description,
                       frequency, avg_similarity
                ORDER BY frequency DESC, avg_similarity DESC
                LIMIT $top_k
            """, limit=top_k * 3, embedding=embedding, top_k=top_k)
            return [dict(r) for r in result]

    def get_semantics_for_attempts(self, attempt_ids: list[str]) -> list[dict]:
        """
        Get semantic patterns from a list of attempts.

        Traverses: Attempt → CAUSES → Issue → ABSTRACTS_TO → Semantic
        """
        if not attempt_ids:
            return []

        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:Attempt)-[:CAUSES]->(i:Issue)-[:ABSTRACTS_TO]->(s:Semantic)
                WHERE a.id IN $attempt_ids
                WITH s, count(DISTINCT i) as issue_count, collect(DISTINCT i.description)[0..3] as sample_issues
                RETURN s.id as id, s.name as name, s.description as description,
                       issue_count, sample_issues
                ORDER BY issue_count DESC
            """, attempt_ids=attempt_ids)
            return [dict(r) for r in result]
