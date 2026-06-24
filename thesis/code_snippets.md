# Code Snippets — Implementation Chapter

Screenshot each code block below and place it in the matching section.
The caption goes directly under the block (italic).

---

## Listing 4.1 — Section 4.2 "The Memory-Assisted Judge"

```python
def judge_with_memory(task, agent_output, graph_manager,
                      goal=None, model="gpt-4o"):
    code_embedding = get_embedding(agent_output)
    # Stage 1-2: contrastive attempts and similar issues
    contrastive = graph_manager.find_contrastive_attempts(
        code_embedding, top_k=3)
    similar_issues = graph_manager.find_similar_issues(
        code_embedding, top_k=5)
    # Stage 3: semantic patterns from the similar issues
    semantic_patterns = graph_manager.find_semantic_patterns(
        code_embedding, top_k=3)
    memory_context = _format_memory_context(
        contrastive, similar_issues, semantic_patterns)
    # ... judge call conditioned on memory_context ...
```

*Listing 4.1 – The three-stage memory retrieval of the MAJ judge.*

---

## Listing 4.2 — Section 4.3 "The MCTS Components"

```python
def uct_score(self, exploration_constant=3.0):
    """Upper Confidence Bound for Trees."""
    if self.visit_count == 0:
        return float('inf')
    parent_visits = self.parent.visit_count if self.parent else 1
    exploitation = self.q_value
    exploration = exploration_constant * math.sqrt(
        math.log(parent_visits) / self.visit_count)
    return exploitation + exploration
```

*Listing 4.2 – The UCT selection score used by the MCTS-Judge.*

---

## Listing 4.3 — Section 4.3 "The MCTS Components"

```python
RETRIEVAL_ACTIONS = [
    {"name": "contrastive_attempts", ...},
    {"name": "similar_issues", ...},
    {"name": "semantic_patterns", ...},
    {"name": "multi_hop_issues_to_fixes", ...},
    {"name": "multi_hop_semantic_to_issues", ...},
    {"name": "multi_hop_policy_to_attempts", ...},
    {"name": "multi_hop_attempt_to_semantic", ...},
]
```

*Listing 4.3 – The seven retrieval actions of MCTS-Retrieval, four of which are multi-hop graph traversals.*

---

## Listing 4.4 — Section 4.4 "The Frozen-Memory Audit"

```python
def snapshot(self) -> dict:
    """Deterministic fingerprint of the current memory state."""
    import hashlib
    # ... collect node_counts, edge_counts, node_ids, edge_keys ...
    h = hashlib.sha256()
    h.update("|".join(node_ids).encode())
    h.update(b"\n")
    h.update("|".join(edge_keys).encode())
    return {
        "node_counts":  node_counts,
        "edge_counts":  edge_counts,
        "total_nodes":  sum(node_counts.values()),
        "total_edges":  sum(edge_counts.values()),
        "fingerprint":  h.hexdigest(),
    }
```

*Listing 4.4 – The memory snapshot and SHA-256 fingerprint.*

---

## Listing 4.5 — Section 4.5 "Structured Outputs"

```python
class SubtaskDecision(BaseModel):
    analysis: str
    decision: bool          # True = correct, False = incorrect

class SelfAssessment(BaseModel):
    useful: bool            # True = subtask improves evaluation

class GlobalVerdict(BaseModel):
    verdict: bool           # True = response passes
    reasoning: str
```

*Listing 4.5 – Pydantic schemas for structured judge outputs.*

---

## Listing 4.6 — Section 4.6 "Handling Transient Failures"

```python
def _call_with_retries(fn, *, max_attempts=5, base=2.0,
                       max_sleep=30.0):
    """Retry fn() on transient errors with backoff + jitter."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if not _is_transient(exc) or attempt == max_attempts:
                raise
            sleep_s = min(max_sleep, base ** attempt) \
                      + random.uniform(0, 0.5)
            time.sleep(sleep_s)
```

*Listing 4.6 – Exponential-backoff retry wrapper for model calls.*

---

## Placement map

| Section | Listing |
|---|---|
| 4.2 The Memory-Assisted Judge | 4.1 |
| 4.3 The MCTS Components | 4.2 and 4.3 |
| 4.4 The Frozen-Memory Audit | 4.4 |
| 4.5 Structured Outputs | 4.5 |
| 4.6 Handling Transient Failures | 4.6 |
