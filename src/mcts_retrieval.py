"""
MCTS-Retrieval: Monte Carlo Tree Search for Memory Graph Exploration

Novel contribution: Use tree search to explore the Neo4j memory graph
instead of single-hop vector similarity.

Instead of: query → top-k similar nodes (one shot)
We do:     query → tree search across retrieval strategies → best context

Actions at each level:
- Vector similarity (attempts, issues, semantics)
- Graph traversal (multi-hop)
- Contrastive retrieval
- Semantic pattern aggregation

The reward is: does the retrieved context lead to a correct judgment?
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from models import get_embedding


# --- Retrieval Actions ---
# Each action is a different way to explore the memory graph

RETRIEVAL_ACTIONS = [
    {
        "name": "contrastive_attempts",
        "description": "Find similar successful and failed attempts for contrast"
    },
    {
        "name": "similar_issues",
        "description": "Find issues similar to this code's potential problems"
    },
    {
        "name": "semantic_patterns",
        "description": "Find high-level semantic patterns from similar issues"
    },
    {
        "name": "multi_hop_issues_to_fixes",
        "description": "Traverse: Similar Issues → Fixes that resolved them"
    },
    {
        "name": "multi_hop_semantic_to_issues",
        "description": "Traverse: Similar Semantics → All issues in that category"
    },
    {
        "name": "multi_hop_policy_to_attempts",
        "description": "Traverse: Similar Policy → All attempts for that policy"
    },
    {
        "name": "multi_hop_attempt_to_semantic",
        "description": "Traverse: Similar Attempts → Their Issues → Semantic categories"
    },
]


@dataclass
class RetrievalNode:
    """A node in the retrieval MCTS tree."""
    action_index: int  # Index into RETRIEVAL_ACTIONS, -1 for root
    parent: Optional["RetrievalNode"] = None
    children: list["RetrievalNode"] = field(default_factory=list)

    # MCTS stats
    visit_count: int = 0
    cumulative_reward: float = 0.0

    # Retrieved data
    retrieved_data: list = field(default_factory=list)
    retrieval_summary: str = ""

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.cumulative_reward / self.visit_count

    def uct_score(self, exploration_constant: float = 2.0) -> float:
        if self.visit_count == 0:
            return float('inf')
        parent_visits = self.parent.visit_count if self.parent else 1
        exploitation = self.q_value
        exploration = exploration_constant * math.sqrt(math.log(parent_visits) / self.visit_count)
        return exploitation + exploration

    def get_unused_actions(self) -> list[int]:
        used = {c.action_index for c in self.children}
        return [i for i in range(len(RETRIEVAL_ACTIONS)) if i not in used]

    def get_trajectory(self) -> list["RetrievalNode"]:
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))


@dataclass
class RetrievalConfig:
    """Configuration for MCTS-Retrieval."""
    max_depth: int = 3          # Max retrieval hops per trajectory
    num_rollouts: int = 4       # Number of MCTS rollouts
    exploration_constant: float = 2.0
    top_k: int = 5              # Results per retrieval action
    similarity_threshold: float = 0.85


class MCTSRetrieval:
    """
    MCTS-Retrieval: Tree search over the memory graph.

    Instead of a fixed retrieval pipeline, we explore different
    combinations of retrieval strategies and select the one that
    produces the best context for judging.
    """

    def __init__(self, graph_manager, config: RetrievalConfig = None):
        self.gm = graph_manager
        self.config = config or RetrievalConfig()

    def retrieve(self, query_text: str, query_embedding: list[float] = None) -> dict:
        """
        Run MCTS to find the best retrieval trajectory through memory.

        Args:
            query_text: The code/text to retrieve context for
            query_embedding: Pre-computed embedding (optional)

        Returns:
            dict with context, trajectory, and stats
        """
        if query_embedding is None:
            query_embedding = get_embedding(query_text)

        root = RetrievalNode(action_index=-1)
        trajectories = []

        for rollout_idx in range(self.config.num_rollouts):
            trajectory = self._run_rollout(root, query_embedding)
            # Score trajectory by diversity and relevance
            score = self._score_trajectory(trajectory)
            self._backpropagate(trajectory, score)
            trajectories.append({
                "nodes": trajectory,
                "score": score,
                "context": self._collect_context(trajectory)
            })

        # Select best trajectory
        best = max(trajectories, key=lambda t: t["score"])

        # Format context from best trajectory
        memory_context = self._format_context(best["context"])

        return {
            "memory_context": memory_context,
            "raw_context": best["context"],
            "trajectory": self._format_trajectory(best["nodes"]),
            "stats": {
                "num_rollouts": self.config.num_rollouts,
                "best_score": best["score"],
                "retrieval_actions": [
                    RETRIEVAL_ACTIONS[n.action_index]["name"]
                    for n in best["nodes"]
                    if n.action_index >= 0
                ]
            }
        }

    def _run_rollout(self, root: RetrievalNode, embedding: list[float]) -> list[RetrievalNode]:
        """Run a single retrieval rollout."""
        trajectory = [root]
        current = root

        for depth in range(self.config.max_depth):
            # Selection
            if current.children:
                current = self._select(current)
                trajectory.append(current)
                continue

            # Expansion
            unused = current.get_unused_actions()
            if not unused:
                break

            action_idx = random.choice(unused)
            child = RetrievalNode(action_index=action_idx, parent=current)
            current.children.append(child)

            # Execute retrieval action
            self._execute_retrieval(child, embedding)
            trajectory.append(child)
            current = child

        return trajectory

    def _select(self, node: RetrievalNode) -> RetrievalNode:
        """Select child with highest UCT score."""
        if not node.children:
            return node
        return max(node.children, key=lambda c: c.uct_score(self.config.exploration_constant))

    def _execute_retrieval(self, node: RetrievalNode, embedding: list[float]):
        """Execute a retrieval action on the memory graph."""
        action = RETRIEVAL_ACTIONS[node.action_index]
        action_name = action["name"]
        top_k = self.config.top_k

        try:
            if action_name == "contrastive_attempts":
                result = self.gm.find_contrastive_attempts(embedding, top_k=top_k)
                node.retrieved_data = result.get("positive", []) + result.get("negative", [])
                pos_count = len(result.get("positive", []))
                neg_count = len(result.get("negative", []))
                node.retrieval_summary = f"Found {pos_count} positive, {neg_count} negative attempts"

            elif action_name == "similar_issues":
                result = self.gm.find_similar_issues(embedding, top_k=top_k)
                node.retrieved_data = result
                node.retrieval_summary = f"Found {len(result)} similar issues"

            elif action_name == "semantic_patterns":
                result = self.gm.find_semantic_patterns(embedding, top_k=top_k)
                node.retrieved_data = result
                node.retrieval_summary = f"Found {len(result)} semantic patterns"

            elif action_name == "multi_hop_issues_to_fixes":
                # Issue → Fix traversal
                issues = self.gm.find_similar_issues(embedding, top_k=top_k)
                fixes = []
                for issue in issues:
                    issue_fixes = self.gm.get_fixes_for_issue(issue['id'])
                    for fix in issue_fixes:
                        fix['source_issue'] = issue['description']
                        fix['issue_score'] = issue.get('score', 0)
                    fixes.extend(issue_fixes)
                node.retrieved_data = fixes
                node.retrieval_summary = f"Found {len(fixes)} fixes via {len(issues)} similar issues"

            elif action_name == "multi_hop_semantic_to_issues":
                # Semantic → Issue traversal
                semantics = self.gm.find_similar_semantics(embedding, top_k=3)
                all_issues = []
                for sem in semantics:
                    sem_issues = self.gm.get_issues_for_semantic(sem['id'])
                    for iss in sem_issues:
                        iss['semantic_name'] = sem['name']
                        iss['semantic_score'] = sem.get('score', 0)
                    all_issues.extend(sem_issues)
                node.retrieved_data = all_issues
                node.retrieval_summary = f"Found {len(all_issues)} issues across {len(semantics)} semantic categories"

            elif action_name == "multi_hop_policy_to_attempts":
                # Policy → Attempt traversal
                policies = self.gm.find_similar_policies(embedding, top_k=3)
                all_attempts = []
                for pol in policies:
                    attempts = self.gm.get_attempts_for_policy(pol['id'])
                    for att in attempts:
                        att['policy_description'] = pol['description']
                        att['policy_score'] = pol.get('score', 0)
                    all_attempts.extend(attempts)
                node.retrieved_data = all_attempts
                node.retrieval_summary = f"Found {len(all_attempts)} attempts across {len(policies)} similar policies"

            elif action_name == "multi_hop_attempt_to_semantic":
                # Attempt → Issue → Semantic traversal
                attempts = self.gm.find_similar_attempts(embedding, top_k=top_k)
                attempt_ids = [a['id'] for a in attempts]
                semantics = self.gm.get_semantics_for_attempts(attempt_ids)
                node.retrieved_data = semantics
                node.retrieval_summary = f"Found {len(semantics)} semantic patterns from {len(attempts)} similar attempts"

            else:
                node.retrieved_data = []
                node.retrieval_summary = f"Unknown action: {action_name}"

        except Exception as e:
            node.retrieved_data = []
            node.retrieval_summary = f"Error: {str(e)}"

    def _score_trajectory(self, trajectory: list[RetrievalNode]) -> float:
        """
        Score a retrieval trajectory based on:
        1. Relevance: How similar are retrieved items
        2. Diversity: How many different types of info
        3. Volume: How much useful data was found
        """
        total_items = 0
        total_relevance = 0.0
        action_types = set()

        for node in trajectory:
            if node.action_index < 0:
                continue

            action_types.add(node.action_index)
            items = node.retrieved_data

            total_items += len(items)
            for item in items:
                # Score relevance from various fields
                score = item.get('score', 0) or item.get('avg_similarity', 0) or \
                        item.get('issue_score', 0) or item.get('semantic_score', 0) or \
                        item.get('policy_score', 0) or 0.5
                total_relevance += score

        if total_items == 0:
            return 0.0

        avg_relevance = total_relevance / total_items
        diversity_bonus = len(action_types) / len(RETRIEVAL_ACTIONS)
        volume_score = min(total_items / 10, 1.0)  # Cap at 10 items

        # Weighted combination
        score = 0.5 * avg_relevance + 0.3 * diversity_bonus + 0.2 * volume_score
        return score

    def _backpropagate(self, trajectory: list[RetrievalNode], reward: float):
        """Propagate reward up the trajectory."""
        for node in trajectory:
            node.visit_count += 1
            node.cumulative_reward += reward

    def _collect_context(self, trajectory: list[RetrievalNode]) -> dict:
        """Collect all retrieved data from trajectory into structured context."""
        context = {
            "contrastive": {"positive": [], "negative": []},
            "issues": [],
            "fixes": [],
            "patterns": [],
            "policy_attempts": [],
        }

        for node in trajectory:
            if node.action_index < 0:
                continue

            action_name = RETRIEVAL_ACTIONS[node.action_index]["name"]

            if action_name == "contrastive_attempts":
                for item in node.retrieved_data:
                    if item.get('is_successful'):
                        context["contrastive"]["positive"].append(item)
                    else:
                        context["contrastive"]["negative"].append(item)

            elif action_name == "similar_issues":
                context["issues"].extend(node.retrieved_data)

            elif action_name == "semantic_patterns":
                context["patterns"].extend(node.retrieved_data)

            elif action_name == "multi_hop_issues_to_fixes":
                context["fixes"].extend(node.retrieved_data)

            elif action_name in ("multi_hop_semantic_to_issues", "multi_hop_attempt_to_semantic"):
                # Add as patterns
                for item in node.retrieved_data:
                    if 'semantic_name' in item:
                        context["patterns"].append({
                            "name": item['semantic_name'],
                            "description": item.get('description', ''),
                            "avg_similarity": item.get('semantic_score', 0)
                        })
                    elif 'name' in item:
                        context["patterns"].append(item)

            elif action_name == "multi_hop_policy_to_attempts":
                context["policy_attempts"].extend(node.retrieved_data)

        return context

    def _format_context(self, context: dict) -> str:
        """Format collected context into a string for the judge prompt."""
        parts = []

        parts.append("IMPORTANT: These are REFERENCE examples only. Each case is UNIQUE.")
        parts.append("Do NOT assume this case will have the same outcome as similar past cases.")
        parts.append("Judge THIS response on its OWN merits against the grading criteria.\n")

        # Contrastive attempts
        positive = context["contrastive"]["positive"]
        negative = context["contrastive"]["negative"]

        if positive:
            parts.append("SUCCESSFUL EXAMPLES (similar responses that passed):")
            for i, a in enumerate(positive[:3], 1):
                score = a.get('score', 0)
                parts.append(f"  {i}. [similarity: {score:.0%}] Response: {str(a.get('agent_output', ''))[:150]}...")
                parts.append(f"     Why passed: {str(a.get('reasoning', ''))[:100]}...")

        if negative:
            parts.append("\nFAILED EXAMPLES (similar responses that failed):")
            for i, a in enumerate(negative[:3], 1):
                score = a.get('score', 0)
                parts.append(f"  {i}. [similarity: {score:.0%}] Response: {str(a.get('agent_output', ''))[:150]}...")
                parts.append(f"     Why failed: {str(a.get('reasoning', ''))[:100]}...")

        # Issues
        issues = context["issues"]
        if issues:
            parts.append("\nPAST ISSUES (only flag if SPECIFICALLY present):")
            for i, issue in enumerate(issues[:5], 1):
                score = issue.get('score', 0)
                parts.append(f"  {i}. [similarity: {score:.0%}] {str(issue.get('description', ''))[:100]}...")

        # Fixes (from multi-hop)
        fixes = context["fixes"]
        if fixes:
            parts.append("\nKNOWN FIXES (from similar past issues):")
            for i, fix in enumerate(fixes[:3], 1):
                source = fix.get('source_issue', 'unknown')
                parts.append(f"  {i}. Issue: {source[:80]}...")
                parts.append(f"     Fix: {str(fix.get('description', ''))[:100]}...")

        # Semantic patterns
        patterns = context["patterns"]
        if patterns:
            # Deduplicate by name
            seen = set()
            unique_patterns = []
            for p in patterns:
                name = p.get('name', '')
                if name not in seen:
                    seen.add(name)
                    unique_patterns.append(p)

            parts.append("\nPATTERNS TO CHECK (warnings only - NOT automatic failures):")
            for i, pattern in enumerate(unique_patterns[:5], 1):
                avg_sim = pattern.get('avg_similarity', 0) or pattern.get('frequency', 0)
                parts.append(f"  {i}. {pattern.get('name', 'Unknown')} [relevance: {avg_sim:.0%}]")
                desc = pattern.get('description', '')
                if desc:
                    parts.append(f"     {desc[:100]}...")

        # Policy attempts (from multi-hop)
        pol_attempts = context["policy_attempts"]
        if pol_attempts:
            successful = [a for a in pol_attempts if a.get('is_successful')]
            failed = [a for a in pol_attempts if not a.get('is_successful')]
            if successful or failed:
                parts.append(f"\nSIMILAR TASK HISTORY: {len(successful)} passed, {len(failed)} failed out of {len(pol_attempts)} attempts")

        if len(parts) <= 3:  # Only the header lines
            return "No relevant past experiences found. Judge based on the criteria alone."

        return "\n".join(parts)

    def _format_trajectory(self, trajectory: list[RetrievalNode]) -> list[dict]:
        """Format trajectory for output."""
        result = []
        for node in trajectory:
            if node.action_index >= 0:
                result.append({
                    "action": RETRIEVAL_ACTIONS[node.action_index]["name"],
                    "items_found": len(node.retrieved_data),
                    "summary": node.retrieval_summary,
                    "visit_count": node.visit_count,
                    "q_value": round(node.q_value, 3)
                })
        return result
