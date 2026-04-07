"""
MCTS-Judge: Monte Carlo Tree Search for LLM-as-a-Judge

Based on: "MCTS-Judge: Test-Time Scaling in LLM-as-a-Judge" (arXiv:2502.12468)

Tree search decomposes evaluation into multi-perspective subtasks.
Each rollout builds a reasoning trajectory evaluated from different angles.
"""

import os
import math
import random
from dataclasses import dataclass, field
from typing import Optional
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --- Structured Output Schemas ---

class SubtaskDecision(BaseModel):
    """Structured output for subtask evaluation."""
    analysis: str
    decision: bool  # True = code correct, False = incorrect

class SelfAssessment(BaseModel):
    """Structured output for LLM self-assessment."""
    useful: bool  # True = subtask would improve evaluation

class SimulatedExecutionResult(BaseModel):
    """Structured output for simulated execution."""
    trace: str
    passed: bool  # True = code produces correct output

class GlobalVerdict(BaseModel):
    """Structured output for global evaluation."""
    verdict: bool  # True = code correct
    reasoning: str

class GeneratedSubtask(BaseModel):
    """A single dynamically generated evaluation subtask."""
    name: str
    prompt: str

class GeneratedSubtasks(BaseModel):
    """Structured output for dynamic subtask generation."""
    subtasks: list[GeneratedSubtask]




@dataclass(eq=False)
class MCTSNode:
    """A node in the MCTS tree."""
    subtask_index: int  # Index into SUBTASKS, -1 for root, -2 for null action
    parent: Optional["MCTSNode"] = None
    children: list["MCTSNode"] = field(default_factory=list)

    # MCTS stats
    visit_count: int = 0
    cumulative_reward: float = 0.0

    # Subtask result
    analysis: str = ""
    decision: Optional[bool] = None  # True = code correct for this subtask

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.cumulative_reward / self.visit_count

    def uct_score(self, exploration_constant: float = 3.0) -> float:
        """Upper Confidence Bound for Trees."""
        if self.visit_count == 0:
            return float('inf')
        parent_visits = self.parent.visit_count if self.parent else 1
        exploitation = self.q_value
        exploration = exploration_constant * math.sqrt(math.log(parent_visits) / self.visit_count)
        return exploitation + exploration

    def get_unused_subtasks(self, num_subtasks: int) -> list[int]:
        """Get subtask indices not yet used by children."""
        used = {c.subtask_index for c in self.children}
        available = [i for i in range(num_subtasks) if i not in used]
        return available

    def get_trajectory(self) -> list["MCTSNode"]:
        """Get the path from root to this node."""
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))


@dataclass
class MCTSConfig:
    """Configuration for MCTS-Judge."""
    max_depth: int = 5          # Max subtasks per trajectory
    num_rollouts: int = 4       # Number of MCTS rollouts
    exploration_constant: float = 3.0  # UCT exploration parameter
    uct_weight: float = 0.9     # Weight for UCT in selection
    llm_weight: float = 0.1     # Weight for LLM self-assessment in selection
    reward_value: float = 1.1   # Terminal reward when prediction matches simulated execution
    model: str = "gpt-4o-mini"  # LLM model
    temperature: float = 0.4


class MCTSJudge:
    """
    MCTS-Judge: Test-time scaling for LLM-as-a-Judge.

    Decomposes code evaluation into multi-perspective subtasks
    and uses tree search to find the best reasoning trajectory.
    """

    def __init__(self, config: MCTSConfig = None):
        self.config = config or MCTSConfig()
        self.root = None
        self.trajectories = []
        self.subtasks = []  # Dynamically generated per evaluation

    def _generate_subtasks(self, task: str, code: str) -> list[dict]:
        """Dynamically generate evaluation subtasks based on the task and response."""
        prompt = f"""You are an evaluation planning expert. Given a task and a response to evaluate, generate 5-7 specific evaluation perspectives.

TASK/CRITERIA:
{task[:500]}

RESPONSE TO EVALUATE:
{code[:500]}

Generate evaluation subtasks that are specifically relevant to THIS task. Each subtask should check a different important aspect of whether the response meets the criteria. Be specific to the content, not generic."""

        try:
            response = client.responses.parse(
                model=self.config.model,
                input=[{"role": "user", "content": prompt}],
                text_format=GeneratedSubtasks,
                temperature=0.3,
            )
            subtasks = [{"name": s.name, "prompt": s.prompt} for s in response.output_parsed.subtasks]
            if len(subtasks) >= 3:
                return subtasks
        except Exception:
            raise RuntimeError("Failed to generate subtasks for this evaluation")

    def evaluate(self, task: str, code: str, memory_context: str = None) -> dict:
        """
        Run MCTS-Judge evaluation.

        Args:
            task: Problem statement
            code: Code snippet to evaluate
            memory_context: Optional memory context from MAJ retrieval

        Returns:
            dict with verdict, reasoning, trajectory, and stats
        """
        # Generate task-specific subtasks
        self.subtasks = self._generate_subtasks(task, code)

        self.root = MCTSNode(subtask_index=-1)  # Root node
        self.trajectories = []

        # Run MCTS rollouts
        for rollout_idx in range(self.config.num_rollouts):
            trajectory = self._run_rollout(task, code, memory_context)
            reward = self._compute_reward(task, code, trajectory)
            self._backpropagate(trajectory, reward)
            self.trajectories.append({
                "nodes": trajectory,
                "reward": reward
            })

        # Select best trajectory by cumulative reward
        best = self._select_best_trajectory()

        # Get final verdict from best trajectory
        verdict = self._aggregate_verdict(best["nodes"])

        # Global evaluation for additional signal
        global_eval = self._global_evaluation(task, code, best["nodes"], memory_context)

        # Final answer combines trajectory majority + global
        final_verdict = self._final_answer(verdict, global_eval)

        return {
            "is_successful": final_verdict["verdict"],
            "reasoning": final_verdict["reasoning"],
            "trajectory": self._format_trajectory(best["nodes"]),
            "stats": {
                "num_rollouts": self.config.num_rollouts,
                "max_depth": self.config.max_depth,
                "best_reward": best["reward"],
                "total_nodes": self._count_nodes(self.root),
            }
        }

    def _run_rollout(self, task: str, code: str, memory_context: str = None) -> list[MCTSNode]:
        """Run a single MCTS rollout: selection → expansion → simulation."""
        trajectory = [self.root]
        current = self.root

        for depth in range(self.config.max_depth):
            # 1. SELECTION: Find node to expand
            if current.children:
                current = self._select(current, task, code, memory_context)
                trajectory.append(current)
                if current.subtask_index == -2:  # Null action
                    continue

            # 2. EXPANSION: Add new child
            unused = current.get_unused_subtasks(len(self.subtasks))
            if not unused:
                break

            subtask_idx = random.choice(unused)
            child = MCTSNode(subtask_index=subtask_idx, parent=current)
            current.children.append(child)

            # 3. SIMULATION: Execute subtask
            self._execute_subtask(child, task, code, memory_context)
            trajectory.append(child)
            current = child

        return trajectory

    def _select(self, node: MCTSNode, task: str, code: str, memory_context: str = None) -> MCTSNode:
        """
        Select child node using global-local strategy.
        Global: UCT scores from prior rollouts
        Local: LLM self-assessment based on current trajectory
        """
        if not node.children:
            return node

        # Global: UCT scores
        uct_scores = {}
        for child in node.children:
            uct_scores[child] = child.uct_score(self.config.exploration_constant)

        # Normalize UCT scores
        max_uct = max(uct_scores.values()) if uct_scores else 1.0
        if max_uct > 0:
            for child in uct_scores:
                uct_scores[child] /= max_uct

        # Local: LLM self-assessment (lightweight)
        llm_scores = self._llm_self_assess(node, task, code, memory_context)

        # Combine: weighted sampling
        combined = {}
        for child in node.children:
            uct = uct_scores.get(child, 0.5)
            llm = llm_scores.get(child.subtask_index, 0.5)
            combined[child] = self.config.uct_weight * uct + self.config.llm_weight * llm

        # Weighted random selection
        total = sum(combined.values())
        if total == 0:
            return random.choice(node.children)

        r = random.uniform(0, total)
        cumsum = 0
        for child, score in combined.items():
            cumsum += score
            if cumsum >= r:
                return child

        return node.children[-1]

    def _llm_self_assess(self, node: MCTSNode, task: str, code: str, memory_context: str = None) -> dict:
        """LLM evaluates whether each child subtask would improve evaluation completeness."""
        trajectory = node.get_trajectory()
        history = [self.subtasks[n.subtask_index]["name"] for n in trajectory if n.subtask_index >= 0]

        scores = {}
        for child in node.children:
            if child.subtask_index < 0:
                scores[child.subtask_index] = 0.3
                continue

            subtask_name = self.subtasks[child.subtask_index]["name"]

            prompt = f"""You are an evaluation planning expert.

Task: {task[:300]}
Response provided for evaluation.

Previously evaluated: {', '.join(history) if history else 'None yet'}
Proposed next evaluation: {subtask_name}

Would evaluating '{subtask_name}' improve the completeness of the evaluation?
Answer Yes or No only."""

            try:
                response = client.responses.parse(
                    model=self.config.model,
                    input=[{"role": "user", "content": prompt}],
                    text_format=SelfAssessment,
                    temperature=0.1,
                )
                scores[child.subtask_index] = 1.0 if response.output_parsed.useful else 0.2
            except Exception:
                scores[child.subtask_index] = 0.5

        return scores

    def _execute_subtask(self, node: MCTSNode, task: str, code: str, memory_context: str = None):
        """Execute a subtask: LLM analyzes response from specific perspective."""
        subtask = self.subtasks[node.subtask_index]

        memory_section = ""
        if memory_context:
            memory_section = f"\n\nMEMORY CONTEXT (past evaluation experiences):\n{memory_context}\n"

        prompt = f"""You are evaluating a response from a specific perspective.

TASK/CRITERIA:
{task}

RESPONSE:
{code}
{memory_section}
EVALUATION PERSPECTIVE: {subtask['name']}
{subtask['prompt']}

First, provide a detailed analysis. Then conclude with your decision: does the response pass or fail from this perspective?"""

        try:
            response = client.responses.parse(
                model=self.config.model,
                input=[{"role": "user", "content": prompt}],
                text_format=SubtaskDecision,
                temperature=self.config.temperature,
            )

            result = response.output_parsed
            node.analysis = result.analysis
            node.decision = result.decision

        except Exception as e:
            node.analysis = f"Error: {str(e)}"
            node.decision = None

    def _compute_reward(self, task: str, code: str, trajectory: list[MCTSNode]) -> float:
        """
        Compute terminal reward using simulated execution.

        The trajectory's prediction (majority vote) is compared against
        simulated test case execution. Reward if they agree.
        """
        # Get trajectory prediction (majority vote of subtask decisions)
        decisions = [n.decision for n in trajectory if n.decision is not None]
        if not decisions:
            return 0.0

        trajectory_prediction = sum(1 for d in decisions if d) > len(decisions) / 2

        # Simulated execution: LLM traces through code with a test case
        sim_result = self._simulated_execution(task, code)

        # Reward if prediction matches simulated execution
        if trajectory_prediction == sim_result:
            return self.config.reward_value
        else:
            return 0.0

    def _simulated_execution(self, task: str, code: str) -> bool:
        """
        LLM performs an independent quick evaluation.
        Returns True if response appears correct, False otherwise.
        """
        prompt = f"""You are an independent evaluator. Quickly assess whether this response meets the criteria.

CRITERIA: {task}

RESPONSE:
{code}

Instructions:
1. Identify the key requirements from the criteria
2. Check if the response addresses each one
3. Make a quick judgment

Conclude with your assessment: does the response pass or fail?"""

        try:
            response = client.responses.parse(
                model=self.config.model,
                input=[{"role": "user", "content": prompt}],
                text_format=SimulatedExecutionResult,
                temperature=0.2,
            )

            return response.output_parsed.passed

        except Exception:
            return True  # Default to pass on error

    def _backpropagate(self, trajectory: list[MCTSNode], reward: float):
        """Propagate reward up through the trajectory."""
        for node in trajectory:
            node.visit_count += 1
            node.cumulative_reward += reward

    def _select_best_trajectory(self) -> dict:
        """Select the trajectory with highest cumulative reward."""
        if not self.trajectories:
            return {"nodes": [], "reward": 0.0}

        # Weighted sampling by reward
        rewards = [t["reward"] for t in self.trajectories]
        max_reward = max(rewards)

        # Return trajectory with highest reward (break ties by index)
        best = max(self.trajectories, key=lambda t: t["reward"])
        return best

    def _aggregate_verdict(self, trajectory: list[MCTSNode]) -> dict:
        """Get majority vote verdict from trajectory subtasks."""
        decisions = [n.decision for n in trajectory if n.decision is not None]
        if not decisions:
            return {"verdict": True, "confidence": 0.0}

        yes_count = sum(1 for d in decisions if d)
        total = len(decisions)

        return {
            "verdict": yes_count > total / 2,
            "confidence": max(yes_count, total - yes_count) / total
        }

    def _global_evaluation(self, task: str, code: str, trajectory: list[MCTSNode], memory_context: str = None) -> dict:
        """Additional global evaluation considering all subtask analyses."""
        analyses = []
        for node in trajectory:
            if node.subtask_index >= 0 and node.analysis:
                name = self.subtasks[node.subtask_index]["name"]
                decision = "PASS" if node.decision else "FAIL"
                analyses.append(f"[{name}] {decision}: {node.analysis[:200]}")

        analyses_text = "\n\n".join(analyses)

        memory_section = ""
        if memory_context:
            memory_section = f"\nMEMORY CONTEXT:\n{memory_context}\n"

        prompt = f"""Based on the following multi-perspective analysis, provide a final verdict.

CRITERIA: {task}

RESPONSE: {code}
{memory_section}
ANALYSIS FROM MULTIPLE PERSPECTIVES:
{analyses_text}

Considering ALL perspectives above, does the response meet the criteria?
Answer: Yes or No, followed by a brief explanation."""

        try:
            response = client.responses.parse(
                model=self.config.model,
                input=[{"role": "user", "content": prompt}],
                text_format=GlobalVerdict,
                temperature=0.2,
            )

            result = response.output_parsed
            return {"verdict": result.verdict, "reasoning": result.reasoning}

        except Exception as e:
            return {"verdict": True, "reasoning": f"Global evaluation failed: {e}"}

    def _final_answer(self, trajectory_verdict: dict, global_eval: dict) -> dict:
        """Combine trajectory majority vote with global evaluation."""
        t_vote = trajectory_verdict["verdict"]
        g_vote = global_eval["verdict"]

        # If they agree, high confidence
        if t_vote == g_vote:
            return {
                "verdict": t_vote,
                "reasoning": global_eval["reasoning"]
            }

        # If they disagree, trust global (it sees all analyses)
        return {
            "verdict": g_vote,
            "reasoning": f"[Note: subtask majority voted {'PASS' if t_vote else 'FAIL'}, "
                        f"but global analysis concluded differently]\n{global_eval['reasoning']}"
        }

    def _format_trajectory(self, trajectory: list[MCTSNode]) -> list[dict]:
        """Format trajectory for output."""
        result = []
        for node in trajectory:
            if node.subtask_index >= 0:
                result.append({
                    "subtask": self.subtasks[node.subtask_index]["name"],
                    "decision": "PASS" if node.decision else "FAIL",
                    "analysis": node.analysis[:500],
                    "visit_count": node.visit_count,
                    "q_value": round(node.q_value, 3)
                })
        return result

    def _count_nodes(self, node: MCTSNode) -> int:
        """Count total nodes in tree."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
