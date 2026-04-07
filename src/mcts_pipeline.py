"""
MCTS Pipeline: Combines all evaluation modes.

Modes:
1. Stateless              - No memory, no MCTS (baseline)
2. MAJ                    - Memory-assisted judge (current system)
3. MCTS-Judge             - Tree search reasoning, no memory
4. MCTS-Judge + Memory    - Tree search reasoning with MAJ memory
5. MCTS-Retrieval + Judge - Tree search retrieval with standard judge
6. Full MCTS              - Tree search retrieval + tree search reasoning (novel)
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

from models import get_embedding, JudgeResult, Policy, Attempt, Issue, Fix
from judge import judge, judge_with_memory, _format_memory_context, classify_issue
from mcts_judge import MCTSJudge, MCTSConfig
from mcts_retrieval import MCTSRetrieval, RetrievalConfig
from prompts import build_judge_prompt, build_judge_with_memory_prompt, DEFAULT_GOAL

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def store_mcts_result(task: str, agent_output: str, result: dict, graph_manager, model: str = "gpt-4o-mini"):
    """Store MCTS evaluation result in the memory graph."""
    policy = Policy(description=task).with_embedding()
    attempt = Attempt(
        agent_output=agent_output,
        is_successful=result["is_successful"],
        reasoning=result["reasoning"][:500]
    ).with_embedding()

    graph_manager.create_policy(policy)
    graph_manager.create_attempt(attempt)
    graph_manager.link_attempt_satisfies_policy(attempt.id, policy.id)

    # Extract issues from trajectory (failed subtasks)
    for t in result.get("trajectory", []):
        if t["decision"] == "FAIL":
            issue = Issue(description=f"{t['subtask']}: {t['analysis'][:200]}").with_embedding()
            graph_manager.create_issue(issue)
            graph_manager.link_attempt_causes_issue(attempt.id, issue.id)

            # Classify into semantic category
            try:
                semantic, is_new = classify_issue(issue, graph_manager, model)
                if is_new:
                    graph_manager.get_or_create_semantic(semantic)
                graph_manager.link_issue_abstracts_to_semantic(issue.id, semantic.id)
            except Exception:
                pass


def run_stateless(task: str, agent_output: str, goal: str = None, model: str = "gpt-4o-mini") -> dict:
    """Mode 1: Baseline stateless evaluation."""
    return judge(task, agent_output, goal, model)


def run_maj(task: str, agent_output: str, graph_manager, goal: str = None, model: str = "gpt-4o-mini") -> dict:
    """Mode 2: MAJ memory-assisted evaluation (current system)."""
    return judge_with_memory(task, agent_output, graph_manager, goal, model)


def run_mcts_judge(task: str, agent_output: str, goal: str = None, config: MCTSConfig = None) -> dict:
    """Mode 3: MCTS-Judge reasoning only, no memory."""
    config = config or MCTSConfig()
    mcts = MCTSJudge(config)
    return mcts.evaluate(task, agent_output)


def run_mcts_judge_with_memory(
    task: str,
    agent_output: str,
    graph_manager,
    goal: str = None,
    mcts_config: MCTSConfig = None,
    model: str = "gpt-4o-mini"
) -> dict:
    """
    Mode 4: MCTS-Judge + MAJ Memory.

    Uses standard MAJ retrieval to get memory context,
    then feeds it into MCTS-Judge for reasoning.
    """
    mcts_config = mcts_config or MCTSConfig(model=model)

    # Standard MAJ retrieval (3-type)
    code_embedding = get_embedding(agent_output)
    contrastive = graph_manager.find_contrastive_attempts(code_embedding, top_k=3)
    similar_issues = graph_manager.find_similar_issues(code_embedding, top_k=5)
    semantic_patterns = graph_manager.find_semantic_patterns(code_embedding, top_k=3)
    memory_context = _format_memory_context(contrastive, similar_issues, semantic_patterns)

    # MCTS reasoning with memory context
    mcts = MCTSJudge(mcts_config)
    result = mcts.evaluate(task, agent_output, memory_context=memory_context)

    result["memory_used"] = {
        "positive_examples": len(contrastive["positive"]),
        "negative_examples": len(contrastive["negative"]),
        "similar_issues": len(similar_issues),
        "semantic_patterns": len(semantic_patterns),
        "retrieval_type": "standard_3type"
    }

    # Store in memory
    store_mcts_result(task, agent_output, result, graph_manager, model)

    return result


def run_mcts_retrieval_with_judge(
    task: str,
    agent_output: str,
    graph_manager,
    goal: str = None,
    retrieval_config: RetrievalConfig = None,
    model: str = "gpt-4o-mini"
) -> dict:
    """
    Mode 5: MCTS-Retrieval + Standard Judge.

    Uses MCTS tree search for retrieval (novel),
    then feeds context into standard single-pass judge.
    """
    goal = goal or DEFAULT_GOAL

    # MCTS retrieval
    retrieval_config = retrieval_config or RetrievalConfig()
    mcts_retrieval = MCTSRetrieval(graph_manager, retrieval_config)
    code_embedding = get_embedding(agent_output)
    retrieval_result = mcts_retrieval.retrieve(agent_output, code_embedding)

    memory_context = retrieval_result["memory_context"]

    # Standard judge with MCTS-retrieved context
    prompt = build_judge_with_memory_prompt(
        task=task,
        agent_output=agent_output,
        goal=goal,
        memory_context=memory_context
    )

    response = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt}],
        text_format=JudgeResult,
    )

    data = response.output_parsed

    result = {
        "is_successful": data.is_successful,
        "reasoning": data.reasoning,
        "issue_fix_pairs": [{"issue": p.issue, "fix": p.fix} for p in data.issue_fix_pairs],
        "trajectory": [{"subtask": "full_evaluation", "decision": "PASS" if data.is_successful else "FAIL", "analysis": data.reasoning[:200]}],
        "retrieval_trajectory": retrieval_result["trajectory"],
        "retrieval_stats": retrieval_result["stats"],
    }

    # Store in memory
    store_mcts_result(task, agent_output, result, graph_manager, model)

    return result


def run_full_mcts(
    task: str,
    agent_output: str,
    graph_manager,
    goal: str = None,
    retrieval_config: RetrievalConfig = None,
    judge_config: MCTSConfig = None,
    model: str = "gpt-4o-mini"
) -> dict:
    """
    Mode 6: Full MCTS — MCTS-Retrieval + MCTS-Judge.

    Two layers of System-2 thinking:
    1. MCTS explores memory graph to find best context
    2. MCTS explores reasoning paths to find best judgment

    This is the novel contribution.
    """
    judge_config = judge_config or MCTSConfig(model=model)
    retrieval_config = retrieval_config or RetrievalConfig()

    # Layer 1: MCTS Retrieval
    mcts_retrieval = MCTSRetrieval(graph_manager, retrieval_config)
    code_embedding = get_embedding(agent_output)
    retrieval_result = mcts_retrieval.retrieve(agent_output, code_embedding)

    memory_context = retrieval_result["memory_context"]

    # Layer 2: MCTS Judge with retrieved context
    mcts_judge = MCTSJudge(judge_config)
    judge_result = mcts_judge.evaluate(task, agent_output, memory_context=memory_context)

    result = {
        "is_successful": judge_result["is_successful"],
        "reasoning": judge_result["reasoning"],
        "trajectory": judge_result["trajectory"],
        "judge_trajectory": judge_result["trajectory"],
        "judge_stats": judge_result["stats"],
        "retrieval_trajectory": retrieval_result["trajectory"],
        "retrieval_stats": retrieval_result["stats"],
        "mode": "full_mcts"
    }

    # Store in memory
    store_mcts_result(task, agent_output, result, graph_manager, model)

    return result


# --- Convenience: Run all modes for comparison ---

def run_ablation(
    task: str,
    agent_output: str,
    graph_manager,
    goal: str = None,
    model: str = "gpt-4o-mini"
) -> dict:
    """
    Run all 6 modes and return results for ablation study.
    """
    results = {}

    # Mode 1: Stateless
    r1 = run_stateless(task, agent_output, goal, model)
    results["stateless"] = {"is_successful": r1["attempt"].is_successful, "reasoning": r1["attempt"].reasoning}

    # Mode 2: MAJ
    r2 = run_maj(task, agent_output, graph_manager, goal, model)
    results["maj"] = {"is_successful": r2["attempt"].is_successful, "reasoning": r2["attempt"].reasoning}

    # Mode 3: MCTS-Judge only
    r3 = run_mcts_judge(task, agent_output, goal, MCTSConfig(model=model, num_rollouts=2, max_depth=3))
    results["mcts_judge"] = {"is_successful": r3["is_successful"], "reasoning": r3["reasoning"]}

    # Mode 4: MCTS-Judge + Memory
    r4 = run_mcts_judge_with_memory(task, agent_output, graph_manager, goal,
                                      MCTSConfig(model=model, num_rollouts=2, max_depth=3), model)
    results["mcts_judge_memory"] = {"is_successful": r4["is_successful"], "reasoning": r4["reasoning"]}

    # Mode 5: MCTS-Retrieval + Judge
    r5 = run_mcts_retrieval_with_judge(task, agent_output, graph_manager, goal,
                                        RetrievalConfig(num_rollouts=2, max_depth=2), model)
    results["mcts_retrieval_judge"] = {"is_successful": r5["is_successful"], "reasoning": r5["reasoning"]}

    # Mode 6: Full MCTS
    r6 = run_full_mcts(task, agent_output, graph_manager, goal,
                        RetrievalConfig(num_rollouts=2, max_depth=2),
                        MCTSConfig(model=model, num_rollouts=2, max_depth=3), model)
    results["full_mcts"] = {"is_successful": r6["is_successful"], "reasoning": r6["reasoning"]}

    return results
