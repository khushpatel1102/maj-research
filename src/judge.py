"""
MAJ Judge - Memory Assisted Judge for evaluating agent outputs.

Two modes:
1. judge() - Stateless evaluation (no memory)
2. judge_with_memory() - Uses past experiences for context
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from models import Policy, Attempt, Issue, Fix, Semantic, JudgeResult, SemanticClassification, get_embedding
from prompts import build_judge_prompt, build_judge_with_memory_prompt, build_classification_prompt, DEFAULT_GOAL

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _format_memory_context(contrastive: dict, similar_issues: list, semantic_patterns: list = None) -> str:
    """Format retrieved memory into prompt context."""
    parts = []

    # FIX 1: Higher similarity thresholds to reduce noise
    # FIX 2: Balance positive/negative - cap negatives to match positives
    positive = [a for a in contrastive['positive'] if a.get('score', 0) >= 0.85]
    negative = [a for a in contrastive['negative'] if a.get('score', 0) >= 0.92]
    issues = [i for i in similar_issues if i.get('score', 0) >= 0.90]

    # FIX 2: Balance - limit negatives to not exceed positives
    # This prevents negative bias from accumulating
    if len(negative) > len(positive) + 1:
        negative = negative[:len(positive) + 1]

    # Stage 3: Semantic patterns - require very high similarity
    patterns = [p for p in (semantic_patterns or []) if p.get('avg_similarity', 0) >= 0.90]

    # FIX 3: Add explicit anti-over-generalization warning at the top
    parts.append("IMPORTANT: These are REFERENCE examples only. Each case is UNIQUE.")
    parts.append("Do NOT assume this case will have the same outcome as similar past cases.")
    parts.append("Judge THIS response on its OWN merits against the grading criteria.\n")

    if positive:
        parts.append("SUCCESSFUL EXAMPLES (similar responses that passed):")
        for i, a in enumerate(positive, 1):
            score = a.get('score', 0)
            parts.append(f"  {i}. [similarity: {score:.0%}] Response excerpt: {a['agent_output'][:150]}...")
            parts.append(f"     Why it passed: {a['reasoning'][:100]}...")

    if negative:
        parts.append("\nFAILED EXAMPLES (similar responses that failed - check if same issue applies):")
        for i, a in enumerate(negative, 1):
            score = a.get('score', 0)
            parts.append(f"  {i}. [similarity: {score:.0%}] Response excerpt: {a['agent_output'][:150]}...")
            parts.append(f"     Why it failed: {a['reasoning'][:100]}...")

    if issues:
        parts.append("\nPAST ISSUES (only flag if SPECIFICALLY present in this response):")
        for i, issue in enumerate(issues, 1):
            score = issue.get('score', 0)
            parts.append(f"  {i}. [similarity: {score:.0%}] {issue['description'][:100]}...")

    if patterns:
        parts.append("\nPATTERNS TO CHECK (warnings only - NOT automatic failures):")
        for i, pattern in enumerate(patterns, 1):
            avg_sim = pattern.get('avg_similarity', 0)
            parts.append(f"  {i}. {pattern['name']} [similarity: {avg_sim:.0%}]")

    if not positive and not negative and not issues and not patterns:
        return "No highly similar past experiences found. Judge based on the criteria alone."

    return "\n".join(parts)


def classify_issue(issue: Issue, graph_manager, model: str = "gpt-4o-mini") -> tuple[Semantic, bool]:
    """
    Classify an issue into a semantic category using LLM.

    Args:
        issue: The Issue to classify
        graph_manager: GraphManager for retrieving existing semantics
        model: OpenAI model to use

    Returns: (semantic, is_new) - the semantic category and whether it's new
    """
    # Get existing semantic categories
    existing = graph_manager.get_all_semantics()

    # Build classification prompt
    prompt = build_classification_prompt(issue.description, existing)

    # Call LLM for classification
    response = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt}],
        text_format=SemanticClassification,
    )

    result = response.output_parsed

    if result.is_new_category:
        # Create new semantic category
        semantic = Semantic(
            name=result.category_name,
            description=result.category_description
        ).with_embedding()
        return semantic, True
    else:
        # Find existing semantic by name
        for s in existing:
            if s['name'] == result.category_name:
                return Semantic(
                    id=s['id'],
                    name=s['name'],
                    description=s['description']
                ), False

        # Fallback: LLM said existing but name not found - create new
        semantic = Semantic(
            name=result.category_name,
            description=result.category_description
        ).with_embedding()
        return semantic, True


def judge(task: str, agent_output: str, goal: str = None, model: str = "gpt-4o-mini") -> dict:
    """
    Judge an agent's output for a given task (stateless, no memory).

    Args:
        task: The task description
        agent_output: The agent's code/response
        goal: What to evaluate for (defaults to DEFAULT_GOAL)
        model: OpenAI model to use (default: gpt-4o-mini)
    """
    goal = goal or DEFAULT_GOAL
    prompt = build_judge_prompt(task=task, agent_output=agent_output, goal=goal)

    response = client.responses.parse(
        model=model,
        input=[
            {"role": "user", "content": prompt}
        ],
        text_format=JudgeResult,
    )

    return _build_result(task, agent_output, response.output_parsed)


def judge_with_memory(task: str, agent_output: str, graph_manager, goal: str = None, model: str = "gpt-4o-mini") -> dict:
    """
    Judge an agent's output using memory of past experiences.

    Args:
        task: The task description
        agent_output: The agent's code/response
        graph_manager: GraphManager instance for memory retrieval
        goal: What to evaluate for (defaults to DEFAULT_GOAL)
        model: OpenAI model to use (default: gpt-4o-mini)
    """
    goal = goal or DEFAULT_GOAL

    # Get embedding for the CODE to find similar implementations
    # (not task - task similarity conflates good/bad implementations of same task)
    code_embedding = get_embedding(agent_output)

    # Stage 2: Retrieve contrastive attempts and similar issues
    contrastive = graph_manager.find_contrastive_attempts(code_embedding, top_k=3)
    similar_issues = graph_manager.find_similar_issues(code_embedding, top_k=5)

    # Stage 3: Retrieve semantic patterns from similar issues
    semantic_patterns = graph_manager.find_semantic_patterns(code_embedding, top_k=3)

    # Format memory context with all stages
    memory_context = _format_memory_context(contrastive, similar_issues, semantic_patterns)

    prompt = build_judge_with_memory_prompt(
        task=task,
        agent_output=agent_output,
        goal=goal,
        memory_context=memory_context
    )

    response = client.responses.parse(
        model=model,
        input=[
            {"role": "user", "content": prompt}
        ],
        text_format=JudgeResult,
    )

    result = _build_result(task, agent_output, response.output_parsed)
    result['memory_used'] = {
        'positive_examples': len(contrastive['positive']),
        'negative_examples': len(contrastive['negative']),
        'similar_issues': len(similar_issues),
        'semantic_patterns': len(semantic_patterns)
    }

    # Classify issues into semantic categories
    semantics = []
    semantic_relationships = []

    for issue in result['issues']:
        semantic, is_new = classify_issue(issue, graph_manager, model)
        semantics.append(semantic)
        semantic_relationships.append({
            "type": "ABSTRACTS_TO",
            "from_id": issue.id,
            "to_id": semantic.id,
            "is_new": is_new
        })

    result['semantics'] = semantics
    result['semantic_relationships'] = semantic_relationships

    return result


def _build_result(task: str, agent_output: str, data: JudgeResult) -> dict:
    """Build the result dict from parsed judge output."""
    policy = Policy(description=task).with_embedding()

    attempt = Attempt(
        agent_output=agent_output,
        is_successful=data.is_successful,
        reasoning=data.reasoning
    ).with_embedding()

    issues = []
    fixes = []
    relationships = []

    relationships.append({"type": "SATISFIES", "from_id": attempt.id, "to_id": policy.id})

    for pair in data.issue_fix_pairs:
        issue = Issue(description=pair.issue).with_embedding()
        fix = Fix(description=pair.fix).with_embedding()

        issues.append(issue)
        fixes.append(fix)

        relationships.append({"type": "CAUSES", "from_id": attempt.id, "to_id": issue.id})
        relationships.append({"type": "RESOLVES", "from_id": fix.id, "to_id": issue.id})

    return {
        "policy": policy,
        "attempt": attempt,
        "issues": issues,
        "fixes": fixes,
        "relationships": relationships
    }
