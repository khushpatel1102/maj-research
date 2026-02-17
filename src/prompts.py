"""
MAJ Judge Prompts

Structure:
- ROLE: Who the judge is (fixed)
- GOAL: What to evaluate for (placeholder)
- INSTRUCTION: Task-specific details (placeholder)
- OUTPUT_SCHEMA: What to return (fixed)
- MEMORY: Past experiences (added when using memory mode)
"""

# --- FIXED: ROLE ---
ROLE = """You are an expert AI Judge who evaluates responses.
You fairly assess whether responses meet the specified criteria."""

# --- PLACEHOLDER: GOAL ---
# This should be provided per task type
GOAL_PLACEHOLDER = "{goal}"

# --- PLACEHOLDER: INSTRUCTION ---
# This contains the task and agent output
INSTRUCTION_PLACEHOLDER = """TASK: {task}

AGENT OUTPUT:
{agent_output}"""

# --- FIXED: OUTPUT SCHEMA ---
OUTPUT_SCHEMA = """Return your evaluation:
1. is_successful: true if the output achieves the GOAL, false otherwise
2. reasoning: explanation of why the attempt succeeded or failed
3. issue_fix_pairs: list of {{issue, fix}} pairs (empty if successful)"""

# --- MEMORY CONTEXT (added when using memory) ---
MEMORY_CONTEXT = """MEMORY CONTEXT (reference examples from past evaluations):
{memory_context}

CRITICAL - How to use this context:
- These are REFERENCE examples only - NOT deterministic rules
- DO NOT let past failures bias you toward rejecting this response
- DO NOT let past successes bias you toward accepting this response
- Each response is UNIQUE and must be judged on ITS OWN MERITS
- Only apply a past pattern if the EXACT SAME issue is present here
- When in doubt, IGNORE the memory and judge based on criteria alone
"""

# --- BUILD PROMPTS ---

def build_judge_prompt(task: str, agent_output: str, goal: str) -> str:
    """Build prompt for stateless judge."""
    return f"""{ROLE}

GOAL: {goal}

{INSTRUCTION_PLACEHOLDER.format(task=task, agent_output=agent_output)}

{OUTPUT_SCHEMA}"""


def build_judge_with_memory_prompt(task: str, agent_output: str, goal: str, memory_context: str) -> str:
    """Build prompt for memory-assisted judge."""
    return f"""{ROLE}

GOAL: {goal}

{INSTRUCTION_PLACEHOLDER.format(task=task, agent_output=agent_output)}

{MEMORY_CONTEXT.format(memory_context=memory_context)}

{OUTPUT_SCHEMA}"""


# --- DEFAULT GOALS ---
# These can be customized per task type

DEFAULT_GOAL = """Evaluate if the solution correctly solves the CORE requirement of the task.
Focus on functionality, not production-readiness (error handling, logging, etc.)."""


# --- SEMANTIC CLASSIFICATION ---

CLASSIFICATION_ROLE = """You are an expert at categorizing software issues into semantic groups.
Focus on ROOT CAUSES, not symptoms."""

CLASSIFICATION_INSTRUCTION = """ISSUE TO CLASSIFY:
{issue_description}

EXISTING SEMANTIC CATEGORIES:
{existing_categories}

Classify this issue:
1. If it matches an existing category, return that category's exact name
2. If no match, propose a NEW category name and description

Examples of good category names (root causes):
- "SQL Injection Vulnerability" (not "query error")
- "Missing Input Validation" (not "crash on bad input")
- "Race Condition" (not "intermittent failure")
- "Weak Cryptography" (not "security issue")
- "Path Traversal Vulnerability" (not "file access error")
"""

CLASSIFICATION_OUTPUT = """Return:
1. category_name: Name of the semantic category (existing or new)
2. category_description: Description of this category (especially if new)
3. is_new_category: true if proposing a new category, false if using existing
4. reasoning: Why this issue belongs to this category"""


def build_classification_prompt(issue_description: str, existing_categories: list[dict]) -> str:
    """Build prompt for classifying an issue into a semantic category."""
    if existing_categories:
        categories_str = "\n".join([
            f"- {cat['name']}: {cat['description']}"
            for cat in existing_categories
        ])
    else:
        categories_str = "(No existing categories yet - propose a new one)"

    return f"""{CLASSIFICATION_ROLE}

{CLASSIFICATION_INSTRUCTION.format(
    issue_description=issue_description,
    existing_categories=categories_str
)}

{CLASSIFICATION_OUTPUT}"""
   