"""
Mutation tests for the frozen-memory audit (round-3 review).

These tests prove, against a live Neo4j, that the hardened audit detects or
blocks every mutation class the reviewer identified:

  1. In-place PROPERTY mutation via raw Cypher (bypassing all wrapper
     methods) changes the full-state fingerprint.        [detection]
  2. Topology mutation via raw Cypher changes it too.    [detection]
  3. Guarded wrapper methods raise under freeze().       [prevention]
  4. Raw write-clause Cypher via gm.driver raises under freeze().
                                                          [prevention]
  5. Read-only Cypher (MATCH, vector queries) still works under freeze().
  6. A FrozenMemoryViolation aborts an audited_eval run instead of being
     recorded as a per-item error.                        [fatality]

Run:  venv/bin/python -m pytest tests/test_audit_mutation.py -v
Requires a running Neo4j (uses a disposable graph; clears it).
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from graph_manager import GraphManager, FrozenMemoryViolation  # noqa: E402
from models import Policy, Attempt  # noqa: E402


@pytest.fixture()
def gm():
    g = GraphManager()
    g.clear_all()
    # tiny fixed graph: one policy, one attempt, one edge
    p = Policy(description="test policy")
    p.embedding = [1.0] + [0.0] * 1535
    a = Attempt(agent_output="test output", is_successful=True,
                reasoning="test reasoning")
    a.embedding = [1.0] + [0.0] * 1535
    g.create_policy(p)
    g.create_attempt(a)
    g.link_attempt_satisfies_policy(a.id, p.id)
    yield g
    g.clear_all()
    g.close()


def test_property_mutation_changes_fingerprint(gm):
    """Flipping a stored label IN PLACE (no topology change) must change
    the fingerprint. This was the reviewer's core gap: the old topology
    fingerprint passed this mutation."""
    before = gm.snapshot()
    with gm.driver.session() as s:
        s.run("MATCH (a:Attempt) SET a.is_successful = NOT a.is_successful")
    after = gm.snapshot()
    assert before["total_nodes"] == after["total_nodes"]      # same topology
    assert before["total_edges"] == after["total_edges"]
    assert before["fingerprint"] != after["fingerprint"]      # yet detected


def test_topology_mutation_changes_fingerprint(gm):
    before = gm.snapshot()
    with gm.driver.session() as s:
        s.run("CREATE (:Attempt {id: 'rogue', agent_output: 'x'})")
    after = gm.snapshot()
    assert before["fingerprint"] != after["fingerprint"]


def test_wrapper_write_raises_under_freeze(gm):
    a = Attempt(agent_output="new", is_successful=False, reasoning="r")
    a.embedding = [1.0] + [0.0] * 1535
    with gm.freeze():
        with pytest.raises(FrozenMemoryViolation):
            gm.create_attempt(a)


def test_raw_driver_write_raises_under_freeze(gm):
    """The reviewer's bypass: code holding gm.driver issues raw write
    Cypher. Under the hardened freeze this must raise, not silently write."""
    with gm.freeze():
        with pytest.raises(FrozenMemoryViolation):
            with gm.driver.session() as s:
                s.run("CREATE (:Attempt {id: 'bypass'})")
        with pytest.raises(FrozenMemoryViolation):
            gm.driver.execute_query("MATCH (a:Attempt) SET a.reasoning = 'x'")


def test_reads_still_work_under_freeze(gm):
    with gm.freeze():
        # plain read
        with gm.driver.session() as s:
            n = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        assert n == 2
        # retrieval path used by the judge (vector index read)
        res = gm.find_similar_attempts([1.0] + [0.0] * 1535, top_k=1)
        assert isinstance(res, list)
        # snapshot itself must be possible while frozen
        snap = gm.snapshot()
        assert "fingerprint" in snap


def test_violation_aborts_audited_eval(gm):
    """A frozen-memory violation must abort the run, not be swallowed as a
    per-item NaN."""
    import pandas as pd
    import exp_common as E

    test_df = pd.DataFrame([{
        "question": "q1", "grading_notes": "notes", "response": "resp",
        "target": "pass", "topic": "t",
    }])

    def violating_predictor(sample):
        # attempt a raw write during the frozen evaluation
        with gm.driver.session() as s:
            s.run("CREATE (:Attempt {id: 'leak'})")
        return True, {}

    with pytest.raises(FrozenMemoryViolation):
        E.audited_eval(test_df, violating_predictor, gm,
                       allow_writes=False, desc="must-abort")


def test_freeze_restores_driver_and_methods(gm):
    with gm.freeze():
        pass
    # after exiting, writes work again
    a = Attempt(agent_output="post-freeze", is_successful=True, reasoning="r")
    a.embedding = [1.0] + [0.0] * 1535
    gm.create_attempt(a)
    with gm.driver.session() as s:
        c = s.run("MATCH (n:Attempt) RETURN count(n) AS c").single()["c"]
    assert c == 2
