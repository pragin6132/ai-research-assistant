"""
Test proving the LangGraph research graph (Step 6A) builds and executes.

This does NOT exercise any real logic (no LLM calls, no Tavily calls,
no summarization/validation) — every node is still a placeholder
pass-through. This just confirms:
  1. `build_research_graph()` compiles without error.
  2. Invoking it with an initial state runs all 8 nodes in the expected
     sequential order, START -> ... -> END.
  3. The state that comes out the other end still carries the fields it
     went in with (since every node is currently a no-op).

Run from backend/:
    python ../tests/test_research_graph.py
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.graph.research_graph import build_research_graph  # noqa: E402
from app.graph.state import create_initial_state  # noqa: E402

PASSED = []
FAILED = []


def check(condition: bool, description: str):
    if condition:
        PASSED.append(description)
        print(f"  PASS: {description}")
    else:
        FAILED.append(description)
        print(f"  FAIL: {description}")


def test_graph_builds_and_runs():
    print("\n=== Test: graph builds and executes the full placeholder flow ===")

    graph = build_research_graph()
    check(graph is not None, "build_research_graph() returned a compiled graph")

    initial_state = create_initial_state("What are the latest advances in solid-state batteries?")
    check(initial_state["user_prompt"].startswith("What are the latest advances"), "initial state carries the user_prompt")
    check(initial_state["research_plan"] is None, "initial state's research_plan starts as None")
    check(initial_state["errors"] == [], "initial state's errors starts empty")
    check(initial_state["retry_count"] == 0, "initial state's retry_count starts at 0")

    final_state = graph.invoke(initial_state)

    check(final_state is not None, "graph.invoke() returned a final state")
    check(
        final_state["user_prompt"] == initial_state["user_prompt"],
        "user_prompt survived the full pass-through flow unchanged",
    )
    check(final_state["research_plan"] is None, "research_plan is still None (no node implements planning yet)")
    check(final_state["search_queries"] == [], "search_queries is still empty (no node implements it yet)")
    check(final_state["search_results"] == [], "search_results is still empty (no node implements it yet)")
    check(final_state["findings"] == [], "findings is still empty (no node implements it yet)")
    check(final_state["validated_findings"] == [], "validated_findings is still empty (no node implements it yet)")
    check(final_state["final_report"] is None, "final_report is still None (no node implements reporting yet)")
    check(final_state["errors"] == [], "errors is still empty (no node raised anything)")
    check(final_state["retry_count"] == 0, "retry_count is still 0 (no node incremented it)")

    # Confirm all 8 expected nodes exist in the compiled graph.
    expected_nodes = {
        "analyze", "plan", "search", "process",
        "summarize", "validate", "quality_check", "report",
    }
    graph_nodes = set(graph.get_graph().nodes.keys()) - {"__start__", "__end__"}
    check(expected_nodes.issubset(graph_nodes), "all 8 expected nodes are present in the compiled graph")


if __name__ == "__main__":
    test_graph_builds_and_runs()

    print("\n=== Summary ===")
    print(f"Passed: {len(PASSED)}")
    print(f"Failed: {len(FAILED)}")

    if FAILED:
        print("\nFailed checks:")
        for item in FAILED:
            print(f"  - {item}")
        sys.exit(1)
    else:
        print("Research graph test passed.")
