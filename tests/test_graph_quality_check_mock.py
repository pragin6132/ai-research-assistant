"""
Mock-based test for Step 6F: the `quality_check` node's retry routing.

Covers three cases:
  1. Pass: validated_findings is sufficient -> routes straight to report.
  2. Retry: validated_findings is empty, under the retry cap -> routes
     back to plan, incrementing retry_count.
  3. Max-retry: validated_findings is empty, retry cap already reached
     -> routes to report anyway (with an explanatory error), instead of
     looping forever.

`quality_check_node` is tested directly (it returns a LangGraph
`Command`), and a full graph run proves the actual loop-then-stop
behavior end-to-end with fake plan/search/summarize services that never
produce a validated finding.

Run from backend/:
    python ../tests/test_graph_quality_check_mock.py
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.graph.research_graph import (  # noqa: E402
    MAX_RETRY_COUNT,
    build_research_graph,
    quality_check_node,
)
from app.graph.state import create_initial_state  # noqa: E402
from app.schemas.models import Finding, ResearchPlan, SearchQuery, SearchResult  # noqa: E402

PASSED = []
FAILED = []


def check(condition: bool, description: str):
    if condition:
        PASSED.append(description)
        print(f"  PASS: {description}")
    else:
        FAILED.append(description)
        print(f"  FAIL: {description}")


def _finding() -> Finding:
    return Finding(
        topic="Battery materials",
        summary="Ceramic electrolytes are the leading approach.",
        key_points=["High conductivity"],
        sources=["https://example.com/materials"],
    )


def test_1_pass_case_routes_to_report():
    print("\n=== Test 1 (pass): sufficient validated_findings routes to 'report' ===")
    state = {"validated_findings": [_finding()], "retry_count": 0, "errors": []}
    command = quality_check_node(state)

    check(command.goto == "report", "routes to 'report' when there is at least one validated finding")
    check(command.update["errors"] == [], "no error is recorded on the pass case")


def test_2_retry_case_routes_to_plan_and_increments():
    print("\n=== Test 2 (retry): insufficient findings under the cap routes back to 'plan' ===")
    state = {"validated_findings": [], "retry_count": 0, "errors": []}
    command = quality_check_node(state)

    check(command.goto == "plan", "routes back to 'plan' when findings are insufficient and under the cap")
    check(command.update["retry_count"] == 1, "retry_count was incremented by exactly 1")
    check(len(command.update["errors"]) == 1, "a retry explanation was recorded")
    check("retrying research" in command.update["errors"][0], "the error message explains the retry")


def test_3_max_retry_case_routes_to_report_anyway():
    print("\n=== Test 3 (max-retry): cap reached routes to 'report' instead of retrying again ===")
    state = {"validated_findings": [], "retry_count": MAX_RETRY_COUNT, "errors": []}
    command = quality_check_node(state)

    check(command.goto == "report", "routes to 'report' once the retry cap has already been reached")
    check(
        "retry_count" not in command.update,
        "retry_count is not incremented further past the cap",
    )
    check(len(command.update["errors"]) == 1, "a max-retries explanation was recorded")
    check("max retries reached" in command.update["errors"][0], "the error message explains the cap was hit")


class FlakyQueryPlanner:
    """Always produces the same plan; used to drive the retry loop end-to-end."""

    def __init__(self, plan: ResearchPlan):
        self.plan = plan
        self.calls = 0

    def create_plan(self, prompt: str) -> ResearchPlan:
        self.calls += 1
        return self.plan


class EmptyTavilyService:
    """Always returns no results, so summarize/validate never produce a finding."""

    def __init__(self):
        self.calls = 0

    def search(self, query: str) -> list[SearchResult]:
        self.calls += 1
        return []


class NoOpSummarizer:
    """Used only so the graph doesn't need a real LLMManager; never called with results."""

    def __init__(self):
        self.calls = 0

    def summarize(self, search_results, main_topic=None, sub_topics=None):
        self.calls += 1
        return []


def test_4_end_to_end_loop_stops_at_cap():
    print("\n=== Test 4 (end-to-end): the graph retries up to the cap, then proceeds to report ===")
    plan = ResearchPlan(
        original_prompt="Research a topic with no coverage.",
        main_topic="A topic with no coverage",
        sub_topics=["Sub-topic"],
        search_queries=[SearchQuery(query="a topic with no coverage", purpose="probe")],
    )
    planner = FlakyQueryPlanner(plan)
    tavily = EmptyTavilyService()
    summarizer = NoOpSummarizer()

    graph = build_research_graph(query_planner=planner, tavily_service=tavily, summarizer=summarizer)
    final_state = graph.invoke(create_initial_state("Research a topic with no coverage."))

    check(final_state["retry_count"] == MAX_RETRY_COUNT, f"retry_count stopped exactly at the cap ({MAX_RETRY_COUNT})")
    check(planner.calls == MAX_RETRY_COUNT + 1, "plan ran once initially plus once per retry (no more, no less)")
    check(final_state["validated_findings"] == [], "validated_findings is still empty (nothing was fabricated)")
    check(
        sum("retrying research" in e for e in final_state["errors"]) == MAX_RETRY_COUNT,
        "exactly MAX_RETRY_COUNT retry messages were recorded",
    )
    check(
        any("max retries reached" in e for e in final_state["errors"]),
        "a final max-retries message was recorded before proceeding",
    )


if __name__ == "__main__":
    test_1_pass_case_routes_to_report()
    test_2_retry_case_routes_to_plan_and_increments()
    test_3_max_retry_case_routes_to_report_anyway()
    test_4_end_to_end_loop_stops_at_cap()

    print("\n=== Summary ===")
    print(f"Passed: {len(PASSED)}")
    print(f"Failed: {len(FAILED)}")

    if FAILED:
        print("\nFailed checks:")
        for item in FAILED:
            print(f"  - {item}")
        sys.exit(1)
    else:
        print("All quality_check retry-routing tests passed.")
