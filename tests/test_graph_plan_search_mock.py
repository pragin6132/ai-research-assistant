"""
Mock-based test for Step 6B: the `plan` and `search` nodes wired to the
real `QueryPlanner` and `TavilySearchService`.

No real LLM/Tavily calls are made: fake `QueryPlanner`/`TavilySearchService`
doubles are injected into `build_research_graph()`.

Run from backend/:
    python ../tests/test_graph_plan_search_mock.py
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.agents.query_planner import QueryPlannerError  # noqa: E402
from app.graph.research_graph import build_research_graph  # noqa: E402
from app.graph.state import create_initial_state  # noqa: E402
from app.schemas.models import ResearchPlan, SearchQuery, SearchResult  # noqa: E402
from app.services.tavily_service import TavilyServiceError  # noqa: E402

PASSED = []
FAILED = []


def check(condition: bool, description: str):
    if condition:
        PASSED.append(description)
        print(f"  PASS: {description}")
    else:
        FAILED.append(description)
        print(f"  FAIL: {description}")


class FakeQueryPlanner:
    """Test double for QueryPlanner: exposes only .create_plan(prompt)."""

    def __init__(self, plan: ResearchPlan | None = None, exception: Exception | None = None):
        self.plan = plan
        self.exception = exception
        self.calls: list[str] = []

    def create_plan(self, prompt: str) -> ResearchPlan:
        self.calls.append(prompt)
        if self.exception is not None:
            raise self.exception
        return self.plan


class FakeTavilyService:
    """Test double for TavilySearchService: exposes only .search(query)."""

    def __init__(self, results_by_query: dict[str, list[SearchResult]] | None = None,
                 exception_for_query: dict[str, Exception] | None = None):
        self.results_by_query = results_by_query or {}
        self.exception_for_query = exception_for_query or {}
        self.calls: list[str] = []

    def search(self, query: str) -> list[SearchResult]:
        self.calls.append(query)
        if query in self.exception_for_query:
            raise self.exception_for_query[query]
        return self.results_by_query.get(query, [])


def _sample_plan() -> ResearchPlan:
    return ResearchPlan(
        original_prompt="Research solid-state batteries.",
        main_topic="Solid-state batteries",
        sub_topics=["Materials", "Manufacturing challenges"],
        search_queries=[
            SearchQuery(query="solid-state battery materials 2026", purpose="Find current materials research"),
            SearchQuery(query="solid-state battery manufacturing challenges", purpose="Understand production hurdles"),
        ],
    )


def test_1_plan_and_search_succeed():
    print("\n=== Test 1: plan + search succeed end-to-end through the graph ===")
    plan = _sample_plan()
    planner = FakeQueryPlanner(plan=plan)
    tavily = FakeTavilyService(
        results_by_query={
            "solid-state battery materials 2026": [
                SearchResult(title="Materials Overview", url="https://example.com/materials", content="...", relevance_score=0.9)
            ],
            "solid-state battery manufacturing challenges": [
                SearchResult(title="Manufacturing Overview", url="https://example.com/manufacturing", content="...", relevance_score=0.8)
            ],
        }
    )

    graph = build_research_graph(query_planner=planner, tavily_service=tavily)
    initial_state = create_initial_state("Research solid-state batteries.")
    final_state = graph.invoke(initial_state)

    check(planner.calls == ["Research solid-state batteries."], "QueryPlanner.create_plan was called with the user_prompt")
    check(final_state["research_plan"] == plan, "research_plan in final state matches the planner's output")
    check(len(final_state["search_queries"]) == 2, "search_queries were copied from the plan")
    check(sorted(tavily.calls) == sorted(q.query for q in plan.search_queries), "TavilySearchService.search was called once per planned query")
    check(len(final_state["search_results"]) == 2, "search_results aggregates results from all queries")
    check(final_state["errors"] == [], "no errors were recorded on the happy path")


def test_2_planner_failure_is_recorded_and_search_skips_gracefully():
    print("\n=== Test 2: QueryPlanner failure is recorded; search node handles the empty plan gracefully ===")
    planner = FakeQueryPlanner(exception=QueryPlannerError("LLM returned invalid JSON output"))
    tavily = FakeTavilyService()

    graph = build_research_graph(query_planner=planner, tavily_service=tavily)
    initial_state = create_initial_state("Research quantum computing.")
    final_state = graph.invoke(initial_state)

    check(final_state["research_plan"] is None, "research_plan stays None when planning fails")
    check(final_state["search_queries"] == [], "search_queries stays empty when planning fails")
    check(len(final_state["errors"]) == 2, "both the plan failure and the resulting empty-query search are recorded")
    check(any("plan:" in e for e in final_state["errors"]), "an error mentioning the plan node is present")
    check(any("search:" in e for e in final_state["errors"]), "an error mentioning the search node is present")
    check(tavily.calls == [], "TavilySearchService.search was never called (no queries to run)")


def test_3_partial_search_failure_keeps_successful_results():
    print("\n=== Test 3: one query failing doesn't lose results from the other queries ===")
    plan = _sample_plan()
    planner = FakeQueryPlanner(plan=plan)
    tavily = FakeTavilyService(
        results_by_query={
            "solid-state battery manufacturing challenges": [
                SearchResult(title="Manufacturing Overview", url="https://example.com/manufacturing", content="...", relevance_score=0.8)
            ],
        },
        exception_for_query={
            "solid-state battery materials 2026": TavilyServiceError("Tavily API error: 503"),
        },
    )

    graph = build_research_graph(query_planner=planner, tavily_service=tavily)
    initial_state = create_initial_state("Research solid-state batteries.")
    final_state = graph.invoke(initial_state)

    check(len(final_state["search_results"]) == 1, "the successful query's result was still kept")
    check(final_state["search_results"][0].title == "Manufacturing Overview", "the surviving result is the one that succeeded")
    check(len(final_state["errors"]) == 1, "exactly one error was recorded, for the failed query")
    check("materials 2026" in final_state["errors"][0], "the error identifies which query failed")


if __name__ == "__main__":
    test_1_plan_and_search_succeed()
    test_2_planner_failure_is_recorded_and_search_skips_gracefully()
    test_3_partial_search_failure_keeps_successful_results()

    print("\n=== Summary ===")
    print(f"Passed: {len(PASSED)}")
    print(f"Failed: {len(FAILED)}")

    if FAILED:
        print("\nFailed checks:")
        for item in FAILED:
            print(f"  - {item}")
        sys.exit(1)
    else:
        print("All plan/search node wiring tests passed.")
