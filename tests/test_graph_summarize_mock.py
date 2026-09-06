"""
Mock-based test for Step 6D: the `summarize` node wired to `Summarizer`.

No real LLM/network calls are made: a fake `Summarizer`-compatible
double is injected into `build_research_graph()`.

Run from backend/:
    python ../tests/test_graph_summarize_mock.py
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.agents.summarizer import SummarizerError  # noqa: E402
from app.graph.research_graph import build_research_graph  # noqa: E402
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


class FakeQueryPlanner:
    def __init__(self, plan: ResearchPlan):
        self.plan = plan

    def create_plan(self, prompt: str) -> ResearchPlan:
        return self.plan


class FakeTavilyService:
    def __init__(self, results: list[SearchResult]):
        self.results = results

    def search(self, query: str) -> list[SearchResult]:
        return self.results


class FakeSummarizer:
    def __init__(self, findings: list[Finding] | None = None, exception: Exception | None = None):
        self.findings = findings or []
        self.exception = exception
        self.calls: list[tuple] = []

    def summarize(self, search_results, main_topic=None, sub_topics=None):
        self.calls.append((search_results, main_topic, sub_topics))
        if self.exception is not None:
            raise self.exception
        return self.findings


def _sample_plan_and_results():
    plan = ResearchPlan(
        original_prompt="Research solid-state batteries.",
        main_topic="Solid-state batteries",
        sub_topics=["Materials"],
        search_queries=[SearchQuery(query="solid-state battery materials", purpose="materials research")],
    )
    results = [
        SearchResult(title="Materials Overview", url="https://example.com/materials", content="details", relevance_score=0.9)
    ]
    return plan, results


def test_1_summarize_success():
    print("\n=== Test 1: summarize node produces findings from cleaned search_results ===")
    plan, results = _sample_plan_and_results()
    finding = Finding(
        topic="Solid-state battery materials",
        summary="Ceramic electrolytes are the leading approach.",
        key_points=["Ceramic offers high conductivity"],
        sources=["https://example.com/materials"],
    )
    summarizer = FakeSummarizer(findings=[finding])

    graph = build_research_graph(
        query_planner=FakeQueryPlanner(plan),
        tavily_service=FakeTavilyService(results),
        summarizer=summarizer,
    )
    final_state = graph.invoke(create_initial_state("Research solid-state batteries."))

    check(len(summarizer.calls) == 1, "Summarizer.summarize was called exactly once")
    passed_results, passed_main_topic, passed_sub_topics = summarizer.calls[0]
    check(passed_main_topic == "Solid-state batteries", "main_topic from the research_plan was passed to the summarizer")
    check(passed_sub_topics == ["Materials"], "sub_topics from the research_plan were passed to the summarizer")
    check(len(final_state["findings"]) == 1, "one Finding is present in the final state")
    check(final_state["findings"][0].topic == "Solid-state battery materials", "the finding's data matches what the summarizer produced")
    check(final_state["errors"] == [], "no errors were recorded on the happy path")


def test_2_summarizer_failure_is_recorded():
    print("\n=== Test 2: Summarizer failure is recorded in errors, findings stays empty ===")
    plan, results = _sample_plan_and_results()
    summarizer = FakeSummarizer(exception=SummarizerError("LLM returned invalid JSON output"))

    graph = build_research_graph(
        query_planner=FakeQueryPlanner(plan),
        tavily_service=FakeTavilyService(results),
        summarizer=summarizer,
    )
    final_state = graph.invoke(create_initial_state("Research solid-state batteries."))

    check(final_state["findings"] == [], "findings stays empty when summarization fails")
    check(len(final_state["errors"]) == 1, "exactly one error was recorded")
    check("summarize:" in final_state["errors"][0], "the error identifies the summarize node")


if __name__ == "__main__":
    test_1_summarize_success()
    test_2_summarizer_failure_is_recorded()

    print("\n=== Summary ===")
    print(f"Passed: {len(PASSED)}")
    print(f"Failed: {len(FAILED)}")

    if FAILED:
        print("\nFailed checks:")
        for item in FAILED:
            print(f"  - {item}")
        sys.exit(1)
    else:
        print("All summarize-node wiring tests passed.")
