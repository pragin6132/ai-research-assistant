"""
Mock-based test for Step 6G: the `report` node wired to `ReportGenerator`.

No real LLM/network calls are made: a fake `ReportGenerator`-compatible
double is injected into `build_research_graph()`.

Run from backend/:
    python ../tests/test_graph_report_mock.py
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.agents.report_generator import ReportGeneratorError  # noqa: E402
from app.graph.research_graph import build_research_graph  # noqa: E402
from app.graph.state import create_initial_state  # noqa: E402
from app.schemas.models import (  # noqa: E402
    Finding,
    ResearchPlan,
    ResearchReport,
    SearchQuery,
    SearchResult,
)

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
    def __init__(self, findings: list[Finding]):
        self.findings = findings

    def summarize(self, search_results, main_topic=None, sub_topics=None):
        return self.findings


class FakeReportGenerator:
    def __init__(self, report: ResearchReport | None = None, exception: Exception | None = None):
        self.report = report
        self.exception = exception
        self.calls: list[tuple] = []

    def generate_report(self, validated_findings, main_topic=None, response_language="English"):
        self.calls.append((validated_findings, main_topic, response_language))
        if self.exception is not None:
            raise self.exception
        return self.report


def _sample_plan_results_and_finding():
    plan = ResearchPlan(
        original_prompt="Research solid-state batteries.",
        main_topic="Solid-state batteries",
        sub_topics=["Materials"],
        search_queries=[SearchQuery(query="solid-state battery materials", purpose="materials research")],
    )
    results = [
        SearchResult(title="Materials Overview", url="https://example.com/materials", content="details", relevance_score=0.9)
    ]
    finding = Finding(
        topic="Solid-state battery materials",
        summary="Ceramic electrolytes are the leading approach.",
        key_points=["Ceramic offers high conductivity"],
        sources=["https://example.com/materials"],
    )
    return plan, results, finding


def test_1_report_success():
    print("\n=== Test 1: report node assembles the final_report from validated_findings ===")
    plan, results, finding = _sample_plan_results_and_finding()
    expected_report = ResearchReport(
        title="Solid-State Battery Research Overview",
        executive_summary="Ceramic electrolytes are the leading materials approach.",
        key_findings=["Ceramic electrolytes offer high conductivity"],
        findings=[finding],
        sources=["https://example.com/materials"],
    )
    report_generator = FakeReportGenerator(report=expected_report)

    graph = build_research_graph(
        query_planner=FakeQueryPlanner(plan),
        tavily_service=FakeTavilyService(results),
        summarizer=FakeSummarizer([finding]),
        report_generator=report_generator,
    )
    final_state = graph.invoke(create_initial_state("Research solid-state batteries."))

    check(len(report_generator.calls) == 1, "ReportGenerator.generate_report was called exactly once")
    passed_findings, passed_main_topic, passed_language = report_generator.calls[0]
    check(passed_findings == [finding], "the exact validated_findings were passed to the report generator")
    check(passed_main_topic == "Solid-state batteries", "main_topic from the research_plan was passed through")
    check(passed_language == "English", "the detected response language was passed to the report generator")
    check(final_state["final_report"] == expected_report, "final_report in state matches what the generator produced")
    check(final_state["errors"] == [], "no errors were recorded on the happy path")


def test_2_report_generator_failure_is_recorded():
    print("\n=== Test 2: ReportGenerator failure is recorded; final_report stays None ===")
    plan, results, finding = _sample_plan_results_and_finding()
    report_generator = FakeReportGenerator(
        exception=ReportGeneratorError("LLM returned invalid JSON output")
    )

    graph = build_research_graph(
        query_planner=FakeQueryPlanner(plan),
        tavily_service=FakeTavilyService(results),
        summarizer=FakeSummarizer([finding]),
        report_generator=report_generator,
    )
    final_state = graph.invoke(create_initial_state("Research solid-state batteries."))

    check(final_state["final_report"] is None, "final_report stays None when report generation fails")
    check(len(final_state["errors"]) == 1, "exactly one error was recorded")
    check("report:" in final_state["errors"][0], "the error identifies the report node")


def test_3_no_validated_findings_skips_llm_call():
    print("\n=== Test 3: no validated_findings skips the report generator call entirely ===")
    plan = ResearchPlan(
        original_prompt="Research a topic with no coverage.",
        main_topic="A topic with no coverage",
        sub_topics=[],
        search_queries=[SearchQuery(query="a topic with no coverage", purpose="probe")],
    )
    report_generator = FakeReportGenerator(report=None)

    graph = build_research_graph(
        query_planner=FakeQueryPlanner(plan),
        tavily_service=FakeTavilyService([]),  # no search results -> no findings -> no validated_findings
        summarizer=FakeSummarizer([]),
        report_generator=report_generator,
    )
    final_state = graph.invoke(create_initial_state("Research a topic with no coverage."))

    check(report_generator.calls == [], "ReportGenerator was never called when there are no validated_findings")
    check(final_state["final_report"] is None, "final_report is None")
    check(
        any("report: no validated_findings" in e for e in final_state["errors"]),
        "an error explains why no report was generated",
    )


if __name__ == "__main__":
    test_1_report_success()
    test_2_report_generator_failure_is_recorded()
    test_3_no_validated_findings_skips_llm_call()

    print("\n=== Summary ===")
    print(f"Passed: {len(PASSED)}")
    print(f"Failed: {len(FAILED)}")

    if FAILED:
        print("\nFailed checks:")
        for item in FAILED:
            print(f"  - {item}")
        sys.exit(1)
    else:
        print("All report-node wiring tests passed.")
